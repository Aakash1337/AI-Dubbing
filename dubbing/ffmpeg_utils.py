"""Thin, well-logged wrappers around the system ``ffmpeg`` / ``ffprobe``.

Three jobs:
  * probe a video's exact duration (so the dub track is built to match),
  * time-stretch a single clip with ``atempo`` (pitch-preserving), chaining
    instances so factors above 2.0 still work, and
  * mux: copy the video stream untouched, encode the new English track as AAC,
    apply single-pass ``loudnorm``, and mark it the single default audio track.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .config import Config

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


class FFmpegError(RuntimeError):
    pass


def ensure_tools() -> None:
    """Raise early with a friendly message if ffmpeg/ffprobe aren't on PATH."""
    missing = [t for t in (FFMPEG, FFPROBE) if shutil.which(t) is None]
    if missing:
        raise FFmpegError(
            f"Required tool(s) not found on PATH: {', '.join(missing)}. "
            "Install ffmpeg (it bundles ffprobe) and re-run."
        )


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-12:])
        raise FFmpegError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{tail}")
    return proc


def probe_duration(video: Path) -> float:
    """Return the VIDEO duration in seconds.

    The dub track is built to this length, so we must match the PICTURE — not
    the (about-to-be-dropped) original audio, which can be slightly longer and
    would otherwise leave a frozen-frame tail. Order: video stream v:0, then the
    container format, then the longest stream as a last resort.
    """
    # 1) the video stream itself (authoritative for picture length)
    out = _run([
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]).stdout.strip()
    dur = _first_float(out)
    if dur and dur > 0:
        return dur

    # 2) container format duration
    out = _run([
        FFPROBE, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]).stdout.strip()
    dur = _first_float(out)
    if dur and dur > 0:
        return dur

    # 3) longest stream as a last resort
    out = _run([
        FFPROBE, "-v", "error",
        "-show_entries", "stream=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]).stdout
    durs = [float(x) for x in out.split() if _is_float(x)]
    if durs:
        return max(durs)
    raise FFmpegError(f"Could not determine duration of {video}")


def atempo_chain(factor: float) -> str:
    """Build an ``atempo=...`` filter string for an arbitrary speed factor.

    Each atempo instance is kept within the well-behaved [0.5, 2.0] range and
    chained, so e.g. 3.0 -> ``atempo=1.732,atempo=1.732``. For our capped use
    (<=1.5) this is a single instance.
    """
    if factor <= 0:
        raise ValueError("atempo factor must be > 0")
    parts: list[float] = []
    remaining = factor
    while remaining > 2.0:
        parts.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        parts.append(0.5)
        remaining /= 0.5
    parts.append(remaining)
    return ",".join(f"atempo={p:.6f}" for p in parts)


def time_stretch_file(src: Path, dst: Path, factor: float, sample_rate: int) -> None:
    """Write ``dst`` = ``src`` sped up by ``factor`` (pitch preserved).

    factor > 1 makes it shorter/faster. factor ~1.0 is a passthrough copy.
    """
    if abs(factor - 1.0) < 1e-3:
        # No meaningful change; just re-encode to the target rate.
        _run([FFMPEG, "-y", "-i", str(src), "-ar", str(sample_rate),
              "-c:a", "pcm_f32le", str(dst)])
        return
    _run([
        FFMPEG, "-y", "-i", str(src),
        "-filter:a", atempo_chain(factor),
        "-ar", str(sample_rate),
        "-c:a", "pcm_f32le",   # 32-bit float keeps full precision through the round-trip
        str(dst),
    ])


def mux(video: Path, dub_wav: Path, out: Path, cfg: Config,
        extra_metadata: Optional[dict[str, str]] = None) -> None:
    """Copy the video stream, attach the AAC-encoded English track, drop the
    original audio, and mark the new track default.

    Single-pass ``loudnorm`` is applied to the new audio. The video is NOT
    re-encoded (``-c:v copy``) so this is fast and lossless on the picture.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    loudnorm = f"loudnorm=I={cfg.loudnorm_i}:TP={cfg.loudnorm_tp}:LRA={cfg.loudnorm_lra}"
    # Atomic write: encode to a temp sibling, rename only on success. A present
    # final file then always means a complete dub (the resume guard depends on it).
    # Keep the real container extension (name.part.mp4) so ffmpeg can pick the muxer.
    tmp = out.with_name(out.stem + ".part" + out.suffix)
    cmd = [
        FFMPEG, "-y",
        "-i", str(video),
        "-i", str(dub_wav),
        "-map", "0:v:0",          # video from source
        "-map", "1:a:0",          # audio from our dub (original audio dropped)
        "-c:v", "copy",           # picture untouched -> fast, lossless
        "-af", loudnorm,
        "-c:a", cfg.aac_codec,
        "-b:a", cfg.audio_bitrate,
        "-ar", str(cfg.sample_rate),
        "-ac", str(cfg.channels),
        "-disposition:a:0", "default",   # the single, default audio track
        "-map_metadata", "0",
        "-movflags", "+faststart",       # web/stream friendly moov atom
        "-shortest",                     # clamp to the picture; no frozen-frame tail
    ]
    if extra_metadata:
        for k, v in extra_metadata.items():
            cmd += ["-metadata", f"{k}={v}"]
    cmd.append(str(tmp))
    try:
        _run(cmd)
        os.replace(str(tmp), str(out))
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def _first_float(text: str) -> Optional[float]:
    """First parseable float in ``text`` (ffprobe may print 'N/A')."""
    for tok in text.split():
        if _is_float(tok):
            return float(tok)
    return None
