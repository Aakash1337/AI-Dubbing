"""Audio DSP: resample, time-fit each clip to its window, and assemble the dub
track by placing clips at their ABSOLUTE timecodes on a silent canvas.

The canvas approach (vs. ffmpeg ``adelay``+``amix`` over hundreds of inputs) is
simpler and bulletproof for sync: a clip's sample offset is ``round(start*sr)``,
full stop, so the assembled track is exactly the source video's length and every
cue lands on its own timestamp regardless of per-cue drift.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from . import ffmpeg_utils


# ── resampling ───────────────────────────────────────────────────────────────
def resample(clip: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample a 1-D float clip from ``src_sr`` to ``dst_sr`` (high quality)."""
    clip = np.asarray(clip, dtype=np.float32).reshape(-1)
    if src_sr == dst_sr or clip.size == 0:
        return clip
    try:
        import torch
        import torchaudio.functional as AF

        t = torch.from_numpy(clip).unsqueeze(0)
        out = AF.resample(t, src_sr, dst_sr).squeeze(0).cpu().numpy()
        return out.astype(np.float32, copy=False)
    except Exception:
        # Last-resort linear interpolation if torchaudio is unavailable.
        n_out = int(round(clip.size * dst_sr / src_sr))
        if n_out <= 0:
            return np.zeros(0, dtype=np.float32)
        x_old = np.linspace(0.0, 1.0, num=clip.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, clip).astype(np.float32)


# ── time-fitting ─────────────────────────────────────────────────────────────
@dataclass
class FitResult:
    clip: np.ndarray
    applied_tempo: float
    needed_tempo: float
    capped: bool
    overrun_seconds: float   # how far the (capped) clip still overruns its window


def fit_clip(clip: np.ndarray, sr: int, window: float,
             max_atempo: float, min_atempo: float = 1.0) -> FitResult:
    """Fit ``clip`` into ``window`` seconds.

    * longer than the window -> sped up via ffmpeg ``atempo`` by
      ``min(needed, max_atempo)``. If the cap bites, the clip stays longer than
      its window; the overrun is reported so the caller can log it.
    * shorter than the window and ``min_atempo`` < 1.0 -> slowed DOWN to fill the
      window (so narration isn't rushed with gaps), but never below
      ``min_atempo`` — a real pause stays partly silent rather than dragging.
    * otherwise -> returned unchanged (the silent canvas supplies trailing pad).
    """
    clip = np.asarray(clip, dtype=np.float32).reshape(-1)
    if clip.size == 0 or window <= 0:
        return FitResult(clip, 1.0, 1.0, False, 0.0)
    clip_len = clip.size / sr if sr else 0.0

    # too long -> speed up (capped, may overrun)
    if clip_len > window + 1e-3:
        needed = clip_len / window
        applied = min(needed, max_atempo)
        stretched = _atempo(clip, sr, applied)
        new_len = stretched.size / sr if sr else 0.0
        capped = needed > max_atempo + 1e-3
        overrun = max(0.0, new_len - window)
        return FitResult(stretched, applied, needed, capped, overrun)

    # too short -> slow down to fill the window, floored at min_atempo
    if min_atempo < 1.0 and clip_len < window - 1e-3:
        needed = clip_len / window                # < 1.0
        applied = max(needed, min_atempo)         # don't over-stretch real pauses
        if applied < 1.0 - 1e-3:
            stretched = _atempo(clip, sr, applied)
            return FitResult(stretched, applied, needed, False, 0.0)

    # fits as-is
    return FitResult(clip, 1.0, _safe_ratio(clip_len, window), False, 0.0)


def _atempo(clip: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """Speed up ``clip`` by ``factor`` using ffmpeg atempo (pitch preserved)."""
    fd_in, p_in = tempfile.mkstemp(suffix=".wav", prefix="dub_in_")
    fd_out, p_out = tempfile.mkstemp(suffix=".wav", prefix="dub_out_")
    os.close(fd_in)
    os.close(fd_out)
    try:
        sf.write(p_in, clip, sr, subtype="FLOAT")
        ffmpeg_utils.time_stretch_file(Path(p_in), Path(p_out), factor, sr)
        out, _ = sf.read(p_out, dtype="float32", always_2d=False)
        if out.ndim > 1:
            out = out.mean(axis=1)
        return out.astype(np.float32, copy=False)
    finally:
        for p in (p_in, p_out):
            try:
                os.remove(p)
            except OSError:
                pass


# ── assembly ─────────────────────────────────────────────────────────────────
def build_canvas(duration_s: float, sr: int) -> np.ndarray:
    """A silent mono float canvas exactly ``duration_s`` long."""
    n = max(0, int(round(duration_s * sr)))
    return np.zeros(n, dtype=np.float32)


def place(canvas: np.ndarray, clip: np.ndarray, start_s: float, sr: int) -> None:
    """Mix ``clip`` into ``canvas`` starting at absolute time ``start_s``.

    Uses additive mixing so any (capped) overrun into the next cue blends rather
    than truncates; the final peak guard prevents clipping. Clips past the canvas
    end are truncated.
    """
    if clip.size == 0:
        return
    offset = int(round(start_s * sr))
    if offset < 0:
        clip = clip[-offset:]
        offset = 0
    if offset >= canvas.size:
        return
    end = min(offset + clip.size, canvas.size)
    n = end - offset
    if n > 0:
        canvas[offset:end] += clip[:n]


def finalize_canvas(canvas: np.ndarray, ceiling: float = 0.99) -> np.ndarray:
    """Guard against clipping from overlapping clips before encode/loudnorm."""
    if canvas.size == 0:
        return canvas
    peak = float(np.max(np.abs(canvas)))
    if peak > ceiling:
        canvas = canvas * (ceiling / peak)
    return canvas.astype(np.float32, copy=False)


def write_wav(path: Path, canvas: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(canvas, dtype=np.float32), sr, subtype="FLOAT")


def _safe_ratio(a: float, b: float) -> float:
    return a / b if b > 0 else 1.0
