"""Self-test for everything that doesn't need the GPU/TTS model.

Exercises the sync-critical DSP + the real ffmpeg atempo/mux round-trips on
synthetic media, so we can trust the pipeline before paying for Chatterbox.

    python selftest.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from dubbing import audio, ffmpeg_utils
from dubbing.config import Config
from dubbing.pairing import episode_key
from dubbing.srt_parser import clean_text, compute_windows, parse_srt

PASS, FAIL = "PASS", "FAIL"
_failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{PASS if cond else FAIL}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        _failures.append(name)


def sine(seconds: float, sr: int, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_episode_key():
    print("episode_key:")
    cases = {
        "01 Understanding Commercial Art": "01",
        "[EN]Mogoon_07": "07",
        "04 Space 1 Perspective": "04",
        "Bonus 1 In-Depth Coloring Insights": "bonus01",
        "[EN]Mogoon_BONUS02": "bonus02",
        "13 Light And Color 2 Diffused Reflection 1": "13",
    }
    for name, want in cases.items():
        got = episode_key(name)
        check(f"{name!r} -> {want}", got == want, f"got {got!r}")


def test_text_clean():
    print("clean_text:")
    check("strips <i> tags", clean_text("<i>hello</i> world") == "hello world")
    check("strips {\\an8}", clean_text("{\\an8}top text") == "top text")
    check("collapses newlines", clean_text("line one\nline two") == "line one line two")


def test_atempo_chain():
    print("atempo_chain:")
    check("1.5x single", ffmpeg_utils.atempo_chain(1.5) == "atempo=1.500000")
    check("3.0x chained", ffmpeg_utils.atempo_chain(3.0) == "atempo=2.000000,atempo=1.500000")
    check("0.25x chained", ffmpeg_utils.atempo_chain(0.25) == "atempo=0.500000,atempo=0.500000")
    # effective factor must equal the product of stages
    for f in (1.25, 1.5, 2.6, 4.0):
        prod = 1.0
        for part in ffmpeg_utils.atempo_chain(f).split(","):
            prod *= float(part.split("=")[1])
        check(f"{f}x product matches", abs(prod - f) < 1e-3, f"product={prod}")


def test_srt():
    print("parse_srt + windows:")
    srt = Path("mogoon/EN SUBS/[EN]Mogoon_01.srt")
    if not srt.exists():
        check("sample srt present", False, f"missing {srt}")
        return
    cues = parse_srt(srt)
    check("parsed cues", len(cues) > 10, f"n={len(cues)}")
    check("starts sorted", all(cues[i].start <= cues[i + 1].start for i in range(len(cues) - 1)))
    check("text is clean", all("\n" not in c.text for c in cues))

    # skip pattern blanks the UI line but keeps the slot
    cues_skipped = parse_srt(srt, skip_patterns=("turn off the subtitles",))
    blanked = sum(1 for c in cues_skipped if not c.speakable)
    check("skip pattern blanks >=1 cue", blanked >= 1, f"blanked={blanked}")

    compute_windows(cues, extend_to_next_cue=True, min_window=0.30, media_duration=1614.5)
    check("windows positive", all(c.window > 0 for c in cues))
    # window equals the REAL gap to the next cue (every positive gap)
    ok = all(abs(cues[i].window - (cues[i + 1].start - cues[i].start)) < 1e-6
             for i in range(len(cues) - 1) if (cues[i + 1].start - cues[i].start) > 0)
    check("window equals real gap to next cue", ok)


def test_windows_degenerate():
    print("compute_windows (degenerate spans):")
    from dubbing.srt_parser import Cue
    # gap 0.1s (< min_window 0.3) must stay 0.1, NOT inflate to 0.3 (would overlap)
    cues = [Cue(1, 0.0, 0.05, "a"), Cue(2, 0.1, 0.2, "b")]
    compute_windows(cues, extend_to_next_cue=True, min_window=0.3, media_duration=10.0)
    check("small gap not inflated", abs(cues[0].window - 0.1) < 1e-9, f"{cues[0].window}")
    # overlapping cues (same start) -> degenerate <=0 -> min_window floor
    cues2 = [Cue(1, 1.0, 1.5, "a"), Cue(2, 1.0, 2.0, "b")]
    compute_windows(cues2, extend_to_next_cue=True, min_window=0.3, media_duration=10.0)
    check("degenerate span uses min_window", abs(cues2[0].window - 0.3) < 1e-9, f"{cues2[0].window}")


def test_cache():
    print("CueCache round-trip (the .tmp-format bug):")
    from dubbing.cache import CueCache
    with tempfile.TemporaryDirectory() as td:
        cache = CueCache(Path(td), "01", {"voice": None, "exaggeration": 0.5})
        clip = sine(0.3, 24000)
        check("miss before put", cache.get(3, "hello") is None)
        cache.put(3, "hello", clip, 24000)        # this raised before the fix
        got = cache.get(3, "hello")
        check("hit after put", got is not None)
        if got is not None:
            data, sr = got
            check("sr preserved", sr == 24000)
            check("float samples preserved", data.shape == clip.shape and np.allclose(data, clip, atol=1e-6))
        check("different text -> miss", cache.get(3, "world") is None)


def test_resample():
    print("resample:")
    clip = sine(1.0, 24000)
    out = audio.resample(clip, 24000, 48000)
    check("24k->48k ~doubles length", abs(out.size - 48000) <= 64, f"len={out.size}")


def test_fit_clip():
    print("fit_clip (real ffmpeg atempo):")
    sr = 48000
    # too long: 2.0s into a 1.0s window, cap 1.5 -> capped, ~1.333s, overruns ~0.333
    fit = audio.fit_clip(sine(2.0, sr), sr, window=1.0, max_atempo=1.5)
    check("capped flag set", fit.capped is True)
    check("applied tempo == cap", abs(fit.applied_tempo - 1.5) < 1e-6, f"{fit.applied_tempo}")
    check("fitted length ~ 2.0/1.5", abs(fit.clip.size / sr - (2.0 / 1.5)) < 0.05,
          f"{fit.clip.size / sr:.3f}s")
    check("overrun ~0.333", abs(fit.overrun_seconds - 0.333) < 0.05, f"{fit.overrun_seconds:.3f}")

    # fits within cap: 1.4s into 1.0s window -> tempo 1.4, not capped, ~1.0s
    fit2 = audio.fit_clip(sine(1.4, sr), sr, window=1.0, max_atempo=1.5)
    check("not capped under cap", fit2.capped is False)
    check("fitted to ~window", abs(fit2.clip.size / sr - 1.0) < 0.05, f"{fit2.clip.size / sr:.3f}s")

    # shorter than window, fill OFF (min_atempo=1.0) -> untouched, tempo 1.0
    short = sine(0.5, sr)
    fit3 = audio.fit_clip(short, sr, window=1.0, max_atempo=1.5)
    check("short clip unchanged (fill off)", fit3.applied_tempo == 1.0 and fit3.clip.size == short.size)

    # shorter than window, fill ON: 0.5s into 1.0s window, floor 0.7 -> applied 0.7
    fitF = audio.fit_clip(sine(0.5, sr), sr, window=1.0, max_atempo=1.5, min_atempo=0.7)
    check("fill applies floor tempo", abs(fitF.applied_tempo - 0.7) < 1e-6, f"{fitF.applied_tempo}")
    check("fill stretches length ~0.5/0.7", abs(fitF.clip.size / sr - (0.5 / 0.7)) < 0.05, f"{fitF.clip.size/sr:.3f}s")
    # within the floor: 0.8s into 1.0s window -> applied 0.8 -> fills ~to window
    fitG = audio.fit_clip(sine(0.8, sr), sr, window=1.0, max_atempo=1.5, min_atempo=0.7)
    check("fill to window when within floor", abs(fitG.clip.size / sr - 1.0) < 0.05, f"{fitG.clip.size/sr:.3f}s")


def test_canvas():
    print("canvas place/finalize:")
    sr = 48000
    canvas = audio.build_canvas(3.0, sr)
    check("canvas exact length", canvas.size == 3 * sr)
    clip = sine(0.5, sr)
    audio.place(canvas, clip, start_s=1.0, sr=sr)
    pre = np.abs(canvas[: sr]).max()
    at = np.abs(canvas[sr: sr + clip.size]).max()
    check("energy only at offset", pre < 1e-6 < at)

    # overlap two loud clips -> finalize prevents clipping
    c2 = audio.build_canvas(1.0, sr)
    loud = (np.ones(sr // 2, dtype=np.float32) * 0.8)
    audio.place(c2, loud, 0.0, sr)
    audio.place(c2, loud, 0.0, sr)  # same spot -> sums to 1.6
    check("pre-finalize clips", np.abs(c2).max() > 1.0)
    c2 = audio.finalize_canvas(c2)
    check("post-finalize <= ceiling", np.abs(c2).max() <= 0.99 + 1e-6)


def test_mux():
    print("mux (real ffmpeg, synthetic media):")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        video = td / "v.mp4"
        dub = td / "dub.wav"
        out = td / "out.mp4"
        # 2s synthetic video
        r = subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=25",
             "-pix_fmt", "yuv420p", "-c:v", "libx264", str(video)],
            capture_output=True, text=True)
        if r.returncode != 0:
            check("make test video", False, r.stderr[-300:])
            return
        sf.write(str(dub), sine(2.0, 48000), 48000, subtype="FLOAT")

        ffmpeg_utils.mux(video, dub, out, Config())
        check("output exists", out.exists())

        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_type,codec_name,channels,disposition=default:format=duration",
             "-of", "default=noprint_wrappers=1", str(out)],
            capture_output=True, text=True).stdout
        check("has aac audio", "codec_name=aac" in probe, probe)
        check("video copied (h264)", "codec_name=h264" in probe, probe)
        check("duration ~2s", any(abs(float(x.split("=")[1]) - 2.0) < 0.3
              for x in probe.splitlines() if x.startswith("duration=")), probe)


def main() -> int:
    for t in (test_episode_key, test_text_clean, test_atempo_chain, test_srt,
              test_windows_degenerate, test_cache, test_resample, test_fit_clip,
              test_canvas, test_mux):
        t()
    print()
    if _failures:
        print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
        return 1
    print("ALL SELF-TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
