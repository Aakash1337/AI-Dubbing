"""The dubbing pipeline: a single-video path and the batch loop around it.

Per video (the part that makes or breaks sync):
  1. parse .srt  ->  cues with absolute start/end and an allotted window
  2. probe the source video's exact duration
  3. for each speakable cue: (cache or) generate TTS, resample, time-fit
  4. place every fitted clip at its absolute timecode on a silent canvas the
     exact length of the video  ->  the dub track can't drift out of the runtime
  5. peak-guard, write the wav
  6. mux: copy video stream, attach AAC dub as the single default track

Batch behaviour: idempotent (skip existing outputs), per-video try/except so one
bad file can't sink the run, CUDA cache cleared between videos, full run report.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from . import audio, ffmpeg_utils
from .cache import CueCache
from .config import Config
from .logging_setup import (CappedCue, RunReport, VideoResult, get_logger,
                            new_run_id, setup_logging)
from .pairing import Pair, pair_inputs
from .srt_parser import compute_windows, parse_srt
from .translate import translate_text

try:  # progress bars are nice-to-have, not required
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **_kw):  # type: ignore
        return it


def process_video(
    pair: Pair,
    cfg: Config,
    narrator,                # dubbing.tts.Narrator (lazily imported by caller)
    report: RunReport,
) -> VideoResult:
    """Dub a single video end-to-end. Returns a :class:`VideoResult`."""
    log = get_logger()
    t0 = time.perf_counter()
    # Preview runs (max_cues) write to a distinct name so a truncated preview
    # never blocks the real full-length dub through the resume guard below.
    preview = cfg.max_cues is not None
    out_name = f"{pair.video.stem}.preview.mp4" if preview else f"{pair.video.stem}.mp4"
    out_path = cfg.output_path / out_name
    result = VideoResult(video=pair.video.name, status="ok", output=str(out_path))

    # Idempotent / resumable: skip only when a COMPLETE output exists. mux()
    # writes atomically (temp + rename), so a present, non-trivial file is "done".
    if out_path.exists() and out_path.stat().st_size > 1024 and not cfg.overwrite:
        log.info("SKIP  %s (output exists)", pair.video.name)
        result.status = "skipped"
        return result

    duration = ffmpeg_utils.probe_duration(pair.video)
    result.duration_s = duration

    cues = parse_srt(pair.srt, skip_patterns=cfg.skip_cue_patterns)
    compute_windows(cues, cfg.extend_to_next_cue, cfg.min_window, media_duration=duration)
    result.n_cues = len(cues)

    canvas = audio.build_canvas(duration, cfg.sample_rate)

    signature = {
        "model": getattr(narrator, "model_id", "chatterbox"),
        "native_sr": getattr(narrator, "sample_rate", 0),
        "reference_wav": _ref_signature(cfg.reference_wav),
        "exaggeration": cfg.exaggeration,
        "cfg_weight": cfg.cfg_weight,
        "temperature": cfg.temperature,
        "seed": cfg.seed,
        "translate": cfg.translate,
    }
    if cfg.seed is not None:
        signature["seed_scheme"] = "per_cue_v1"  # only when seeding actually applies
    cache = CueCache(cfg.cache_path, pair.key, signature)

    n_generated = 0
    n_capped = 0
    desc = f"{pair.key} {pair.video.stem[:28]}"
    for cue in tqdm(cues, desc=desc, unit="cue", leave=False):
        if not cue.speakable:
            continue
        if cfg.max_cues is not None and n_generated >= cfg.max_cues:
            break  # preview mode: leave the rest of the timeline silent
        text = translate_text(cue.text, cfg.target_language, cfg.source_language) if cfg.translate else cue.text
        if not text.strip():
            continue

        cached = cache.get(cue.index, text)
        if cached is not None:
            clip_native, native_sr = cached
        else:
            clip_native = narrator.synthesize(text, cue.index)
            native_sr = narrator.sample_rate
            cache.put(cue.index, text, clip_native, native_sr)

        clip = audio.resample(clip_native, native_sr, cfg.sample_rate)
        fit = audio.fit_clip(clip, cfg.sample_rate, cue.window, cfg.max_atempo, cfg.min_atempo)
        audio.place(canvas, fit.clip, cue.start, cfg.sample_rate)
        n_generated += 1

        if fit.capped:
            n_capped += 1
            report.add_capped(CappedCue(
                video=pair.video.name, cue_index=cue.index,
                start=cue.start, end=cue.end,
                needed_tempo=fit.needed_tempo, applied_tempo=fit.applied_tempo,
                overrun_seconds=fit.overrun_seconds, text=cue.text[:200],
            ))
            log.warning("  cap  cue #%d @%.2fs needs %.2fx > %.2fx (overruns %.2fs): %s",
                        cue.index, cue.start, fit.needed_tempo, cfg.max_atempo,
                        fit.overrun_seconds, cue.text[:60])

    canvas = audio.finalize_canvas(canvas)

    # Write the assembled track to a temp wav next to the cache, then mux.
    dub_wav = cfg.cache_path / cache.dir.name / "_dub_track.wav"
    audio.write_wav(dub_wav, canvas, cfg.sample_rate)
    ffmpeg_utils.mux(pair.video, dub_wav, out_path, cfg,
                     extra_metadata={"comment": "English dub (Chatterbox TTS)"})
    try:
        dub_wav.unlink()
    except OSError:
        pass

    result.n_generated = n_generated
    result.n_capped = n_capped
    result.elapsed_s = time.perf_counter() - t0
    log.info("OK    %s  (%d/%d cues voiced, %d capped, %.1fs)",
             pair.video.name, n_generated, len(cues), n_capped, result.elapsed_s)
    return result


def run_batch(cfg: Config) -> RunReport:
    """Discover, pair, and dub every video. Never raises on a single bad video."""
    cfg.validate()
    run_id = new_run_id()
    log, log_file = setup_logging(cfg.log_path, run_id)
    from datetime import datetime
    report = RunReport(run_id=run_id, started_at=datetime.now().isoformat(timespec="seconds"))

    log.info("Run %s  |  log: %s", run_id, log_file)
    ffmpeg_utils.ensure_tools()

    # Route big intermediates (atempo temp wavs, the full dub-track wav) under the
    # cache dir so they land on the same roomy drive — not a near-full system temp.
    import tempfile
    work_tmp = cfg.cache_path / "_tmp"
    work_tmp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(work_tmp)

    pairing = pair_inputs(cfg)
    _log_pairing(cfg, pairing, log)

    if cfg.dry_run:
        log.info("Dry run: %d pair(s) would be processed. Nothing generated.", len(pairing.pairs))
        report.write(cfg.log_path)
        return report

    if not pairing.pairs:
        log.warning("No video/subtitle pairs found — nothing to do.")
        report.write(cfg.log_path)
        return report

    cfg.output_path.mkdir(parents=True, exist_ok=True)

    # Load the model ONCE and reuse across all videos.
    from .tts import Narrator
    log.info("Loading Chatterbox model on %s ...", cfg.device)
    narrator = Narrator(cfg)
    log.info("Model ready (native %d Hz). Voice: %s",
             narrator.sample_rate, cfg.reference_wav or "built-in default")

    for pair in pairing.pairs:
        try:
            result = process_video(pair, cfg, narrator, report)
        except Exception as exc:  # one bad video must not sink the batch
            log.exception("FAIL  %s: %s", pair.video.name, exc)
            result = VideoResult(video=pair.video.name, status="failed", error=str(exc))
        report.add_video(result)
        narrator.reset()  # clear CUDA cache between videos to avoid VRAM creep

    paths = report.write(cfg.log_path)
    counts = report.counts()
    log.info("Done. ok=%d skipped=%d failed=%d | capped cues=%d",
             counts["ok"], counts["skipped"], counts["failed"], counts["capped_cues"])
    log.info("Summary: %s | Capped report: %s", paths["summary"], paths["capped"])
    return report


# ── helpers ──────────────────────────────────────────────────────────────────
def _ref_signature(reference_wav: Optional[str]) -> Optional[str]:
    if not reference_wav:
        return None
    p = Path(reference_wav)
    try:
        st = p.stat()
        return f"{p.name}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return reference_wav


def _log_pairing(cfg: Config, pairing, log) -> None:
    log.info("Matched %d video/subtitle pair(s) [strategy=%s]:",
             len(pairing.pairs), cfg.match_strategy)
    for p in pairing.pairs:
        log.info("  [%s]  %s  <->  %s", p.key, p.video.name, p.srt.name)
    for v in pairing.videos_without_srt:
        log.warning("  no subtitle for video: %s", v.name)
    for s in pairing.srts_without_video:
        log.warning("  no video for subtitle: %s", s.name)
    for key, names in pairing.duplicate_keys.items():
        log.warning("  duplicate subtitle key %r: %s", key, ", ".join(names))
    # Warn if two source videos share a stem and would collide on one output .mp4.
    out_stems: dict[str, list[str]] = {}
    for p in pairing.pairs:
        out_stems.setdefault(p.video.stem.lower(), []).append(p.video.name)
    for stem, names in out_stems.items():
        if len(names) > 1:
            log.warning("  output collision (same stem -> one .mp4): %s", ", ".join(names))
