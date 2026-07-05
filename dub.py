#!/usr/bin/env python
"""AI Dubbing — batch-replace foreign audio with English Chatterbox TTS.

Examples
--------
  # 0. See how videos pair with subtitles (generates nothing):
  python dub.py --dry-run

  # 1. Single-video sanity check (recommended first run):
  python dub.py --only 01

  # 1b. ...with a cloned narrator voice held constant across videos:
  python dub.py --only 01 --reference-wav voice.wav

  # 2. Full batch (idempotent — re-running skips finished outputs):
  python dub.py

  # CPU fallback / custom dirs / re-do everything:
  python dub.py --device cpu
  python dub.py --input-dir mycourse --output-dir dubbed --overwrite
"""

from __future__ import annotations

import argparse
import sys

from dubbing import __version__
from dubbing.config import Config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dub.py",
        description="Batch-dub foreign videos into English using Chatterbox TTS + their .srt files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--version", action="version", version=f"AI Dubbing {__version__}")
    p.add_argument("--config", metavar="PATH", help="YAML config file (overridden by any CLI flags).")

    # Paths
    g = p.add_argument_group("paths")
    g.add_argument("--input-dir", dest="input_dir")
    g.add_argument("--output-dir", dest="output_dir")
    g.add_argument("--subtitle-dir", dest="subtitle_dir",
                   help="Where .srt files live. Default: search recursively under input dir.")
    g.add_argument("--reference-wav", dest="reference_wav",
                   help="5-10s narrator voice to clone across ALL videos.")
    g.add_argument("--cache-dir", dest="cache_dir")
    g.add_argument("--log-dir", dest="log_dir")

    # Pairing / selection
    g = p.add_argument_group("pairing & selection")
    g.add_argument("--match", dest="match_strategy", choices=["episode", "stem"])
    g.add_argument("--only", dest="only", nargs="*", metavar="KEY",
                   help="Process only these episode keys/stems, e.g. --only 01 bonus01")
    g.add_argument("--max-cues", dest="max_cues", type=int, metavar="N",
                   help="Preview: voice only the first N cues per video (fast sanity check).")

    # Translation
    g = p.add_argument_group("translation (SRTs assumed already English)")
    g.add_argument("--translate", dest="translate", action=argparse.BooleanOptionalAction, default=None,
                   help="Run each cue through dubbing.translate.translate_text before TTS.")
    g.add_argument("--target-language", dest="target_language")
    g.add_argument("--source-language", dest="source_language")

    # Chatterbox knobs
    g = p.add_argument_group("chatterbox")
    g.add_argument("--exaggeration", type=float)
    g.add_argument("--cfg-weight", dest="cfg_weight", type=float)
    g.add_argument("--temperature", type=float)
    g.add_argument("--seed", type=int)

    # Fitting / assembly
    g = p.add_argument_group("time-fitting & assembly")
    g.add_argument("--max-atempo", dest="max_atempo", type=float,
                   help="Hardest allowed speed-up (default 1.5).")
    g.add_argument("--min-atempo", dest="min_atempo", type=float,
                   help="Slowest playback for SHORT clips, to fill gaps (1.0=off; e.g. 0.7).")
    g.add_argument("--extend-to-next-cue", dest="extend_to_next_cue",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="Use the gap to the next cue as each window (more natural).")
    g.add_argument("--min-window", dest="min_window", type=float)
    g.add_argument("--sample-rate", dest="sample_rate", type=int)
    g.add_argument("--channels", type=int, choices=[1, 2])

    # Loudness / encoding
    g = p.add_argument_group("loudness & encoding")
    g.add_argument("--loudnorm-i", dest="loudnorm_i", type=float)
    g.add_argument("--loudnorm-tp", dest="loudnorm_tp", type=float)
    g.add_argument("--loudnorm-lra", dest="loudnorm_lra", type=float)
    g.add_argument("--audio-bitrate", dest="audio_bitrate")

    # Runtime
    g = p.add_argument_group("runtime")
    g.add_argument("--device", choices=["cuda", "cpu"])
    g.add_argument("--cuda-mem-fraction", dest="cuda_mem_fraction", type=float, metavar="F",
                   help="Cap this process to fraction F of VRAM (for parallel workers).")
    g.add_argument("--overwrite", dest="overwrite", action="store_true", default=None,
                   help="Re-dub videos even if their output already exists.")
    g.add_argument("--dry-run", dest="dry_run", action="store_true", default=None,
                   help="Discover + pair + report only; generate nothing.")
    g.add_argument("--skip-cue-pattern", dest="skip_cue_patterns", action="append", metavar="REGEX",
                   help="Skip cues whose text matches REGEX (repeatable).")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # defaults < YAML < CLI
    cfg = Config.load_yaml(args.config) if args.config else Config()

    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("config",) and v is not None}
    # argparse gives a list for --only / --skip-cue-pattern; config wants tuples.
    for key in ("only", "skip_cue_patterns"):
        if key in overrides and isinstance(overrides[key], list):
            overrides[key] = tuple(overrides[key])
    cfg = cfg.merged_with(overrides)

    try:
        cfg.validate()
    except Exception as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    from dubbing.pipeline import run_batch
    report = run_batch(cfg)
    counts = report.counts()
    # Non-zero exit if anything failed, so this composes in scripts/CI.
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
