"""Per-run logging: a console stream plus a timestamped file in ``log_dir``.

Also provides :class:`RunReport`, a small accumulator for the structured
end-of-run summary the spec asks for (which videos processed, which cues hit the
speed cap, which videos failed).
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

LOGGER_NAME = "dubbing"


def setup_logging(log_dir: Path, run_id: str, verbose: bool = True) -> tuple[logging.Logger, Path]:
    """Configure the package logger with console + file handlers.

    Returns the logger and the path to this run's log file.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{run_id}.log"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # idempotent across repeated calls in one process
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO if verbose else logging.WARNING)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    return logger, log_file


def new_run_id() -> str:
    """A filesystem-safe id, e.g. ``20260626-141502-12345`` (pid suffix keeps
    parallel workers from colliding on the same log file)."""
    return datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


@dataclass
class CappedCue:
    """A cue that needed more speed-up than ``max_atempo`` allowed."""
    video: str
    cue_index: int
    start: float
    end: float
    needed_tempo: float
    applied_tempo: float
    overrun_seconds: float
    text: str


@dataclass
class VideoResult:
    video: str
    status: str                      # "ok" | "skipped" | "failed"
    output: Optional[str] = None
    n_cues: int = 0
    n_generated: int = 0             # cues actually voiced (after filters/empties)
    n_capped: int = 0
    duration_s: float = 0.0
    elapsed_s: float = 0.0
    error: Optional[str] = None


@dataclass
class RunReport:
    run_id: str
    started_at: str
    videos: list[VideoResult] = field(default_factory=list)
    capped: list[CappedCue] = field(default_factory=list)

    def add_video(self, result: VideoResult) -> None:
        self.videos.append(result)

    def add_capped(self, cue: CappedCue) -> None:
        self.capped.append(cue)

    # ── persistence ──────────────────────────────────────────────────────────
    def write(self, log_dir: Path) -> dict[str, Path]:
        """Write the JSON summary and the capped-cue CSV. Returns their paths."""
        log_dir.mkdir(parents=True, exist_ok=True)
        summary_path = log_dir / f"run_{self.run_id}_summary.json"
        capped_path = log_dir / f"run_{self.run_id}_capped.csv"

        summary = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "counts": self.counts(),
            "videos": [asdict(v) for v in self.videos],
        }
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)

        with open(capped_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["video", "cue_index", "start", "end",
                             "needed_tempo", "applied_tempo", "overrun_seconds", "text"])
            for c in self.capped:
                writer.writerow([c.video, c.cue_index, f"{c.start:.3f}", f"{c.end:.3f}",
                                 f"{c.needed_tempo:.3f}", f"{c.applied_tempo:.3f}",
                                 f"{c.overrun_seconds:.3f}", c.text])

        return {"summary": summary_path, "capped": capped_path}

    def counts(self) -> dict[str, int]:
        return {
            "total": len(self.videos),
            "ok": sum(v.status == "ok" for v in self.videos),
            "skipped": sum(v.status == "skipped" for v in self.videos),
            "failed": sum(v.status == "failed" for v in self.videos),
            "capped_cues": len(self.capped),
        }
