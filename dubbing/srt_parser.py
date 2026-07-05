"""Parse ``.srt`` files into clean, TTS-ready cues.

Uses the `srt` library for parsing and adds:
  * robust encoding handling (UTF-8 with/without BOM, UTF-16, cp1252 fallback),
  * text cleanup (strip HTML/SSA tags, collapse newlines for natural speech),
  * optional regex-based cue skipping (e.g. the course's "turn off subtitles"
    UI line), and
  * per-cue allotted-window computation honouring ``extend_to_next_cue``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import srt  # type: ignore

_TAG_RE = re.compile(r"<[^>]+>")          # <i>, <b>, <font ...>
_BRACE_RE = re.compile(r"\{[^}]*\}")      # {\an8}, {\i1} SSA/ASS overrides
_WS_RE = re.compile(r"\s+")


@dataclass
class Cue:
    index: int            # 1-based cue number as it appears in the file
    start: float          # seconds
    end: float            # seconds
    text: str             # cleaned, single-line text ("" if nothing to speak)
    window: float = 0.0   # allotted duration (s); filled by compute_windows()

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def speakable(self) -> bool:
        return bool(self.text.strip())


def _read_text(path: Path) -> str:
    """Decode an .srt file, trying the encodings these files realistically use."""
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    # latin-1 never raises, but keep a definite return for type-checkers.
    return raw.decode("latin-1", errors="replace")


def clean_text(text: str) -> str:
    """Flatten subtitle markup into a single natural-speech line."""
    text = _TAG_RE.sub("", text)
    text = _BRACE_RE.sub("", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = _WS_RE.sub(" ", text)
    return text.strip()


def parse_srt(
    path: Path,
    skip_patterns: tuple[str, ...] = (),
) -> list[Cue]:
    """Parse a single .srt file into a time-ordered list of :class:`Cue`.

    Cues matching any ``skip_patterns`` regex (matched against the cleaned text,
    case-insensitive) keep their timing but are blanked so they stay silent.
    """
    text = _read_text(path)
    compiled = [re.compile(p, re.IGNORECASE) for p in skip_patterns]

    cues: list[Cue] = []
    for sub in srt.parse(text):
        clean = clean_text(sub.content)
        if clean and any(rx.search(clean) for rx in compiled):
            clean = ""  # drop the words, preserve the time slot as silence
        cues.append(Cue(
            index=sub.index if sub.index is not None else len(cues) + 1,
            start=sub.start.total_seconds(),
            end=sub.end.total_seconds(),
            text=clean,
        ))

    cues.sort(key=lambda c: (c.start, c.end))
    return cues


def compute_windows(
    cues: list[Cue],
    extend_to_next_cue: bool,
    min_window: float,
    media_duration: Optional[float] = None,
) -> None:
    """Fill each cue's ``window`` (allotted seconds) in place.

    With ``extend_to_next_cue`` the window runs to the next cue's start (more
    room for natural speech; the slack becomes trailing silence). Otherwise it
    is the cue's own start->end span. Windows never fall below ``min_window``.
    """
    n = len(cues)
    for i, cue in enumerate(cues):
        if extend_to_next_cue:
            if i + 1 < n:
                upper = cues[i + 1].start
            elif media_duration is not None:
                upper = media_duration
            else:
                upper = cue.end
            real = upper - cue.start
        else:
            real = cue.end - cue.start
        # Use the REAL allotted span: a clip must never silently claim time that
        # belongs to the next cue (which would overlap two voices, unlogged).
        # min_window only rescues a degenerate (<=0) span from zero-length or
        # overlapping cues so fit_clip's tempo math has a positive divisor.
        cue.window = real if real > 0 else min_window
