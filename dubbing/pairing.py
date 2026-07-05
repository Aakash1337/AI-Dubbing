"""Discover videos and subtitles and pair them.

The real Mogoon dataset does NOT pair by filename stem:

    mogoon/01 Understanding Commercial Art.m4v
    mogoon/EN SUBS/[EN]Mogoon_01.srt

so the default strategy pairs by an "episode key" parsed from each name
(``01`` .. ``23`` plus ``bonus01``/``bonus02``). A classic same-stem strategy
is available via ``match_strategy="stem"`` for folders that follow the
``lecture01.mp4`` <-> ``lecture01.srt`` convention.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import Config


@dataclass(frozen=True)
class Pair:
    key: str
    video: Path
    srt: Path


def episode_key(name: str) -> Optional[str]:
    """Parse a canonical episode key from a file name (no extension needed).

    Examples
    --------
    >>> episode_key("01 Understanding Commercial Art")
    '01'
    >>> episode_key("[EN]Mogoon_07")
    '07'
    >>> episode_key("04 Space 1 Perspective")   # the leading number wins
    '04'
    >>> episode_key("Bonus 1 In-Depth Coloring Insights")
    'bonus01'
    >>> episode_key("[EN]Mogoon_BONUS02")
    'bonus02'
    """
    s = name.lower()
    if "bonus" in s:
        m = re.search(r"bonus\D*(\d+)", s)
        if m:
            return f"bonus{int(m.group(1)):02d}"
        # "bonus" with no following number: fall through to a plain number if any
    m = re.search(r"(\d+)", s)
    if m:
        return f"{int(m.group(1)):02d}"
    return None


def _key_for(name: str, strategy: str) -> Optional[str]:
    if strategy == "stem":
        return name
    return episode_key(name)


def find_videos(cfg: Config) -> list[Path]:
    exts = {e.lower() for e in cfg.video_exts}
    vids = [
        p for p in sorted(cfg.input_path.rglob("*"))
        if p.is_file() and p.suffix.lower() in exts
    ]
    return vids


def find_subtitles(cfg: Config) -> list[Path]:
    # Recursive so a nested "EN SUBS" folder is found whether subtitle_dir is set
    # to the parent or left as the input dir.
    return [p for p in sorted(cfg.subtitle_path.rglob("*.srt")) if p.is_file()]


@dataclass
class PairingResult:
    pairs: list[Pair]
    videos_without_srt: list[Path]
    srts_without_video: list[Path]
    duplicate_keys: dict[str, list[str]]  # key -> list of conflicting file names


def pair_inputs(cfg: Config) -> PairingResult:
    """Pair every video with its subtitle file using the configured strategy."""
    strategy = cfg.match_strategy
    videos = find_videos(cfg)
    srts = find_subtitles(cfg)

    # Build key -> path indexes, recording collisions instead of silently losing.
    srt_index: dict[str, Path] = {}
    dup: dict[str, list[str]] = {}
    for s in srts:
        k = _key_for(s.stem, strategy)
        if k is None:
            continue
        if k in srt_index:
            dup.setdefault(k, [srt_index[k].name]).append(s.name)
        else:
            srt_index[k] = s

    pairs: list[Pair] = []
    used_srt_keys: set[str] = set()
    videos_without_srt: list[Path] = []

    only = {o.lower() for o in cfg.only}
    for v in videos:
        k = _key_for(v.stem, strategy)
        if k is None:
            videos_without_srt.append(v)
            continue
        if only and k.lower() not in only:
            continue
        srt = srt_index.get(k)
        if srt is None:
            videos_without_srt.append(v)
            continue
        pairs.append(Pair(key=k, video=v, srt=srt))
        used_srt_keys.add(k)

    srts_without_video = [
        s for s in srts
        if (_key_for(s.stem, strategy) or "\x00") not in used_srt_keys
    ]
    # When --only is active, the "unused srt" list is meaningless; drop it.
    if only:
        srts_without_video = []

    pairs.sort(key=lambda p: p.key)
    return PairingResult(
        pairs=pairs,
        videos_without_srt=videos_without_srt,
        srts_without_video=srts_without_video,
        duplicate_keys=dup,
    )
