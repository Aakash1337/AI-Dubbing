"""Per-cue TTS clip cache for cheap resume.

These lectures are ~27 minutes each with hundreds of cues, so a crash (or an
OOM, or a Ctrl-C) partway through a video shouldn't throw away minutes of GPU
work. Each generated clip is written to disk keyed by a content hash of the
text + voice + generation params; on the next run identical cues load instantly
and only the missing ones are generated.

Clips are stored at the model's NATIVE sample rate (resampling is deterministic
and cheap, and keeping native rate means changing the output sample_rate does
not invalidate the cache).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


class CueCache:
    """Disk cache for one video's generated cue clips."""

    def __init__(self, root: Path, video_key: str, signature: dict):
        # ``signature`` holds everything that affects generation but the text:
        # voice ref, exaggeration, cfg_weight, temperature, seed, model id, etc.
        self.dir = Path(root) / _safe(video_key)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._sig = json.dumps(signature, sort_keys=True, ensure_ascii=False)

    def _hash(self, cue_index: int, text: str) -> str:
        h = hashlib.sha1()
        h.update(self._sig.encode("utf-8"))
        h.update(b"\x00")
        h.update(str(cue_index).encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        return h.hexdigest()[:16]

    def _path(self, cue_index: int, text: str) -> Path:
        return self.dir / f"{cue_index:04d}_{self._hash(cue_index, text)}.wav"

    def get(self, cue_index: int, text: str) -> Optional[tuple[np.ndarray, int]]:
        path = self._path(cue_index, text)
        if not path.is_file():
            return None
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception:
            return None
        if data.ndim > 1:  # stored mono, but be defensive
            data = data.mean(axis=1)
        return data.astype(np.float32, copy=False), int(sr)

    def put(self, cue_index: int, text: str, clip: np.ndarray, sr: int) -> Path:
        path = self._path(cue_index, text)
        tmp = path.with_suffix(".wav.tmp")
        # Pass format/subtype explicitly: the ".tmp" extension defeats soundfile's
        # format-from-extension inference, and FLOAT keeps the model's native
        # float32 precision (the WAV default would quantize to 16-bit PCM).
        sf.write(str(tmp), np.asarray(clip, dtype=np.float32), sr,
                 format="WAV", subtype="FLOAT")
        tmp.replace(path)  # atomic-ish: never leave a half-written cache file
        return path


def _safe(name: str) -> str:
    """Make an episode key safe as a folder name."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
