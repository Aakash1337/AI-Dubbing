"""Chatterbox TTS wrapper (Resemble AI, MIT).

Verified against chatterbox-tts master (PyPI 0.1.7). The English model's
signature is::

    generate(text, repetition_penalty=1.2, min_p=0.05, top_p=1.0,
             audio_prompt_path=None, exaggeration=0.5, cfg_weight=0.5,
             temperature=0.8)

Two things that bite people, handled here:
  * ``audio_prompt_path`` is the *5th* positional arg, so EVERYTHING except
    ``text`` is passed by keyword.
  * the model has no ``seed`` argument — reproducibility is seeded externally.

``generate()`` returns a CPU float32 tensor shaped ``(1, N)`` at ``model.sr``
(24000 Hz). We expose it as a 1-D float32 numpy array; resampling to the
pipeline's working rate happens in :mod:`dubbing.audio`.

Note: outputs carry Resemble's inaudible Perth watermark by design — that's
expected and survives the atempo/loudnorm/mux steps untouched.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

import numpy as np

from .config import Config

log = logging.getLogger("dubbing")


def resolve_device(requested: str) -> str:
    """Honour the requested device, falling back to CPU if CUDA is absent."""
    import torch

    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        log.warning("device='cuda' requested but no CUDA GPU is available — using CPU (slow).")
        return "cpu"
    return "cpu"


class Narrator:
    """Loads Chatterbox once and synthesizes one cue at a time."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._reference = cfg.reference_wav or None

        import torch
        from chatterbox.tts import ChatterboxTTS

        self._torch = torch
        self.device = resolve_device(cfg.device)

        # Cap VRAM so several workers co-fit on one GPU (run_parallel.py).
        if self.device == "cuda" and cfg.cuda_mem_fraction:
            try:
                torch.cuda.set_per_process_memory_fraction(float(cfg.cuda_mem_fraction))
            except Exception as exc:  # non-fatal: just means no cap
                log.warning("could not set CUDA mem fraction: %s", exc)

        # Reproducibility is seeded PER CUE in synthesize() (the model has no seed
        # arg). Per-cue derivation keeps each clip a pure function of (seed, index,
        # text, params) — independent of batch order, cache hits, and --only/preview
        # — so cached clips stay reproducible and the --seed contract holds.

        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self.sample_rate = int(getattr(self.model, "sr", 24000))
        self.model_id = f"ChatterboxTTS@{self.device}"

        # Only forward kwargs this installed version actually accepts — the
        # generate() signature has gained params across releases.
        try:
            self._accepted = set(inspect.signature(self.model.generate).parameters)
        except (ValueError, TypeError):
            self._accepted = set()

    def _generate_kwargs(self) -> dict[str, Any]:
        wanted = {
            "audio_prompt_path": self._reference,
            "exaggeration": self.cfg.exaggeration,
            "cfg_weight": self.cfg.cfg_weight,
            "temperature": self.cfg.temperature,
        }
        if not self._accepted:  # couldn't introspect; trust these stable names
            return {k: v for k, v in wanted.items() if v is not None}
        return {k: v for k, v in wanted.items()
                if k in self._accepted and v is not None}

    def synthesize(self, text: str, cue_index: int = 0) -> np.ndarray:
        """Return mono float32 audio at :attr:`sample_rate` for one cue's text."""
        text = text.strip()
        if not text:
            return np.zeros(int(0.05 * self.sample_rate), dtype=np.float32)

        if self.cfg.seed is not None:
            self._seed_for(cue_index)
        wav = self.model.generate(text, **self._generate_kwargs())
        return _to_mono_f32(wav)

    def _seed_for(self, cue_index: int) -> None:
        """Seed all RNGs from (seed, cue_index) so each cue is reproducible
        regardless of generation order or cache state."""
        import random
        s = (int(self.cfg.seed) * 1000003 + int(cue_index)) & 0x7FFFFFFF
        random.seed(s)
        np.random.seed(s)
        self._torch.manual_seed(s)
        if self.device == "cuda":
            self._torch.cuda.manual_seed_all(s)

    def reset(self) -> None:
        """Free cached VRAM between videos to avoid creep over a long batch."""
        if self.device == "cuda":
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass


def _to_mono_f32(wav: Any) -> np.ndarray:
    """Coerce Chatterbox's (1, N) tensor (or any array) to a 1-D float32 array."""
    try:
        import torch
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().to("cpu").numpy()
    except Exception:
        pass
    arr = np.asarray(wav, dtype=np.float32)
    if arr.ndim > 1:
        # (1, N) -> (N,); a true multi-channel clip is averaged to mono.
        arr = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
    return np.ascontiguousarray(arr, dtype=np.float32)
