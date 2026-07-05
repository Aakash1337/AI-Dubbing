"""Central configuration for the dubbing pipeline.

Every knob the spec asks for lives on :class:`Config`. Values resolve in three
layers, lowest precedence first:

    dataclass defaults  <  YAML file (``--config``)  <  CLI flags

Nothing here imports torch/chatterbox so the config can be inspected (e.g.
``--dry-run``) without paying the model-import cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, MISSING
from pathlib import Path
from typing import Any, Optional


# Default video containers to look for in the input directory. The Mogoon course
# ships as .m4v; the rest are here so the tool is useful on other folders too.
DEFAULT_VIDEO_EXTS = (".m4v", ".mp4", ".mkv", ".mov", ".webm", ".avi")


@dataclass
class Config:
    """All pipeline settings. See ``config.example.yaml`` for an annotated copy."""

    # ── Paths ────────────────────────────────────────────────────────────────
    input_dir: str = "mogoon"
    """Folder holding the source videos."""

    output_dir: str = "output"
    """Where dubbed .mp4 files are written."""

    subtitle_dir: Optional[str] = None
    """Folder holding the .srt files. ``None`` -> search recursively under
    ``input_dir`` (finds the Mogoon ``EN SUBS`` subfolder automatically)."""

    reference_wav: Optional[str] = None
    """A single 5-10s reference .wav used as the cloned narrator voice across
    ALL videos for consistency. ``None`` -> Chatterbox's built-in default voice."""

    cache_dir: str = ".dub_cache"
    """Per-cue TTS clips are cached here so a crash mid-video resumes cheaply."""

    log_dir: str = "logs"
    """Per-run logs and the capped-cue report land here."""

    # ── Pairing ──────────────────────────────────────────────────────────────
    match_strategy: str = "episode"
    """How videos are paired with subtitles:
      - "episode": match by episode number/BONUS token parsed from each name
                   (works for ``01 ....m4v`` <-> ``[EN]Mogoon_01.srt``).
      - "stem":    exact filename-stem match (the classic ``lecture01`` case)."""

    video_exts: tuple[str, ...] = DEFAULT_VIDEO_EXTS

    only: tuple[str, ...] = ()
    """If non-empty, process only these episode keys / stems (e.g. ('01',) or
    ('bonus01',)). Great for the single-video sanity check."""

    max_cues: Optional[int] = None
    """Debug/preview: voice only the first N speakable cues per video (the rest
    stay silent). Lets you hear sync/voice on a 27-min lecture in seconds without
    generating the whole thing. ``None`` = no limit."""

    # ── Translation (point #1 assumption: SRTs are already English) ──────────
    translate: bool = False
    """When True, each cue's text is run through the (swappable) translation
    function in ``dubbing.translate`` before TTS. Off by default."""

    target_language: str = "en"
    source_language: Optional[str] = None  # None -> auto/unknown

    # ── Chatterbox TTS knobs ─────────────────────────────────────────────────
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    seed: Optional[int] = None  # set for reproducible generations

    # ── Time-fitting / assembly ──────────────────────────────────────────────
    max_atempo: float = 1.5
    """Hardest allowed speed-up. Clips needing more than this are clamped and
    logged (they will slightly overrun their window)."""

    min_atempo: float = 0.8
    """Gap control — the SLOWEST playback allowed for a clip SHORTER than its
    window. 1.0 = off (short clips keep natural speed; leftover time is silence,
    which reads as gaps between lines because TTS speaks faster than the
    original). Below 1.0 stretches short clips to FILL their window so narration
    flows: the default 0.8 lets a clip play up to ~1.25x longer (cut measured
    gaps ~half on the Mogoon course); 0.7/0.6 fill more aggressively. The floor
    stops unnatural over-slowing when a window is much larger than the clip, so
    real pauses stay partly silent."""

    extend_to_next_cue: bool = True
    """Window = (next cue start - this start) instead of (end - start). Gives
    natural speech more room and reduces how often we hit the speed cap. The
    extra room is filled with trailing silence when the clip is short."""

    min_window: float = 0.30
    """Floor (seconds) for a cue's allotted window, guarding against zero/negative
    durations from overlapping or malformed cues."""

    sample_rate: int = 48000
    """Common working + output sample rate. All TTS clips are resampled to this."""

    channels: int = 2
    """Output audio channels (2 = stereo, the safest "plays everywhere" choice)."""

    # ── Loudness / encoding (point #1 of the 'extra points': AAC + default) ──
    loudnorm_i: float = -16.0   # integrated loudness target (LUFS)
    loudnorm_tp: float = -1.5   # true-peak ceiling (dBTP)
    loudnorm_lra: float = 11.0  # loudness range
    audio_bitrate: str = "192k"
    aac_codec: str = "aac"      # ffmpeg native AAC; broadly compatible

    # ── Runtime ──────────────────────────────────────────────────────────────
    device: str = "cuda"        # "cuda" | "cpu"; auto-falls back to cpu if no GPU
    cuda_mem_fraction: Optional[float] = None
    """Cap THIS process's VRAM to a fraction (0-1) of the GPU's total. Single-
    stream Chatterbox only uses ~35% of a 4080's compute, so run_parallel.py runs
    several worker processes at once and caps each one's memory so they co-fit.
    None = no cap (use all of VRAM)."""
    overwrite: bool = False     # if False, skip videos whose output already exists
    dry_run: bool = False       # discover + pair + report, generate nothing
    skip_cue_patterns: tuple[str, ...] = ()
    """Regexes; any cue whose text matches is skipped (left silent). Handy for
    UI lines like 'To turn off the subtitles, please go to ...'."""

    # ── Normalisation ────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        # Lower-case extensions and ensure a leading dot, so a natural YAML value
        # like ["mp4", ".MKV"] still matches Path.suffix (".mp4"/".mkv").
        norm = []
        for e in self.video_exts:
            e = str(e).strip().lower()
            if e and not e.startswith("."):
                e = "." + e
            if e:
                norm.append(e)
        self.video_exts = tuple(norm)

    # ── Derived helpers ──────────────────────────────────────────────────────
    @property
    def input_path(self) -> Path:
        return Path(self.input_dir)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def subtitle_path(self) -> Path:
        return Path(self.subtitle_dir) if self.subtitle_dir else self.input_path

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir)

    @property
    def log_path(self) -> Path:
        return Path(self.log_dir)

    # ── Construction helpers ─────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Build a Config from a plain dict, ignoring unknown keys and coercing
        list -> tuple for the tuple-typed fields."""
        known = {f.name for f in fields(cls)}
        tuple_fields = {"video_exts", "only", "skip_cue_patterns"}
        clean: dict[str, Any] = {}
        for k, v in (data or {}).items():
            if k not in known:
                raise KeyError(f"Unknown config key: {k!r}")
            if k in tuple_fields and isinstance(v, list):
                v = tuple(v)
            clean[k] = v
        return cls(**clean)

    @classmethod
    def load_yaml(cls, path: str | Path) -> "Config":
        import yaml  # local import so PyYAML isn't needed for pure-CLI use

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    def merged_with(self, overrides: dict[str, Any]) -> "Config":
        """Return a copy with ``overrides`` applied (None values are ignored so
        unset CLI flags don't clobber YAML/defaults)."""
        current = {f.name: getattr(self, f.name) for f in fields(self)}
        for k, v in overrides.items():
            if v is None:
                continue
            current[k] = v
        return Config.from_dict(current)

    def validate(self) -> None:
        if self.match_strategy not in ("episode", "stem"):
            raise ValueError(f"match_strategy must be 'episode' or 'stem', got {self.match_strategy!r}")
        if self.device not in ("cuda", "cpu"):
            raise ValueError(f"device must be 'cuda' or 'cpu', got {self.device!r}")
        if self.max_atempo < 1.0:
            raise ValueError(f"max_atempo must be >= 1.0, got {self.max_atempo}")
        if not (0.0 < self.min_atempo <= 1.0):
            raise ValueError(f"min_atempo must be in (0, 1.0], got {self.min_atempo}")
        if self.cuda_mem_fraction is not None and not (0.0 < self.cuda_mem_fraction <= 1.0):
            raise ValueError(f"cuda_mem_fraction must be in (0, 1.0], got {self.cuda_mem_fraction}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.channels not in (1, 2):
            raise ValueError(f"channels must be 1 or 2, got {self.channels}")
        if not self.video_exts:
            raise ValueError("video_exts must be non-empty")
        if self.max_cues is not None and self.max_cues < 1:
            raise ValueError(f"max_cues must be >= 1 or null, got {self.max_cues}")
        if self.reference_wav and not Path(self.reference_wav).is_file():
            raise FileNotFoundError(f"reference_wav not found: {self.reference_wav}")


def default_field_help() -> dict[str, str]:
    """Map of field name -> default repr, used by the CLI ``--help`` epilog."""
    out: dict[str, str] = {}
    for f in fields(Config):
        if f.default is not MISSING:
            out[f.name] = repr(f.default)
        elif f.default_factory is not MISSING:  # type: ignore[misc]
            out[f.name] = repr(f.default_factory())  # type: ignore[misc]
    return out
