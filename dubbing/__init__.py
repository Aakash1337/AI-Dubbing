"""AI Dubbing — batch-replace foreign-language audio with English Chatterbox TTS.

The pipeline reads a folder of videos plus their matching ``.srt`` files,
generates an English voiceover per subtitle cue with Chatterbox TTS, fits each
clip to its time window, places clips by absolute timecode on a silent canvas
the exact length of the source video, then muxes the new track back onto the
untouched video stream.

See :mod:`dubbing.pipeline` for the per-video and batch entry points and
:mod:`dubbing.config` for every tunable knob.
"""

__version__ = "1.0.0"
