"""Swappable translation backend.

ASSUMPTION (point #1 of the spec): the .srt files are already English, so the
default path does NOT translate — ``translate_text`` is the identity function
and ``cfg.translate`` defaults to False.

When you DO want translation, flip ``translate: true`` in the config and replace
the body of :func:`translate_text` with a call to your LLM or translation API.
The signature is intentionally stable so the rest of the pipeline never changes.

Example (Anthropic) — wire your own key/client and uncomment:

    from anthropic import Anthropic
    _client = Anthropic()

    def translate_text(text, target_language="en", source_language=None):
        if not text.strip():
            return text
        msg = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=(
                "You are a subtitle translator. Translate the user's line into "
                f"{target_language}. Reply with ONLY the translation, no quotes, "
                "preserving tone and keeping it concise enough to be spoken aloud."
            ),
            messages=[{"role": "user", "content": text}],
        )
        return msg.content[0].text.strip()

Batch tip: for hundreds of cues, translate a whole file's cues in one request
(numbered lines in, numbered lines out) to cut latency and cost — keep the
per-line :func:`translate_text` for simplicity, or add ``translate_batch`` and
call it from the pipeline.
"""

from __future__ import annotations

from typing import Optional


def translate_text(
    text: str,
    target_language: str = "en",
    source_language: Optional[str] = None,
) -> str:
    """Return ``text`` translated into ``target_language``.

    Default implementation is the identity function (no-op), because the SRTs
    are assumed to already be in English. SWAP THIS OUT to enable translation.
    """
    return text
