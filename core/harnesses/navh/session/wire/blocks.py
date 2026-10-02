"""Content blocks, usage and the CLI's environment — the small things every session shares.

A hop's prompt is either a string or a list of Anthropic content blocks
(text and base64 images); this is the one place those are built. Usage
comes back on the SDK's ``ResultMessage`` and is read the same way for every
kind of session.
"""

from __future__ import annotations

import os
from typing import Any

Blocks = list[dict[str, Any]]


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def image_block(png_b64: str, media_type: str = "image/png") -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": png_b64}}


def count_images(prompt: str | Blocks) -> int:
    if isinstance(prompt, str):
        return 0
    return sum(1 for b in prompt if isinstance(b, dict) and b.get("type") == "image")


def prompt_text(prompt: str | Blocks) -> str:
    """The words of a prompt, images left out."""
    if isinstance(prompt, str):
        return prompt
    return "\n".join(str(b.get("text") or "") for b in prompt if isinstance(b, dict) and b.get("type") == "text")


def in_tokens(d: dict | None) -> int:
    d = d or {}
    return (int(d.get("input_tokens", 0) or 0)
            + int(d.get("cache_creation_input_tokens", 0) or 0)
            + int(d.get("cache_read_input_tokens", 0) or 0))


def ctx_tokens(usage: Any) -> int:
    """Prompt footprint of the LAST turn of a hop (``usage['iterations'][-1]``);
    the top level is a session aggregate, which mis-measures context by 10x."""
    if not isinstance(usage, dict):
        return 0
    its = usage.get("iterations") or []
    return in_tokens(its[-1]) if its else in_tokens(usage)


def peak_tokens(usage: Any) -> int:
    """The largest prompt any turn of the hop carried."""
    if not isinstance(usage, dict):
        return 0
    its = usage.get("iterations") or []
    return max([in_tokens(usage)] + [in_tokens(i) for i in its])


def out_tokens(usage: Any) -> int:
    return int((usage or {}).get("output_tokens", 0) or 0) if isinstance(usage, dict) else 0


def subscription_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The CLI's environment: the process's, minus an API key (an API key
    present means API billing), plus what the caller adds."""
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    env.update({k: str(v) for k, v in (extra or {}).items()})
    return env
