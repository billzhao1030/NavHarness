"""Anthropic — Claude models, anthropic-messages wire."""

from __future__ import annotations

from .base import BaseProvider


class AnthropicProvider(BaseProvider):
    name = "anthropic"
    wire = "anthropic-messages"
    key_env = "ANTHROPIC_API_KEY"

    def matches(self, model_id: str, extra: dict | None = None) -> bool:
        model = (model_id or "").lower()
        return any(s in model for s in ("anthropic", "claude"))
