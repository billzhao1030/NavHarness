"""OpenAI — GPT models, openai-chat wire."""

from __future__ import annotations

from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    name = "openai"
    key_env = "OPENAI_API_KEY"

    def matches(self, model_id: str, extra: dict | None = None) -> bool:
        model = (model_id or "").lower()
        return model.startswith(("gpt", "openai/"))
