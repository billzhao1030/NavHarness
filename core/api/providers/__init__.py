"""Providers — the bottom layer: pure knowledge of who gets called.

One vendor per file: identity (name, wire), key resolution (canonical
variable + legacy fallbacks), model-id recognition (``matches``), default
endpoint. Declarative only — nothing here imports the endpoint machinery
above it (imports point strictly downward).

``PROVIDERS`` order is the resolution order: local prefixes → explicit
api_base → name substrings. Qwen must precede OpenAI — its model ids are
``openai/qwen…``.
"""

from __future__ import annotations

from .anthropic import AnthropicProvider
from .base import BaseProvider, CallSpec  # noqa: F401
from .local import HostedVllmProvider, OllamaProvider
from .openai import OpenAIProvider
from .qwen import QwenProvider

PROVIDERS: tuple[BaseProvider, ...] = (
    OllamaProvider(),
    HostedVllmProvider(),
    QwenProvider(),
    AnthropicProvider(),
    OpenAIProvider(),
)


def provider_of(model_id: str, extra: dict | None = None) -> BaseProvider | None:
    """Resolve a harness-facing model id (+model extra) to its provider."""
    for provider in PROVIDERS:
        if provider.matches(model_id, extra):
            return provider
    return None
