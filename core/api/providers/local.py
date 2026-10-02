"""Locally served models — keyless; pure knowledge of the local vendors.

The serving machinery lives ABOVE this layer (core/api/ollama.py and
core/api/vllm.py) and is dispatched from the api entry (``ensure_serving``)
— these classes only declare identity and where the default endpoint lives.

The endpoint-URL constants are knowledge and belong here; the machinery
above imports them downward.
"""

from __future__ import annotations

import os

from .base import BaseProvider

# ollama: env-overridable for hosts where 11434 is held by a system server
# that cannot be replaced — everything then targets the given URL.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
# vllm: "" = no external endpoint configured (a tunnel or operator-started
# server sets this; a local launch gets its port from VllmServer's pool).
VLLM_URL = os.environ.get("VLLM_URL", "")


class OllamaProvider(BaseProvider):
    name = "ollama"
    wire = "ollama"
    default_api_base = OLLAMA_URL

    def matches(self, model_id: str, extra: dict | None = None) -> bool:
        return (model_id or "").lower().startswith("ollama")


class HostedVllmProvider(BaseProvider):
    name = "hosted_vllm"
    default_api_base = VLLM_URL or None

    def matches(self, model_id: str, extra: dict | None = None) -> bool:
        return (model_id or "").lower().startswith("hosted_vllm/")
