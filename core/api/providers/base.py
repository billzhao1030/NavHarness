"""Provider base — the unified face of every platform we call.

One provider = one platform, one file each under ``core/api/providers``. The base
class unifies identity (name, wire format), key resolution (canonical
variable + legacy fallbacks), model-id recognition (``matches``), and call
preparation (``prepare`` → CallSpec), plus a litellm-backed ``complete()``
for driver-side utility calls (e.g. benchmark judges).

Harnesses keep their own execution paths — they consume a provider's DATA
(key, api_base) through ``prepare``/``resolve_key``, they never route their
agent loops through ``complete()``.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CallSpec:
    """What a caller needs to reach one model on a provider."""

    model_id: str
    api_base: str | None = None
    api_key: str | None = None    # set only when the CANONICAL variable is set
    key_env: str | None = None    # canonical key variable (None = keyless)


class BaseProvider(ABC):
    name: str = ""
    wire: str = "openai-chat"               # native wire format
    key_env: str = ""                       # canonical key variable ("" = keyless)
    key_fallbacks: tuple[str, ...] = ()     # legacy variables honored if canonical unset

    @abstractmethod
    def matches(self, model_id: str, extra: dict | None = None) -> bool:
        """Does this harness-facing model id (+model extra) belong to us?"""

    def key_chain(self, extra: dict | None = None) -> tuple[str, ...]:
        """Key variables to try, in order. ``extra`` lets a provider pick a
        different chain per call (e.g. a vendor whose site is read off the
        model's api_base, when sites' keys are not interchangeable)."""
        return tuple(v for v in (self.key_env, *self.key_fallbacks) if v)

    def resolve_key(self, extra: dict | None = None) -> tuple[str | None, str | None]:
        """(key value, variable it came from) — canonical first, then fallbacks.

        (None, None) for keyless providers and when nothing is set; the
        caller decides whether that is fatal (a harness may raise before
        episode 0).
        """
        for var in self.key_chain(extra):
            if os.environ.get(var):
                return os.environ[var], var
        return None, None

    def prepare(self, model_id: str, extra: dict | None = None) -> CallSpec:
        """Resolve everything a caller needs for one model, without calling.

        ``api_key`` is set only when the key came from a canonical (non-
        legacy) variable — a key riding a legacy fallback keeps flowing
        implicitly through the environment.
        """
        key, src = self.resolve_key(extra)
        return CallSpec(
            model_id=model_id,
            api_base=(extra or {}).get("api_base"),
            api_key=key if (key and src not in self.key_fallbacks) else None,
            key_env=self.key_env or None,
        )

    # knowledge only: where this vendor's endpoint lives when the model
    # carries no api_base (local vendors); None for remote APIs. The serving
    # MACHINERY lives above this layer and is dispatched from the api entry.
    default_api_base: str | None = None

    def complete(self, model_id: str, messages: list[dict[str, Any]],
                 extra: dict | None = None, **kwargs: Any) -> Any:
        """Unified one-shot call for driver-side utilities (litellm-backed)."""
        import litellm

        spec = self.prepare(model_id, extra)
        if spec.api_base:
            kwargs.setdefault("api_base", spec.api_base)
        if spec.api_key:
            kwargs.setdefault("api_key", spec.api_key)
        return litellm.completion(model=model_id, messages=messages, **kwargs)
