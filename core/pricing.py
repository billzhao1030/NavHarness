"""pricing — one billing convention for every harness: tokens x litellm's list price.

A harness records the usage its provider reports, in that provider's shape.
``billing_from_usage`` turns any of those shapes into ONE ledger keyed by
model id::

    {"<model id>": {input, cache_read, cache_write_5m, cache_write_1h, output}}

(``input`` = uncached input tokens; ``output`` includes reasoning; the two
cache-write tiers are Anthropic's 5-minute / 1-hour TTLs, other vendors
only ever fill the first). Shapes recognised, by their keys — not by
harness name, so a new harness needs nothing here as long as it records
what its provider sends:

- already a ledger (``{"billing": {...}}``) — e.g. navh: summed per call
  by ``add_litellm_usage`` / ``add_anthropic_usage`` as the calls happen
- codex ``turn.completed`` totals: ``input_tokens`` (cached included),
  ``cached_input_tokens``, ``cache_write_input_tokens``, ``output_tokens``
- Anthropic / Claude Code ``ResultMessage.usage``: ``input_tokens`` (uncached),
  ``cache_read_input_tokens``, ``cache_creation`` {ephemeral_5m/1h} (or the
  flat ``cache_creation_input_tokens``), ``output_tokens``
- OpenAI-chat / litellm ``usage``: ``prompt_tokens`` (cached and cache-write
  included), ``prompt_tokens_details.cached_tokens``,
  ``cache_creation_input_tokens``, ``completion_tokens``

``rates_for`` reads ``litellm.model_cost`` (the same table litellm itself
bills with); ``price`` applies the five rates. A model without a table row
prices as None — recorded as unpriced, never as zero.
"""

from __future__ import annotations

from typing import Any

LEDGER_KEYS = ("input", "cache_read", "cache_write_5m", "cache_write_1h", "output")
LOCAL_PREFIXES = ("ollama/", "ollama_chat/")  # local inference: no API bill


def empty_ledger() -> dict[str, int]:
    return dict.fromkeys(LEDGER_KEYS, 0)


def _int(d: Any, key: str) -> int:
    try:
        return int((d or {}).get(key) or 0)
    except (TypeError, ValueError, AttributeError):
        return 0


def _add(ledger: dict[str, Any], model: str, row: dict[str, int]) -> None:
    bucket = ledger.setdefault(model, empty_ledger())
    for k in LEDGER_KEYS:
        bucket[k] += max(0, int(row.get(k, 0)))


# ── per-call accumulators (a harness that sees each response calls these) ──


def add_litellm_usage(ledger: dict[str, Any], model: str, usage: Any) -> None:
    """One litellm / OpenAI-chat response's ``usage`` into the ledger."""
    if usage is None:
        return
    u = usage if isinstance(usage, dict) else _obj_to_dict(usage)
    details = u.get("prompt_tokens_details") or {}
    if not isinstance(details, dict):
        details = _obj_to_dict(details)
    prompt = _int(u, "prompt_tokens")
    cached = _int(details, "cached_tokens") or _int(u, "cache_read_input_tokens")
    write = _int(u, "cache_creation_input_tokens")
    _add(ledger, model, {
        "input": prompt - cached - write,
        "cache_read": cached,
        "cache_write_5m": write,
        "cache_write_1h": 0,
        "output": _int(u, "completion_tokens"),
    })


def add_anthropic_usage(ledger: dict[str, Any], model: str, usage: Any) -> None:
    """One Anthropic Messages / Claude SDK ``usage`` into the ledger."""
    if not isinstance(usage, dict):
        return
    _add(ledger, model, _anthropic_row(usage))


def _anthropic_row(u: dict[str, Any]) -> dict[str, int]:
    cw = u.get("cache_creation") or {}
    w5, w1 = _int(cw, "ephemeral_5m_input_tokens"), _int(cw, "ephemeral_1h_input_tokens")
    if not (w5 or w1):
        w5 = _int(u, "cache_creation_input_tokens")
    return {
        "input": _int(u, "input_tokens"),
        "cache_read": _int(u, "cache_read_input_tokens"),
        "cache_write_5m": w5,
        "cache_write_1h": w1,
        "output": _int(u, "output_tokens"),
    }


def _obj_to_dict(obj: Any) -> dict[str, Any]:
    for attr in ("model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return dict(fn())
            except Exception:
                pass
    return dict(getattr(obj, "__dict__", {}) or {})


# ── one episode's recorded usage -> the ledger ──


def billing_from_usage(usage: Any, model: str) -> dict[str, dict[str, int]] | None:
    """The ledger of one episode from whatever ``agent.usage`` holds;
    ``model`` names the seat's model for the shapes that carry no id.
    None when nothing billable was recorded."""
    if not isinstance(usage, dict) or not usage:
        return None
    if isinstance(usage.get("billing"), dict):
        ledger: dict[str, dict[str, int]] = {}
        for m, row in usage["billing"].items():
            if isinstance(row, dict):
                _add(ledger, str(m), {k: _int(row, k) for k in LEDGER_KEYS})
        return ledger or None
    ledger = {}
    if "cached_input_tokens" in usage:  # codex totals
        cached = _int(usage, "cached_input_tokens")
        _add(ledger, model, {
            "input": _int(usage, "input_tokens") - cached,
            "cache_read": cached,
            "cache_write_5m": _int(usage, "cache_write_input_tokens"),
            "cache_write_1h": 0,
            "output": _int(usage, "output_tokens"),
        })
        return ledger
    if "cache_read_input_tokens" in usage or "cache_creation" in usage or "input_tokens" in usage:
        _add(ledger, model, _anthropic_row(usage))
        return ledger
    if "prompt_tokens" in usage:
        add_litellm_usage(ledger, model, usage)
        return ledger
    return None


def ledger_totals(ledger: dict[str, dict[str, int]] | None) -> dict[str, int]:
    tot = empty_ledger()
    for row in (ledger or {}).values():
        for k in LEDGER_KEYS:
            tot[k] += int(row.get(k, 0))
    return tot


# ── the price table ──


def is_local(model: str) -> bool:
    return str(model).startswith(LOCAL_PREFIXES)


def rates_for(model: str, price_key: str | None = None) -> dict[str, Any] | None:
    """The five per-token rates for a model from ``litellm.model_cost``:
    the explicit ``price_key`` (a models-table row's ``price:``) if given,
    else the model id itself. None when the table has no row."""
    try:
        import litellm
    except ImportError:
        return None
    table = litellm.model_cost
    key = price_key or model
    row = table.get(key)
    if row is None and "/" in key:
        # a litellm routing prefix (anthropic/…, openai/…) is not a table key;
        # the bare id usually is — the resolved key is recorded either way
        key = key.split("/", 1)[1]
        row = table.get(key)
    if row is None:
        return None
    inp = float(row.get("input_cost_per_token") or 0.0)
    if not inp and not float(row.get("output_cost_per_token") or 0.0):
        return None  # a row with no rates (dashscope/… placeholders) is no price
    w5 = row.get("cache_creation_input_token_cost")
    w1 = row.get("cache_creation_input_token_cost_above_1hr")
    return {
        "key": key,
        "input": inp,
        "cache_read": float(row.get("cache_read_input_token_cost") or 0.0),
        # no cache-write rate in the table = the vendor has no cache write
        # tier (OpenAI); a 5m rate without a 1h one = Anthropic's published
        # 2x-input multiplier for the 1h TTL, noted in the record
        "cache_write_5m": float(w5 or 0.0),
        "cache_write_1h": float(w1) if w1 is not None else (2.0 * inp if w5 else 0.0),
        "cache_write_1h_rule": "table" if w1 is not None else ("2x_input" if w5 else "none"),
        "output": float(row.get("output_cost_per_token") or 0.0),
    }


def price(row: dict[str, int], rates: dict[str, Any]) -> float:
    return sum(int(row.get(k, 0)) * float(rates.get(k, 0.0)) for k in LEDGER_KEYS)


def price_ledger(
    ledger: dict[str, dict[str, int]], price_keys: dict[str, str] | None = None
) -> tuple[float | None, str, dict[str, Any]]:
    """USD for a whole ledger: (usd, source, rates-per-model). source is
    ``priced`` when every model priced, ``local`` when every model is local
    inference, ``unpriced`` when any API model has no table row (usd None:
    a partial bill is not a bill)."""
    price_keys = price_keys or {}
    usd, rates_used, missing, api_models = 0.0, {}, [], 0
    for model, row in ledger.items():
        if is_local(model):
            rates_used[model] = None
            continue
        api_models += 1
        override = price_keys.get(model) or price_keys.get(model.split("/", 1)[-1])
        rates = rates_for(model, override)
        rates_used[model] = rates
        if rates is None:
            missing.append(model)
            continue
        usd += price(row, rates)
    if missing:
        return None, "unpriced", rates_used
    if api_models == 0:
        return None, "local", rates_used
    return round(usd, 6), "priced", rates_used
