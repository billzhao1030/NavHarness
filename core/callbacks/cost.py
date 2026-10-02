"""CostLogger — what the run cost, per episode and in total, on ONE
convention for every harness: the tokens the provider reported x litellm's
list price (core/pricing.py). The harness's own figure, when it has one
(Claude Code's CLI-settled ``total_cost_usd``, litellm's meter under the
mini-swe-agent core), is
kept beside it as ``settled_usd`` with the deviation ``delta_pct`` — a
cross-check, never the number.

    record["agent"]["cost"] = {
        usd, source,            # priced | local | unpriced | unmetered | fake
        settled_usd, delta_pct, # the harness's own bill, and (usd - settled) / settled
        tokens,                 # input · cache_read · cache_write_5m · cache_write_1h · output
        billing,                # the same, per model id (a judge model bills separately)
        price,                  # the per-token rates used, per model id — the number is traceable
        api_calls, by_tool, usd_per_turn,
    }

source: ``priced`` — every model in the ledger has a table row; ``local``
— ollama, no API bill; ``unpriced`` — an API model litellm has no row for
(usd None, a warning line, never zero); ``unmetered`` — the harness
recorded no usage; ``fake`` — the scripted harness or endpoint, usd 0.

For the Claude SDK stream every AssistantMessage's usage (deduped by
message_id) becomes one API call with a billing weight and the tool it
issued, using the SAME weights reporting/run_stats.py apportions with; the
episode's usd is split across those calls into ``by_tool``. At run end
``summary["cost"]`` = total / mean / median / min / max over the priced
episodes + sources + settled total + token totals + the price snapshot +
``by_issuing_tool`` (2 dp, like stats.json).

Run budget: ``run.budget_usd=<float>`` makes this callback raise StopRun
once the priced episodes so far (resumed ones included) exceed it — the
runner drains, and ``summary.cost.stopped_by_budget`` says so. Removing
this callback removes exactly those keys; the harness totals the runner
records (``agent.usage`` / ``agent.total_cost_usd``) stay.

Re-price a finished run (a price table update, a models-table ``price:``
fix, a run recorded without this callback)::

    python -m core.callbacks.cost <run_dir> [--price <model>=<litellm key> …]

rewrites every ``agent.cost`` and ``summary.cost`` in summary.json (with
``cost.priced_at``) and regenerates stats.json / stats.html.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from core.callbacks.base import Callback, StopRun
from core.pricing import billing_from_usage, ledger_totals, price_ledger


def _stats(vals: list[float]) -> dict[str, float]:
    vs = sorted(vals)
    return {"total": round(sum(vs), 4), "mean": round(sum(vs) / len(vs), 4),
            "median": round(vs[len(vs) // 2], 4),
            "min": round(vs[0], 4), "max": round(vs[-1], 4)}


def price_episode(
    agent: dict[str, Any],
    model: str,
    *,
    price_keys: dict[str, str] | None = None,
    fake: bool = False,
    calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One episode's ``agent.cost`` from its recorded usage. ``calls`` are
    the weighted API calls (tool, weight) the usd is apportioned over; None
    when the stream was not read (the re-price CLI reads them off raw/)."""
    settled = agent.get("total_cost_usd")
    ledger = billing_from_usage(agent.get("usage"), model)
    rates: dict[str, Any] = {}
    if fake:
        usd, source = 0.0, "fake"
    elif ledger is None:
        usd, source = None, "unmetered"
    else:
        usd, source, rates = price_ledger(ledger, price_keys)
        if source == "unpriced":
            missing = [m for m, r in rates.items() if r is None and not m.startswith("ollama")]
            print(f"[std] cost: no litellm price for {missing} — usd left null "
                  f"(models table: add `price: <litellm key>` to the row)")
    delta = None
    if usd is not None and settled:
        delta = round((usd - float(settled)) / float(settled) * 100.0, 1)
    by_tool: dict[str, float] | None = None
    if calls:
        total_w = sum(c["weight"] for c in calls)
        if usd is not None and total_w > 0:
            by_tool = {}
            for c in calls:
                key = c["tool"] or "final_text"
                by_tool[key] = round(by_tool.get(key, 0.0) + usd * c["weight"] / total_w, 6)
    turns = agent.get("num_turns")
    return {
        "usd": usd,
        "source": source,
        "settled_usd": settled,
        "delta_pct": delta,
        "tokens": ledger_totals(ledger) if ledger else None,
        "billing": ledger,
        "price": rates or None,
        "api_calls": len(calls) if calls is not None else None,
        "by_tool": by_tool,
        "usd_per_turn": round(usd / turns, 6) if usd and turns else None,
    }


def summarize(episodes: list[dict[str, Any]], *, budget_usd: float | None = None,
              stopped_by_budget: bool = False) -> dict[str, Any]:
    """``summary["cost"]`` over episode records."""
    costs = [rec["agent"]["cost"] for rec in episodes if (rec.get("agent") or {}).get("cost")]
    usds = [float(c["usd"]) for c in costs if c.get("usd") is not None]
    settled = [float(c["settled_usd"]) for c in costs if c.get("settled_usd") is not None]
    sources: dict[str, int] = {}
    by_tool: dict[str, float] = {}
    tokens: dict[str, int] = {}
    prices: dict[str, Any] = {}
    for c in costs:
        sources[c.get("source", "?")] = sources.get(c.get("source", "?"), 0) + 1
        for tool, usd in (c.get("by_tool") or {}).items():
            by_tool[tool] = by_tool.get(tool, 0.0) + usd
        for k, v in (c.get("tokens") or {}).items():
            tokens[k] = tokens.get(k, 0) + int(v)
        for m, r in (c.get("price") or {}).items():
            prices.setdefault(m, r)
    out: dict[str, Any] = {**(_stats(usds) if usds else {}), "priced_episodes": len(usds),
                           "sources": sources}
    if settled:
        out["settled_total"] = round(sum(settled), 4)
        if usds and len(usds) == len(settled) and sum(settled) > 0:
            out["delta_pct"] = round((sum(usds) - sum(settled)) / sum(settled) * 100.0, 1)
    out["tokens"] = tokens or None
    out["price"] = prices or None
    out["by_issuing_tool"] = {k: round(v, 2) for k, v in sorted(by_tool.items(), key=lambda kv: -kv[1])}
    out["budget_usd"] = budget_usd
    out["stopped_by_budget"] = stopped_by_budget
    return out


def _is_fake(app: dict[str, Any]) -> bool:
    harness = (app.get("harness") or {}).get("root") or (app.get("agent") or {}).get("harness", {}).get("root")
    api = ((app.get("agent") or {}).get("api") or {}).get("api_gateway")
    # a scripted harness, the fake endpoint, or a run flagged run.fake=true
    return harness == "fake" or api == "fake" or bool((app.get("run") or {}).get("fake"))


class CostLogger(Callback):
    writes = ("agent.cost", "cost")

    def __init__(self) -> None:
        # the apportioning weights live with the report that shares them
        from reporting.run_stats import _CACHE_WRITE_WEIGHTS, _TOKEN_WEIGHTS
        self._token_w = _TOKEN_WEIGHTS
        self._cache_w = _CACHE_WRITE_WEIGHTS
        self._calls: dict[int, list[dict[str, Any]]] = {}
        self._seen: dict[int, set[str]] = {}
        self.budget_usd: float | None = None
        self.spent_usd = 0.0
        self.stopped_by_budget = False
        self._model = ""
        self._price_keys: dict[str, str] = {}
        self._fake = False

    # ── per run ──

    def on_run_start(self, run: Any) -> None:
        budget = run.cfg.get("budget_usd")
        self.budget_usd = float(budget) if budget is not None else None
        model = (run.app.get("agent") or {}).get("model") or {}
        self._model = str(model.get("id") or "")
        # a models-table row's `price: <litellm key>` names the table row for
        # an id litellm does not know under that name (codex's gpt-5.6-sol …)
        if model.get("price"):
            self._price_keys = {self._model: str(model["price"])}
        self._fake = _is_fake(run.app)
        self.spent_usd = 0.0
        for rec in run.prior_episodes.values():
            agent = rec.get("agent") or {}
            usd = (agent.get("cost") or {}).get("usd")
            if usd is None:
                usd = agent.get("total_cost_usd")
            self.spent_usd += float(usd or 0.0)

    # ── per API call (Claude SDK stream) ──

    def on_episode_start(self, ep: Any) -> None:
        self._calls[ep.index] = []
        self._seen[ep.index] = set()

    def on_raw(self, ep: Any, rec: Any) -> None:
        calls = self._calls.get(ep.index)
        if calls is None or not isinstance(rec, dict):
            return
        if rec.get("type") == "AssistantMessage":
            msg = rec.get("msg") or {}
            mid = msg.get("message_id")
            usage = msg.get("usage") or {}
            tool = None
            for block in (msg.get("content") or []):
                if isinstance(block, dict) and block.get("_type") == "ToolUseBlock":
                    tool = str(block.get("name", "?")).replace("mcp__env__", "")
            seen = self._seen[ep.index]
            if mid and mid not in seen and usage:
                seen.add(mid)
                cw = usage.get("cache_creation") or {}
                weight = sum(usage.get(k, 0) * w for k, w in self._token_w.items())
                weight += sum(cw.get(k, 0) * w for k, w in self._cache_w.items())
                calls.append({"tool": tool, "weight": weight})
            elif mid in seen and tool and calls and calls[-1]["tool"] is None:
                calls[-1]["tool"] = tool  # tool block arrived in a later batch

    # ── per episode ──

    def on_episode_end(self, ep: Any, record: dict[str, Any] | None) -> None:
        calls = self._calls.pop(ep.index, [])
        self._seen.pop(ep.index, None)
        if record is None:
            return
        agent = record.setdefault("agent", {})
        cost = price_episode(agent, self._model, price_keys=self._price_keys,
                             fake=self._fake, calls=calls)
        agent["cost"] = cost
        usd = cost["usd"]
        if usd:
            self.spent_usd += float(usd)
        if self.budget_usd is not None and self.spent_usd > self.budget_usd:
            self.stopped_by_budget = True
            raise StopRun(f"run budget {self.budget_usd:.2f} USD exceeded "
                          f"({self.spent_usd:.2f} spent)")

    def on_run_end(self, run: Any, summary: dict[str, Any]) -> None:
        summary["cost"] = summarize(list(run.episodes.values()), budget_usd=self.budget_usd,
                                    stopped_by_budget=self.stopped_by_budget)


# ── re-price a finished run ──


def reprice(run_dir: Path, price_keys: dict[str, str] | None = None, *, stats: bool = True) -> dict[str, Any]:
    from reporting.run_stats import _raw_calls

    path = run_dir / "summary.json"
    summary = json.loads(path.read_text())
    config = summary.get("config") or {}
    model = str(config.get("model") or "")
    fake = summary.get("harness") == "fake" or (config.get("extra") or {}).get("api_gateway") == "fake" \
        or str(summary.get("run_name", "")).startswith("fake")
    old = summary.get("cost") or {}
    for rec in summary.get("episodes") or []:
        agent = rec.setdefault("agent", {})
        calls = _raw_calls(run_dir / "raw" / f"episode_{rec.get('index')}.jsonl")
        agent["cost"] = price_episode(agent, model, price_keys=price_keys, fake=fake,
                                      calls=calls if calls else None)
    summary["cost"] = summarize(summary.get("episodes") or [], budget_usd=old.get("budget_usd"),
                                stopped_by_budget=bool(old.get("stopped_by_budget")))
    summary["cost"]["priced_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        import litellm
        summary["cost"]["litellm"] = getattr(litellm, "version", None) or _dist_version("litellm")
    except Exception:
        pass
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    if stats:
        from reporting.run_stats import generate
        generate(run_dir)
    return summary["cost"]


def _dist_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="re-price a finished run from its recorded usage")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--price", action="append", default=[], metavar="MODEL=KEY",
                        help="litellm table row for a model id (repeatable)")
    parser.add_argument("--no-stats", action="store_true", help="skip stats.json / stats.html")
    args = parser.parse_args(argv)
    keys = dict(kv.split("=", 1) for kv in args.price)
    cost = reprice(args.run_dir, keys or None, stats=not args.no_stats)
    print(json.dumps({k: cost.get(k) for k in ("total", "mean", "priced_episodes", "sources",
                                                 "settled_total", "delta_pct", "tokens")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
