"""How a task closes, and what the loop hands back.

The loop itself is ``Orchestrator._run_task`` (``orchestrator.py``); this
module holds what it returns (``TaskOutcome``), how a run aborts when the
provider refuses (``ProviderUnavailable``, ``check_provider``), and the two
ways the harness closes a task against the env: ``stop`` after the agent's
close request went through the gates, ``force_stop`` when the task ends
without one (``TERMINATIONS``: a spent turn or step budget, or a model that
has stopped acting). The agent's own close request is answered by the
judge's gate (``judge.py``); the bridge only records it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..env.bench import Bench
from ..env.types import TaskEnd
from ..session import Hop


@dataclass
class TaskOutcome:
    end: TaskEnd
    claim: dict
    frames: list[dict]
    steps_used: int
    marks: list[str]
    tool_calls: dict
    turns: int
    peak_prompt_tokens: int
    meta: dict = field(default_factory=dict)


class ProviderUnavailable(RuntimeError):
    """The provider refused (session / rate limit, auth). Stop the run:
    burning the turn budget against a refusal produces fake failed tasks."""


REFUSAL_WORDS = ("session limit", "usage limit", "rate limit", "quota",
                 "out of credit", "insufficient credit", "too many requests",
                 # Auth, not budget, but the same thing has to happen: an OAuth
                 # token that expires mid-run makes the CLI answer every turn
                 # with this sentence instead of a tool call, and every remaining
                 # task would burn to the turn cap as a failure.
                 "failed to authenticate", "oauth", "not logged in",
                 "authentication_error", "invalid api key", "unauthorized")


def check_provider(hop: Hop) -> None:
    """Raise if the hop stopped because we are out of budget, not out of ideas.
    A non-success subtype alone is NOT a refusal: ``error_max_turns`` is the
    ordinary end of a hop capped at ``chunk_turns``."""
    tail = " ".join(str(s.get("text") or "") for s in hop.story[-3:] if s.get("role") == "assistant")
    low = (hop.text + " " + tail + " " + (hop.error or "")).lower()
    if any(w in low for w in REFUSAL_WORDS):
        raise ProviderUnavailable((hop.text or hop.error or "").strip()[:200] or "provider refused")


#: How a task ends when the harness, not a closure, ends it — ``end_reason`` and
#: the words its record carries in place of a claim. None of them claims anything.
TERMINATIONS: dict[str, str] = {
    "turns": "turn budget exhausted",
    "budget": "step budget exhausted",
    "inactive": "inactive: hops in a row without a tool call",
}


def claim_status(status: str) -> str:
    return "incomplete" if str(status).lower().startswith("in") else "complete"


def stop(bench: Bench, claimed: str) -> TaskEnd:
    """The STOP: the env action that ends the task, issued by the harness once
    the agent's close request has passed the gates."""
    out = bench.env.call(bench.fn("step"), {"action": 0})
    if isinstance(out, dict) and "error" in out:
        raise RuntimeError(f"STOP refused by the env: {out['error']}")
    _, end = bench.parse_step(out, claimed)
    if end is None:
        end = TaskEnd(end_reason="stop", score=dict(out.get("metrics") or {}),
                      chain_finished=bool(out.get("terminated")))
    return end


def force_stop(bench: Bench, reason: str = "turns") -> TaskEnd:
    """The task ends without a closure (``reason`` is one of ``TERMINATIONS``):
    close it the way the harness closes it, scored where the robot stands.
    The orchestrator takes the stop views and the map before calling this."""
    out = bench.env.call(bench.fn("step"), {"action": 0})
    _, end = bench.parse_step(out, "stop")
    if end is None:
        end = TaskEnd(end_reason=reason, score=dict(out.get("metrics") or {}),
                      chain_finished=bool(out.get("terminated")))
    end.end_reason = reason
    return end


def end_from_state(bench: Bench, st: dict) -> TaskEnd:
    """The env ended the task inside a tool call (the step budget): the
    bridge kept the env's payload; read the TaskEnd off it."""
    payload = st.get("end_payload") or {}
    claimed = (st.get("close_request") or {}).get("status")
    _, end = bench.parse_step(payload, claimed)
    if end is None:
        end = TaskEnd(end_reason=str(st.get("end_reason") or "budget"),
                      score=dict(payload.get("metrics") or {}), chain_finished=bool(payload.get("terminated")))
    return end
