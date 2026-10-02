"""The judge — two checks at the task boundary, one class, and the gate the orchestrator runs.

"The closing check" is two mechanisms with two different powers, and they are
two switches:

``precheck``   BEFORE the STOP is issued. When the agent asks to close, the
               judge looks at the same clean-room evidence and, if its verdict
               is definite and contradicts the claim, the harness returns that
               counter-evidence to the agent INSTEAD of stopping (``unknown``
               never blocks). The
               agent decides: keep going, or ask again. A second request is
               honoured unconditionally (the judge can veto work, never the
               conclusion). This is the mechanism that can change the
               current task's outcome, so it is the one that may move
               s-SR / E-SR.

``certify``    AFTER the STOP is scored. Decides what the ledger says about
               the task: CERTIFIED / NOT DONE / UNRESOLVED. It cannot
               change the current task; it can only protect the next one.
               Its endpoint is successor success, not E-SR.

Judge modes for either: ``none`` | ``model`` (the clean-room judge session).
No mode reads the environment's ground truth: that is for scoring only.

Clean room means: goal + the stop views and a uniform sample of this task's
earlier frames + the map + the mechanical record. The record never contains
the agent's claim.

The model path is a session — ``session/judge.py``'s ``PrecheckSession`` or
``CertifySession`` — and so is ``Verdict``; this module is the mode switch,
the gate on a first close request (``judge_gate``) and the refusal the
agent reads.
"""

from __future__ import annotations

from typing import Any

from ..env.bench import Bench
from ..env.types import Goal
from ..session.judge import (
    CertifySession,
    JudgeSession,
    PrecheckSession,
    Verdict,
    parse_verdict,
    stop_first_sample,
    uniform_sample,
)

MODES = ("none", "model")
KINDS = {"precheck": PrecheckSession, "certify": CertifySession}


class Judge:
    """One judge, two uses. ``mode`` decides where the verdict comes from;
    under ``model`` the verdict is a session's (``**session``: what a
    ``BaseSession`` is opened with — loop, cwd, trace, rec, meter)."""

    def __init__(self, mode: str, model: str | None = None, *, kind: str = "certify",
                 max_frames: int = 12, **session: Any) -> None:
        if mode not in MODES:
            raise ValueError("judge mode must be none | model")
        if mode == "model" and not model:
            raise ValueError("mode=model needs a judge model")
        if kind not in KINDS:
            raise ValueError("judge kind must be precheck | certify")
        self.mode = mode
        self.model = model
        self.kind = kind
        self.session: JudgeSession | None = (
            KINDS[kind](model, max_frames=max_frames, **session) if mode == "model" else None)

    @property
    def model_name(self) -> str | None:
        return self.model if self.mode == "model" else None

    def run(self, goal: Goal, frames: list[dict], record: dict, *, claim: dict | None,
            map_b64: str | None = None) -> Verdict:
        if self.mode == "none":
            # no certification ran: the row stays CLAIMED whatever the agent said
            return Verdict(verdict="none", judge="none", evidence="agent's own claim, not checked")
        rec = dict(record)
        rec.pop("agent_claimed", None)          # the judge never sees the agent's own verdict
        assert self.session is not None
        return self.session.judge(goal, frames, rec, map_b64)


def refusal_text(v: Verdict, claim_status: str) -> str:
    """What the agent reads when its first close request is not honoured."""
    if v.verdict == "incomplete" and claim_status == "complete":
        head = "The closing check does not see the goal where you stopped."
    else:
        head = "The closing check disagrees with your claim."
    tail = " If you still believe you are right, call close_task again with the same status; the second request is honoured."
    ev = f" Evidence: {v.evidence}" if v.evidence else ""
    return head + ev + tail


def judge_gate(gate: Judge | None, bench: Bench, goal: Goal, claim: dict, *, frames: list[dict],
               record: dict) -> tuple[Verdict | None, str | None]:
    """The judge's gate on a FIRST close request: ``(verdict, refusal)``. The
    verdict is None when there is no gate; the refusal is None when the close
    may go through. The judge sees the frames taken so far (the look-around
    the bridge took on the request included), the mechanical record and the
    same map the certification will see."""
    if gate is None or gate.mode == "none":
        return None, None
    try:           # the gate sees the same map the certification will
        map_b64 = (bench.env.call(bench.fn("get_map"), {}) or {}).get("map")
    except Exception:
        map_b64 = None
    v = gate.run(goal, frames, record, claim=claim, map_b64=map_b64)
    if v.disagrees_with(claim["status"]):
        return v, refusal_text(v, claim["status"])
    return v, None


__all__ = ["KINDS", "MODES", "Judge", "Verdict", "judge_gate", "parse_verdict", "refusal_text",
           "stop_first_sample", "uniform_sample"]
