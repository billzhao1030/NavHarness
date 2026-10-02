"""Recovery as a scope, not a switch.

The binary restart answers *when* an attempt should end and can answer it one
way only: throw the conversation away and start the same task again. That is the right
move for exactly one failure — a mind that has talked itself into a corner —
and the wrong move for every other one. A robot that has been carried into a
different room does not need a clear head; it needs to find out where it is.
One that came back the next day does not need either; it needs to reassess
what it knew about the house.

So instead of a binary restart there is a ladder, under one rule:

    A recovery invalidates the smallest state scope consistent with the
    observed failure.

The ladder is ordered by how much it throws away:

    continue  <  attempt  <  place  <  room  <  floor  <  house

and each rung is a CONTRACT (``SCOPES``) saying which state it invalidates and
which it keeps. That contract is the whole point: a "restart" defined by what
code it runs drifts in meaning every time the code changes. Here a rung is
defined by the state it touches, and the code is whatever carries that out.

Two parties, and they are not equals. The model PROPOSES a rung, because it is
the only one that knows the conversation has gone in circles. The harness
DECIDES, because it is the only one that can see whether the body actually
stopped making progress — and because a model that is confused is, by
construction, not a reliable witness about its own confusion. See
``RecoveryController``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

# ── the ladder ───────────────────────────────────────────────────────────

#: Ordered rungs. The index IS the ordering; escalation walks up this list.
LADDER: tuple[str, ...] = ("continue", "attempt", "place", "room", "floor", "house")


@dataclass(frozen=True)
class Scope:
    """One rung: what it is for, and precisely what it invalidates.

    ``invalidates`` and ``keeps`` are the contract the rest of the harness
    reads. Nothing else may decide what a recovery clears — if a new organ
    holds state, it declares itself here or it is not part of a recovery."""
    name: str
    when: str                      # the failure this rung is the right answer to
    invalidates: tuple[str, ...]
    keeps: tuple[str, ...]
    costs_steps: bool              # does carrying it out move the body?


_ALL = ("context", "pose_belief", "route_belief", "floor_belief",
        "map", "markers", "ledger", "archive")


def _keeps(*invalidated: str) -> tuple[str, ...]:
    return tuple(s for s in _ALL if s not in invalidated)


SCOPES: dict[str, Scope] = {
    "continue": Scope(
        "continue", "nothing is wrong; the attempt is young, not lost",
        (), _keeps(), False),
    "attempt": Scope(
        "attempt", "the reasoning is stuck: re-reading the same views, re-walking cleared ground",
        ("context",), _keeps("context"), False),
    "place": Scope(
        "place", "the body was moved but is still in a room it knows; the pose is what is wrong",
        ("context", "pose_belief"), _keeps("context", "pose_belief"), True),
    "room": Scope(
        "room", "we are in the wrong room, or cannot say which room this is",
        ("context", "pose_belief", "route_belief"),
        _keeps("context", "pose_belief", "route_belief"), True),
    "floor": Scope(
        "floor", "we are on a different storey than the map layer we are planning on",
        ("context", "pose_belief", "route_belief", "floor_belief"),
        _keeps("context", "pose_belief", "route_belief", "floor_belief"), True),
    "house": Scope(
        "house", "we have come back after an interruption or on another day",
        ("context", "pose_belief", "route_belief", "floor_belief"),
        _keeps("context", "pose_belief", "route_belief", "floor_belief"), True),
}
# `house` invalidates no more state than `floor`, and carrying it out is the
# same: the same retained map and the same place-graph brief, nothing reloaded.
# What makes it the higher rung is the breadth of the reassessment the recovery
# prompt asks for — the whole house rather than the storey (prompts.WHAT_WAS_CLEARED).

MODEL_CHOICES: tuple[str, ...] = LADDER[1:]      # the model never proposes "continue"


def rank(scope: str) -> int:
    try:
        return LADDER.index(scope)
    except ValueError:
        return LADDER.index("attempt")           # unknown proposal: the cheapest real rung


def escalate(scope: str) -> str:
    """One rung up. A recovery that did not work is evidence the failure was
    larger than we thought, so repeating it is the one thing not worth doing."""
    return LADDER[min(rank(scope) + 1, len(LADDER) - 1)]


# ── what the harness can see for itself ──────────────────────────────────

@dataclass
class Signals:
    """Observables, all of them from the body rather than from the story.

    Every field is something the model cannot fake by being confident: they
    come from the env's own report and from the harness's own bookkeeping."""
    turns_on_attempt: int = 0
    d_coverage_m2: float = 0.0     # newly explored area since the last checkpoint
    d_evidence: int = 0            # new markers + frames that showed something new
    revisits: int = 0              # checkpoints spent within a metre of a place already stood on
    slam_lost: bool = False
    floor_mismatch: bool = False   # planning layer != the layer the body is on
    steps_left: int = 10 ** 9
    turns_left: int = 10 ** 9
    first_visit: bool = True       # nothing known about this house yet (no places, no notes)
    recoveries_done: int = 0
    last_scope: str = ""

    def progressing(self, *, area: float = 1.5, evidence: int = 1) -> bool:
        """Is this attempt still buying anything? Either new ground or new
        evidence counts; a task can be going well while standing still."""
        return self.d_coverage_m2 >= area or self.d_evidence >= evidence


# ── the controller ───────────────────────────────────────────────────────

@dataclass
class Decision:
    scope: str                     # what will actually happen ("continue" = nothing)
    proposed: str                  # what the model asked for ("" = it did not ask)
    why: str
    confidence: float = 0.0
    adjusted: str = ""             # "downgraded" | "escalated" | "" if honoured as asked

    def acts(self) -> bool:
        return self.scope != "continue"

    def as_record(self) -> dict:
        return asdict(self)


class RecoveryController:
    """u = π_harness(û, c, signals).

    The rules are deliberately few and each one exists because the model's
    proposal is unreliable in a specific, nameable way:

      * it cannot see that it is still making progress -> refuse a premature ask;
      * it cannot tell "I am confused" from "I do not know where I am" -> a lost
        pose escalates past `attempt`, since a clear head does not relocalise;
      * it does not know whether it has been here before -> `house` on a first
        visit has no earlier knowledge of the house to reassess, so it is `room`;
      * it does not know the same rung already failed -> repeats escalate.

    The harness may also act with no proposal at all (``reset_after_turns``):
    the floor under a model that never asks. With ``allow_escalation`` off, a
    repeat recovery is applied at the scope decided for it, even when that
    scope already failed once.
    """

    def __init__(self, *, fallback_turns: int = 80, max_recoveries: int = 2,
                 min_turns: int = 12, allow_escalation: bool = True,
                 ceiling: str = "house") -> None:
        self.fallback_turns = int(fallback_turns or 0)
        self.max_recoveries = int(max_recoveries or 0)
        self.min_turns = int(min_turns or 0)
        self.allow_escalation = bool(allow_escalation)
        self.ceiling = ceiling if ceiling in SCOPES else "house"

    def turns_to_fallback(self, turns_on_attempt: int, recoveries_done: int) -> int | None:
        """Turns the attempt may still take before the fallback threshold is
        reached, or None when the fallback cannot fire (off, the task's
        recoveries spent) or is already due. The threshold is read between
        hops only, so the orchestrator caps a hop at this many turns: the
        fallback then fires at ``fallback_turns``, not at the end of a hop
        that ran past it."""
        if not self.fallback_turns or recoveries_done >= self.max_recoveries:
            return None
        left = self.fallback_turns - int(turns_on_attempt)
        return left if left > 0 else None

    def decide(self, proposal: dict | None, s: Signals) -> Decision:
        asked = bool(proposal)
        want = str((proposal or {}).get("scope") or "attempt")
        conf = float((proposal or {}).get("confidence") or 0.0)
        if want not in SCOPES or want == "continue":
            want = "attempt"

        hit_fallback = bool(self.fallback_turns and s.turns_on_attempt >= self.fallback_turns)
        if not asked and not hit_fallback:
            return Decision("continue", "", "nothing asked, threshold not reached")

        if s.recoveries_done >= self.max_recoveries:
            return Decision("continue", want if asked else "", "recovery budget spent", conf)

        # ── the model asked ──────────────────────────────────────────────
        if asked and not hit_fallback:
            if s.turns_on_attempt < self.min_turns:
                return Decision("continue", want, f"asked at turn {s.turns_on_attempt}, "
                                f"before the attempt is {self.min_turns} turns old", conf, "downgraded")
            if s.progressing() and not (s.slam_lost or s.floor_mismatch):
                return Decision("continue", want, "still gaining ground or evidence", conf, "downgraded")

        scope = want if asked else "attempt"
        why = "model asked" if asked else f"no ask; {s.turns_on_attempt} turns on this attempt"

        # ── corrections the body's own state forces ──────────────────────
        if s.slam_lost and rank(scope) < rank("place"):
            scope, why = "place", why + "; pose is lost, a clear head cannot relocalise"
        if s.floor_mismatch and rank(scope) < rank("floor"):
            scope, why = "floor", why + "; planning layer is not the storey we are on"
        if scope == "house" and s.first_visit:
            scope, why = "room", why + "; first visit to this house, nothing earlier to reassess"
        if (self.allow_escalation and s.last_scope
                and rank(scope) <= rank(s.last_scope) and s.recoveries_done):
            scope = escalate(s.last_scope)
            why += f"; `{s.last_scope}` was already tried and did not hold"

        if rank(scope) > rank(self.ceiling):
            scope, why = self.ceiling, why + f"; capped at `{self.ceiling}`"

        adjusted = ("" if scope == want and asked else
                    "escalated" if rank(scope) > rank(want) else
                    "downgraded" if asked and rank(scope) < rank(want) else "")
        return Decision(scope, want if asked else "", why, conf, adjusted)
