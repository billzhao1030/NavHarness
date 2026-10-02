"""Memory — the two halves, and why they are different objects.

Spatial memory lives in the env process: occupancy layers, the ORB-SLAM3
session, the world anchor, and the agent's markers. It never enters the
context; the agent reaches it only through get_map / mark / preview_path.
It is the part of memory a language model cannot hold, and the part that
must survive any amount of compression. ``SpatialMemory`` here is only a
description of it for the record.

The ledger is the textual half: one row per closed task, written by the
harness after certification, never by the agent directly. What the next
task inherits besides the map. It grows with tasks, not with steps.

The ledger keeps five kinds of thing apart, because blending them is how a
confident mistake becomes doctrine: what the instruments OBSERVED, what the
agent CLAIMED, what was CERTIFIED by the judge, what stays UNRESOLVED, and
NEGATIVE evidence (where the target was looked for and not found). Only
CERTIFIED lines are rendered as facts; a claim never becomes a fact by
being written down. Coordinates are scrubbed from anything the agent
wrote (the workspace's scrubber): a SLAM re-anchor moves the map, text
does not move with it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from ..memory.workspace import scrub_coordinates

VERDICTS = ("complete", "incomplete", "unknown", "none")

@dataclass
class TaskRow:
    index: int
    goal: str
    modality: str
    claim: str                    # complete | incomplete | stop (no claim)
    claim_evidence: str
    verdict: str                  # certify verdict: complete | incomplete | unknown | none
    verdict_evidence: str
    searched: list[str]
    markers: list[str]
    summary: str                  # the agent's own handover (a CLAIM, not a fact)
    steps: int
    end_reason: str               # stop | budget | turns | inactive (close.TERMINATIONS)
    # pre-commit gate bookkeeping
    precheck_verdict: str = ""    # verdict on the FIRST close request, if any
    close_requests: int = 0       # 1 = honoured first time; 2 = refused once, insisted
    # env truth — recorded, NEVER shown to the agent
    truth_success: float | None = None
    truth_distance: float | None = None
    extra: dict = field(default_factory=dict)

    def render_line(self) -> str:
        """The one-line index entry: enough to decide whether to open the full row.

        Verdict, goal, and the places it touched — nothing else. A room name is
        what a later task actually searches on ("was the fridge in the utility
        room?"), so the markers earn their place here while the evidence and the
        handover paragraph do not.
        """
        k = self.index + 1
        tag = {"complete": "CERTIFIED", "incomplete": "NOT DONE",
               "unknown": "UNRESOLVED"}.get(self.verdict, "CLAIMED")
        where = f" · {', '.join(self.markers[:3])}" if self.markers else ""
        goal = (self.goal or "")[:64]
        return f"  {k:>3}. {tag:<10} {goal}{where}"

    def render(self) -> str:
        """One ledger entry the agent reads. Truth is not rendered.

        CERTIFIED  goal + evidence pointer; the handover line stays marked as the robot's account
        NOT DONE   negative evidence: where it was searched, judge's reason
        UNRESOLVED the handover line, marked as the robot's own claim
        CLAIMED    no certification ran; the claim, marked as a claim
        """
        k = self.index + 1
        mk = f"  markers dropped: {', '.join(self.markers)}" if self.markers else ""
        if self.verdict == "complete":
            lines = [f"Task {k} · CERTIFIED complete · goal: {self.goal}"]
            if self.verdict_evidence:
                lines.append(f"  certified by: {self.verdict_evidence}")
            if self.summary:   # only the completion predicate is certified; the account stays a claim
                lines.append(f"  the robot's own account (unverified): {self.summary}")
        elif self.verdict == "incomplete":
            lines = [f"Task {k} · NOT DONE · goal: {self.goal}"]
            if self.searched:
                lines.append(f"  searched without finding it: {', '.join(self.searched)}")
            if self.verdict_evidence:
                lines.append(f"  reason: {self.verdict_evidence}")
            if self.summary:
                lines.append(f"  the robot's own account (unverified): {self.summary}")
        elif self.verdict == "unknown":
            lines = [f"Task {k} · UNRESOLVED · goal: {self.goal}",
                     "  completion could not be confirmed; treat as open"]
            if self.summary:
                lines.append(f"  the robot claimed (unverified): {self.summary}")
        else:  # none: no certification ran
            tag = {"complete": "CLAIMED complete", "incomplete": "CLAIMED not done"}.get(self.claim, "closed")
            lines = [f"Task {k} · {tag} (not checked) · goal: {self.goal}"]
            if self.summary:
                lines.append(f"  the robot's own account: {self.summary}")
        if mk:
            lines.append(mk)
        return "\n".join(lines)


class Ledger:
    def __init__(self) -> None:
        self.rows: list[TaskRow] = []
        self.redactions = 0

    def add(self, row: TaskRow) -> TaskRow:
        row.summary, k1 = scrub_coordinates(row.summary or "")
        row.claim_evidence, k2 = scrub_coordinates(row.claim_evidence or "")
        row.verdict_evidence, k3 = scrub_coordinates(row.verdict_evidence or "")
        self.redactions += k1 + k2 + k3
        self.rows.append(row)
        return row

    def render(self, full_last: int = 0) -> str:
        """The ledger as the agent sees it.

        ``full_last=0`` renders every closed task in full. That is O(k) in the
        number of tasks and it does not stop growing — a few hundred tokens per
        row, so on a long chain it becomes a large share of the prompt. A work
        journal is not something you carry in your head; it is something you
        look up.

        ``full_last=n`` keeps the last n rows in full — recency is what a task
        usually needs — and renders everything older as one line each: verdict,
        goal, and the places it touched. The full row of any task is still
        reachable with ``recall("ledger/<k>")``. Bounded head, complete record.
        """
        if not self.rows:
            return "No tasks closed yet in this house."
        head = (f"Record of the {len(self.rows)} task(s) you have closed in this house. "
                "Only lines marked CERTIFIED were confirmed by the closing check; "
                "CLAIMED and UNRESOLVED lines are your own earlier account and may be wrong.")
        if full_last <= 0 or len(self.rows) <= full_last:
            return head + "\n" + "\n".join(r.render() for r in self.rows)
        old_rows, recent = self.rows[:-full_last], self.rows[-full_last:]
        parts = [head, "",
                 f"Tasks 1-{len(old_rows)}, one line each. "
                 f'Open any of them in full with recall("ledger/<number>"):']
        parts += [r.render_line() for r in old_rows]
        parts += ["", f"The last {len(recent)} in full:"]
        parts += [r.render() for r in recent]
        return "\n".join(parts)

    def row(self, k: int) -> TaskRow | None:
        """The task numbered k as the agent counts them (1-based)."""
        i = int(k) - 1
        return self.rows[i] if 0 <= i < len(self.rows) else None

    def to_json(self) -> list[dict]:
        return [asdict(r) for r in self.rows]

    def __len__(self) -> int:
        return len(self.rows)


@dataclass
class SpatialMemory:
    """Description of the env-side memory, for the record. It holds no data."""
    persists_across_tasks: bool
    pose_source: str = "orb-slam3"
    representation: str = "occupancy layers + markers"
    agent_tools: tuple[str, ...] = ("get_map", "mark", "preview_path")

    def describe(self) -> dict[str, Any]:
        return asdict(self)


def json_dumps(o: Any) -> str:
    return json.dumps(o, ensure_ascii=False, default=str)
