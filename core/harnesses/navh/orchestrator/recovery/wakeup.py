"""Waking up somewhere, coarse to fine.

A robot that has been moved cannot be relocalised in one step with one
outcome: when that fails there is nothing to say about WHY, and "the next
task failed" is compatible with four completely different breakdowns. So
waking up is staged, and each stage is reported:

    1. place recognised    which room / storey is this, from the views alone
    2. pose relocalised    metric lock, searched only in that candidate's area
    3. old state reused    the markers and relations for that place come back
    4. successor succeeds  the next task actually works

This is the funnel. Reporting only the last number hides that a run can fail at
stage 1 (no idea where we are) and a run can fail at stage 4 having sailed
through 1-3 (knew exactly where it was, still could not do the task) — and
those two ask for opposite fixes.

Stage 1 is answered with what the harness already banks: every marker carries
the views taken at that spot, so "where am I" is a comparison between what the
camera sees now and those photo sets. The comparison itself is delegated to a
``WakeupSession`` — this module owns the staging, the candidate set and the
record, not the perception.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ...memory.placegraph import PlaceGraph
from ...session import image_block

PLACE_QUESTION = """\
The robot has just been moved and does not know where it is. The first images \
are what it sees now, taken by turning on the spot. The images after them are \
photographs it took earlier at places it named in this house.

Which of the named places is the robot standing in or next to now? Answer with \
JSON only:

  {"place": "<one of the labels below, or empty if none matches>",
   "confidence": <0.0-1.0>,
   "why": "<one sentence, by what you can see>"}

Labels, in the order their photographs appear: %s"""


@dataclass
class Funnel:
    """One wake-up, stage by stage. ``None`` means "did not get this far"."""
    place_recognised: bool | None = None
    place_guess: str = ""
    place_confidence: float = 0.0
    place_correct: bool | None = None     # only knowable when a ground truth exists
    pose_relocalised: bool | None = None
    state_reused: bool | None = None
    markers_restored: int = 0
    steps_to_first_room: int = -1
    successor_success: float | None = None
    candidates: list[str] = field(default_factory=list)
    notes: str = ""

    def as_record(self) -> dict:
        return asdict(self)

    def deepest_stage(self) -> str:
        if not self.place_recognised:
            return "place"
        if not self.pose_relocalised:
            return "pose"
        if not self.state_reused:
            return "state"
        return "task"


class CoarseToFine:
    """Stages a wake-up over the place graph.

    The saving is not only diagnostic. A blind relocalisation searches the
    whole house; once stage 1 has named a candidate, stage 2 only has to agree
    with that one place — which is both cheaper and less likely to lock onto a
    room that merely looks similar."""

    def __init__(self, graph: PlaceGraph, session, *, top_k: int = 4,
                 photos_per_place: int = 3) -> None:
        self.graph = graph
        self.session = session          # a WakeupSession, or None when no model judges
        self.top_k = int(top_k)
        self.photos_per_place = int(photos_per_place)

    # ── stage 1 ──────────────────────────────────────────────────────────

    def recognise(self, views_b64: list[str], photos_for, *, floor_hint: int | None = None) -> Funnel:
        """Name the place from the views, before anything metric is attempted.

        ``photos_for(label, k) -> list[str]`` hands back that place's banked
        views; it is passed in rather than imported so this works against both
        the live marker photos and the long-term archive."""
        f = Funnel()
        pool = (self.graph.candidates_on(floor_hint) if floor_hint is not None
                else sorted(self.graph.places.values(), key=lambda p: (-p.photos, -p.visits)))
        pool = [p for p in pool if p.photos][:self.top_k]
        f.candidates = [p.label for p in pool]
        if not pool or not views_b64 or self.session is None:
            f.place_recognised = False
            f.notes = "no banked photographs to compare against" if not pool else "nothing to compare"
            return f
        # the session's own block shape (session/wire/blocks.py): a block built any
        # other way arrives as no picture at all, and the model then answers
        # from the labels while sounding as if it had looked
        content: list[dict] = [{"type": "text", "text": "What the robot sees now:"}]
        for b in views_b64[:4]:
            content.append(image_block(b))
        for p in pool:
            content.append({"type": "text", "text": f"Photographs taken at: {p.address()}"})
            for b in photos_for(p.label, self.photos_per_place):
                content.append(image_block(b))
        content.append({"type": "text", "text": PLACE_QUESTION % ", ".join(f.candidates)})
        try:
            import json as _json
            import re as _re
            raw = self.session.ask(content)
            m = _re.search(r"\{.*\}", raw or "", _re.S)
            d = _json.loads(m.group(0)) if m else {}
        except Exception as e:
            f.place_recognised = False
            f.notes = f"place question failed: {str(e)[:120]}"
            return f
        guess = str(d.get("place") or "").strip()
        f.place_guess = guess if guess in self.graph.places else ""
        f.place_confidence = float(d.get("confidence") or 0.0)
        f.place_recognised = bool(f.place_guess)
        f.notes = str(d.get("why") or "")[:200]
        return f

    # ── stage 3 (stage 2, the metric pose, is the tracker's) ─────────────

    def restore(self, f: Funnel) -> Funnel:
        """Stage 3: bring back what is known about the place we decided we are in."""
        if not f.place_guess:
            f.state_reused = False
            return f
        p = self.graph.places.get(f.place_guess)
        if p is None:
            f.state_reused = False
            return f
        siblings = self.graph.rooms().get(p.room, [])
        f.markers_restored = len(siblings)
        f.state_reused = True
        return f

    def brief(self, f: Funnel) -> str:
        """What the agent is told after a wake-up. Never a coordinate, and
        never more certainty than the funnel earned."""
        if not f.place_recognised:
            return ("You have been moved and cannot yet tell where you are. Turn on the spot, "
                    "look for a landmark you named before, and re-establish which room this is "
                    "before planning any route.")
        p = self.graph.places[f.place_guess]
        line = f"You have been moved. From what you can see you are at or beside {p.address()}."
        if f.place_confidence < 0.6:
            line += " That identification is not certain — confirm it before relying on it."
        sib = [s for s in self.graph.rooms().get(p.room, []) if s != p.label]
        if sib:
            line += " Other places you named in this room: " + ", ".join(sib) + "."
        if not f.pose_relocalised:
            line += (" The map has not re-locked onto your position, so treat routes as unverified "
                     "until a landmark confirms them.")
        return line
