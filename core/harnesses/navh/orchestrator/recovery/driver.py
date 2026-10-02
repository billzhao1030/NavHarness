"""The ladder, carried out.

The orchestrator (``Orchestrator._recover``) hands over the body's state as
read from the bridge's state file and the note the session wrote to its file;
this driver does the rest of a recovery: it decides the rung
(``RecoveryController``), decides what the next attempt is handed
(``carry``, by ``LadderConfig.recovery_handover``), carries the rung out
against the env, writes the lesson file and the record.

Two facts are set from outside by the orchestrator, because they belong to
the house rather than to the driver: ``places`` (the place graph) and
``marker_photos`` (the views banked at each marker: this run's off the
bridge's state file, earlier runs' from the house's photo sets);
``first_visit`` says whether an archive exists for this house at all.
"""

from __future__ import annotations

import base64
import contextlib
import json
from pathlib import Path

from ...config import LadderConfig
from ...env.bench import Bench, map_floor
from ...env.types import Goal
from ...session.wakeup import WakeupSession
from .ladder import SCOPES, Decision, RecoveryController, Signals
from .lam import (
    HANDOVERS,
    Banked,
    Handover,
    Lambda,
    load_bank,
    mean_words,
    note_words,
    pick_other,
    read_note,
    trim_words,
)
from .wakeup import CoarseToFine

#: Four views back from ``look_around``, whichever way the env words them.
_VIEW_KEYS = ("view_front", "view_right", "view_back", "view_left")


def _views_of(got: dict) -> list[str]:
    views = [v for v in (got.get("views") or []) if isinstance(v, str) and v]
    if not views:
        views = [got[k] for k in _VIEW_KEYS if isinstance(got.get(k), str) and got[k]]
    return views[:4]


class RecoveryDriver:
    def __init__(self, *, rec, out_dir: Path | None, wake: WakeupSession | None,
                 ladder: LadderConfig | None = None) -> None:
        ladder = ladder or LadderConfig()
        self.rec = rec
        self.out_dir = out_dir
        # the one session this driver opens: the wake-up question of a rung
        # above `attempt` (λ is the orchestrator's: a file the agent's own
        # session writes on its last hop)
        self.wake = wake
        # recovery off means off: no fallback either
        self.ctrl = RecoveryController(
            fallback_turns=int(ladder.reset_after_turns or 0) if ladder.recovery else 0,
            max_recoveries=int(ladder.max_resets_per_task or 0),
            min_turns=int(ladder.recovery_min_turns or 0),
            allow_escalation=bool(ladder.recovery_escalation), ceiling=ladder.recovery_ceiling)
        self.handover = str(ladder.recovery_handover or "structured")
        if self.handover not in HANDOVERS:
            raise ValueError(f"recovery_handover must be one of {HANDOVERS}, got {self.handover!r}")
        # `other` lends from the bank and from this run's own notes as they are
        # written; `summary` is cut to the bank's mean length
        bank = load_bank(ladder.recovery_handover_bank) if self.handover in ("other", "summary") else []
        self.pool: list[Banked] = list(bank)
        self.summary_words = mean_words(bank) if self.handover == "summary" else 0
        if self.handover == "summary" and not self.summary_words:
            raise ValueError("recovery_handover=summary matches the summary to the length of structured "
                             "notes: recovery_handover_bank must name an earlier run that wrote some")
        self.places = None                       # PlaceGraph, set by the chain runner
        self.marker_photos: dict = {}            # label -> views: the house's photo sets, then this run's
        self.first_visit = True
        self._last_check: dict | None = None

    # ── reading the body ─────────────────────────────────────────────

    def new_task(self, area: float = 0.0) -> None:
        """A task starts: its first deltas are measured from here. ``area`` is
        the map's explored area as the body reads it at the task's start — the
        map the task inherits — so ground mapped before the task is not
        counted as ground the task gained. Frames and markers are the task's
        own and start from zero."""
        self._last_check = {"area": float(area or 0.0), "frames": 0, "marks": 0}

    def explored(self, bench: Bench) -> float:
        """The whole map's explored area as the body reports it now: read off
        ``observe``, whose reply carries the body's readings as a motion reply
        does; 0 when the env reports none. Best-effort, like ``_look``."""
        try:
            out = bench.env.call(bench.fn("observe"), {}) or {}
        except Exception as e:
            self.rec.emit("warning", kind_detail="baseline_area", error=str(e)[:200])
            return 0.0
        info = out.get("info") if isinstance(out.get("info"), dict) else {}
        try:
            return float(out.get("explored_m2", info.get("explored_m2")) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def signals(self, *, body: dict, frames: int, marks: int, steps_used: int, step_budget: int,
                attempt_turns: int, turns: int, max_turns: int, resets: list[dict]) -> Signals:
        """Read the body, not the story. Deltas are measured against the last
        checkpoint so that "progress" means recent progress, not lifetime. A
        body that has reported no area yet this task (no motion reply) has the
        area of the last checkpoint."""
        prev = self._last_check or {"area": 0.0, "frames": 0, "marks": 0}
        area = (float(body["explored_m2"]) if body.get("explored_m2") is not None
                else float(prev.get("area") or 0.0))
        sig = Signals(
            turns_on_attempt=attempt_turns,
            d_coverage_m2=max(0.0, area - float(prev.get("area") or 0.0)),
            d_evidence=max(0, (frames - int(prev.get("frames") or 0)) // 4)
            + max(0, marks - int(prev.get("marks") or 0)),
            slam_lost=str(body.get("slam") or "ok") not in ("ok", "", "None"),
            floor_mismatch=bool(body.get("floor_mismatch")),
            steps_left=max(0, int(step_budget) - int(steps_used or 0)),
            turns_left=max(0, int(max_turns) - int(turns)),
            first_visit=bool(self.first_visit),
            recoveries_done=len(resets),
            last_scope=(resets[-1].get("scope", "") if resets else ""))
        self._last_check = {"area": area, "frames": frames, "marks": marks}
        return sig

    def refuse(self, k: int, dec: Decision, attempt_turns: int) -> None:
        self.rec.emit("recovery_refused", index=k, scope=dec.proposed, why=dec.why,
                      turns=attempt_turns, confidence=dec.confidence)

    # ── the note that crosses ────────────────────────────────────────

    def writes_note(self) -> bool:
        """Does the dying attempt write a note? Every handover but ``empty``
        asks for one: ``other`` banks it without handing it on, ``summary``
        asks for prose instead of λ."""
        return self.handover != "empty"

    def carry(self, goal: str, written: str, attempt: int) -> Handover:
        """What the next attempt reads, from what the dying attempt ``written``.

        ``structured`` hands the note on as written; ``empty`` hands nothing;
        ``summary`` hands the prose on, cut to the bank's mean length;
        ``other`` banks the attempt's own note and hands on a note written for
        a different goal (nothing, when there is none to lend)."""
        if self.handover == "empty":
            return Handover(Lambda(), "empty", "")
        if self.handover == "summary":
            text = trim_words(written, self.summary_words)
            return Handover(Lambda(), "summary" if text else "summary-empty", text, words=note_words(text))
        own = read_note(written)
        if self.handover == "structured":
            return Handover(own, "own" if not own.is_empty() else "own-empty", written,
                            words=note_words(written))
        if not own.is_empty():
            self.pool.append(Banked(goal=goal, lam=own, words=note_words(written)))
        got = pick_other(self.pool, goal, attempt)
        if got is None:
            self.rec.emit("warning", kind_detail="no_note_to_lend", goal=goal[:80], pool=len(self.pool))
            return Handover(Lambda(), "other-unavailable", "", own=own)
        text = json.dumps(got.lam.as_record(), ensure_ascii=False, indent=1)
        return Handover(got.lam, "other", text, own=own, words=note_words(text))

    # ── carrying the rung out ────────────────────────────────────────

    def _look(self, bench: Bench, warn: str) -> tuple[list[str], int | None]:
        """Four views and the map's level, each best-effort: a recovery runs deep
        inside a chain that has already cost hours, and a picture that cannot
        be taken must cost the brief, not the run."""
        views: list[str] = []
        try:
            views = _views_of(bench.env.call(bench.fn("look_around"), {}) or {})
        except Exception as e:
            self.rec.emit("warning", kind_detail=warn, error=str(e)[:200])
        floor = None
        with contextlib.suppress(Exception):
            floor = map_floor(bench.env.call(bench.fn("get_map"), {}) or {})
        return views, floor

    def _funnel(self, views: list[str], floor: int | None, out: dict) -> None:
        """Stage 1 of the coarse-to-fine funnel (which place, from the views)
        and stage 3 (what is known there). The metric pose is the tracker's;
        recognising a room does not determine it."""
        try:
            c2f = CoarseToFine(self.places, self.wake)
            f = c2f.recognise(views, self.photos_for, floor_hint=floor)
            f = c2f.restore(f)
            out["funnel"] = f.as_record()
            out["brief"] = c2f.brief(f)
        except Exception as e:
            self.rec.emit("warning", kind_detail="recovery_funnel", error=str(e)[:200])

    def apply_scope(self, scope: str, bench: Bench) -> dict:
        """Carry out the rung, honouring SCOPES as the single source of truth
        about what a recovery may touch.

        What the body can contribute here is narrower than it looks.
        ``look_around`` is SLAM-silent — four views, back to the entry pose,
        nothing fed to the map — so it can say what the place LOOKS like but
        cannot re-lock a lost pose. Rungs above ``attempt`` therefore run stage
        1 of the wake-up funnel and are honest that stages 2-3 did not happen."""
        s = SCOPES[scope]
        out: dict = {"scope": scope, "invalidated": list(s.invalidates), "brief": ""}
        if not s.costs_steps:
            return out
        views, floor = self._look(bench, "recovery_look_around")
        if floor is not None:
            out["floor"] = int(floor)
        if self.places is not None and views:
            self._funnel(views, floor, out)
        if views and not (out.get("funnel") or {}).get("place_recognised"):
            # no saved photograph named the place: the navigation session
            # reasons about its location from the views themselves
            out["views"] = views
        if not out["brief"]:
            out["brief"] = ("Re-establish which room you are in before planning any route: turn on "
                            "the spot and look for a landmark you named before.")
        if scope in ("floor", "house") and self.places is not None:
            shape = self.places.render()
            if shape:
                out["brief"] += "\n\n" + shape
        return out

    def photos_for(self, label: str, k: int) -> list[str]:
        """The views banked at a named place. The banked entry holds a PATH, not
        the picture: a deployment drops dozens of markers and they are read on
        the one occasion a wake-up asks for them. An unreadable file costs that
        place its vote, not the recovery."""
        out: list[str] = []
        for v in ((self.marker_photos or {}).get(label) or [])[:k]:
            if not isinstance(v, dict):
                continue
            if v.get("b64"):
                out.append(v["b64"])
                continue
            try:
                out.append(base64.b64encode(Path(v["path"]).read_bytes()).decode())
            except (OSError, KeyError, TypeError):
                continue
        return out

    # ── the record ───────────────────────────────────────────────────

    def record(self, k: int, goal: Goal, *, dec: Decision, sig: Signals, note: Handover,
               applied: dict, turns: int, attempt_turns: int,
               steps_now: int, reason: str, resets: list[dict]) -> dict:
        rec = {"attempt": len(resets) + 1, "trigger": "self" if dec.proposed else "cap",
               "scope": dec.scope, "at_turn": turns, "attempt_turns": attempt_turns,
               "at_step": steps_now, "lam": note.lam.as_record(), "lam_source": note.source,
               "lam_words": note.words, "handover": self.handover,
               "decision": dec.as_record(),
               "applied": applied, "signals": sig.__dict__.copy(),
               "lesson": note.render(), "reason": reason}
        if note.own is not None:
            rec["lam_own"] = note.own.as_record()
        if self.handover == "summary":
            rec["summary_words"] = self.summary_words
        resets.append(rec)
        self.rec.emit("recovery", index=k, **{x: rec[x] for x in
                                              ("attempt", "trigger", "scope", "at_turn",
                                               "attempt_turns", "at_step", "applied", "lam_source")},
                      why=dec.why, adjusted=dec.adjusted, proposed=dec.proposed)
        self.write_lesson(k, goal, rec)
        return rec

    def write_lesson(self, k: int, goal: Goal, reset: dict) -> None:
        """Failed attempts are experience too: one append-only file per run."""
        if self.out_dir is None:
            return
        row = {"task": k, "goal": goal.text, "modality": goal.modality, **reset}
        try:
            with (self.out_dir / "lessons.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except OSError as e:
            self.rec.emit("warning", kind_detail="lesson_write_failed", error=str(e)[:200])
