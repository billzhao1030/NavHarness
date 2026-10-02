"""The recovery driver of navh on a stub env and a scripted wake-up session.

No server, no model, no tokens: the driver is exercised on a stub env, a stub
recorder and a scripted wake-up session. The wake-up floor tests run
against the driver itself; the rest checks reading the body, carrying λ
and its handover variants, the switches handed to the controller, and
applying a rung. The ladder through the orchestrator is
``tests/test_navh_orchestrator.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.harnesses.navh.config import LadderConfig
from core.harnesses.navh.env.types import Goal
from core.harnesses.navh.orchestrator.recovery.driver import RecoveryDriver
from core.harnesses.navh.orchestrator.recovery.lam import Handover, Lambda, note_words


class _Rec:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, kind: str, **payload) -> None:
        self.events.append((kind, payload))

    def kinds(self) -> list[str]:
        return [k for k, _ in self.events]


class _Env:
    def __init__(self, replies: dict | None = None) -> None:
        self.calls: list = []
        self.replies = replies or {}

    def call(self, verb: str, inputs: dict) -> dict:
        self.calls.append((verb, dict(inputs)))
        return dict(self.replies.get(verb, {}))


def bench_on(replies: dict | None = None):
    return SimpleNamespace(env=_Env(replies), fn=lambda n: n)


class _Wake:
    """A wake-up session that answers with a script, one text per question."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.asked: list = []

    def ask(self, content):
        self.asked.append(content)
        return self.answers.pop(0) if self.answers else ""


def driver(tmp_path: Path | None = None, rec=None, wake=None, **ladder) -> RecoveryDriver:
    return RecoveryDriver(rec=rec or _Rec(), out_dir=tmp_path, wake=wake, ladder=LadderConfig(**ladder))


GOAL = Goal(index=0, total=3, text='find a "chair"')
FOUND = {"place_recognised": True, "place_guess": "upstairs landing", "place_confidence": 0.88}


# ── reading the body ──


def test_signals_are_deltas_against_the_last_checkpoint() -> None:
    d = driver()
    common = dict(steps_used=40, step_budget=500, turns=20, max_turns=100, resets=[])
    s1 = d.signals(body={"explored_m2": 10.0}, frames=8, marks=1, attempt_turns=20, **common)
    assert s1.d_coverage_m2 == 10.0 and s1.d_evidence == 2 + 1 and s1.steps_left == 460
    s2 = d.signals(body={"explored_m2": 10.5, "slam": "lost"}, frames=9, marks=1, attempt_turns=25, **common)
    assert s2.d_coverage_m2 == 0.5 and s2.d_evidence == 0 and s2.slam_lost is True
    d.new_task(area=10.5)                    # the retained map, as the body reads it at the next task's start
    s3 = d.signals(body={"explored_m2": 11.0}, frames=0, marks=0, attempt_turns=1, **common)
    assert s3.d_coverage_m2 == 0.5, "a new task measures from the map it inherits, not from an empty one"
    d.new_task(area=10.5)
    s4 = d.signals(body={}, frames=0, marks=0, attempt_turns=1, **common)
    assert s4.d_coverage_m2 == 0.0, "no motion reply yet: the area is the start's"
    s5 = d.signals(body={"explored_m2": 12.0}, frames=4, marks=0, attempt_turns=2, **common)
    assert s5.d_coverage_m2 == 1.5 and s5.d_evidence == 1
    d.new_task()
    assert d.signals(body={"explored_m2": 3.0}, frames=0, marks=0, attempt_turns=1,
                     **common).d_coverage_m2 == 3.0, "an empty map at the start: everything is new"
    d.first_visit = False
    assert d.signals(body={}, frames=0, marks=0, attempt_turns=1, **common).first_visit is False


def test_the_baseline_area_is_the_bodys_reading_off_observe() -> None:
    rec = _Rec()
    d = driver(rec=rec)
    assert d.explored(bench_on({"observe": {"rgb": "x", "explored_m2": 42.5}})) == 42.5
    assert d.explored(bench_on({"observe": {"rgb": "x", "info": {"explored_m2": 7.0}}})) == 7.0
    assert d.explored(bench_on({"observe": {"rgb": "x"}})) == 0.0, "a bare body reports no area"

    class _Dead:
        def call(self, verb, inputs):
            raise RuntimeError("env gone")

    assert d.explored(SimpleNamespace(env=_Dead(), fn=lambda n: n)) == 0.0
    assert rec.kinds() == ["warning"]


# ── the note that crosses ──


def test_carry_hands_on_the_attempt_s_own_note() -> None:
    written = json.dumps({"searched": ["kitchen"], "ruled_out": [], "last_anchor": "", "untried": [],
                          "uncertain": [], "recommended_scope": "attempt"})
    d = driver()
    note = d.carry(GOAL.text, written, 1)
    assert note.source == "own" and note.lam.searched == ["kitchen"] and note.text == written
    assert note.words == 1 and d.writes_note()
    assert d.carry(GOAL.text, "", 1).source == "own-empty"


# ── carrying the rung out ──


def test_attempt_costs_no_steps_and_place_looks_around() -> None:
    d = driver()
    bench = bench_on({"look_around": {"views": ["F", "R", "B", "L"]}, "get_map": {"floor": 1}})
    out = d.apply_scope("attempt", bench)
    assert out == {"scope": "attempt", "invalidated": ["context"], "brief": ""} and bench.env.calls == []
    out = d.apply_scope("place", bench)
    assert [v for v, _ in bench.env.calls] == ["look_around", "get_map"]
    assert out["invalidated"] == ["context", "pose_belief"] and out["brief"].startswith("Re-establish")
    assert out["views"] == ["F", "R", "B", "L"], "no saved photograph named the place: the session gets the views"


def test_views_are_read_from_the_list_or_from_the_four_keys() -> None:
    from core.harnesses.navh.orchestrator.recovery.driver import _views_of

    assert _views_of({"views": ["a", "b"]}) == ["a", "b"]
    assert _views_of({"view_front": "f", "view_right": "r", "view_back": "b", "view_left": "l"}) == ["f", "r", "b", "l"]
    assert _views_of({"views": [], "view_front": "f"}) == ["f"]
    assert _views_of({}) == []


def test_record_appends_emits_and_writes_the_lesson_file(tmp_path: Path) -> None:
    from core.harnesses.navh.orchestrator.recovery.ladder import Decision, Signals

    rec = _Rec()
    d = driver(tmp_path, rec=rec)
    resets: list = []
    dec = Decision("place", "place", "model asked", 0.7)
    note = Handover(Lambda(untried=["hall"]), "own", '{"untried": ["hall"]}', words=1)
    row = d.record(2, GOAL, dec=dec, sig=Signals(turns_on_attempt=15), note=note,
                   applied={"scope": "place", "brief": "x"}, turns=40,
                   attempt_turns=15, steps_now=120, reason="circles", resets=resets)
    assert resets == [row] and row["attempt"] == 1 and row["trigger"] == "self" and row["lam_source"] == "own"
    assert rec.kinds() == ["recovery"] and rec.events[0][1]["scope"] == "place"
    lines = (tmp_path / "lessons.jsonl").read_text().splitlines()
    assert len(lines) == 1 and '"task": 2' in lines[0]


# ── the switches the driver hands the controller ──


def test_recovery_off_means_no_fallback_either() -> None:
    assert driver(recovery=True, reset_after_turns=80).ctrl.fallback_turns == 80
    assert driver(recovery=False, reset_after_turns=80).ctrl.fallback_turns == 0


def test_escalation_is_a_switch_that_reaches_the_controller() -> None:
    from core.harnesses.navh.orchestrator.recovery.ladder import Signals

    on, off = driver(), driver(recovery_escalation=False)
    assert on.ctrl.allow_escalation is True and off.ctrl.allow_escalation is False
    on.ctrl.max_recoveries = off.ctrl.max_recoveries = 2
    on.ctrl.min_turns = off.ctrl.min_turns = 0
    again = Signals(turns_on_attempt=30, recoveries_done=1, last_scope="attempt")
    ask = {"scope": "attempt", "confidence": 0.8}
    assert on.ctrl.decide(ask, again).scope == "place", "a repeat at the same rung goes one up"
    assert off.ctrl.decide(ask, again).scope == "attempt", "without escalation it is applied as decided"


def test_a_floor_mismatch_sends_a_recovery_to_the_floor_scope() -> None:
    """The body reports it is off the floor layer the map plans on: the
    reading reaches the controller, which lifts the progress guard and
    raises the rung to ``floor`` — asked for or not."""
    from core.harnesses.navh.orchestrator.recovery.ladder import RecoveryController, Signals

    d = driver(recovery_min_turns=12, max_resets_per_task=1)
    common = dict(steps_used=40, step_budget=500, turns=20, max_turns=200, resets=[])
    d.new_task(area=0.0)
    sig = d.signals(body={"explored_m2": 30.0, "floor_mismatch": True}, frames=8, marks=1,
                    attempt_turns=20, **common)
    assert sig.floor_mismatch is True and sig.progressing(), "still gaining ground, on the wrong layer"
    dec = d.ctrl.decide({"scope": "attempt", "confidence": 0.6}, sig)
    assert dec.scope == "floor" and dec.adjusted == "escalated" and "storey" in dec.why
    assert d.ctrl.decide({"scope": "room"}, sig).scope == "floor", "below floor is raised to it"
    d.new_task(area=0.0)
    level = d.signals(body={"explored_m2": 30.0, "floor_mismatch": False}, frames=8, marks=1,
                      attempt_turns=20, **common)
    assert d.ctrl.decide({"scope": "attempt"}, level).scope == "continue", "on its layer, progress refuses"
    fallback = RecoveryController(fallback_turns=80, max_recoveries=1, min_turns=12)
    assert fallback.decide(None, Signals(turns_on_attempt=80, floor_mismatch=True)).scope == "floor"
    capped = RecoveryController(fallback_turns=0, max_recoveries=1, min_turns=0, ceiling="room")
    assert capped.decide({"scope": "attempt"}, Signals(turns_on_attempt=20, floor_mismatch=True)).scope == "room"


def test_house_reassesses_and_reloads_nothing() -> None:
    from core.harnesses.navh import prompts as P
    from core.harnesses.navh.orchestrator.recovery.ladder import SCOPES, RecoveryController, Signals

    assert SCOPES["house"].invalidates == SCOPES["floor"].invalidates
    assert SCOPES["house"].keeps == SCOPES["floor"].keeps
    house = P.WHAT_WAS_CLEARED["house"]
    assert "reload" not in house and "reassess" in house and "whole house" in house
    d = RecoveryController(fallback_turns=0, max_recoveries=1, min_turns=0)
    first = d.decide({"scope": "house"}, Signals(turns_on_attempt=20, first_visit=True))
    assert first.scope == "room" and "reload" not in first.why
    bench = bench_on({"look_around": {"views": ["F", "R", "B", "L"]}, "get_map": {"floor": 0}})
    floor, house_out = driver().apply_scope("floor", bench), driver().apply_scope("house", bench)
    assert {k: v for k, v in floor.items() if k != "scope"} == {k: v for k, v in house_out.items() if k != "scope"}


# ── what crosses: the handover variants ──


def _note(goal: str, *untried: str) -> dict:
    return {"searched": ["hall"], "ruled_out": [], "last_anchor": "hall", "untried": list(untried),
            "uncertain": [], "recommended_scope": "attempt"}


def _bank(tmp_path: Path) -> Path:
    """An earlier structured run: two chains' lessons.jsonl, one row per recovery."""
    rows = [
        {"task": 0, "goal": GOAL.text, "lam": _note(GOAL.text, "the attic"), "lam_source": "own", "lam_words": 4},
        {"task": 1, "goal": "find a bed", "lam": _note("find a bed", "the guest room"), "lam_source": "own",
         "lam_words": 8},
        {"task": 2, "goal": "find a sink", "lam": _note("find a sink", "the scullery"), "lam_source": "empty"},
    ]
    for i, row in enumerate(rows):
        d = tmp_path / "run" / f"navh_{i}"
        d.mkdir(parents=True)
        (d / "lessons.jsonl").write_text(json.dumps(row) + "\n")
    return tmp_path / "run"


def test_load_bank_reads_a_run_directory_and_a_json_list(tmp_path: Path) -> None:
    from core.harnesses.navh.orchestrator.recovery.lam import load_bank, mean_words

    bank = load_bank(str(_bank(tmp_path)))
    assert [b.goal for b in bank] == [GOAL.text, "find a bed"], "only notes the attempt handed over as its own"
    assert mean_words(bank) == 6
    listed = tmp_path / "notes.json"
    listed.write_text(json.dumps([_note("x", "porch"), {"unrelated": 1}]))
    got = load_bank(str(listed))
    assert len(got) == 1 and got[0].lam.untried == ["porch"] and got[0].words == got[0].lam.words()
    assert load_bank("") == []
    with pytest.raises(FileNotFoundError):
        load_bank(str(tmp_path / "missing"))


def test_empty_handover_writes_nothing_and_hands_nothing() -> None:
    d = driver(recovery_handover="empty")
    assert not d.writes_note()
    note = d.carry(GOAL.text, "", 1)
    assert note.source == "empty" and note.text == "" and note.lam.is_empty()


def test_other_hands_on_a_note_written_for_a_different_goal(tmp_path: Path) -> None:
    d = driver(recovery_handover="other", recovery_handover_bank=str(_bank(tmp_path)))
    own = json.dumps(_note(GOAL.text, "the garage"))
    note = d.carry(GOAL.text, own, 1)
    assert note.source == "other" and note.lam.untried == ["the guest room"], "never this goal's own note"
    assert note.own is not None and note.own.untried == ["the garage"]
    assert json.loads(note.text)["untried"] == ["the guest room"] and note.words == note_words(note.text)
    assert any(b.lam.untried == ["the garage"] for b in d.pool), "the attempt's own note is banked for others"
    # without a bank, only this run's own notes can be lent
    rec = _Rec()
    solo = driver(rec=rec, recovery_handover="other")
    first = solo.carry("find a bed", json.dumps(_note("find a bed", "loft")), 1)
    assert first.source == "other-unavailable" and first.text == "" and "warning" in rec.kinds()
    second = solo.carry(GOAL.text, json.dumps(_note(GOAL.text, "porch")), 1)
    assert second.source == "other" and second.lam.untried == ["loft"]


def test_summary_is_prose_cut_to_the_banks_mean_length(tmp_path: Path) -> None:
    d = driver(recovery_handover="summary", recovery_handover_bank=str(_bank(tmp_path)))
    assert d.summary_words == 6 and d.writes_note()
    written = "I walked the hall twice and then\nchecked the garage, where nothing was."
    note = d.carry(GOAL.text, written, 1)
    assert note.source == "summary" and note.lam.is_empty()
    assert note.text == "I walked the hall twice and" and note.words == 6
    assert note.render() == note.text
    assert d.carry(GOAL.text, "", 1).source == "summary-empty"
    with pytest.raises(ValueError):
        driver(recovery_handover="summary")              # nothing to match the length to
    with pytest.raises(ValueError):
        driver(recovery_handover="prose")


def test_note_words_counts_content_not_json() -> None:
    from core.harnesses.navh.orchestrator.recovery.lam import trim_words

    assert note_words(json.dumps(_note("x", "the guest room"))) == 1 + 1 + 3
    assert note_words("two words") == 2 and note_words("") == 0
    assert trim_words("a b\nc d", 3) == "a b\nc" and trim_words("a b", 5) == "a b"
