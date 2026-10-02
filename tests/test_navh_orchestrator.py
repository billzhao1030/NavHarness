"""The orchestrator over scripted sessions — the gates, the ladder with λ as a file, the seam, the day end.

No server, no CLI, no tokens: every session's wire is a script that plays the bridge's part by editing
the state file and the workspace the way the real tools would, then answers as a ``Hop``; the env is a
stub bench that scores a STOP. What is checked is what the orchestrator does BETWEEN hops: the file
gate, then the judge's gate (a precheck session of its own), then the STOP; a recovery that runs the
session's last hop for λ and reopens it; the session kept across tasks under ``compact: token``; the
env ending a task inside a hop; a provider refusal aborting the run; the consolidate session writing
the notes; the summary's sessions report.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from core.harnesses.navh import prompts as P
from core.harnesses.navh.config import HarnessConfig
from core.harnesses.navh.env.types import Goal, TaskEnd
from core.harnesses.navh.orchestrator import Orchestrator
from core.harnesses.navh.orchestrator.close import ProviderUnavailable
from core.harnesses.navh.session import Hop, prompt_text
from core.harnesses.navh.session.base import BaseSession

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
PNG = base64.b64encode(PNG_BYTES).decode()


# ── the stub env and bench ──

class _Env:
    url = "http://stub"

    def __init__(self) -> None:
        self.calls: list = []

    def call(self, fn: str, inputs: dict | None = None, config: dict | None = None) -> dict:
        self.calls.append((fn, dict(inputs or {})))
        if fn == "step" and int((inputs or {}).get("action", -1)) == 0:
            return {"info": {"subtask_result": {"success": 1.0, "distance_to_goal": 0.4}}, "terminated": False}
        if fn == "get_map":
            return {"map": PNG, "current_floor": 0}
        if fn == "look_around":
            return {"views": [PNG] * 4}
        return {}

    def stops(self) -> int:
        return sum(1 for fn, a in self.calls if fn == "step" and a.get("action") == 0)


class _Bench:
    house_id = "stub-house"

    def __init__(self, n: int = 1) -> None:
        self.env = _Env()
        self.n = n
        self.k = 0

    @staticmethod
    def fn(name: str) -> str:
        return name

    def _goal(self, k: int) -> Goal:
        return Goal(index=k, total=self.n, text=f"find thing {k}", modality="object")

    def seat(self, arg, persist_map, skip=0) -> Goal:
        self.k = skip
        return self._goal(self.k)

    def parse_step(self, out: dict, claimed) -> tuple[dict, TaskEnd | None]:
        info = out.get("info") or {}
        res = info.get("subtask_result")
        if not res:
            return info, None
        return info, TaskEnd(end_reason=str(info.get("end_reason") or "stop"), score=dict(res), chain_finished=False)

    def after_end(self, end: TaskEnd, persist_map) -> TaskEnd:
        self.k += 1
        if self.k >= self.n:
            end.chain_finished, end.next_goal = True, None
        else:
            end.next_goal = self._goal(self.k)
        return end

    def chain_report(self) -> dict:
        return {}


# ── the scripted wire: it plays the bridge, one handler per session kind ──

class _Wire:
    orch: Orchestrator | None = None          # set by the test
    handlers: dict[str, Any] = {}             # kind -> fn(wire, prompt) -> Hop
    opened_kinds: list[str] = []

    def __init__(self, loop, options, *, on_message=None) -> None:
        self.session = on_message.__self__   # the BaseSession this wire serves
        self.kind = self.session.kind
        self.opened = False
        self.prompts: list = []
        self.limits: list = []                 # the turn cap each hop was given

    @property
    def is_open(self) -> bool:
        return self.opened

    def open(self) -> dict:
        self.opened = True
        _Wire.opened_kinds.append(self.kind)
        return {"mcp": "connected"}

    def hop(self, prompt, max_turns=None) -> Hop:
        self.prompts.append(prompt)
        self.limits.append(max_turns)
        return self.handlers[self.kind](self, prompt)

    def close(self) -> None:
        self.opened = False


def _hop(text: str = "DONE", turns: int = 1, *tools: str) -> Hop:
    return Hop(text=text, num_turns=turns, subtype="success", session_id="sid",
               usage={"input_tokens": 100, "output_tokens": 10, "iterations": [{"input_tokens": 120}]},
               tool_uses=[{"id": f"t{i}", "name": t, "input": {}} for i, t in enumerate(tools)])


def _frame(orch: Orchestrator, st: dict, kind: str) -> None:
    live = orch.out_dir / "live"
    live.mkdir(parents=True, exist_ok=True)
    fid = len(st["frames"]) + 1
    p = live / f"obs_{fid:04d}.png"
    p.write_bytes(PNG_BYTES)
    st["frames"].append({"id": fid, "path": str(p), "step": st["steps_used"], "kind": kind})


def _close_request(orch: Orchestrator, status: str = "complete") -> None:
    st = orch.state.read()
    for _ in range(4):
        _frame(orch, st, "stop")
    st["close_request"] = {"status": status, "evidence": "it is in view", "steps_used": st["steps_used"]}
    st["close_requests"] += 1
    st["calls"]["close_task"] = st["calls"].get("close_task", 0) + 1
    orch.state.write(st)


def _write(orch: Orchestrator, rel: str, text: str) -> None:
    st = orch.state.read()
    out = orch.ws.session_write(rel, text, set(st["writable"]))
    assert "written" in out, out
    st["written"].append(rel)
    orch.state.write(st)


def _walk(orch: Orchestrator, steps: int = 3) -> None:
    st = orch.state.read()
    _frame(orch, st, "observe")
    st["steps_used"] += steps
    st["body"] = {"explored_m2": 1.0, "slam": "ok"}
    st["calls"] = {"observe": 1, "step": 1}
    orch.state.write(st)


def _agent_closes_properly(wire: _Wire, prompt) -> Hop:
    """A model that walks, is refused once for the missing handover, writes it, and closes."""
    orch = _Wire.orch
    text = prompt_text(prompt)
    if text.startswith("Task"):
        _walk(orch)
        _close_request(orch)
        return _hop("DONE", 3, "observe", "step", "close_task")
    if text.startswith("Not closed yet"):
        _write(orch, orch.ws.rel(orch.ws.handover_path(orch.k)), "Went to the kitchen; the thing was there.")
        _close_request(orch)
        return _hop("DONE", 2, "write_file", "close_task")
    if "closing check" in text:
        _close_request(orch)
        return _hop("DONE", 1, "close_task")
    raise AssertionError(f"unexpected prompt: {text[:80]}")


def _judge_says(verdict: str):
    def handler(wire: _Wire, prompt) -> Hop:
        assert isinstance(prompt, list) and any(b.get("type") == "image" for b in prompt), "the judge sees frames"
        return _hop(json.dumps({"verdict": verdict, "evidence": "the frames", "confidence": "high"}), 1)
    return handler


def _consolidates(wire: _Wire, prompt) -> Hop:
    orch = _Wire.orch
    text = prompt_text(prompt)
    assert "house/journal/" in text and "run/ledger.md" in text and "run/tasks" in text
    st = orch.consolidate_state.read()
    writable = set(st["writable"])
    for path, body in (("house/house.md", "kitchen connects to the hall; the thing is in the kitchen"),
                       ("house/rooms/kitchen.md", "the thing is here (CERTIFIED, task 1)"),
                       ("house/MEMORY.md", "- house — the overview\n- rooms/kitchen — where the thing is")):
        out = orch.ws.session_write(path, body, writable)
        assert "written" in out, out
        st["written"].append(path)
    assert "error" in orch.ws.session_write("run/ledger.md", "x", writable), "only the notes are writable"
    orch.consolidate_state.write(st)
    return _hop("DONE", 3, "read_file", "write_file")


@pytest.fixture
def scripted(monkeypatch):
    monkeypatch.setattr(BaseSession, "wire_cls", _Wire)
    _Wire.orch = None
    _Wire.handlers = {"agent": _agent_closes_properly, "precheck": _judge_says("incomplete"),
                      "certify": _judge_says("complete"), "consolidate": _consolidates,
                      "wakeup": lambda w, p: _hop('{"place": ""}', 1)}
    _Wire.opened_kinds = []
    yield _Wire
    _Wire.orch = None


def _orch(tmp_path: Path, bench: _Bench, **switches) -> Orchestrator:
    switches.setdefault("precheck", "none")       # the gate is tested on its own, below
    switches.setdefault("max_turns_per_task", 40)
    cfg = HarnessConfig.from_flat(benchmark="es", chain_id=0, memory_root=str(tmp_path / "mem"),
                                  place_graph=False, **switches)
    o = Orchestrator(cfg, "claude-test", None, bench, tmp_path / "out", bridge=tmp_path / "bridge.py")
    _Wire.orch = o
    return o


# ── the gates ──

def test_the_file_gate_then_the_judges_gate_then_the_stop(tmp_path: Path, scripted) -> None:
    bench = _Bench(n=1)
    o = _orch(tmp_path, bench, precheck="model", certify="model", notes=True)
    summary = o.run()
    assert summary["exit_status"] == "chain_complete" and bench.env.stops() == 1
    kinds = [json.loads(line)["kind"] for line in (o.out_dir / "episode_0.jsonl").read_text().splitlines()]
    assert kinds.index("handover_prompted") < kinds.index("precheck_refused") < kinds.index("boundary")
    assert o.ws.handover_path(0).read_text().startswith("Went to the kitchen")
    assert (o.ws.ledger.read_text().count("\n") == 1) and o.ws.verdict_path(0).exists()
    row = o.rec.rows[0]
    assert row["claim"]["status"] == "complete" and row["precheck"]["close_requests"] == 3
    assert row["verdict"]["judge"] == "gate-reused", "the agent did not move after the gate: certification reused"
    assert row["context"]["handover_prompts"] == 1 and row["context"]["files_written"] == ["run/tasks/0/handover.md"]
    report = summary["sessions"]
    assert report["agent"]["opened"] == 1 and report["precheck"]["opened"] == 1 and report["certify"]["opened"] == 0
    assert report["consolidate"]["opened"] == 1 and summary["notes"]["model_written"] is True
    assert o.notes.overview.read_text().startswith("kitchen connects")
    # the index is the session's own: the harness adds nothing to the notes
    assert o.notes.index.read_text() == "- house — the overview\n- rooms/kitchen — where the thing is\n"
    assert summary["notes"]["files"] == ["house/house.md", "house/rooms/kitchen.md", "house/MEMORY.md"]
    journal = o.notes.journal_today.read_text()
    assert "task 1 · NOT DONE" in journal and "account: Went to the kitchen" in journal  # the gate said incomplete
    assert _Wire.opened_kinds == ["agent", "precheck", "consolidate"]
    assert not o.loop.thread.is_alive(), "the sessions' loop is stopped with the chain"


def test_without_a_precheck_the_second_request_goes_straight_to_the_stop(tmp_path: Path, scripted) -> None:
    bench = _Bench(n=1)
    o = _orch(tmp_path, bench, precheck="none", certify="model", notes=False)
    summary = o.run()
    assert bench.env.stops() == 1 and o.rec.rows[0]["precheck"]["close_requests"] == 2
    assert summary["sessions"]["certify"]["opened"] == 1 and o.rec.rows[0]["verdict"]["verdict"] == "complete"
    assert summary["notes"] is None and not o.notes.index.exists()
    assert o.notes.journal_today.exists(), "the journal is a task record: written with or without consolidation"


# ── the session's lifetime ──

def test_compact_token_keeps_the_session_across_tasks(tmp_path: Path, scripted) -> None:
    o = _orch(tmp_path, _Bench(n=2), notes=False, compact="token")
    o.run()
    assert o.agent_session.runs == 1 and len(o.rec.rows) == 2
    o2 = _orch(tmp_path / "b", _Bench(n=2), notes=False, compact="boundary")
    o2.run()
    assert o2.agent_session.runs == 2


def test_the_env_ending_a_task_inside_a_hop_needs_no_stop(tmp_path: Path, scripted) -> None:
    def agent(wire, prompt):
        orch = _Wire.orch
        _walk(orch, steps=50)
        st = orch.state.read()
        st.update(task_over=True, end_reason="budget",
                  end_payload={"info": {"subtask_result": {"success": 0.0}, "end_reason": "budget"}})
        orch.state.write(st)
        return _hop("DONE", 4, "step")
    _Wire.handlers["agent"] = agent
    bench = _Bench(n=1)
    o = _orch(tmp_path, bench, notes=False)
    o.run()
    assert bench.env.stops() == 0 and o.rec.rows[0]["end_reason"] == "budget"
    assert o.rec.rows[0]["claim"]["status"] == "stop"


def test_a_provider_refusal_aborts_the_run_and_stops_the_loop(tmp_path: Path, scripted) -> None:
    _Wire.handlers["agent"] = lambda w, p: _hop("You have hit your usage limit for today.", 1)
    o = _orch(tmp_path, _Bench(n=1), notes=False)
    with pytest.raises(ProviderUnavailable):
        o.run()
    assert not o.loop.thread.is_alive()


def test_three_hops_without_a_turn_are_the_provider_and_abort_the_run(tmp_path: Path, scripted) -> None:
    """A hop that produced no turn at all is the provider refusing, not the
    model thinking: three in a row abort the run instead of ending the task
    as a failure. A hop with a turn in between starts the count again."""
    hops = iter([_hop("", 0), _hop("", 0), _hop("DONE", 2, "observe"), _hop("", 0), _hop("", 0), _hop("", 0)])
    seen: list = []

    def agent(wire, prompt):
        seen.append(prompt)
        return next(hops)

    _Wire.handlers["agent"] = agent
    bench = _Bench(n=1)
    o = _orch(tmp_path, bench, notes=False)
    with pytest.raises(ProviderUnavailable, match="3 hops in a row produced no turn"):
        o.run()
    assert len(seen) == 6 and bench.env.stops() == 0 and o.rec.rows == []
    assert not o.loop.thread.is_alive()


# ── the ladder, with λ as the session's last hop ──

def test_the_ladder_runs_through_the_orchestrator_and_lambda_is_the_sessions_last_hop(tmp_path: Path, scripted) -> None:
    seen: list[str] = []

    def agent(wire, prompt):
        orch = _Wire.orch
        text = prompt_text(prompt)
        seen.append(text[:24])
        if text.startswith("Task"):
            _walk(orch)
            st = orch.state.read()
            st["recovery_request"] = {"scope": "attempt", "reason": "going in circles", "confidence": 0.9}
            orch.state.write(st)
            return _hop("DONE", 6, "observe", "step", "request_recovery")
        if "write the note your next self will read" in text:
            st = orch.state.read()
            assert st["only_files"] is True and "run/tasks/0/lambda.md" in st["writable"]
            assert wire.session.runs == 1, "λ is written by the SAME session, before it is closed"
            _write(orch, "run/tasks/0/lambda.md", json.dumps({"searched": ["kitchen"], "ruled_out": [],
                                                              "last_anchor": "hall", "untried": ["garage"],
                                                              "uncertain": [], "recommended_scope": "attempt"}))
            return _hop("DONE", 1, "write_file")
        if text.startswith("This is a fresh attempt"):
            assert wire.session.runs == 2, "a recovery at `attempt` voids the context: a new session"
            assert "run/tasks/0/lambda.md" in text and orch.state.read()["only_files"] is False
            _write(orch, "run/tasks/0/handover.md", "Second attempt: found it in the garage.")
            _close_request(orch)
            return _hop("DONE", 3, "read_file", "write_file", "close_task")
        raise AssertionError(text[:80])

    _Wire.handlers["agent"] = agent
    bench = _Bench(n=1)
    o = _orch(tmp_path, bench, notes=False, recovery=True, max_resets_per_task=1, recovery_min_turns=0)
    o.run()
    assert bench.env.stops() == 1 and o.agent_session.runs == 2
    row = o.rec.rows[0]
    assert len(row["resets"]) == 1 and row["resets"][0]["scope"] == "attempt"
    assert row["resets"][0]["lam"]["untried"] == ["garage"] and row["resets"][0]["lam_source"] == "own"
    kinds = [json.loads(line)["kind"] for line in (o.out_dir / "episode_0.jsonl").read_text().splitlines()]
    assert kinds.index("lambda_ask") < kinds.index("lambda_file") < kinds.index("recovery")
    assert (o.out_dir / "lessons.jsonl").exists()


def test_a_young_or_progressing_attempt_is_refused_a_recovery(tmp_path: Path, scripted) -> None:
    asks = {"n": 0}

    def agent(wire, prompt):
        orch = _Wire.orch
        text = prompt_text(prompt)
        if text.startswith("Task") or text == "Continue.":
            asks["n"] += 1
            st = orch.state.read()
            st["body"] = {"explored_m2": 10.0 * asks["n"]}          # still gaining ground
            st["recovery_request"] = {"scope": "room", "reason": "lost", "confidence": 0.5}
            orch.state.write(st)
            if asks["n"] >= 2:
                _write(orch, "run/tasks/0/handover.md", "gave up")
                _close_request(orch, "incomplete")
                return _hop("DONE", 2, "write_file", "close_task")
            return _hop("DONE", 20, "step", "request_recovery")
        raise AssertionError(text[:80])

    _Wire.handlers["agent"] = agent
    o = _orch(tmp_path, _Bench(n=1), notes=False, recovery=True, max_resets_per_task=2, recovery_min_turns=12)
    o.run()
    kinds = [json.loads(line)["kind"] for line in (o.out_dir / "episode_0.jsonl").read_text().splitlines()]
    assert "recovery_refused" in kinds and "recovery" not in kinds and o.agent_session.runs == 1


def test_a_later_task_measures_new_area_from_the_map_it_inherits(tmp_path: Path, scripted) -> None:
    """The second task starts on a 50 m² map and gains nothing: its first
    checkpoint sees no new area, so a request for recovery is not refused as
    progress for the ground the first task mapped."""
    class _MappedEnv(_Env):
        def call(self, fn: str, inputs: dict | None = None, config: dict | None = None) -> dict:
            if fn == "observe":
                self.calls.append((fn, dict(inputs or {})))
                return {"rgb": PNG, "info": {"explored_m2": 50.0}}
            return super().call(fn, inputs, config)

    def agent(wire, prompt):
        orch = _Wire.orch
        text = prompt_text(prompt)
        if orch.k == 0:
            return _agent_closes_properly(wire, prompt)
        if text.startswith("Task"):
            st = orch.state.read()
            st["body"] = {"explored_m2": 50.0, "slam": "ok"}
            st["recovery_request"] = {"scope": "attempt", "reason": "going in circles", "confidence": 0.9}
            orch.state.write(st)
            return _hop("DONE", 6, "step", "request_recovery")
        if "write the note your next self will read" in text:
            _write(orch, "run/tasks/1/lambda.md", json.dumps({"searched": ["hall"], "untried": ["garage"]}))
            return _hop("DONE", 1, "write_file")
        if text.startswith("This is a fresh attempt"):
            _write(orch, "run/tasks/1/handover.md", "Found it in the garage.")
            _close_request(orch)
            return _hop("DONE", 3, "write_file", "close_task")
        raise AssertionError(text[:80])

    _Wire.handlers["agent"] = agent
    bench = _Bench(n=2)
    bench.env = _MappedEnv()
    o = _orch(tmp_path, bench, notes=False, recovery=True, max_resets_per_task=1, recovery_min_turns=0)
    o.run()
    assert [fn for fn, _ in bench.env.calls].count("observe") == 2, "the baseline is read at each task's start"
    second = o.rec.rows[1]
    assert len(second["resets"]) == 1 and second["resets"][0]["signals"]["d_coverage_m2"] == 0.0
    kinds = [json.loads(line)["kind"] for line in (o.out_dir / "episode_1.jsonl").read_text().splitlines()]
    assert "recovery_refused" not in kinds


# ── the switches: recovery off, the handover variants, inactivity, the exact turn budget ──

def _keeps_walking(log: list):
    """A model that walks and never closes: every hop takes all the turns it is given.
    A λ ask is answered with a one-turn note."""
    def agent(wire, prompt):
        orch = _Wire.orch
        text = prompt_text(prompt)
        log.append((text[:24], wire.limits[-1]))
        if "write the note your next self will read" in text:
            _write(orch, "run/tasks/0/lambda.md", json.dumps({"searched": ["hall"], "untried": ["garage"]}))
            return _hop("DONE", 1, "write_file")
        _walk(orch)
        return _hop("DONE", wire.limits[-1], "observe", "step")
    return agent


def test_recovery_off_switches_the_fallback_off_too(tmp_path: Path, scripted) -> None:
    log: list = []
    _Wire.handlers["agent"] = _keeps_walking(log)
    o = _orch(tmp_path, _Bench(n=1), notes=False, recovery=False, reset_after_turns=10, max_resets_per_task=1)
    o.run()
    kinds = [json.loads(line)["kind"] for line in (o.out_dir / "episode_0.jsonl").read_text().splitlines()]
    assert "recovery" not in kinds and "lambda_ask" not in kinds and o.agent_session.runs == 1
    row = o.rec.rows[0]
    assert row["resets"] == [] and row["end_reason"] == "turns" and row["turns"] == 40
    assert row["claim"] == {"status": "stop", "evidence": "turn budget exhausted"}


def test_the_turn_budget_is_exact_and_the_note_hop_spends_from_it(tmp_path: Path, scripted) -> None:
    """B = 40, H = 25, T_f = 20: the first hop stops at the fallback threshold,
    the λ hop may take only what is left of the task, and so may the fresh
    attempt, whose recoveries are spent."""
    log: list = []
    _Wire.handlers["agent"] = _keeps_walking(log)
    o = _orch(tmp_path, _Bench(n=1), notes=False, recovery=True, reset_after_turns=20,
              max_resets_per_task=1, recovery_min_turns=0)
    o.run()
    assert [limit for _, limit in log] == [20, 20, 19], log
    assert log[1][0].startswith("You have spent") and log[2][0].startswith("This is a fresh attempt")
    row = o.rec.rows[0]
    assert row["turns"] == 40 and row["end_reason"] == "turns" and len(row["resets"]) == 1
    assert row["resets"][0]["attempt_turns"] == 20 and row["resets"][0]["trigger"] == "cap"
    # without a recovery the same task stops at 40 too, never at 25 + 25
    log2: list = []
    _Wire.handlers["agent"] = _keeps_walking(log2)
    o2 = _orch(tmp_path / "b", _Bench(n=1), notes=False, recovery=False)
    o2.run()
    assert [limit for _, limit in log2] == [25, 15] and o2.rec.rows[0]["turns"] == 40


def test_the_fallback_fires_at_its_threshold_not_at_the_end_of_a_hop(tmp_path: Path, scripted) -> None:
    """The paper's settings, B = 200, H = 25, T_f = 80: the fourth hop takes
    the 5 turns left to the threshold, so the fallback fires at turn 80 of the
    attempt rather than at 100; once the task's one recovery is spent, hops
    run at H again until the budget is."""
    log: list = []
    _Wire.handlers["agent"] = _keeps_walking(log)
    o = _orch(tmp_path, _Bench(n=1), notes=False, recovery=True, reset_after_turns=80,
              max_resets_per_task=1, recovery_min_turns=12, max_turns_per_task=200)
    o.run()
    assert [limit for _, limit in log[:4]] == [25, 25, 25, 5], log
    assert log[4][0].startswith("You have spent") and log[5][0].startswith("This is a fresh attempt")
    assert [limit for _, limit in log[5:]] == [25, 25, 25, 25, 19], log
    row = o.rec.rows[0]
    reset = row["resets"][0]
    assert reset["trigger"] == "cap" and reset["attempt_turns"] == 80 and reset["at_turn"] == 80
    assert row["turns"] == 200 and row["end_reason"] == "turns"


def test_the_fallback_does_not_cap_a_hop_it_can_no_longer_fire_in(tmp_path: Path, scripted) -> None:
    """With the recovery switched off, or the task's recoveries spent, the
    threshold caps nothing: hops run at H."""
    for sub, switches in (("off", {"recovery": False}), ("spent", {"recovery": True, "max_resets_per_task": 0})):
        log: list = []
        _Wire.handlers["agent"] = _keeps_walking(log)
        o = _orch(tmp_path / sub, _Bench(n=1), notes=False, reset_after_turns=20, **switches)
        o.run()
        assert [limit for _, limit in log] == [25, 15], (sub, log)
        assert o.rec.rows[0]["resets"] == []


def test_three_hops_without_a_tool_call_end_the_task_as_inactive(tmp_path: Path, scripted) -> None:
    from core.harnesses.navh.env.record_sink import exit_content
    from core.harnesses.navh.orchestrator.close import TERMINATIONS

    _Wire.handlers["agent"] = lambda w, p: _hop("Let me think about where the thing could be.", 2)
    bench = _Bench(n=1)
    o = _orch(tmp_path, bench, notes=False)
    summary = o.run()
    row = o.rec.rows[0]
    assert row["end_reason"] == "inactive" and row["turns"] == 6 and bench.env.stops() == 1
    assert row["claim"] == {"status": "stop", "evidence": TERMINATIONS["inactive"]}
    assert "budget" not in row["claim"]["evidence"], "an idle model has not spent its budget"
    assert o.state.read()["end_reason"] == "inactive"
    assert json.loads(exit_content(summary, 1))["end_reason"] == "inactive"


# ── the certification's evidence: taken before the STOP ──

ROUTE, AFTER = base64.b64encode(b"route").decode(), base64.b64encode(b"after").decode()


class _RouteEnv(_Env):
    """A STOP starts the next task's route: the map drawn after it no longer
    shows the route this task drove (on IR2R-CE the boundary also moves the body)."""

    def __init__(self) -> None:
        super().__init__()
        self.stopped = False

    def call(self, fn: str, inputs: dict | None = None, config: dict | None = None) -> dict:
        out = super().call(fn, inputs, config)
        if fn == "get_map":
            return {"map": AFTER if self.stopped else ROUTE, "current_floor": 0}
        if fn == "step" and int((inputs or {}).get("action", -1)) == 0:
            self.stopped = True
        return out


def _certify_sees(seen: list):
    """A certification that keeps the map it was shown and its frames' tags."""
    def handler(wire, prompt) -> Hop:
        blocks = list(prompt)
        at = next((i for i, b in enumerate(blocks)
                   if b.get("type") == "text" and b["text"].startswith("Top-down map")), None)
        seen.append({"map": blocks[at + 1]["source"]["data"] if at is not None else None,
                     "stop_views": sum(1 for b in blocks if b.get("type") == "text"
                                       and "look-around at the stopping point" in b["text"])})
        return _hop(json.dumps({"verdict": "complete", "evidence": "the frames", "confidence": "high"}), 1)
    return handler


@pytest.mark.parametrize("persist", [True, False])
def test_the_certification_sees_the_map_with_the_route_before_the_stop(tmp_path: Path, scripted,
                                                                         persist: bool) -> None:
    seen: list = []
    _Wire.handlers["certify"] = _certify_sees(seen)
    bench = _Bench(n=1)
    bench.env = _RouteEnv()
    o = _orch(tmp_path, bench, notes=False, certify="model", persist_map=persist)
    o.run()
    assert [s["map"] for s in seen] == [ROUTE], "the route it drove, not the map after the STOP"
    assert seen[0]["stop_views"] == 8, "the views close_task took, on each of the two requests"
    calls = [(fn, a.get("action")) for fn, a in bench.env.calls]
    assert calls.index(("get_map", None)) < calls.index(("step", 0))
    assert calls.count(("look_around", None)) == 0, "close_task took the stop views; none are taken again"


def _ends_by(how: str):
    """A model whose task the harness ends: it keeps walking (``turns``), stops
    calling tools (``inactive``), spends the step budget (``budget``), or walks
    into the env's own cap (``env``)."""
    def agent(wire, prompt):
        orch = _Wire.orch
        if how == "inactive":
            return _hop("Let me think about where the thing could be.", 2)
        _walk(orch)
        if how in ("budget", "env"):
            st = orch.state.read()
            if how == "budget":
                st["budget_exhausted"] = True
            else:
                st.update(task_over=True, end_reason="budget",
                          end_payload={"info": {"subtask_result": {"success": 0.0}, "end_reason": "budget"}})
            orch.state.write(st)
        return _hop("DONE", wire.limits[-1], "observe", "step")
    return agent


@pytest.mark.parametrize("how,reason", [("turns", "turns"), ("inactive", "inactive"),
                                        ("budget", "budget"), ("env", "budget")])
def test_a_task_a_limit_ends_is_certified_with_the_stop_views(tmp_path: Path, scripted, how: str,
                                                              reason: str) -> None:
    seen: list = []
    _Wire.handlers["agent"] = _ends_by(how)
    _Wire.handlers["certify"] = _certify_sees(seen)
    bench = _Bench(n=1)
    bench.env = _RouteEnv()
    o = _orch(tmp_path, bench, notes=False, certify="model")
    o.run()
    row = o.rec.rows[0]
    assert row["end_reason"] == reason
    assert seen == [{"map": ROUTE, "stop_views": 4}]
    calls = [fn for fn, _ in bench.env.calls]
    assert calls.count("look_around") == 1
    if how != "env":                             # the env ended that one itself: no STOP is issued
        assert calls.index("look_around") < calls.index("get_map") < bench.env.calls.index(("step", {"action": 0}))
    stop = [f for f in o.state.read()["frames"] if f["kind"] == "stop"]
    assert [f["id"] for f in stop] == o.state.read()["stop_frames"] and len(stop) == 4
    assert row["frames"] == len(o.state.read()["frames"])
    kept = sorted(p.name for p in (o.out_dir / "live_0").glob("obs_*.png"))
    assert kept[-4:] == [Path(f["path"]).name for f in stop], "filed beside the task's other frames"


def test_without_a_certification_no_stop_evidence_is_taken(tmp_path: Path, scripted) -> None:
    _Wire.handlers["agent"] = _ends_by("turns")
    bench = _Bench(n=1)
    o = _orch(tmp_path, bench, notes=False, certify="none")
    o.run()
    calls = [fn for fn, _ in bench.env.calls]
    assert "look_around" not in calls
    assert calls[:calls.index("step") + 1] == ["observe", "step"], "nothing between the hop and the STOP"


def _asks_for_recovery_then_closes(seen: dict):
    """Asks for a recovery on its first hop, writes whatever note it is asked
    for, and closes on the fresh attempt; every prompt is kept in ``seen``."""
    def agent(wire, prompt):
        orch = _Wire.orch
        text = prompt_text(prompt)
        seen.setdefault("prompts", []).append(text)
        if text.startswith("Task"):
            _walk(orch)
            st = orch.state.read()
            st["recovery_request"] = {"scope": "attempt", "reason": "going in circles", "confidence": 0.9}
            orch.state.write(st)
            return _hop("DONE", 6, "observe", "step", "request_recovery")
        if "write the note your next self will read" in text:
            _write(orch, "run/tasks/0/lambda.md", json.dumps({"searched": ["kitchen"], "ruled_out": [],
                                                              "last_anchor": "hall", "untried": ["garage"],
                                                              "uncertain": [], "recommended_scope": "attempt"}))
            return _hop("DONE", 1, "write_file")
        if "write a summary of this attempt" in text:
            _write(orch, "run/tasks/0/lambda.md",
                   "I walked from the hall to the kitchen and back and saw nothing like it.")
            return _hop("DONE", 1, "write_file")
        if text.startswith("This is a fresh attempt"):
            seen["recovered"] = text
            p = orch.ws.lambda_path(0, 1)
            seen["note"] = p.read_text() if p.exists() else None
            _write(orch, "run/tasks/0/handover.md", "Second attempt: found it.")
            _close_request(orch)
            return _hop("DONE", 3, "read_file", "write_file", "close_task")
        raise AssertionError(text[:80])
    return agent


def _bank(tmp_path: Path) -> str:
    """An earlier structured run's lessons: one note for another goal (6 words), one for this goal."""
    d = tmp_path / "earlier" / "navh_0"
    d.mkdir(parents=True)
    rows = [{"goal": "find a bed", "lam_source": "own", "lam_words": 6,
             "lam": {"searched": ["landing"], "ruled_out": [], "last_anchor": "landing",
                     "untried": ["the guest room"], "uncertain": [], "recommended_scope": "attempt"}},
            {"goal": "find thing 0", "lam_source": "own", "lam_words": 6,
             "lam": {"searched": ["porch"], "ruled_out": [], "last_anchor": "porch",
                     "untried": ["the shed"], "uncertain": [], "recommended_scope": "attempt"}}]
    (d / "lessons.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(tmp_path / "earlier")


def _recover_once(tmp_path: Path, **switches) -> tuple[Orchestrator, dict, list[str]]:
    seen: dict = {}
    _Wire.handlers["agent"] = _asks_for_recovery_then_closes(seen)
    o = _orch(tmp_path, _Bench(n=1), notes=False, recovery=True, max_resets_per_task=1,
              recovery_min_turns=0, **switches)
    o.run()
    kinds = [json.loads(line)["kind"] for line in (o.out_dir / "episode_0.jsonl").read_text().splitlines()]
    return o, seen, kinds


def test_an_empty_handover_asks_for_no_note_and_names_none(tmp_path: Path, scripted) -> None:
    o, seen, kinds = _recover_once(tmp_path, recovery_handover="empty")
    assert "lambda_ask" not in kinds and len(seen["prompts"]) == 2, "no note hop"
    assert "Required reading" not in seen["recovered"] and "lambda" not in seen["recovered"]
    assert seen["note"] is None
    reset = o.rec.rows[0]["resets"][0]
    assert reset["lam_source"] == "empty" and reset["handover"] == "empty" and reset["lam_words"] == 0


def test_another_tasks_handover_replaces_the_note_the_attempt_wrote(tmp_path: Path, scripted) -> None:
    o, seen, kinds = _recover_once(tmp_path, recovery_handover="other", recovery_handover_bank=_bank(tmp_path))
    assert kinds.index("lambda_ask") < kinds.index("lambda_handed_over") < kinds.index("recovery")
    assert "run/tasks/0/lambda.md" in seen["recovered"] and "ruled-out lines" in seen["recovered"]
    lent = json.loads(seen["note"])
    assert lent["untried"] == ["the guest room"], "another goal's note, never this goal's"
    reset = o.rec.rows[0]["resets"][0]
    assert reset["lam_source"] == "other" and reset["lam"]["untried"] == ["the guest room"]
    assert reset["lam_own"]["untried"] == ["garage"], "the attempt's own note is kept in the record"


def test_a_matched_length_summary_is_prose_cut_to_the_structured_length(tmp_path: Path, scripted) -> None:
    o, seen, _kinds = _recover_once(tmp_path, recovery_handover="summary", recovery_handover_bank=_bank(tmp_path))
    ask = next(p for p in seen["prompts"] if "write a summary of this attempt" in p)
    assert "at most 6 words" in ask and "JSON" not in ask and "ruled_out" not in ask
    assert seen["note"] == "I walked from the hall to"
    assert "the summary your previous attempt wrote" in seen["recovered"]
    assert "ruled-out lines" not in seen["recovered"]
    reset = o.rec.rows[0]["resets"][0]
    assert reset["lam_source"] == "summary" and reset["summary_words"] == 6 and reset["lam_words"] == 6
    assert reset["lesson"] == "I walked from the hall to"


# ── a new day in a mapped house ──

def _restored_house(tmp_path: Path) -> Path:
    """What an earlier day left: the map file, the place graph beside it and a photo set."""
    from core.harnesses.navh.memory.longterm import HouseNotes
    from core.harnesses.navh.memory.placegraph import PlaceGraph

    maps = tmp_path / "maps"
    maps.mkdir()
    (maps / "house.pkl").write_bytes(b"map")
    g = PlaceGraph()
    g.note_mark("kitchen", 0, photos=4, task=1)
    g.save(maps / "house.places.json")
    views = []
    for i, h in enumerate(("front", "right", "back", "left")):
        src = tmp_path / f"v{i}.png"
        src.write_bytes(PNG_BYTES)
        views.append({"path": str(src), "step": 9, "heading": h, "pose": {"x": 1.0, "z": 2.0, "yaw_deg": 0.0}})
    HouseNotes(tmp_path / "mem" / "stub-house", day="2026-09-21").attach_photos("kitchen", views, 1)
    return maps / "house.pkl"


def test_a_run_that_opens_on_a_restored_map_starts_after_a_relocation(tmp_path: Path, scripted) -> None:
    first: list = []

    def agent(wire, prompt):
        if not first:
            first.append(prompt)
        return _agent_closes_properly(wire, prompt)

    _Wire.handlers["agent"] = agent
    bench = _Bench(n=2)
    o = _orch(tmp_path, bench, notes=False, map_state=str(_restored_house(tmp_path)))
    o.run()
    opening = first[0]
    assert isinstance(opening, list), "the four views travel with the first instruction"
    assert sum(1 for b in opening if b.get("type") == "image") == 4
    assert P.RELOCATED.strip() in prompt_text(opening)
    calls = [fn for fn, _ in bench.env.calls]
    assert calls.index("load_spatial") < calls.index("look_around"), "views taken after the map is restored"
    assert not {"set_body_pose", "set_floor", "reset_spatial"} & set(calls), "the map is not re-registered"
    assert o.rec.rows[0]["after_wakeup"] is True and o.rec.rows[1]["after_wakeup"] is False


def test_the_first_day_in_a_house_is_not_a_relocation(tmp_path: Path, scripted) -> None:
    first: list = []

    def agent(wire, prompt):
        first.append(prompt)
        return _agent_closes_properly(wire, prompt)

    _Wire.handlers["agent"] = agent
    o = _orch(tmp_path, _Bench(n=1), notes=False, map_state=str(tmp_path / "maps" / "none.pkl"))
    o.run()
    assert isinstance(first[0], str) and P.RELOCATED.strip() not in first[0]
    assert o.rec.rows[0]["after_wakeup"] is False


def test_the_wake_up_compares_against_the_photos_of_earlier_days(tmp_path: Path, scripted) -> None:
    o = _orch(tmp_path, _Bench(n=1), notes=False, map_state=str(_restored_house(tmp_path)))
    o.seat()
    views = o.driver.marker_photos["kitchen"]
    assert [v["heading"] for v in views] == ["front", "right", "back", "left"]
    assert o.driver.photos_for("kitchen", 3) == [PNG] * 3, "the wake-up session is shown the stored views"
    o.shutdown()


# ── the task's accounting ──

def _hop_with(context: int, out: int, text: str = "DONE", turns: int = 1, *tools: str) -> Hop:
    """A hop whose largest turn carried ``context`` prompt tokens."""
    h = _hop(text, turns, *tools)
    h.usage = {"input_tokens": context // 2, "cache_read_input_tokens": context // 2, "output_tokens": out,
               "iterations": [{"input_tokens": context // 2, "cache_read_input_tokens": context // 2}]}
    return h


def test_the_peak_prompt_is_over_every_session_the_task_ran_in(tmp_path: Path, scripted) -> None:
    """A recovery that voids the context opens a new session mid-task; the
    task's peak is the largest context any of its turns carried, in either."""
    def agent(wire, prompt):
        orch = _Wire.orch
        text = prompt_text(prompt)
        if text.startswith("Task"):
            _walk(orch)
            st = orch.state.read()
            st["recovery_request"] = {"scope": "attempt", "reason": "going in circles", "confidence": 0.9}
            orch.state.write(st)
            return _hop_with(9000, 50, "DONE", 6, "observe", "step", "request_recovery")
        if "write the note your next self will read" in text:
            _write(orch, "run/tasks/0/lambda.md", json.dumps({"searched": ["kitchen"], "untried": ["garage"]}))
            return _hop_with(9400, 20, "DONE", 1, "write_file")
        if text.startswith("This is a fresh attempt"):
            _write(orch, "run/tasks/0/handover.md", "Found it in the garage.")
            _close_request(orch)
            return _hop_with(2000, 30, "DONE", 3, "write_file", "close_task")
        raise AssertionError(text[:80])

    _Wire.handlers["agent"] = agent
    _Wire.handlers["certify"] = lambda w, p: _hop_with(
        700, 5, json.dumps({"verdict": "complete", "evidence": "x", "confidence": "high"}))
    o = _orch(tmp_path, _Bench(n=1), notes=False, certify="model", recovery=True, max_resets_per_task=1,
              recovery_min_turns=0)
    summary = o.run()
    assert o.agent_session.runs == 2
    row = o.rec.rows[0]
    assert row["context"]["peak_prompt_tokens"] == 9400, "the first session's turns count, λ's too"
    tokens = row["context"]["tokens"]
    assert tokens["agent"] == {"calls": 2, "prompt_tokens": 9000 + 2000, "cached_tokens": 4500 + 1000,
                               "completion_tokens": 80, "peak_prompt_tokens": 9000}
    assert tokens["note"] == {"calls": 1, "prompt_tokens": 9400, "cached_tokens": 4700,
                              "completion_tokens": 20, "peak_prompt_tokens": 9400}, "the λ hop, on its own line"
    assert tokens["certify"]["calls"] == 1 and tokens["certify"]["prompt_tokens"] == 700
    assert summary["aggregate"]["tokens"]["agent"]["prompt_tokens"] == 11000
    assert summary["run_tokens"] == {}, "no consolidation ran"


def test_each_task_is_charged_its_own_sessions_and_consolidation_the_run(tmp_path: Path, scripted) -> None:
    calls = {"n": 0}

    def agent(wire, prompt):
        calls["n"] += 1
        h = _agent_closes_properly(wire, prompt)
        h.usage = {"input_tokens": 1000 * calls["n"], "output_tokens": 1}
        return h

    _Wire.handlers["agent"] = agent
    _Wire.handlers["consolidate"] = lambda w, p: (lambda h: (h.usage.update(input_tokens=5000), h)[1])(
        _consolidates(w, p))
    o = _orch(tmp_path, _Bench(n=2), notes=True, certify="model")
    summary = o.run()
    first, second = (r["context"]["tokens"]["agent"] for r in o.rec.rows)
    assert first["calls"] == second["calls"] == 2
    assert first["prompt_tokens"] == 1000 + 2000 and second["prompt_tokens"] == 3000 + 4000
    assert o.rec.rows[1]["context"]["peak_prompt_tokens"] == 4000, "a task's peak is its own"
    assert summary["run_tokens"]["consolidate"]["prompt_tokens"] == 5000
    assert all("consolidate" not in r["context"]["tokens"] for r in o.rec.rows)
    assert [r["subtask"] for r in o.rec.rows] == [None, None] and "episode_id" in o.rec.rows[0]
