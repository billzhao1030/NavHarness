"""The session policies and the memory-access rows — what each one gives the session, and what runs around it.

No server, no CLI, no tokens. The rows of the paper's comparison are switches of this one harness, so
each is checked where it bites: the config holds what a context-only policy rules out; the bridge
registers the observation and action tools alone (``act``) and, under ``ledger=none``, its file tools
read only the current task's directory; the orchestrator, over scripted sessions, opens the session the
policy says, sends the instruction it says, and runs (or does not run) the file gate, the two checks, the
ladder and consolidation; the place graph goes with the map; and the three seats hand their core the
same tool list. The rows:

    Independent sessions                      session_mode=independent precheck=none
    Independent sessions + pre-stop check     session_mode=independent precheck=model
    Single session                            session_mode=single precheck=none
    Clear map after each task (keep records)  persist_map=false
    Hide earlier task records (keep map)      ledger=none
    No cross-task memory                      persist_map=false ledger=none
"""

from __future__ import annotations

import base64
import importlib
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core.harnesses.navh import prompts as P
from core.harnesses.navh.bridge import FILE_TOOL_NAMES, TaskState, tool_names
from core.harnesses.navh.config import CONTEXT_ONLY, FLAT_ORDER, HarnessConfig
from core.harnesses.navh.env.types import Goal, TaskEnd
from core.harnesses.navh.orchestrator import Orchestrator, agent_tool_names
from core.harnesses.navh.session import Hop, prompt_text
from core.harnesses.navh.session.base import BaseSession
from core.harnesses.navharness import NavHarnessAdapter
from runner import build, compose_app

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "exp_workspace" / "lhnav_es" / "configs"
BRIDGE = REPO / "core" / "harnesses" / "navh" / "bridge" / "bridge.py"
MODULE = "core.harnesses.navh.bridge.bridge"
ACT = ["observe", "step", "get_goal", "close_task"]
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
PNG = base64.b64encode(PNG_BYTES).decode()


def _cfg(**switches: Any) -> HarnessConfig:
    return HarnessConfig.from_flat(benchmark="es", chain_id=0, **switches)


# ── the config: the policy holds what it rules out ──

def test_the_reference_is_navharness_with_every_record_in_view() -> None:
    cfg = _cfg()
    assert cfg.session.session_mode == "harness" and cfg.memory.ledger == "text"
    assert not cfg.context_only and cfg.held == {}
    # appended at the end of their groups, so the record's order is otherwise unchanged
    assert FLAT_ORDER.index("session_mode") == FLAT_ORDER.index("chunk_turns") + 1
    assert FLAT_ORDER.index("ledger") == FLAT_ORDER.index("day") + 1
    assert agent_tool_names(_cfg(recovery=True, instruments=2)) == tool_names(recovery=True)


@pytest.mark.parametrize("mode", ["independent", "single"])
def test_a_context_only_policy_holds_what_it_rules_out(mode: str) -> None:
    # the reference switches as the yaml gives them: the policy has to win over every one
    cfg = _cfg(session_mode=mode, precheck="model", certify="model", recovery=True, reset_after_turns=80,
               max_resets_per_task=1, place_graph=True, notes=True, compact="boundary", instruments=2)
    assert cfg.context_only
    assert cfg.judge.certify == "none" and cfg.judge.precheck == "model", "the pre-stop check stays the row's"
    assert cfg.ladder.recovery is False and cfg.ladder.reset_after_turns == 0 and cfg.ladder.max_resets_per_task == 0
    assert cfg.memory.place_graph is False and cfg.memory.archive_policy is False and cfg.memory.notes is False
    assert cfg.memory.ledger == "none"
    assert cfg.session.compact == ("boundary" if mode == "independent" else "token")
    assert set(cfg.held) == {"certify", "recovery", "reset_after_turns", "max_resets_per_task", "place_graph",
                             "notes", "ledger"} | ({"compact"} if mode == "single" else set())
    assert set(CONTEXT_ONLY) <= set(FLAT_ORDER)
    assert agent_tool_names(cfg) == ACT == tool_names(toolset="act")
    # the effective config is what the record carries, and it reads back to itself
    assert _cfg(**{k: v for k, v in cfg.flat().items() if k not in ("benchmark", "chain_id")}).flat() == cfg.flat()


def test_an_unknown_policy_or_ledger_is_refused() -> None:
    with pytest.raises(ValueError):
        _cfg(session_mode="mip")
    with pytest.raises(ValueError):
        _cfg(ledger="summary")
    with pytest.raises(ValueError):
        NavHarnessAdapter(session_mode="mip")
    with pytest.raises(ValueError):
        NavHarnessAdapter(ledger="summary")
    with pytest.raises(ValueError):
        tool_names(toolset="env")


# ── the bridge: the act toolset, and the read scope of ledger=none ──

class _EnvHandler(BaseHTTPRequestHandler):
    def log_message(self, *a: Any) -> None:
        pass

    def do_POST(self) -> None:
        fn = self.path.rsplit("__", 1)[-1]
        out = {"look_around": {"views": [PNG] * 4}, "observe_egocentric": {"rgb": PNG}}.get(fn, {})
        body = json.dumps({"outputs": out}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def bridge_env(tmp_path: Path, monkeypatch):
    """Import the bridge in-process under the environment a session would give it."""
    srv = HTTPServer(("127.0.0.1", 0), _EnvHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    ws, house, state = tmp_path / "workspace", tmp_path / "house", tmp_path / "state.json"
    for k, v in {"NAVH_SERVER_URL": f"http://127.0.0.1:{srv.server_port}", "NAVH_LIVE_DIR": str(tmp_path / "live"),
                 "NAVH_MARK_DIR": str(tmp_path / "markers"), "NAVH_STATE_FILE": str(state),
                 "NAVH_WORKSPACE": str(ws), "NAVH_HOUSE_DIR": str(house), "NAVH_STEP_BUDGET": "50",
                 "NAVH_STOP_VIEWS": "4", "NAVH_BENCH": "goat", "NAVH_RECOVERY": "1"}.items():
        monkeypatch.setenv(k, v)

    def load(toolset: str, ledger: str = "text", task: int = 0):
        monkeypatch.setenv("NAVH_TOOLSET", toolset)
        monkeypatch.setenv("NAVH_LEDGER", ledger)
        TaskState(state).reset(task, writable=[f"run/tasks/{task}/handover.md"])
        sys.modules.pop(MODULE, None)
        return importlib.import_module(MODULE)

    yield load, ws, house
    srv.shutdown()
    sys.modules.pop(MODULE, None)


def _registered(B) -> dict[str, str]:
    return {t.name: t.description for t in B.mcp._tool_manager.list_tools()}


def test_a_context_only_bridge_registers_the_observation_and_action_tools_alone(bridge_env) -> None:
    load, *_ = bridge_env
    B = load("act")
    tools = _registered(B)
    assert sorted(tools) == sorted(ACT), "no map, markers, route preview, recovery proposal or files"
    assert "handover" not in tools["close_task"] and "file" not in tools["close_task"]
    res = B.close_task("complete", "the bed is in view")
    assert res["close_requested"] is True and "file" not in res["note"]
    assert sorted(_registered(load("all"))) == sorted(tool_names(recovery=True))


def test_ledger_none_lets_the_file_tools_read_only_the_current_tasks_directory(bridge_env) -> None:
    load, ws, house = bridge_env
    B = load("all", ledger="none", task=1)
    (ws / "run" / "tasks" / "0").mkdir(parents=True, exist_ok=True)
    (ws / "run" / "tasks" / "1").mkdir(parents=True, exist_ok=True)
    (ws / "run" / "ledger.md").write_text("- task 1 | CERTIFIED | the bed\n")
    (ws / "run" / "tasks" / "0" / "handover.md").write_text("The bed is upstairs.\n")
    (ws / "run" / "tasks" / "1" / "goal.md").write_text("find the sink\n")
    (house / "journal").mkdir(parents=True, exist_ok=True)
    (house / "journal" / "2026-01-01.md").write_text("- task 1 · CERTIFIED · the bed\n")

    def ls(p: str) -> dict:
        return json.loads(B.list_files(p))

    def cat(p: str) -> dict:
        return json.loads(B.read_file(p)[-1])      # [image,] json: the text part is last

    assert ls("")["entries"] == ["run/"] and ls("run")["entries"] == ["tasks/"]
    assert ls("run/tasks")["entries"] == ["1/"] and ls("run/tasks/1")["entries"] == ["goal.md"]
    assert cat("run/tasks/1/goal.md")["text"] == "find the sink\n"
    for hidden in ("run/ledger.md", "run/tasks/0/handover.md", "house/journal/2026-01-01.md", "house/MEMORY.md"):
        assert "not available" in cat(hidden)["error"], hidden
    for hidden in ("house", "house/journal", "run/tasks/0"):
        assert "not available" in ls(hidden)["error"], hidden
    assert "outside the workspace" in cat("../secrets")["error"]
    assert "written" in json.loads(B.write_file("run/tasks/1/handover.md", "Found the sink."))
    # the same files under the reference: every record can be read
    B = load("all", ledger="text", task=1)
    assert cat("run/ledger.md")["text"].startswith("- task 1") and "house/" in ls("")["entries"]


# ── the orchestrator over scripted sessions ──

class _Env:
    url = "http://stub"

    def __init__(self) -> None:
        self.calls: list = []
        self.map_rebuilt = False

    def call(self, fn: str, inputs: dict | None = None, config: dict | None = None) -> dict:
        self.calls.append((fn, dict(inputs or {})))
        if fn == "step" and int((inputs or {}).get("action", -1)) == 0:
            return {"info": {"subtask_result": {"success": 1.0, "distance_to_goal": 0.4}}, "terminated": False}
        if fn == "get_map":
            return {} if self.map_rebuilt else {"map": PNG, "current_floor": 0}
        if fn == "look_around":
            return {"views": [PNG] * 4}
        return {}

    def stops(self) -> int:
        return sum(1 for fn, a in self.calls if fn == "step" and a.get("action") == 0)


class _Bench:
    house_id = "stub-house"

    def __init__(self, n: int) -> None:
        self.env = _Env()
        self.n, self.k = n, 0

    @staticmethod
    def fn(name: str) -> str:
        return name

    def _goal(self, k: int) -> Goal:
        return Goal(index=k, total=self.n, text=f"find thing {k}", modality="object")

    def seat(self, arg, persist_map, skip=0) -> Goal:
        self.k = skip
        return self._goal(self.k)

    def parse_step(self, out: dict, claimed) -> tuple[dict, TaskEnd | None]:
        res = (out.get("info") or {}).get("subtask_result")
        return {}, (TaskEnd(end_reason="stop", score=dict(res), chain_finished=False) if res else None)

    def after_end(self, end: TaskEnd, persist_map) -> TaskEnd:
        # the env's reset_spatial: a map rebuilt at the boundary has nothing on it
        self.env.map_rebuilt = not persist_map
        self.k += 1
        end.chain_finished, end.next_goal = (True, None) if self.k >= self.n else (False, self._goal(self.k))
        return end

    def chain_report(self) -> dict:
        return {}


class _Wire:
    handlers: dict[str, Any] = {}
    opened: list[str] = []

    def __init__(self, loop, options, *, on_message=None) -> None:
        self.session = on_message.__self__
        self.is_open = False

    def open(self) -> dict:
        self.is_open = True
        _Wire.opened.append(self.session.kind)
        return {}

    def hop(self, prompt, max_turns=None) -> Hop:
        return self.handlers[self.session.kind](self, prompt)

    def close(self) -> None:
        self.is_open = False


def _hop(text: str = "DONE", turns: int = 1, *tools: str) -> Hop:
    return Hop(text=text, num_turns=turns, subtype="success", session_id="sid",
               usage={"input_tokens": 100, "output_tokens": 10},
               tool_uses=[{"id": f"t{i}", "name": t, "input": {}} for i, t in enumerate(tools)])


def _close(o: Orchestrator, status: str = "complete") -> None:
    """What close_task does to the state file: the look-around, then the request."""
    st = o.state.read()
    live = o.out_dir / "live"
    live.mkdir(parents=True, exist_ok=True)
    for _ in range(4):
        p = live / f"obs_{len(st['frames']) + 1:04d}.png"
        p.write_bytes(PNG_BYTES)
        st["frames"].append({"id": len(st["frames"]) + 1, "path": str(p), "step": st["steps_used"], "kind": "stop"})
    st["close_request"] = {"status": status, "evidence": "it is in view", "steps_used": st["steps_used"]}
    st["close_requests"] += 1
    o.state.write(st)


def _walk(o: Orchestrator, steps: int = 3, marks: tuple[str, ...] = ()) -> None:
    o.bench.env.map_rebuilt = False               # walking maps: the env's map has this task's route on it
    st = o.state.read()
    st["steps_used"] += steps
    st["body"] = {"explored_m2": 1.0, "slam": "ok"}
    for lab in marks:
        st["marks"].append(lab)
        st["marker_photos"][lab] = [{"id": f"{lab}:0", "path": str(o.out_dir / "none.png"), "step": 0}]
    o.state.write(st)


def _judge(verdict: str, seen: list | None = None):
    def handler(wire, prompt) -> Hop:
        if seen is not None:
            seen.append(prompt_text(prompt))
        return _hop(json.dumps({"verdict": verdict, "evidence": "the frames", "confidence": "high"}))
    return handler


@pytest.fixture
def scripted(monkeypatch):
    monkeypatch.setattr(BaseSession, "wire_cls", _Wire)
    _Wire.opened = []
    _Wire.handlers = {"precheck": _judge("incomplete"), "certify": _judge("complete"),
                      "consolidate": lambda w, p: _hop("DONE"), "wakeup": lambda w, p: _hop('{"place": ""}')}
    yield _Wire


def _orch(tmp_path: Path, n: int, **switches: Any) -> Orchestrator:
    # the reference switches the yaml sets, so a row is checked against what a launch would get
    row = {"precheck": "model", "certify": "model", "recovery": True, "reset_after_turns": 80,
           "max_resets_per_task": 1, "recovery_min_turns": 12, "notes": True, "place_graph": True,
           "max_turns_per_task": 200, "instruments": 2, **switches}
    cfg = HarnessConfig.from_flat(benchmark="es", chain_id=0, memory_root=str(tmp_path / "mem"), **row)
    return Orchestrator(cfg, "claude-test", None, _Bench(n), tmp_path / "out", bridge=BRIDGE)


def _stream(o: Orchestrator) -> list[dict]:
    """Every task's events, in task order (one episode_<k>.jsonl per task)."""
    files = sorted(o.out_dir.glob("episode_*.jsonl"), key=lambda f: int(f.stem.split("_")[1]))
    return [json.loads(line) for f in files for line in f.read_text().splitlines()]


def _kinds(o: Orchestrator) -> list[str]:
    return [e["kind"] for e in _stream(o)]


def _events(o: Orchestrator, kind: str) -> list[dict]:
    return [e for e in _stream(o) if e["kind"] == kind]


def test_independent_sessions_a_fresh_context_only_session_per_task(tmp_path: Path, scripted) -> None:
    prompts: list[str] = []

    def agent(wire, prompt) -> Hop:
        text = prompt_text(prompt)
        prompts.append(text)
        if text.startswith("Task"):
            # long enough that the reference's fallback (80 turns) would have recovered it
            _walk(o, marks=("hall",))
            return _hop("searching", 90, "observe", "step")
        assert text == P.CONTINUE, text
        _close(o)
        return _hop("DONE", 2, "close_task")

    _Wire.handlers["agent"] = agent
    o = _orch(tmp_path, 2, session_mode="independent", precheck="none")
    assert o.agent_session.system_prompt == P.SESSION_SYSTEM and o.agent_session.tools == ACT
    env = o.agent_session.mcp["env"]["env"]
    assert env["NAVH_TOOLSET"] == "act" and env["NAVH_LEDGER"] == "none" and env["NAVH_RECOVERY"] == "0"
    summary = o.run()

    assert o.bench.env.stops() == 2 and o.agent_session.runs == 2, "one fresh session per task"
    assert _Wire.opened == ["agent", "agent"], "no judge, no wake-up, no consolidation"
    first = [t for t in prompts if t.startswith("Task")]
    assert first == [P.SESSION_INSTRUCTION.format(k=i + 1, n=2, goal=f"find thing {i}", photo="") for i in (0, 1)]
    kinds = _kinds(o)
    for absent in ("handover_prompted", "precheck_refused", "lambda_ask", "recovery", "notes_consolidated"):
        assert absent not in kinds, absent
    assert all(e["required"] == [] for e in _events(o, "instruction"))
    rows = o.rec.rows
    assert [r["verdict"]["judge"] for r in rows] == ["none", "none"] and all(r["resets"] == [] for r in rows)
    assert summary["tools"] == ACT and summary["notes"] is None
    assert summary["session_policy"]["session_mode"] == "independent"
    assert summary["session_policy"]["held"]["certify"] == "none"
    assert summary["config"]["session_mode"] == "independent" and summary["config"]["recovery"] is False
    assert o.house.places is None, "no place graph"
    # the harness still keeps its own record of every task, for scoring
    assert o.ws.ledger.read_text().count("\n") == 2


def test_independent_sessions_with_the_pre_stop_check(tmp_path: Path, scripted) -> None:
    def agent(wire, prompt) -> Hop:
        text = prompt_text(prompt)
        assert text.startswith("Task") or "closing check" in text, text
        _close(o)
        return _hop("DONE", 2, "observe", "close_task")

    _Wire.handlers["agent"] = agent
    o = _orch(tmp_path, 1, session_mode="independent", precheck="model")
    o.run()
    assert o.bench.env.stops() == 1 and _Wire.opened == ["agent", "precheck"], "the gate, and no certification"
    kinds = _kinds(o)
    assert "precheck_refused" in kinds and "handover_prompted" not in kinds
    row = o.rec.rows[0]
    assert row["precheck"]["close_requests"] == 2 and row["verdict"]["judge"] == "none"
    assert row["context"]["judge_calls"] == 1


def test_single_session_one_context_only_session_across_tasks(tmp_path: Path, scripted) -> None:
    def agent(wire, prompt) -> Hop:
        assert prompt_text(prompt).startswith("Task")
        _walk(o)
        _close(o)
        return _hop("DONE", 3, "observe", "step", "close_task")

    _Wire.handlers["agent"] = agent
    o = _orch(tmp_path, 3, session_mode="single", precheck="none")
    assert o.cfg.session.compact == "token" and o.agent_session.tools == ACT
    o.run()
    assert o.bench.env.stops() == 3 and o.agent_session.runs == 1, "the conversation carries over"
    assert _Wire.opened == ["agent"]
    assert [r["context"]["compact"] for r in o.rec.rows] == ["token"] * 3
    assert "handover_prompted" not in _kinds(o)


def test_the_reference_session_keeps_its_gates_and_its_files(tmp_path: Path, scripted) -> None:
    def agent(wire, prompt) -> Hop:
        text = prompt_text(prompt)
        if text.startswith("Not closed yet"):
            out = o.ws.session_write(o.ws.rel(o.ws.handover_path(o.k)), "Found it.", set(o.state.read()["writable"]))
            assert "written" in out
        _close(o)
        return _hop("DONE", 2, "close_task")

    _Wire.handlers["agent"] = agent
    o = _orch(tmp_path, 1, precheck="none", notes=False)
    assert o.agent_session.system_prompt.startswith(P.SYSTEM)   # + where the notes are, on Claude Code
    assert o.agent_session.tools == tool_names(recovery=True) and set(FILE_TOOL_NAMES) <= set(o.agent_session.tools)
    assert o.agent_session.mcp["env"]["env"]["NAVH_TOOLSET"] == "all"
    assert o.agent_session.mcp["env"]["env"]["NAVH_LEDGER"] == "text"
    o.run()
    assert "handover_prompted" in _kinds(o) and _Wire.opened == ["agent", "certify"]


def test_hide_earlier_task_records_keeps_the_gates_and_withholds_the_reading(tmp_path: Path, scripted) -> None:
    firsts: list[str] = []

    def agent(wire, prompt) -> Hop:
        text = prompt_text(prompt)
        if text.startswith("Task"):
            firsts.append(text)
        if text.startswith("Not closed yet"):
            o.ws.session_write(o.ws.rel(o.ws.handover_path(o.k)), f"Task {o.k}: found it.",
                               set(o.state.read()["writable"]))
        _close(o)
        return _hop("DONE", 2, "close_task")

    _Wire.handlers["agent"] = agent
    o = _orch(tmp_path, 2, ledger="none", precheck="none", notes=False)
    assert o.cfg.held == {} and o.agent_session.tools == tool_names(recovery=True)
    assert o.agent_session.mcp["env"]["env"]["NAVH_LEDGER"] == "none"
    o.run()
    assert all(e["required"] == [] for e in _events(o, "instruction"))
    assert P.RECORDS_WITHHELD in firsts[1] and "run/tasks/0/handover.md" not in firsts[1]
    assert "run/ledger.md" not in firsts[1] and "run/tasks/1/handover.md" in firsts[1], "its own file is named"
    # the gates and the certification run as in the reference; the records are still written
    assert _kinds(o).count("handover_prompted") == 2 and _Wire.opened.count("certify") == 2
    assert o.ws.ledger.read_text().count("\n") == 2 and o.ws.handover_path(0).exists()


@pytest.mark.parametrize("persist", [True, False])
def test_the_place_graph_goes_with_the_map(tmp_path: Path, scripted, persist: bool) -> None:
    seen_cert: list[str] = []
    at_second: dict = {}

    def agent(wire, prompt) -> Hop:
        text = prompt_text(prompt)
        if text.startswith("Task 2"):
            at_second.update(places=sorted(o.house.places.places), photos=dict(o.driver.marker_photos))
            _walk(o)
        elif text.startswith("Task 1"):
            _walk(o, marks=("kitchen",))
        else:
            o.ws.session_write(o.ws.rel(o.ws.handover_path(o.k)), "Marked the kitchen.",
                               set(o.state.read()["writable"]))
        _close(o)
        return _hop("DONE", 2, "close_task")

    _Wire.handlers["agent"] = agent
    _Wire.handlers["certify"] = _judge("complete", seen_cert)
    o = _orch(tmp_path, 2, persist_map=persist, precheck="none", notes=False)
    o.run()
    if persist:
        assert at_second["places"] == ["kitchen"] and "kitchen" in at_second["photos"]
    else:
        assert at_second == {"places": [], "photos": {}}, "no place or banked view crosses the boundary"
        saved = json.loads((o.out_dir / "places.json").read_text())
        assert saved["places"] == [] and saved["connectors"] == []
    # the certification judges each task on that task's own map, also when the boundary rebuilt it
    assert len(seen_cert) == 2 and all("Top-down map" in t for t in seen_cert)


def test_no_cross_task_memory_is_both_interventions(tmp_path: Path, scripted) -> None:
    def agent(wire, prompt) -> Hop:
        if prompt_text(prompt).startswith("Not closed yet"):
            o.ws.session_write(o.ws.rel(o.ws.handover_path(o.k)), "done", set(o.state.read()["writable"]))
        else:
            _walk(o, marks=("hall",))
        _close(o)
        return _hop("DONE", 2, "close_task")

    _Wire.handlers["agent"] = agent
    o = _orch(tmp_path, 2, persist_map=False, ledger="none", precheck="none", notes=False)
    o.run()
    assert o.agent_session.mcp["env"]["env"]["NAVH_LEDGER"] == "none" and not o.house.places.places
    assert all(e["required"] == [] for e in _events(o, "instruction"))
    assert _kinds(o).count("handover_prompted") == 2, "it still keeps its own records"


# ── the three seats: one tool list, handed over three ways ──

def test_the_three_cores_are_handed_the_act_toolset(tmp_path: Path, monkeypatch) -> None:
    from core.harnesses.navh.session.wire import home
    from core.harnesses.navh.session.wire.codex import CodexSpec, CodexWire
    from core.harnesses.navh.session.wire.mini import MiniSpec

    # Claude Code: the allow-list is the session's tools, the server the act bridge
    o = _orch(tmp_path / "sdk", 1, session_mode="independent", precheck="none")
    monkeypatch.setattr(home, "ensure_relocated", lambda *a, **k: {})
    opts = o.agent_session.options()
    assert opts.allowed_tools == [f"mcp__env__{t}" for t in ACT] and opts.system_prompt == P.SESSION_SYSTEM
    assert opts.mcp_servers["env"]["env"]["NAVH_TOOLSET"] == "act"
    o.shutdown()

    # Codex: no allow-list — the surface is what the bridge registers, and it is told act
    o = _orch(tmp_path / "codex", 1, session_mode="independent", precheck="none", transport="codex-session")
    spec = CodexSpec.from_session(o.agent_session)
    argv = " ".join(CodexWire(None, lambda: spec)._argv(spec, "go", [], None))
    assert 'NAVH_TOOLSET = "act"' in argv and 'NAVH_LEDGER = "none"' in argv
    assert json.dumps(P.SESSION_SYSTEM, ensure_ascii=False) in argv, "the policy's system prompt"
    o.shutdown()

    # mini: the menu is listed off the real bridge process, spawned with the session's environment
    o = _orch(tmp_path / "mini", 1, session_mode="single", precheck="none", transport="mini-session")
    spec = MiniSpec.from_session(o.agent_session)
    assert spec.system_prompt == P.SESSION_SYSTEM
    sys.path.insert(0, str(REPO / "core" / "harnesses" / "mini"))
    from bridge_toolset import BridgeToolSet  # noqa: E402

    path, env = spec.bridge
    ts = BridgeToolSet(Path(path), {**env, "EH_EXECUTOR": "mini-swe", "PYTHONPATH": str(REPO)})
    try:
        assert sorted(ts.tool_names()) == sorted(ACT)
    finally:
        ts.close()
        o.shutdown()


# ── the seats, as a launch builds them ──

ROWS = {
    "": ("harness", "text", True),
    "harness.session_mode=independent harness.precheck=none": ("independent", "none", False),
    "harness.session_mode=independent harness.precheck=model": ("independent", "none", False),
    "harness.session_mode=single harness.precheck=none": ("single", "none", False),
    "harness.persist_map=false": ("harness", "text", True),
    "harness.ledger=none": ("harness", "none", True),
    "harness.persist_map=false harness.ledger=none": ("harness", "none", True),
}


@pytest.mark.parametrize("experiment", ["std_goat_slam_es_lhnav", "std_ivlnce_tour_slam_es_lhnav"])
@pytest.mark.parametrize("seat", ["harness=navh model=opus-5", "harness=navhcodex model=gpt-6",
                                  "harness=navhmini model=qwen3.8-27b"])
@pytest.mark.parametrize("row", list(ROWS))
def test_every_row_launches_on_every_seat(experiment: str, seat: str, row: str, tmp_path: Path) -> None:
    app = compose_app(CONFIGS / f"{experiment}.yaml", [*seat.split(), *row.split(), "suffix=_row"])
    adapter, *_ = build(app)
    assert app["run"]["name"].endswith("_lhnav_row"), "a row's runs get a directory of their own"
    ctx = SimpleNamespace(index=0, run_dir=tmp_path, workdir=tmp_path / "w", max_turns=1500, benchmark="goat",
                          step_budget=5000, instruments=2, effort="default", arm=None)
    seen = adapter.describe(ctx)
    mode, ledger, files = ROWS[row]
    cfg = seen["navh_config"]
    assert cfg["session_mode"] == mode and cfg["ledger"] == ledger
    assert cfg["max_turns_per_task"] == 200 and cfg["step_budget"] == 500, "the same task budgets in every row"
    assert seen["navh_tools"] == (tool_names(recovery=True) if files else ACT)
    if mode != "harness":
        assert cfg["certify"] == "none" and cfg["recovery"] is False and cfg["notes"] is False
        assert cfg["precheck"] == ("model" if "precheck=model" in row else "none")


def test_the_three_yamls_default_to_the_reference_policy() -> None:
    import yaml

    for name in ("navh", "navhcodex", "navhmini"):
        cfg = yaml.safe_load((CONFIGS / "harness" / f"{name}.yaml").read_text())
        assert cfg["session_mode"] == "harness" and cfg["ledger"] == "text", name
    assert NavHarnessAdapter().switches["session_mode"] == "harness"
    assert NavHarnessAdapter().switches["ledger"] == "text"
