"""mini-swe-agent as the reasoning core — can it actually reach the workspace?

The claim the mini wire rests on is that the tools are not its own: they come
from the same MCP bridge the two CLIs spawn, so `list_files` / `read_file` /
`write_file` arrive on the model's menu without the wire granting anything.
That claim is only worth as much as a test that spawns the REAL bridge as a
real subprocess, lists its tools over real MCP, and calls them.

So everything here is real except two things:

    the env      a stub HTTP server answering the nodeset's verbs, so no
                 simulator is needed (the same stub the bridge tests use)
    the model    scripted. The model id is the local Qwen so the wiring is the
                 one a local run would take, but nothing is ever asked of it:
                 a smoke test that depends on what a 4B model decides to do is
                 not testing the harness.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.harnesses.navh.bridge import TaskState  # noqa: E402
from core.harnesses.navh.session.wire.mini import MiniSpec, MiniWire  # noqa: E402

MODEL = "ollama_chat/qwen3.5:4b-bf16-std"        # the local seat's id, never called
BRIDGE = REPO / "core" / "harnesses" / "navh" / "bridge" / "bridge.py"
PNG = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


class _Env:
    """The nodeset's verbs, answered without a simulator."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, fn: str, inputs: dict) -> dict:
        self.calls.append((fn, dict(inputs)))
        if fn == "observe_egocentric":
            return {"rgb": PNG, "explored_m2": 2.0}
        if fn == "step_discrete":
            return {"info": {"collision": False, "slam": "ok"}, "terminated": False, "truncated": False}
        if fn == "look_around":
            return {"views": [PNG] * 4}
        if fn == "get_map":
            return {"map": PNG, "floor": 1, "explored_m2": 3.5}
        if fn == "mark":
            return {"marked": {"label": inputs.get("label")}, "n_markers": 1}
        if fn == "get_goal":
            return {"text": "find the kettle", "modality": "object", "goal_index": 0, "n_goals": 2}
        raise KeyError(fn)


@pytest.fixture
def world(tmp_path: Path):
    """A stub env, a laid-out workspace, and the bridge's environment — the
    three things the orchestrator hands a wire."""
    env = _Env()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:
            pass

        def do_POST(self) -> None:
            fn = self.path.rsplit("__", 1)[-1]
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            try:
                out = json.dumps({"outputs": env(fn, body.get("inputs") or {})}).encode()
            except KeyError:
                self.send_response(404); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    ws, house, state = tmp_path / "workspace", tmp_path / "house", tmp_path / "state.json"
    (ws / "run" / "tasks" / "0").mkdir(parents=True)
    (ws / "house").mkdir(parents=True, exist_ok=True)
    house.mkdir(parents=True, exist_ok=True)
    (ws / "run" / "ledger.md").write_text("- task 1 | CERTIFIED | the kettle, on the hob\n")
    (house / "notes.md").write_text("# the house\nThe kitchen is through the arch.\n")
    (ws / "run" / "tasks" / "0" / "goal.md").write_text("find the kettle\n")
    TaskState(state).reset(0, writable=["run/tasks/0/handover.md"])

    spec = MiniSpec(
        model=MODEL, cwd=str(ws),
        system_prompt="You are a home robot.",
        mcp={"env": {"command": sys.executable, "args": [str(BRIDGE)], "env": {
            "NAVH_SERVER_URL": f"http://127.0.0.1:{srv.server_port}",
            "NAVH_LIVE_DIR": str(tmp_path / "live"), "NAVH_MARK_DIR": str(tmp_path / "markers"),
            "NAVH_STATE_FILE": str(state), "NAVH_WORKSPACE": str(ws), "NAVH_HOUSE_DIR": str(house),
            "NAVH_STEP_BUDGET": "50", "NAVH_STOP_VIEWS": "4", "NAVH_BENCH": "goat",
            "NAVH_RECOVERY": "1", "NAVH_TOOLSET": "all", "PYTHONPATH": str(REPO),
        }}},
        max_turns=4)
    yield spec, env, ws, house, state
    srv.shutdown()


def _toolset(spec: MiniSpec):
    """The bridge, spawned and listed — what the model's menu is made of."""
    sys.path.insert(0, str(REPO / "core" / "harnesses" / "mini"))
    from bridge_toolset import BridgeToolSet  # noqa: E402

    path, env = spec.bridge
    return BridgeToolSet(Path(path), {**env, "EH_EXECUTOR": "mini-swe"})


# ── the menu ──


def test_the_bridge_puts_the_file_tools_on_minis_menu(world) -> None:
    """Nothing in the wire names these tools. They are discovered."""
    spec, *_ = world
    ts = _toolset(spec)
    try:
        names = set(ts.tool_names())
    finally:
        ts.close()
    assert {"list_files", "read_file", "write_file"} <= names, sorted(names)
    assert {"observe", "step", "close_task", "get_goal"} <= names, sorted(names)
    # the schemas the model is shown are the bridge's, not ours
    schemas = {s["name"]: s for s in ts.tool_schemas()} if hasattr(ts, "tool_schemas") else {}
    if schemas:
        assert "path" in json.dumps(schemas.get("read_file", {}))


# ── reading the memory ──


def test_mini_reads_the_ledger_the_notes_and_the_goal(world) -> None:
    """The three files a task opens with, read through the bridge rather than
    pasted into a prompt. This is the whole point of the file-connected memory."""
    spec, _env, ws, house, _state = world
    ts = _toolset(spec)
    try:
        top = ts.execute("list_files", {})            # one directory, not a walk
        listed = ts.execute("list_files", {"path": "run"})
        led = ts.execute("read_file", {"path": "run/ledger.md"})
        notes = ts.execute("read_file", {"path": "house/notes.md"})
        goal = ts.execute("read_file", {"path": "run/tasks/0/goal.md"})
    finally:
        ts.close()
    assert "run/" in _text(top) and "house/" in _text(top), _text(top)
    assert "ledger.md" in _text(listed), _text(listed)
    assert "the kettle, on the hob" in _text(led)
    assert "through the arch" in _text(notes)
    assert "find the kettle" in _text(goal)


def test_the_fence_holds_for_mini_exactly_as_for_the_clis(world) -> None:
    """`Workspace.resolve` is one implementation, shared by all three wires:
    the dataset, the run's own output and the rest of the disk are not this
    session's to read, whichever core is asking."""
    spec, *_ = world
    ts = _toolset(spec)
    try:
        for escape in ("../../etc/passwd", "/etc/passwd", "run/../../secrets",
                       "house/../../../etc/hostname"):
            out = _text(ts.execute("read_file", {"path": escape}))
            assert "outside the workspace" in out or "error" in out.lower(), f"{escape} -> {out[:120]}"
    finally:
        ts.close()


# ── writing the memory ──


def test_mini_writes_only_the_file_the_instruction_named(world) -> None:
    """A session writes its handover and nothing else. The allowed set is the
    orchestrator's, written into the state file between hops — the same gate
    the CLIs meet."""
    spec, _env, ws, _house, _state = world
    ts = _toolset(spec)
    try:
        ok = ts.execute("write_file", {"path": "run/tasks/0/handover.md",
                                       "text": "The kettle is on the hob, left of the sink."})
        refused = ts.execute("write_file", {"path": "run/ledger.md", "text": "nope"})
    finally:
        ts.close()
    assert "kettle" in (ws / "run" / "tasks" / "0" / "handover.md").read_text()
    assert "error" in _text(ok).lower() or "wrote" in _text(ok).lower() or _text(ok)
    # the ledger belongs to the harness; a session that could edit it could
    # edit the record of its own judging
    assert (ws / "run" / "ledger.md").read_text().startswith("- task 1")
    assert "error" in _text(refused).lower() or "not yours" in _text(refused).lower() \
        or "outside" in _text(refused).lower(), _text(refused)[:150]


# ── the env tools, on the same menu ──


def test_the_env_tools_work_through_the_same_toolset(world) -> None:
    """The file tools are not a separate surface: one bridge, one menu."""
    spec, env, *_ = world
    ts = _toolset(spec)
    try:
        ts.execute("observe", {})
        ts.execute("step", {"actions": [1, 1]})       # plural: one call is several moves
        ts.execute("mark", {"label": "kitchen arch"})
    finally:
        ts.close()
    verbs = [fn for fn, _ in env.calls]
    assert "observe_egocentric" in verbs, verbs
    assert verbs.count("step_discrete") == 2, f"one step call, two moves: {verbs}"
    # `mark` banks a photo set, so it drives look_around too — one tool, two verbs
    assert "mark" in verbs and "look_around" in verbs, verbs


def _text(result: Any) -> str:
    """A `ToolResult` as the model reads it: the text parts of `content`, plus
    `info`, which is the machine-readable half the environment acts on."""
    content = getattr(result, "content", None)
    info = getattr(result, "info", None)
    parts = []
    if isinstance(content, list):
        parts += [str(p.get("text") or "") for p in content if isinstance(p, dict)]
    if info:
        parts.append(json.dumps(info, ensure_ascii=False, default=str))
    if parts:
        return "\n".join(parts)
    return json.dumps(result, ensure_ascii=False, default=str)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ── the hop itself ──


def test_a_hop_advances_one_conversation_and_hands_control_back(world, monkeypatch) -> None:
    """The wire's own contract, with only the model's answer scripted.

    The model object is mini's REAL `NavToolsModel` — only `query` is replaced,
    so message formatting, observation folding and usage accounting stay the
    ones a local run would use. Three things are checked, none of them being
    what a 4B model would choose to do:

    * `DefaultAgent.run` is never used — it clears `messages`, so using it would
      make every hop a fresh session and the ledger unreadable across a task;
    * a hop stops at its model-call budget and returns, leaving the
      conversation for the next hop to continue;
    * what happened comes back as a `Hop` in the same shape the two CLI wires
      report, so the orchestrator above cannot tell which core it was.
    """
    spec, _env, ws, *_ = world
    spec.max_turns = 2
    sys.path.insert(0, str(REPO / "core" / "harnesses" / "mini"))
    import model as mini_model                                   # noqa: E402

    seen_messages: list[list[dict]] = []
    n = {"i": 0}

    def scripted(self, messages, **kw):
        seen_messages.append(list(messages))
        n["i"] += 1
        # the real accounting path: mini has no per-call usage, it accumulates
        # into `model.billing` (a test must not invent a `last_usage`
        # attribute the model does not have).
        from core.pricing import add_litellm_usage
        add_litellm_usage(self.billing, self.config.model_name,
                          {"prompt_tokens": 11, "completion_tokens": 3})
        # the shape `NavToolsModel._parse_actions` produces: mini's agent runs
        # `extra.actions`, and a message without it executes nothing at all
        return {"role": "assistant", "content": f"turn {n['i']}",
                "extra": {"actions": [{"tool": "read_file",
                                       "args": {"path": "run/ledger.md"},
                                       "tool_call_id": f"c{n['i']}"}]}}

    monkeypatch.setattr(mini_model.NavToolsModel, "query", scripted, raising=True)

    wire = MiniWire(None, lambda: spec)
    heard: list[dict] = []
    wire.on_message = heard.append

    wire.open()
    try:
        assert {"list_files", "read_file", "write_file"} <= set(wire.info["tools"])
        first = wire.hop("Task 1 of 2. Read run/ledger.md, then act.")
        before = len(wire.agent.messages)
        second = wire.hop("Continue.")
        after = len(wire.agent.messages)
    finally:
        wire.close()

    assert first.num_turns == 2, f"a hop stops at its budget: {first.subtype} {first.error}"
    assert first.tool_uses and first.tool_uses[0]["name"] == "read_file"
    assert any("the kettle, on the hob" in " ".join(r["texts"]) for r in first.tool_results), \
        "the tool result is the real bridge's: mini read the ledger off the disk"
    # the conversation carried: the second hop saw everything the first left
    assert after > before
    assert len(seen_messages[-1]) > len(seen_messages[0]), \
        "messages grew across hops; run() would have cleared them"
    assert second.num_turns == 2
    # one hop, two model calls, 11 prompt + 3 completion each — read off the
    # ledger by difference, in the key names the meter and `in_tokens` speak
    assert first.usage.get("input_tokens") == 22, first.usage
    assert first.usage.get("output_tokens") == 6, first.usage
    from core.harnesses.navh.session.wire.blocks import ctx_tokens, out_tokens
    assert ctx_tokens(first.usage) == 22 and out_tokens(first.usage) == 6
    assert second.usage.get("input_tokens") == 22, "each hop is billed its own turns, not the total"
    assert heard, "NavAgent's event hook reaches the recorder through the wire"


# ── the three wires are one shape ──


def test_the_three_wires_are_interchangeable() -> None:
    """`BaseSession` drives whichever wire the transport names, so a difference
    in their surface is a difference the orchestrator would meet at runtime,
    not at import. These are the four things it relies on."""
    import inspect

    from core.harnesses.navh.session.wire.client import SdkWire
    from core.harnesses.navh.session.wire.codex import CodexWire

    wires = (SdkWire, CodexWire, MiniWire)
    for w in wires:
        # 1. built the same way, and each supplies its own options builder
        sig = list(inspect.signature(w.__init__).parameters)
        assert sig[:3] == ["self", "loop", "options"], (w.__name__, sig)
        assert "on_message" in sig, w.__name__
        assert callable(getattr(w, "options_for", None)), w.__name__
        # 2. the same three verbs, plus the two the session reads
        for name in ("open", "hop", "close"):
            assert callable(getattr(w, name, None)), (w.__name__, name)
        # a hop can be given a lower turn cap: what is left of the task's budget
        assert "max_turns" in inspect.signature(w.hop).parameters, w.__name__
        assert isinstance(getattr(w, "is_open", None), property), w.__name__
    # 3. a hop returns the same object, whichever core produced it
    for w in wires:
        src = inspect.getsource(w)
        assert "-> Hop" in src or "Hop()" in src, w.__name__


def test_every_wire_reports_the_fields_the_accounting_reads(world, monkeypatch) -> None:
    """`BaseSession._account` reads five fields off a `Hop` and feeds them to
    the run record and the billing meter. A wire that leaves one empty does not
    fail — it under-reports, which is worse. Checked here on the mini wire,
    which is the one that had to build its usage by hand."""
    spec, *_ = world
    spec.max_turns = 1
    sys.path.insert(0, str(REPO / "core" / "harnesses" / "mini"))
    import model as mini_model  # noqa: E402

    def scripted(self, messages, **kw):
        from core.pricing import add_litellm_usage
        add_litellm_usage(self.billing, self.config.model_name,
                          {"prompt_tokens": 7, "completion_tokens": 2})
        return {"role": "assistant", "content": "ok",
                "extra": {"actions": [{"tool": "list_files", "args": {}, "tool_call_id": "c1"}]}}

    monkeypatch.setattr(mini_model.NavToolsModel, "query", scripted, raising=True)
    wire = MiniWire(None, lambda: spec)
    wire.open()
    try:
        hop = wire.hop("go")
    finally:
        wire.close()
    assert hop.num_turns == 1
    assert hop.usage and hop.usage.get("input_tokens") == 7
    assert hop.subtype in ("success", "exit")
    assert hop.latency_s >= 0
    assert hop.tool_uses and hop.tool_results, "both halves of a tool call are recorded"
    assert hop.story, "the hop as text, in order, for the trace"


def test_a_mini_session_can_be_named_in_the_record(world, monkeypatch) -> None:
    """`BaseSession._account` copies `hop.session_id` into `run.id`, and the
    trace index and the run records name a session by it. The two CLIs get one from
    the CLI; this wire has no CLI, so it mints one per open — and a reopened
    session (what a recovery does) must be a different one."""
    spec, *_ = world
    spec.max_turns = 1
    sys.path.insert(0, str(REPO / "core" / "harnesses" / "mini"))
    import model as mini_model  # noqa: E402

    monkeypatch.setattr(mini_model.NavToolsModel, "query",
                        lambda self, messages, **kw: {"role": "assistant", "content": "ok",
                                                      "extra": {"actions": []}}, raising=True)
    wire = MiniWire(None, lambda: spec)
    wire.open()
    first_id = wire.hop("go").session_id
    wire.close()
    assert wire.session_id is None, "a closed session names nothing"
    wire.open()
    second_id = wire.hop("go").session_id
    wire.close()
    assert first_id and second_id and first_id != second_id
    assert first_id.startswith("mini-")


def test_the_transport_moves_all_five_kinds_not_just_the_agent() -> None:
    """A seat is one word in a yaml. If it moved the agent's session and left
    the judge, the wake-up or the consolidation on the other wire, the run
    would say one core and be two — and nothing would fail to say so."""
    from core.harnesses.navh.session.agent import AgentSession
    from core.harnesses.navh.session.consolidate import ConsolidateSession
    from core.harnesses.navh.session.judge import CertifySession, PrecheckSession
    from core.harnesses.navh.session.wakeup import WakeupSession
    from core.harnesses.navh.session.wire.client import SdkWire
    from core.harnesses.navh.session.wire.codex import CodexWire

    kinds = (AgentSession, PrecheckSession, CertifySession, WakeupSession, ConsolidateSession)
    for transport, wire in (("sdk-session", SdkWire), ("codex-session", CodexWire),
                            ("mini-session", MiniWire)):
        for cls in kinds:
            s = cls("m", loop=None, cwd="/tmp", transport=transport)
            assert s.wire_cls is wire, f"{cls.__name__} on {transport} -> {s.wire_cls.__name__}"


def test_a_mini_hop_takes_the_lower_cap_it_is_given(world, monkeypatch) -> None:
    """The session's cap is 3; a hop given 1 makes one model call, and the
    next hop, given none, is back to the session's own cap."""
    spec, *_ = world
    spec.max_turns = 3
    sys.path.insert(0, str(REPO / "core" / "harnesses" / "mini"))
    import model as mini_model

    monkeypatch.setattr(mini_model.NavToolsModel, "query",
                        lambda self, messages, **kw: {"role": "assistant", "content": "ok",
                                                      "extra": {"actions": [{"tool": "list_files", "args": {},
                                                                             "tool_call_id": "c"}]}},
                        raising=True)
    wire = MiniWire(None, lambda: spec)
    wire.open()
    try:
        capped = wire.hop("go", max_turns=1)
        full = wire.hop("Continue.")
    finally:
        wire.close()
    assert capped.num_turns == 1 and full.num_turns == 3 and spec.max_turns == 3
