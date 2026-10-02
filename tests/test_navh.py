"""NavHarness — the package core/harnesses/navh/ and its seat.

No server, no tokens. The package stands alone and carries none of the pieces its design rules out
(an executor, a head the harness rebuilds, a summary session, a litellm model, a harness-held
transcript, a stateless SDK wrapper): its sessions are one class per kind under one base over a live
client, its tools are a bridge process, and it is reached by the seat word ``navh`` through
``core/harnesses/navharness.py``.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from core.harnesses.navh.config import FLAT_ORDER, HarnessConfig
from core.harnesses.navharness import TRANSPORTS, NavHarnessAdapter
from runner import build, compose_app

REPO = Path(__file__).resolve().parents[1]
NAVH = REPO / "core" / "harnesses" / "navh"
EXPERIMENT = REPO / "exp_workspace" / "lhnav_es" / "configs" / "std_goat_slam_es_lhnav.yaml"

LAYERS = ("orchestrator", "orchestrator/recovery", "session", "session/wire", "bridge", "memory", "env")
def _files(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*.py") if "__pycache__" not in p.parts}


def test_the_package_stands_alone() -> None:
    # This repository carries one harness package; a stray copy of another
    # would make "which code produced this run" ambiguous.
    others = [d.name for d in (NAVH.parent).iterdir()
              if d.is_dir() and d.name.startswith("navh") and d.name != "navh"]
    assert not others, others
    for rel in sorted(_files(NAVH)):
        for n, line in enumerate((NAVH / rel).read_text().splitlines(), 1):
            if "from harness." in line:
                raise AssertionError(f"{rel}:{n}: {line.strip()}")


def test_every_layer_says_what_it_is_for() -> None:
    for layer in LAYERS:
        init = NAVH / layer / "__init__.py"
        assert init.exists(), layer
        assert init.read_text().lstrip().startswith('"""'), f"{layer}/__init__.py opens with a docstring"


def test_three_roles_and_who_calls_whom() -> None:
    """The bridge never imports a session or the orchestrator; a session never
    imports the bridge or the orchestrator: a tool cannot open a session and a
    session cannot run the chain."""
    for rel in ("bridge/bridge.py", "bridge/state.py"):
        text = (NAVH / rel).read_text()
        assert not re.search(r"\.(session|orchestrator)\b", text), rel
    for p in (NAVH / "session").rglob("*.py"):
        text = p.read_text()
        assert "from ..bridge" not in text and "from ..orchestrator" not in text, p.name
        assert "...bridge" not in text and "...orchestrator" not in text, p.name
    # the kinds stand on base.py and base.py on wire/: no kind reaches into wire's modules by name
    for p in (NAVH / "session").glob("*.py"):
        if p.name not in ("base.py", "__init__.py"):
            assert "from .wire." not in p.read_text(), p.name


def test_the_session_kinds_are_classes_of_one_base() -> None:
    from core.harnesses.navh.session.agent import AgentSession
    from core.harnesses.navh.session.base import BaseSession
    from core.harnesses.navh.session.consolidate import ConsolidateSession
    from core.harnesses.navh.session.judge import CertifySession, PrecheckSession
    from core.harnesses.navh.session.wakeup import WakeupSession

    leaves = [AgentSession, PrecheckSession, CertifySession, WakeupSession, ConsolidateSession]
    assert all(issubclass(c, BaseSession) for c in leaves)
    assert [c.kind for c in leaves] == ["agent", "precheck", "certify", "wakeup", "consolidate"]
    for c in leaves:
        d = c.describe()
        assert d["kind"] and d["writes"] and d["lifetime"] in ("call", "task", "chain"), c.__name__


def test_a_one_question_session_is_one_hop_on_a_live_wire() -> None:
    from core.harnesses.navh.session import Hop, SessionLoop
    from core.harnesses.navh.session.wakeup import WakeupSession

    class _Wire:
        def __init__(self, loop, options, *, on_message=None) -> None:
            self.opened = False
            self.prompts: list = []

        @property
        def is_open(self) -> bool:
            return self.opened

        def open(self) -> dict:
            self.opened = True
            return {"mcp": None}

        def hop(self, prompt, max_turns=None) -> Hop:
            self.prompts.append(prompt)
            return Hop(text='{"place": "kitchen"}', num_turns=1, subtype="success", session_id="s-1",
                       usage={"input_tokens": 10, "output_tokens": 5})

        def close(self) -> None:
            self.opened = False

    loop = SessionLoop()
    try:
        s = WakeupSession("m", loop=loop, cwd="/tmp")
        s.wire_cls = _Wire
        assert s.ask([{"type": "text", "text": "which room"}]) == '{"place": "kitchen"}'
        assert s.runs == 1 and not s.is_open
        assert s.last_usage == {"prompt_tokens": 10, "completion_tokens": 5}
    finally:
        loop.stop()


def test_the_switches_are_flat_and_grouped() -> None:
    cfg = HarnessConfig.from_flat(benchmark="es", chain_id=0, notes=False, compact="token", chunk_turns=7)
    assert cfg.memory.notes is False and cfg.session.compact == "token" and cfg.session.chunk_turns == 7
    assert cfg.session.transport == "sdk-session"
    assert tuple(cfg.flat()) == FLAT_ORDER
    for gone in ("rollover_at", "longterm", "orchestrator", "reset_lesson", "token_budget",
                 "image_window", "keep_reasoning", "judge_max_tokens", "lesson_max_tokens", "context_window"):
        assert gone not in FLAT_ORDER
        with pytest.raises(TypeError):
            HarnessConfig.from_flat(benchmark="es", chain_id=0, **{gone: 1})


def test_the_bridge_names_the_tools_a_session_is_given() -> None:
    from core.harnesses.navh.bridge import FILE_TOOL_NAMES, TOOL_NAMES, tool_names

    assert tool_names() == [*TOOL_NAMES, *FILE_TOOL_NAMES]
    assert tool_names(recovery=True)[-4] == "request_recovery"
    assert "get_map" not in tool_names(instruments=0) and "observe" in tool_names(instruments=0)
    assert tool_names(toolset="files") == list(FILE_TOOL_NAMES)


def test_adapter_refuses_what_it_cannot_honour() -> None:
    # Three reasoning cores, and nothing else: two CLIs that hold a conversation
    # of their own, and mini-swe-agent, whose loop is ours. The mini seat reaches
    # a provider through mini-swe-agent's own model layer, outside this package.
    # A transport this harness cannot honour must fail at the seat rather than
    # half-run.
    assert TRANSPORTS == ("sdk-session", "codex-session", "mini-session")
    with pytest.raises(ValueError):
        NavHarnessAdapter(transport="api")
    with pytest.raises(ValueError):
        NavHarnessAdapter(transport="sdk-stateless")
    assert NavHarnessAdapter(transport="codex-session").transport == "codex-session"
    assert NavHarnessAdapter(transport="mini-session").transport == "mini-session"
    a = NavHarnessAdapter()
    assert a.name == "navh" and a.switches["notes"] is True and a.switches["chunk_turns"] == 25


@pytest.mark.parametrize(
    "seat",
    [
        "harness=navh model=opus-5",
        "harness=navh model=opus-5 api=fake",
        "harness=navh model=opus-5 effort=high",
    ],
)
def test_the_navh_seat_composes_and_builds(seat: str) -> None:
    app = compose_app(EXPERIMENT, seat.split())
    adapter, arm, module_cls, _callbacks = build(app)
    assert arm.word == "lhnav" and module_cls.__name__ == "EsGoat"
    assert app["agent"]["harness"]["root"] == "navh" and adapter.name == "navh"
    assert app["agent"]["model"]["id"].startswith("claude-"), "a Claude Code seat takes bare Claude ids"
    assert app["run"]["dir"].endswith(f"/navharness/{app['run']['name']}")


# ── the Codex line ──


def test_the_codex_line_is_the_same_harness_with_one_wire_swapped() -> None:
    """Codex holds the session instead of Claude Code. Everything above the
    wire is untouched: the five kinds, the three gates, the ladder, the
    workspace. What changes is one class, chosen by the transport."""
    from core.harnesses.navh.session.base import BaseSession
    from core.harnesses.navh.session.wire.client import SdkWire
    from core.harnesses.navh.session.wire.codex import CodexWire

    s = BaseSession("gpt-6-astra", loop=None, cwd="/tmp", transport="codex-session")
    assert s.wire_cls is CodexWire
    assert BaseSession("claude-opus-5", loop=None, cwd="/tmp").wire_cls is SdkWire
    with pytest.raises(ValueError):
        BaseSession("x", loop=None, cwd="/tmp", transport="api")

    # each wire builds its own options; the session does not know which
    assert CodexWire.options_for(s)().model == "gpt-6-astra"
    assert SdkWire.options_for(s) == s.options   # the session's own builder, handed back


def test_the_codex_hop_is_capped_by_counting_not_by_asking() -> None:
    """Codex has no turn cap. The SDK enforces max_turns; here the wire follows
    the thread's rollout, counts model requests, and stops the process — only
    between turns, because one exec can carry several MCP calls and cutting at
    a result drops the rest of that exec."""
    from core.harnesses.navh.session.wire import codex as C

    src = Path(C.__file__).read_text()
    assert "roll.turns >= spec.max_turns" in src
    assert "roll.pending == 0" in src, "the cap must wait for the outstanding calls"
    assert "killpg" in src and "SIGKILL" in src, "a process that ignores SIGTERM is still ours to end"

    r = C._Rollout(None)
    assert r.pending == 0
    r.calls.add("a")
    assert r.pending == 1
    r.outputs.add("a")
    assert r.pending == 0


def test_the_codex_seat_gives_it_the_bridge_and_nothing_else() -> None:
    """The tool surface is the MCP bridge. A shell is the whole disk, and the
    dataset with the goal positions is on it."""
    from core.harnesses.navh.session.wire.codex import DISABLED_FEATURES, CodexSpec, CodexWire

    assert "shell_tool" in DISABLED_FEATURES and "unified_exec" in DISABLED_FEATURES
    spec = CodexSpec(model="gpt-6-astra", cwd="/tmp", system_prompt="BRIEF",
                     mcp={"env": {"command": "/py", "args": ["/bridge.py"],
                                  "env": {"NAVH_STATE_FILE": "/tmp/s.json"}}})
    argv = CodexWire(None, lambda: spec)._argv(spec, "go", [], None)
    assert "--sandbox" in argv and argv[argv.index("--sandbox") + 1] == "read-only"
    assert any("mcp_servers.env.command" in a for a in argv)
    assert any("developer_instructions" in a for a in argv), "the brief rides as developer instructions"
    assert argv[-2:] == ["--", "go"]
    # a resumed hop does NOT repeat the brief: the thread already has it
    again = CodexWire(None, lambda: spec)._argv(spec, "Continue.", [], "thread-1")
    assert "resume" in again and not any("developer_instructions" in a for a in again)
    assert spec.state_file == Path("/tmp/s.json")


def test_the_seat_word_default_is_not_a_codex_effort() -> None:
    """The seat's "no preference" is the string `default`. The SDK drops it;
    Codex takes an enum and refuses the word — and refuses it per request, so
    the failure would be empty hops and a provider-outage abort, not a clear
    error at startup."""
    from core.harnesses.navh.session.wire.codex import codex_args

    for unset in (None, "", "default"):
        assert "\"medium\"" in " ".join(codex_args("gpt-6-astra", unset))
    assert "\"high\"" in " ".join(codex_args("gpt-6-astra", "high"))


def test_codex_home_is_outside_the_source_tree() -> None:
    """`.codex-home` holds the login AND the rollout store, and a rollout
    inlines every MCP image result, so it reaches hundreds of MB within a few
    tasks. Under `core/` the code snapshot would copy it into every run
    directory."""
    from core.harnesses.navh.session.wire.codex import codex_home

    home = Path(codex_home()).resolve()
    assert home.parent.name == "outputs", home
    assert "core" not in home.relative_to(REPO).parts, home


def test_mini_is_the_third_core_and_reaches_the_memory_the_same_way() -> None:
    """mini-swe-agent as the reasoning core. The honest difference is that its
    loop is ours, not a CLI's; the tools are not — they come from the same MCP
    bridge, so `list_files` / `read_file` / `write_file` arrive on the model's
    menu without this wire granting anything, and the fence is the one in
    `Workspace.resolve`, shared by all three."""
    from core.harnesses.navh.session.base import BaseSession
    from core.harnesses.navh.session.wire.mini import MiniWire

    s = BaseSession("ollama_chat/qwen3.5:9b", loop=None, cwd="/tmp", transport="mini-session",
                    mcp={"env": {"command": "/py", "args": ["/bridge.py"],
                                 "env": {"NAVH_SERVER_URL": "http://x"}}})
    assert s.wire_cls is MiniWire
    spec = MiniWire.options_for(s)()
    assert spec.bridge == ("/bridge.py", {"NAVH_SERVER_URL": "http://x"})

    src = Path(MiniWire.__module__.replace(".", "/") + ".py")
    text = (REPO / src).read_text()
    # the menu is the bridge's, never a list written here
    assert "tool_schemas()" in text and "tool_names()" in text
    assert "list_files" not in text.split('"""', 2)[2], "the tools are discovered, not named"
    # DefaultAgent.run would clear the conversation, making every hop a new one
    assert "agent.step()" in text and "agent.run(" not in text


def test_all_three_cores_are_told_the_same_budgets() -> None:
    """One harness, three cores: the seats may differ only in who holds the
    session. A turn budget that drifts between them turns a core comparison
    into a budget comparison."""
    import yaml

    seats = {}
    for name in ("navh", "navhcodex", "navhmini"):
        seats[name] = yaml.safe_load(
            (REPO / "exp_workspace/lhnav_es/configs/harness" / f"{name}.yaml").read_text())
    base = dict(seats["navh"])
    for name, cfg in seats.items():
        differs = {k for k in set(base) | set(cfg) if base.get(k) != cfg.get(k)}
        assert differs <= {"root", "transport"}, f"{name} also differs in {differs - {'root', 'transport'}}"
    # the paper's settings: H = 25, T_f = 80, t_min = 12, at most one recovery,
    # 500 steps and 200 turns per task, both completion checks, four stop views
    assert base["chunk_turns"] == 25 and base["reset_after_turns"] == 80 and base["recovery_min_turns"] == 12
    assert base["max_resets_per_task"] == 1 and base["step_budget_per_task"] == 500
    assert base["max_turns_per_task"] == 200 and base["stop_views"] == 4
    assert base["precheck"] == "model" and base["certify"] == "model"
    # recovery on, with escalation, handing over the structured note
    assert base["recovery"] is True and base["recovery_escalation"] is True
    assert base["recovery_handover"] == "structured" and base["recovery_handover_bank"] == ""


# ── the paper's recovery rows, as switches ──


def test_the_recovery_switches_reach_the_seat_and_refuse_what_cannot_run(tmp_path: Path) -> None:
    a = NavHarnessAdapter()
    assert a.switches["recovery_handover"] == "structured" and a.switches["recovery_escalation"] is True
    assert a.switches["recovery_handover_bank"] == ""
    with pytest.raises(ValueError):
        NavHarnessAdapter(recovery_handover="prose")
    with pytest.raises(ValueError):
        NavHarnessAdapter(recovery_handover="summary")      # no structured notes to match in length
    with pytest.raises(FileNotFoundError):
        NavHarnessAdapter(recovery_handover="other", recovery_handover_bank=str(tmp_path / "missing"))
    bank = tmp_path / "lessons.jsonl"
    bank.write_text(json.dumps({"goal": "find a bed", "lam_source": "own",
                                "lam": {"untried": ["the guest room"], "last_anchor": "landing"}}) + "\n")
    a = NavHarnessAdapter(recovery_handover="summary", recovery_handover_bank=str(bank),
                          recovery_escalation=False)
    cfg = HarnessConfig.from_flat(benchmark="es", chain_id=0, **a.switches)
    assert cfg.ladder.recovery_handover == "summary" and cfg.ladder.recovery_escalation is False
    assert cfg.ladder.recovery_handover_bank == str(bank)
    assert NavHarnessAdapter(recovery_handover="other").switches["recovery_handover"] == "other"


@pytest.mark.parametrize(
    "row",
    [
        "harness.recovery=false",
        "harness.recovery_handover=empty",
        "harness.recovery_handover=other",
        "harness.recovery_escalation=false",
    ],
)
def test_the_recovery_rows_compose_and_build(row: str) -> None:
    app = compose_app(EXPERIMENT, ["harness=navh", "model=opus-5", row])
    adapter, *_ = build(app)
    key, value = row.split("=", 1)
    got = adapter.switches[key.split(".", 1)[1]]
    assert str(got).lower() == value


# ── the SDK wire's per-hop turn cap ──


class _ScriptedClient:
    """A connected client that plays turns the way the CLI streams them: each
    turn as two assistant messages under one id (a thinking block, then a tool
    call), then the call's result, until the script ends or it is interrupted.
    Like the CLI, it counts a turn it cancelled."""

    def __init__(self, turns: int, ids: bool = True) -> None:
        self.turns = turns
        self.ids = ids
        self.done = 0
        self.interrupted_after: int | None = None

    async def query(self, prompt) -> None:
        return None

    async def interrupt(self) -> None:
        self.interrupted_after = self.done

    async def receive_response(self):
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )
        for t in range(self.turns):
            if self.interrupted_after is not None:
                break
            mid = f"msg-{t}" if self.ids else None
            yield AssistantMessage(content=[ThinkingBlock(thinking="hm", signature="")], model="m",
                                   message_id=mid)
            yield AssistantMessage(content=[ToolUseBlock(id=f"c{t}", name="mcp__env__observe", input={})],
                                   model="m", message_id=mid)
            self.done += 1                      # the turn is done once its result is out
            yield UserMessage(content=[ToolResultBlock(tool_use_id=f"c{t}", content="ok")])
        cut = self.interrupted_after is not None
        yield ResultMessage(subtype="error_during_execution" if cut else "success", duration_ms=1,
                            duration_api_ms=1, is_error=cut, num_turns=self.done + (1 if cut else 0),
                            session_id="s-1")


def test_the_sdk_hop_stops_at_a_lower_cap_between_turns() -> None:
    """The client was connected with max_turns 25; a hop given 3 counts its
    turns by message id and interrupts once the third turn's tool call has its
    result, reporting what the CLI's own cap reports."""
    from core.harnesses.navh.session.wire.client import SdkWire

    wire = SdkWire(None, lambda: None)
    wire._cap = 25
    client = _ScriptedClient(turns=10)
    hop = asyncio.run(wire._hop(client, "go", 3))
    assert client.interrupted_after == 3, "two messages per turn are one turn, and the third turn finishes"
    assert hop.num_turns == 3 and hop.subtype == "error_max_turns" and hop.error is None
    assert len(hop.tool_uses) == len(hop.tool_results) == 3
    # a CLI that sends no message ids: a turn is what streams between two results
    client = _ScriptedClient(turns=10, ids=False)
    assert asyncio.run(wire._hop(client, "go", 2)).num_turns == 2 and client.interrupted_after == 2
    # at the connection's own cap the CLI stops the hop; the wire does not interrupt
    client = _ScriptedClient(turns=4)
    hop = asyncio.run(wire._hop(client, "go", 25))
    assert client.interrupted_after is None and hop.num_turns == 4 and hop.subtype == "success"
