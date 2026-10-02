"""The completion checks on every core, a judge on another core, and the gate scored where it refused.

Three things are held here, each against stubs that stand where a network would:

* every session kind opens, and the tool-less ones answer, on each of the three wires — Claude Code
  (``claude_agent_sdk.ClaudeSDKClient`` replaced), the Codex CLI (a ``codex`` executable on PATH that
  speaks ``--json``), mini-swe-agent (``litellm.completion`` replaced; mini's own model and loop, and
  the real bridge, run as they do in a chain). A navhmini chain once died at its first judged
  ``close_task`` because the mini wire would not open without a bridge; ``test_a_mini_chain_*`` runs
  that chain end to end;
* the judge may be another model on another core (the cross-model judging comparison), routed to
  the core that runs it, and refused at the seat when its core cannot;
* each refused close request keeps the evaluator's truth at the pose it was made from
  (``stop_truth``), and the chain aggregate scores the gate from it.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.harnesses.navh.bridge import TaskState  # noqa: E402
from core.harnesses.navh.config import HarnessConfig  # noqa: E402
from core.harnesses.navh.env.types import Goal, TaskEnd  # noqa: E402
from core.harnesses.navh.memory.longterm import NOTE_PATTERNS, HouseNotes  # noqa: E402
from core.harnesses.navh.memory.workspace import Workspace  # noqa: E402
from core.harnesses.navh.orchestrator import Orchestrator  # noqa: E402
from core.harnesses.navh.session import Hop, SessionLoop  # noqa: E402
from core.harnesses.navh.session.agent import AgentSession  # noqa: E402
from core.harnesses.navh.session.base import BaseSession  # noqa: E402
from core.harnesses.navh.session.consolidate import ConsolidateSession  # noqa: E402
from core.harnesses.navh.session.judge import CertifySession, PrecheckSession  # noqa: E402
from core.harnesses.navh.session.wakeup import WakeupSession  # noqa: E402
from core.harnesses.navh.session.wire.mini import MiniWire, openai_parts  # noqa: E402

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
BRIDGE = REPO / "core" / "harnesses" / "navh" / "bridge" / "bridge.py"
QWEN = "ollama_chat/qwen3.8-27b-q4-std"
OPUS = "claude-opus-5"
VERDICT = {"verdict": "incomplete", "evidence": "no kettle in the stop views", "confidence": "high"}
TRANSPORTS = ("sdk-session", "codex-session", "mini-session")
MODEL_OF = {"sdk-session": OPUS, "codex-session": "gpt-6-astra", "mini-session": QWEN}


# ── the stubs ───────────────────────────────────────────────────────────


class _HttpEnv:
    """The bridge's env verbs over HTTP (``/call/<nodeset>__<fn>``), no simulator."""

    def __init__(self) -> None:
        env = self
        self.calls: list[str] = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def do_POST(self) -> None:
                fn = self.path.rsplit("__", 1)[-1]
                n = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(n)
                env.calls.append(fn)
                out = {"views": [PNG] * 4} if fn == "look_around" else {}
                body = json.dumps({"outputs": out}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.srv = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}"

    def stop(self) -> None:
        self.srv.shutdown()


def _completion(script):
    """``litellm.completion`` replaced by a script: (model, messages, tools) -> reply. A reply is a
    string (an answer), or ``(tool, args)`` (one tool call). The response is a real litellm
    ``ModelResponse``, so mini's own parsing, cost and billing paths run on it."""
    import litellm

    calls: list[dict] = []

    def completion(*, model, messages, tools=None, **kw):
        calls.append({"model": model, "messages": messages, "tools": tools, "kwargs": kw})
        reply = script(model, messages, [t["function"]["name"] for t in (tools or [])])
        if isinstance(reply, str):
            message, finish = {"role": "assistant", "content": reply}, "stop"
        else:
            tool, args = reply
            message = {"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{len(calls)}", "type": "function",
                 "function": {"name": tool, "arguments": json.dumps(args)}}]}
            finish = "tool_calls"
        return litellm.ModelResponse(choices=[{"message": message, "finish_reason": finish}],
                                     usage={"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14})

    completion.calls = calls
    return completion


class _SdkClient:
    """``ClaudeSDKClient`` with the CLI taken out: it connects, reports the MCP server, and answers
    every query with ``_SdkClient.answer`` as one assistant turn and a result."""

    answer = json.dumps(VERDICT)
    queries: list[Any] = []

    def __init__(self, options=None) -> None:
        self.options = options

    async def connect(self) -> None:
        return None

    async def get_mcp_status(self) -> dict:
        return {"mcpServers": [{"name": "env", "status": "connected"}]}

    async def query(self, prompt) -> None:
        if isinstance(prompt, str):
            _SdkClient.queries.append(prompt)
            return
        async for msg in prompt:
            _SdkClient.queries.append(msg["message"]["content"])

    async def receive_response(self):
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

        yield AssistantMessage(content=[TextBlock(text=_SdkClient.answer)], model=OPUS)
        yield ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False, num_turns=1,
                            session_id="sdk-1", usage={"input_tokens": 9, "output_tokens": 4})

    async def disconnect(self) -> None:
        return None


FAKE_CODEX = r'''#!PYTHON
"""`codex exec --json`, scripted: one thread, one turn, one answer; argv kept for the test."""
import json, os, pathlib, sys, uuid
argv = sys.argv[1:]
with pathlib.Path(os.environ["FAKE_CODEX_LOG"]).open("a") as fh:
    fh.write(json.dumps(argv) + "\n")
thread = argv[argv.index("resume") + 1] if "resume" in argv else "th-" + uuid.uuid4().hex[:8]
day = pathlib.Path(os.environ["CODEX_HOME"]) / "sessions" / "2026" / "10" / "02"
day.mkdir(parents=True, exist_ok=True)
usage = {"last_token_usage": {"input_tokens": 12, "output_tokens": 5}}
with (day / f"rollout-x-{thread}.jsonl").open("a") as fh:
    fh.write(json.dumps({"type": "event_msg", "payload": {"type": "token_usage_record", "info": usage}}) + "\n")
print(json.dumps({"type": "thread.started", "thread_id": thread}), flush=True)
answer = {"type": "agent_message", "text": os.environ["FAKE_CODEX_ANSWER"]}
print(json.dumps({"type": "item.completed", "item": answer}), flush=True)
print(json.dumps({"type": "turn.completed"}), flush=True)
'''


@pytest.fixture
def cores(tmp_path: Path, monkeypatch):
    """All three cores stubbed at the network's edge."""
    import claude_agent_sdk

    from core.harnesses.navh.session.wire import home

    monkeypatch.setattr(home, "ensure_relocated", lambda *a, **k: {"dir": str(tmp_path / "claude")})
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", _SdkClient)
    _SdkClient.queries = []
    _SdkClient.answer = json.dumps(VERDICT)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    codex = bin_dir / "codex"
    codex.write_text(FAKE_CODEX.replace("PYTHON", sys.executable, 1))
    codex.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("NAVH_CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setenv("FAKE_CODEX_LOG", str(tmp_path / "codex_argv.jsonl"))
    monkeypatch.setenv("FAKE_CODEX_ANSWER", json.dumps(VERDICT))

    import litellm

    def script(model, messages, tools):
        if not tools:
            return json.dumps(VERDICT)
        if "observe" in tools:
            return "DONE"                                       # the agent, opened and told to wait
        st = messages[-1]
        if st.get("role") == "tool":
            return "DONE"
        return ("write_file", {"path": "house/house.md", "text": "The kitchen is through the arch."})

    completion = _completion(script)
    monkeypatch.setattr(litellm, "completion", completion)

    env = _HttpEnv()
    loop = SessionLoop()
    yield {"tmp": tmp_path, "env": env, "loop": loop, "completion": completion}
    loop.stop()
    env.stop()


def _bridge(tmp_path: Path, url: str, state: Path, toolset: str) -> dict:
    ws = tmp_path / "workspace"
    return {"env": {"type": "stdio", "command": sys.executable, "args": [str(BRIDGE)], "env": {
        "NAVH_SERVER_URL": url, "NAVH_LIVE_DIR": str(tmp_path / "live"), "NAVH_MARK_DIR": str(tmp_path / "marks"),
        "NAVH_STATE_FILE": str(state), "NAVH_WORKSPACE": str(ws), "NAVH_HOUSE_DIR": str(ws / "house"),
        "NAVH_STEP_BUDGET": "50", "NAVH_STOP_VIEWS": "4", "NAVH_BENCH": "goat", "NAVH_RECOVERY": "0",
        "NAVH_TOOLSET": toolset, "PYTHONPATH": str(REPO)}}}


def _frames(tmp_path: Path) -> list[dict]:
    live = tmp_path / "frames"
    live.mkdir(exist_ok=True)
    out = []
    for i in range(1, 5):
        p = live / f"obs_{i:04d}.png"
        p.write_bytes(base64.b64decode(PNG))
        out.append({"id": i, "path": str(p), "step": 3, "kind": "stop"})
    return out


# ── every kind, every wire ──────────────────────────────────────────────


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_every_session_kind_opens_and_the_tool_less_ones_answer_on_every_wire(transport: str, cores) -> None:
    tmp, env, loop = cores["tmp"], cores["env"], cores["loop"]
    model = MODEL_OF[transport]
    ws = Workspace(tmp / "workspace", tmp / "workspace" / "house")
    common = {"loop": loop, "cwd": ws.root, "transport": transport}
    goal = Goal(index=0, total=1, text="find the kettle", modality="object")

    # the agent's session: the whole bridge, opened and closed
    agent_state = TaskState(tmp / "state_agent.json")
    agent_state.reset(0, writable=[])
    agent = AgentSession(model, system_prompt="You are a home robot.", tools=["observe"], max_turns=4,
                         mcp=_bridge(tmp, env.url, agent_state.path, "all"), **common)
    agent.open()
    assert agent.is_open
    agent.close()

    # the two completion checks: no tools, the frames and the map in the question
    for cls in (PrecheckSession, CertifySession):
        judge = cls(model, **common)
        assert judge.mcp is None and judge.wire_cls.__name__ in ("SdkWire", "CodexWire", "MiniWire")
        v = judge.judge(goal, _frames(tmp), {"task": 1, "of": 1, "steps_used": 3}, map_b64=PNG)
        assert v.verdict == "incomplete" and v.judge == "model" and not v.error, (cls.kind, transport, v)
        assert v.evidence.startswith("no kettle") and not judge.is_open and judge.runs == 1
        assert v.frames_shown == [1, 2, 3, 4]

    # the place question: no tools, views and banked photographs in one message
    wake = WakeupSession(model, **common)
    raw = wake.ask([{"type": "text", "text": "What the robot sees now:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}},
                    {"type": "text", "text": "Which place?"}])
    assert json.loads(raw)["verdict"] == "incomplete" and wake.runs == 1 and not wake.is_open

    # the consolidation: the file tools alone
    cstate = TaskState(tmp / "state_consolidate.json")
    cstate.reset(-1, writable=list(NOTE_PATTERNS))
    notes = HouseNotes(ws.house, day="2026-10-02")
    notes.journal(0, "find the kettle", "NOT DONE", 3, [], "searched the kitchen")
    cons = ConsolidateSession(model, mcp=_bridge(tmp, env.url, cstate.path, "files"),
                              tools=["list_files", "read_file", "write_file"], **common)
    report = cons.consolidate(ws, cstate, notes)
    assert not report["error"], report
    assert report["turns"] >= 1 and cons.runs == 1 and not cons.is_open

    # what each core was actually sent: the images rode along as that core takes them
    if transport == "sdk-session":
        sent = [q for q in _SdkClient.queries if isinstance(q, list)]
        assert sent and any(b.get("type") == "image" for b in sent[0])
    elif transport == "codex-session":
        argvs = [json.loads(line) for line in (tmp / "codex_argv.jsonl").read_text().splitlines()]
        judged = [a for a in argvs if "-i" in a]
        assert judged and not any("mcp_servers.env.command" in x for x in judged[0]), "a judge has no bridge"
    else:
        asked = [c for c in cores["completion"].calls if not c["tools"]]
        parts = asked[0]["messages"][-1]["content"]
        assert any(p.get("type") == "image_url" and p["image_url"]["url"].startswith("data:image/png;base64,")
                   for p in parts), "an Anthropic image block reaches mini as no picture at all"
        assert not any(p.get("type") == "image" for p in parts)
        # the consolidation wrote through the real bridge, and its "DONE" ended the hop there
        assert report["model_written"] is True and "kitchen" in notes.overview.read_text()
        assert report["turns"] == 2, report


def test_a_judge_that_cannot_be_reached_is_unknown_and_the_chain_goes_on(tmp_path: Path) -> None:
    """Every judge failure is the same verdict: ``unknown``, which never refuses a closure."""

    class _Dead:
        def __init__(self, loop, options, *, on_message=None) -> None:
            pass

        is_open = False

        def open(self) -> dict:
            raise RuntimeError("the judge's endpoint is down")

        def close(self) -> None:
            pass

    goal = Goal(index=0, total=1, text="find the kettle", modality="object")
    judge = PrecheckSession("m", loop=None, cwd=tmp_path)
    judge.wire_cls = _Dead
    v = judge.judge(goal, [], {"task": 1})
    assert v.verdict == "unknown" and "endpoint is down" in v.error and not v.disagrees_with("complete")
    wake = WakeupSession("m", loop=None, cwd=tmp_path)
    wake.wire_cls = _Dead
    with pytest.raises(RuntimeError):
        wake.ask([{"type": "text", "text": "which room"}])      # the funnel records why


def test_a_failed_mini_answer_is_reported_not_parsed(tmp_path: Path, monkeypatch) -> None:
    import litellm

    def down(**kw):
        raise litellm.exceptions.AuthenticationError("no key", llm_provider="ollama", model=QWEN)

    monkeypatch.setattr(litellm, "completion", down)
    judge = CertifySession(QWEN, loop=None, cwd=tmp_path, transport="mini-session")
    v = judge.judge(Goal(index=0, total=1, text="find the kettle", modality="object"), [], {"task": 1})
    assert v.verdict == "unknown" and "AuthenticationError" in v.error


class _Answers:
    """A judge's wire that answers each hop from a script of ``Hop``s."""

    script: list = []
    asked: list = []

    def __init__(self, loop, options, *, on_message=None) -> None:
        self.opened = False

    @property
    def is_open(self) -> bool:
        return self.opened

    def open(self) -> dict:
        self.opened = True
        return {}

    def hop(self, prompt, max_turns=None) -> Hop:
        _Answers.asked.append(prompt)
        return _Answers.script.pop(0)

    def close(self) -> None:
        self.opened = False


def _said(text: str, thinking: str = "") -> Hop:
    return Hop(text=text, thinking=thinking, num_turns=1, subtype="success", session_id="j",
               usage={"input_tokens": 50, "output_tokens": 5})


def _judge_on(tmp_path: Path, *answers: Hop):
    _Answers.script, _Answers.asked = list(answers), []
    judge = CertifySession("m", loop=None, cwd=tmp_path)
    judge.wire_cls = _Answers
    return judge.judge(Goal(index=0, total=1, text="find the kettle", modality="object"), [], {"task": 1})


def test_the_verdict_is_read_from_the_answer_and_a_failed_parse_is_asked_once_more(tmp_path: Path) -> None:
    """The answer is parsed; when it does not parse, the same conversation is
    asked once for the JSON alone; when that does not parse either, the
    verdict is unknown. The model's reasoning is never read for a verdict."""
    from core.harnesses.navh import prompts as P

    complete = json.dumps({"verdict": "complete", "evidence": "kettle on the counter", "confidence": "high"})
    v = _judge_on(tmp_path, _said(complete))
    assert v.verdict == "complete" and v.answered_by == "prose" and len(_Answers.asked) == 1

    v = _judge_on(tmp_path, _said("I looked at the frames.", thinking=json.dumps(VERDICT)),
                  _said(complete))
    assert len(_Answers.asked) == 2 and _Answers.asked[1] == P.VERDICT_ONLY
    assert v.verdict == "complete" and v.answered_by == "retry", "the thinking's verdict is not taken"
    assert v.usage["retried"] is True and v.usage["completion_tokens"] == 10

    v = _judge_on(tmp_path, _said("Hard to say.", thinking=json.dumps(VERDICT)),
                  _said("Still hard to say.", thinking=json.dumps(VERDICT)))
    assert v.verdict == "unknown" and v.error and not v.disagrees_with("complete")
    assert len(_Answers.asked) == 2, "asked once more, not again"


def test_openai_parts_turns_every_image_block_into_an_image_url() -> None:
    parts = openai_parts([{"type": "text", "text": "a"},
                          {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "QQ=="}}])
    assert parts == [{"type": "text", "text": "a"},
                     {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QQ=="}}]


def test_the_codex_wire_names_another_provider_only_when_the_session_says_so() -> None:
    from core.harnesses.navh.session.wire.codex import CodexWire

    s = BaseSession("gpt-6-astra", loop=None, cwd="/tmp", transport="codex-session",
                    endpoint={"codex": ["-c", 'model_provider = "acfake"']})
    spec = CodexWire.options_for(s)()
    argv = CodexWire(None, lambda: spec)._argv(spec, "go", [], None)
    assert 'model_provider = "acfake"' in argv and argv[-2:] == ["--", "go"]
    plain = BaseSession("gpt-6-astra", loop=None, cwd="/tmp", transport="codex-session")
    spec = CodexWire.options_for(plain)()
    assert not any("model_provider" in a for a in CodexWire(None, lambda: spec)._argv(spec, "go", [], None))
    mini = BaseSession(QWEN, loop=None, cwd="/tmp", transport="mini-session",
                       endpoint={"mini": {"api_base": "http://127.0.0.1:11434"}})
    assert MiniWire.options_for(mini)().model_kwargs == {"api_base": "http://127.0.0.1:11434"}


# ── a navhmini chain, judged, end to end ────────────────────────────────


class _Bench:
    """One task; the STOP is scored a success; ``stop_truth`` says the refused stop would have failed."""

    house_id = "stub-house"

    def __init__(self, url: str) -> None:
        bench = self

        class _Env:
            def __init__(self) -> None:
                self.url = url
                self.calls: list[str] = []

            def call(self, fn: str, inputs: dict | None = None, config: dict | None = None) -> dict:
                self.calls.append(fn)
                if fn == "step" and int((inputs or {}).get("action", -1)) == 0:
                    return {"info": {"subtask_result": {"success": 1.0, "distance_to_goal": 0.4}}}
                if fn == "get_map":
                    return {"map": PNG}
                if fn == "look_around":
                    return {"views": [PNG] * 4}
                if fn == "stop_truth":
                    return bench.truth
                return {}

        self.env = _Env()
        self.truth: dict = {"success": 0.0, "distance_to_goal": 2.613, "success_distance": 1.0}

    @staticmethod
    def fn(name: str) -> str:
        return name

    def seat(self, arg, persist_map, skip=0) -> Goal:
        return Goal(index=0, total=1, text="find the kettle", modality="object")

    def parse_step(self, out: dict, claimed) -> tuple[dict, TaskEnd | None]:
        res = (out.get("info") or {}).get("subtask_result")
        return {}, (TaskEnd(end_reason="stop", score=dict(res), chain_finished=False) if res else None)

    def after_end(self, end: TaskEnd, persist_map) -> TaskEnd:
        end.chain_finished, end.next_goal = True, None
        return end

    def chain_report(self) -> dict:
        return {}


def test_a_mini_chain_closes_a_judged_task_and_keeps_the_truth_where_it_was_refused(tmp_path: Path,
                                                                                     monkeypatch) -> None:
    """navhmini with both checks on: the agent writes its handover and asks to close, the judge (a
    tool-less mini session) sends the request back, the refusal keeps the evaluator's truth, the second
    request is honoured, and the day ends with the consolidation writing a note — all on the mini core,
    through the real bridge."""
    import litellm

    def script(model, messages, tools):
        if not tools:
            parts = messages[-1]["content"]
            assert any(p.get("type") == "image_url" for p in parts), "the judge sees the stop views"
            return json.dumps(VERDICT)
        last = messages[-1]
        if "observe" not in tools:                              # the consolidation's files-only bridge
            if last.get("role") == "tool":
                return "DONE"
            return ("write_file", {"path": "house/house.md", "text": "The kettle was not in the kitchen."})
        if last.get("role") == "tool" and "close_requested" in json.dumps(last.get("content")):
            return "DONE"                                       # as the tool asks: the turn ends here
        wrote = any("handover.md" in json.dumps(m.get("tool_calls") or "") for m in messages)
        if not wrote:
            return ("write_file", {"path": "run/tasks/0/handover.md", "text": "Searched the kitchen."})
        return ("close_task", {"status": "complete", "evidence": "the kettle is on the hob"})

    completion = _completion(script)
    monkeypatch.setattr(litellm, "completion", completion)
    env = _HttpEnv()
    try:
        cfg = HarnessConfig.from_flat(benchmark="es", chain_id=0, memory_root=str(tmp_path / "mem"),
                                      transport="mini-session", precheck="model", certify="model", notes=True,
                                      place_graph=False, max_turns_per_task=40, chunk_turns=6, step_budget=50)
        bench = _Bench(env.url)
        o = Orchestrator(cfg, QWEN, None, bench, tmp_path / "out", bridge=BRIDGE)
        summary = o.run()
    finally:
        env.stop()

    assert summary["exit_status"] == "chain_complete", summary.get("exit_status")
    assert bench.env.calls.count("step") == 1, "one STOP, after the second request"
    row = summary["tasks"][0]
    assert row["claim"]["status"] == "complete" and row["truth"]["success"] == 1.0
    pre = row["precheck"]
    assert pre["refused_once"] is True and pre["close_requests"] == 2
    assert pre["refusals"] == [{"by": "judge", "turn": pre["refusals"][0]["turn"], "steps_used": 0,
                                "claim": "complete", "verdict": "incomplete",
                                "truth": {"success": 0.0, "distance_to_goal": 2.613, "success_distance": 1.0}}]
    assert row["verdict"]["verdict"] == "incomplete"
    assert row["context"]["tokens"]["precheck"]["calls"] == 1, "the mini judge is metered under its check"
    sessions = summary["sessions"]
    assert sessions["precheck"]["opened"] == 1 and sessions["precheck"]["transport"] == "mini-session"
    assert sessions["agent"]["model"] == QWEN and sessions["consolidate"]["opened"] == 1
    assert summary["notes"]["model_written"] is True and summary["judge_transport"] == "mini-session"
    agg = summary["aggregate"]
    assert agg["precheck_refusals"] == 1
    assert (agg["gate_refused_tasks"], agg["gate_truth_read"], agg["gate_correct_refusals"],
            agg["gate_rescues"], agg["gate_harms"]) == (1, 1, 1, 1, 0)
    # the truth is the record's alone: nothing the agent could read carries it
    for p in (tmp_path / "out" / "workspace").rglob("*"):
        if p.is_file():
            assert "2.613" not in p.read_text(errors="ignore"), p
    for c in completion.calls:
        assert "distance_to_goal" not in json.dumps(c["messages"]), "the truth reached a prompt"


# ── the gate's truth, against the scripted wire ─────────────────────────


class _Scripted:
    """A wire that plays the bridge's part by editing the state file, one handler per kind."""

    orch: Orchestrator | None = None
    handlers: dict = {}

    def __init__(self, loop, options, *, on_message=None) -> None:
        self.kind = on_message.__self__.kind
        self.opened = False

    @property
    def is_open(self) -> bool:
        return self.opened

    def open(self) -> dict:
        self.opened = True
        return {}

    def hop(self, prompt, max_turns=None) -> Hop:
        return self.handlers[self.kind](prompt)

    def close(self) -> None:
        self.opened = False


def _ask_to_close(o: Orchestrator) -> None:
    st = o.state.read()
    st["close_request"] = {"status": "complete", "evidence": "here", "steps_used": st["steps_used"]}
    st["close_requests"] += 1
    o.state.write(st)


def test_the_file_gate_and_an_env_without_the_verb_are_recorded_too(tmp_path: Path, monkeypatch) -> None:
    """A refusal by the file gate is kept as well (``by: handover``); an env that cannot report the
    truth leaves its reason, and the aggregate counts that refusal without guessing its outcome."""
    from core.harnesses.navh.prompts import HANDOVER_REQUIRED

    def agent(prompt) -> Hop:
        o = _Scripted.orch
        text = prompt if isinstance(prompt, str) else ""
        if text.startswith(HANDOVER_REQUIRED.split("{")[0][:20]):
            o.ws.session_write("run/tasks/0/handover.md", "searched", set(o.state.read()["writable"]))
        _ask_to_close(o)
        return Hop(text="DONE", num_turns=1, subtype="success", tool_uses=[{"id": "t", "name": "close_task"}])

    def judge(prompt) -> Hop:
        return Hop(text=json.dumps(VERDICT), num_turns=1, subtype="success")

    monkeypatch.setattr(BaseSession, "wire_cls", _Scripted)
    _Scripted.handlers = {"agent": agent, "precheck": judge, "certify": judge}
    bench = _Bench("http://stub")
    bench.truth = {"error": "no verb stop_truth"}
    cfg = HarnessConfig.from_flat(benchmark="es", chain_id=0, memory_root=str(tmp_path / "mem"),
                                  precheck="model", certify="model", notes=False, place_graph=False,
                                  max_turns_per_task=40)
    o = Orchestrator(cfg, OPUS, None, bench, tmp_path / "out", bridge=BRIDGE)
    _Scripted.orch = o
    summary = o.run()
    refusals = summary["tasks"][0]["precheck"]["refusals"]
    assert [r["by"] for r in refusals] == ["handover", "judge"]
    assert all(r["truth"] == {"error": "no verb stop_truth"} for r in refusals)
    agg = summary["aggregate"]
    assert agg["gate_refused_tasks"] == 1 and agg["gate_truth_read"] == 0 and agg["gate_rescues"] == 0


# ── a judge on another core ─────────────────────────────────────────────


def test_the_judge_can_sit_on_another_core_and_only_the_checks_move(tmp_path: Path) -> None:
    """Opus 5 navigating on Claude Code, Qwen3.8-27B judging on mini-swe-agent: both checks move, the
    wake-up and the consolidation stay the navigator's, and a vendor's effort word does not cross."""
    cfg = HarnessConfig.from_flat(benchmark="es", chain_id=0, memory_root=str(tmp_path / "mem"),
                                  judge_transport="mini-session", place_graph=False)
    o = Orchestrator(cfg, OPUS, QWEN, _Bench("http://stub"), tmp_path / "out", bridge=BRIDGE, effort="high",
                     endpoint={}, judge_endpoint={"mini": {"api_base": "http://127.0.0.1:11434"}})
    try:
        for check in (o.pre.session, o.cert.session):
            assert (check.model, check.transport, check.wire_cls) == (QWEN, "mini-session", MiniWire)
            assert check.effort is None and check.endpoint == {"mini": {"api_base": "http://127.0.0.1:11434"}}
        for own in (o.agent_session, o.driver.wake, o.consolidate_session):
            assert (own.model, own.transport, own.effort) == (OPUS, "sdk-session", "high")
        report = o._sessions_report()
        assert report["precheck"]["transport"] == "mini-session" and report["agent"]["transport"] == "sdk-session"
        # the swapped judge is metered under its check, like any other
        o.pre.session.meter(QWEN, {"input_tokens": 50, "output_tokens": 5})
        o.cert.session.meter(QWEN, {"input_tokens": 70, "output_tokens": 7})
        assert o.tally.get("precheck", "prompt_tokens") == 50 and o.tally.get("certify", "completion_tokens") == 7
        assert o.tally.get("agent", "calls") == 0
    finally:
        o.shutdown()
    # by default the judge is the seat, on the seat's core, with the seat's effort and endpoint
    cfg = HarnessConfig.from_flat(benchmark="es", chain_id=0, memory_root=str(tmp_path / "mem2"),
                                  transport="mini-session", place_graph=False)
    o = Orchestrator(cfg, QWEN, None, _Bench("http://stub"), tmp_path / "out2", bridge=BRIDGE,
                     endpoint={"mini": {"api_base": "http://x"}})
    try:
        assert (o.pre.session.model, o.pre.session.transport) == (QWEN, "mini-session")
        assert o.pre.session.endpoint == {"mini": {"api_base": "http://x"}}
    finally:
        o.shutdown()


def test_the_seat_refuses_a_judge_its_core_cannot_run() -> None:
    from core.harnesses.navharness import NavHarnessAdapter

    # the paper's two directions build
    a = NavHarnessAdapter(transport="mini-session", judge_model=OPUS, judge_transport="sdk-session")
    assert a.switches["judge_transport"] == "sdk-session"
    b = NavHarnessAdapter(transport="sdk-session", judge_model=QWEN, judge_transport="mini-session")
    assert b.settings["judge_model"] == QWEN
    assert NavHarnessAdapter().switches["judge_transport"] == "", "the reference: the seat's own core"
    # a litellm id named to a CLI, a Claude model to Codex, a GPT to Claude Code, a core that does not exist
    for kw in ({"transport": "sdk-session", "judge_model": QWEN},
               {"transport": "mini-session", "judge_model": QWEN, "judge_transport": "codex-session"},
               {"transport": "codex-session", "judge_model": OPUS},
               {"transport": "sdk-session", "judge_model": "gpt-6-astra"},
               {"judge_transport": "api"}):
        with pytest.raises(ValueError):
            NavHarnessAdapter(**kw)


class _Gateway:
    port = 4999

    def anthropic_env(self) -> dict:
        return {"ANTHROPIC_BASE_URL": "http://127.0.0.1:4999", "ANTHROPIC_AUTH_TOKEN": "k"}

    def codex_args(self) -> list[str]:
        return ["-c", 'model_provider = "acfake"']

    def litellm_kwargs(self) -> dict:
        return {"api_base": "http://127.0.0.1:4999/v1", "api_key": "k", "custom_llm_provider": "openai"}

    def describe(self) -> dict:
        return {"backend": "fake", "auth": "none"}

    def stop(self) -> None:
        pass


@pytest.fixture
def seat_env(monkeypatch, tmp_path: Path):
    import core.harnesses.navh.session as S

    import core.harnesses.navharness as A

    monkeypatch.setattr(S, "ensure_relocated", lambda *a, **k: {"dir": str(tmp_path)})
    served: list = []      # a local model's serving check, recorded instead of run
    monkeypatch.setattr(A, "ensure_serving", lambda model_id, extra=None: served.append(model_id) or None)
    monkeypatch.served = served
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.setenv(key, "set-by-test")
        monkeypatch.delenv(key)
    yield monkeypatch


def test_both_judge_swaps_are_routed_to_the_core_that_runs_the_judge(seat_env) -> None:
    from core.api.providers.local import OLLAMA_URL
    from core.episode import ModelSpec
    from core.harnesses.navharness import NavHarnessAdapter

    # Qwen3.8-27B navigating on mini, Opus 5 judging on Claude Code (its subscription)
    seat_env.setenv("ANTHROPIC_API_KEY", "sk-would-bill-the-api")
    a = NavHarnessAdapter(transport="mini-session", judge_model=OPUS, judge_transport="sdk-session")
    a.prepare(ModelSpec(QWEN))
    assert a._endpoint == {"mini": {"api_base": OLLAMA_URL}} and a._judge_endpoint == {}
    assert "ANTHROPIC_API_KEY" not in os.environ, "a Claude Code judge runs on the subscription"
    assert (a.inherent["judge_model"], a.inherent["judge_transport"]) == (OPUS, "sdk-session")
    assert seat_env.served == [QWEN], "a local model on mini is checked for its pinned context and sampling"
    # Opus 5 navigating on Claude Code, Qwen3.8-27B judging on mini
    b = NavHarnessAdapter(transport="sdk-session", judge_model=QWEN, judge_transport="mini-session")
    b.prepare(ModelSpec(OPUS))
    assert b._endpoint == {} and b._judge_endpoint == {"mini": {"api_base": OLLAMA_URL}}
    # the seat's own model moved to a core that cannot run it is refused before anything starts
    with pytest.raises(ValueError):
        NavHarnessAdapter(transport="mini-session", judge_transport="sdk-session").prepare(ModelSpec(QWEN))
    # a Claude model judging on mini beside a Claude Code seat keeps its key, which the seat's
    # subscription takes out of the environment
    seat_env.setenv("ANTHROPIC_API_KEY", "sk-judge-key")
    d = NavHarnessAdapter(transport="sdk-session", judge_transport="mini-session")
    d.prepare(ModelSpec(OPUS))
    assert d._judge_endpoint == {"mini": {"api_key": "sk-judge-key"}} and "ANTHROPIC_API_KEY" not in os.environ
    # the default: one model fills both checks, on the seat's own core
    c = NavHarnessAdapter(transport="mini-session")
    c.prepare(ModelSpec("openai/qwen-max", extra={"api_base": "https://dashscope-intl.aliyuncs.com/v1"}))
    assert c._judge_endpoint == c._endpoint and c._endpoint["mini"]["api_base"].startswith("https://dashscope")


def test_the_scripted_endpoint_takes_every_session_on_every_core(seat_env) -> None:
    import core.harnesses.navharness as N
    from core.episode import ModelSpec

    gw = _Gateway()
    seat_env.setattr(N, "gateway_for", lambda spec, wire, tag: gw)
    seat_env.setattr(N, "fake_gateway_for", lambda spec, tag: gw)
    fake = {"api_gateway": "fake", "gateway_task": "goat"}
    a = N.NavHarnessAdapter(transport="mini-session", judge_model="gpt-6-astra", judge_transport="codex-session")
    a.prepare(ModelSpec(QWEN, extra=fake))
    assert a._endpoint == {"mini": gw.litellm_kwargs()} and a._judge_endpoint == {"codex": gw.codex_args()}
    assert os.environ["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4999", "a Claude Code session would reach it too"
    a.finalize(Path("/tmp"))
    assert "ANTHROPIC_BASE_URL" not in os.environ
    # a cross-API seat's gateway serves its own model only: a different judge on its core is refused
    seat_env.setattr(N, "gateway_for", lambda spec, wire, tag: gw)
    b = N.NavHarnessAdapter(transport="sdk-session", judge_model="claude-sonnet-5")
    with pytest.raises(RuntimeError):
        b.prepare(ModelSpec("gpt-5-alias", extra={"api_gateway": "litellm", "gateway_route": "openai/gpt-5"}))


# ── the env's verb ──────────────────────────────────────────────────────


def test_stop_truth_reads_the_score_a_stop_would_get_without_issuing_it() -> None:
    from types import SimpleNamespace

    from exp_workspace.lhnav_es.mcp.env import SlamEsEnvManager

    class SequenceNavMetrics:
        def __init__(self, inner) -> None:
            self.env, self._sd = inner, 1.0

    base = SimpleNamespace(episode=SimpleNamespace(goal=SimpleNamespace(kind="sequence")), goal_index=2)

    class Outer:
        def __init__(self) -> None:
            self.env = SequenceNavMetrics(SimpleNamespace(env=None))
            self.unwrapped = base

    m = SlamEsEnvManager()
    m._line, m._split, m._placed, m._env = "goat", "val_unseen", True, Outer()
    m._live = ("goat", "val_unseen")
    m._info = {"distance_to_goal": 0.7}
    assert m.stop_truth() == {"success": 1.0, "distance_to_goal": 0.7, "success_distance": 1.0,
                              "goal_index": 2, "step_count": 0}
    m._info = {"distance_to_goal": 1.0}
    assert m.stop_truth()["success"] == 0.0, "inside the radius, strictly — as the wrapper scores it"
    m._info = {}
    assert m.stop_truth()["success"] is None
    m._episode_done = True
    assert "error" in m.stop_truth()
    m._executor.shutdown(wait=False)


def test_both_judge_swaps_launch_as_runs_of_their_own() -> None:
    """The documented overrides compose, build, and name a run apart from the seat's own-judge run —
    whose episodes they would otherwise replace."""
    from runner import apply_judge_name, build, compose_app

    exp = REPO / "exp_workspace" / "lhnav_es" / "configs" / "std_goat_slam_es_lhnav.yaml"
    own = apply_judge_name(compose_app(exp, ["harness=navhmini", "model=qwen3.8-27b"]))
    swaps = {
        "_judge-claude-opus-5-sdk": ["harness=navhmini", "model=qwen3.8-27b",
                                     "harness.judge_model=claude-opus-5", "harness.judge_transport=sdk-session"],
        "_judge-ollama-chat-qwen3.8-27b-q4-std-mini": ["harness=navh", "model=opus-5", f"harness.judge_model={QWEN}",
                                                "harness.judge_transport=mini-session"],
    }
    for tail, overrides in swaps.items():
        app = apply_judge_name(compose_app(exp, overrides))
        assert app["run"]["name"].endswith(tail) and app["run"]["dir"].endswith(app["run"]["name"])
        adapter, *_ = build(app)
        assert adapter.settings["judge_model"] == app["agent"]["harness"]["judge_model"]
        assert adapter.switches["judge_transport"] == app["agent"]["harness"]["judge_transport"]
    assert own["run"]["name"] == "std_goat_slam_es_navhmini_qwen3.8-27b_default_lhnav"


def test_the_judge_tag_and_a_repeats_suffix_compose() -> None:
    from runner import apply_judge_name, compose_app

    exp = REPO / "exp_workspace" / "lhnav_es" / "configs" / "std_goat_slam_es_lhnav.yaml"
    swap = ["harness=navhmini", "model=qwen3.8-27b", "harness.judge_model=claude-opus-5",
            "harness.judge_transport=sdk-session"]
    app = apply_judge_name(compose_app(exp, [*swap, "suffix=_s2"]))
    assert app["run"]["name"] == "std_goat_slam_es_navhmini_qwen3.8-27b_default_lhnav_s2_judge-claude-opus-5-sdk"
    app = apply_judge_name(compose_app(exp, [*swap, "suffix=_judge-claude-opus-5-sdk"]))
    assert app["run"]["name"].count("_judge-") == 1, "the judge is named once"
