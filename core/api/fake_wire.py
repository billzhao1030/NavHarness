"""fake wire endpoint — a scripted "model" behind both wires, for harness checks.

``api=fake`` (the experiment's api/ group) runs a REAL harness (Claude Code, the codex CLI or
mini-swe-agent) against this endpoint instead of a vendor: the adapter, its
loop, the tool schemas it registers, the bridge it spawns, the records it
writes — everything the harness owns — is exercised; only the decisions are
a fixed script. No tokens, no key, no billing; usage is reported as zero.

Three wires on one port, so every harness speaks its native one:

    POST /v1/messages              anthropic-messages (Claude Code; SSE when asked)
    POST /v1/messages/count_tokens an estimate
    POST /v1/responses             openai-responses (codex, as a custom provider; SSE) —
                                   codex 0.153+ code mode: the script is played through its JS exec tool
    POST /v1/chat/completions      openai-chat (mini via litellm)

The endpoint is stateless: the turn index is the number of tool results in
the request so far, the tool names are whatever the request registers
(``mcp__env__observe`` from Claude Code, ``observe`` from mini — matched by
the segment after the last ``__``), and a request that registers none of
the script's tools (a side call) gets a short text reply.

Config rides env vars: AC_FAKE_TASK (the task key — picks the closing
move), AC_FAKE_SCRIPT (JSON ``[[tool, args], …]``; empty = the default
walk), AC_FAKE_KEY (the bearer token the harnesses present). Every call is
one line in the gateway log (outputs/_api_gateway/<tag>_<port>.log);
AC_FAKE_DEBUG=1 in the launching shell also keeps every codex request whole.

Runs as its own subprocess (``FakeWireGateway`` at the bottom, started by
core.api.fake_gateway_for).
"""

# ruff: noqa: I002
# NB: no `from __future__ import annotations` here — FastAPI resolves the
# handler annotations at runtime, and with postponed annotations the locally
# imported Request type stops resolving (the parameter degrades to a query
# field and every call 422s — same as anthropic_wire.py). Python 3.10
# handles the `X | None` forms natively.
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from .anthropic_wire import synthesize_sse
from .openai_wire import CODEX_KEY_ENV, codex_provider_args
from .service import MASTER_KEY, REPO_ROOT, LoopbackService

Script = list[tuple[str, dict[str, Any]]]
Move = tuple[str, dict[str, Any]]


# ── the script ──


_HANDOVER = re.compile(r"run/tasks/(\d+)/handover\.md")


def handover_of(body: dict[str, Any]) -> str:
    """The handover file the request's task names (the highest task index
    mentioned: the required reading names the previous task's), or ""."""
    ks = [int(k) for k in _HANDOVER.findall(json.dumps(body, ensure_ascii=False))]
    return f"run/tasks/{max(ks)}/handover.md" if ks else ""


def default_script(benchmark: str, tools: dict[str, set[str]], handover: str = "") -> Script:
    """A short walk on whatever the bridge registered (tool name → its
    argument names), ending the task the way the tool surface does
    (handover file, then close_task / STOP). ``benchmark`` is the task key."""
    script: Script = []
    if "observe" in tools:
        script.append(("observe", {}))
    if "step" in tools:
        script.append(("step", {"actions": [2, 1, 3]}))
    if "write_file" in tools and handover:
        script.append(("write_file", {"path": handover,
                                      "text": "Scripted walk: turned left, forward, right; goal not checked."}))
    if "close_task" in tools:
        script.append(("close_task", {"status": "complete", "evidence": "scripted walk"}))
    elif "step" in tools:
        script.append(("step", {"actions": [0]}))
    elif "stop" in tools:
        script.append(("stop", {}))
    return script


def parse_script(raw: Any) -> Script | None:
    """``[[tool, args], …]`` as JSON text or a list; None = the default walk."""
    if not raw:
        return None
    if isinstance(raw, str):
        raw = json.loads(raw)
    return [(str(name), dict(args or {})) for name, args in raw]


def short_name(tool: str) -> str:
    """``mcp__env__observe`` / ``env__observe`` / ``observe`` → ``observe``."""
    return tool.rsplit("__", 1)[-1]


def next_move(
    turn: int, registered: dict[str, set[str]], benchmark: str, script: Script | None,
    handover: str = "",
) -> Move | None:
    """The script's ``turn``-th call by SHORT tool name (``registered``: short
    name → argument names); None once the script is done, or when the request
    registers none of the script's tools (a side call)."""
    steps = [
        (t, a) for t, a in (script or default_script(benchmark, registered, handover)) if t in registered
    ]
    return steps[turn] if turn < len(steps) else None


# ── anthropic-messages wire ──


def _anthropic_tools(body: dict[str, Any]) -> tuple[dict[str, str], dict[str, set[str]]]:
    """(short → full name, short → argument names) of the request's tools."""
    full: dict[str, str] = {}
    props: dict[str, set[str]] = {}
    for tool in body.get("tools") or []:
        name = tool.get("name") if isinstance(tool, dict) else None
        if not name:
            continue
        short = short_name(name)
        full.setdefault(short, name)
        props.setdefault(short, set((tool.get("input_schema") or {}).get("properties") or {}))
    return full, props


def _anthropic_turn(body: dict[str, Any]) -> int:
    n = 0
    for msg in body.get("messages") or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            n += sum(1 for b in content if isinstance(b, dict) and b.get("type") == "tool_result")
    return n


def _anthropic_since_prompt(body: dict[str, Any]) -> int:
    """Tool results since the last prompt (a user message that is not tool
    results): the script restarts at every prompt, so a session carried
    across tasks walks it once per task."""
    n = 0
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        results = sum(1 for b in content if isinstance(b, dict) and b.get("type") == "tool_result") \
            if isinstance(content, list) else 0
        n = n + results if results else 0
    return n


def anthropic_reply(body: dict[str, Any], benchmark: str, script: Script | None) -> dict[str, Any]:
    """One COMPLETE anthropic response for the request: the next scripted
    tool_use, or an end_turn text once the script is done."""
    full, props = _anthropic_tools(body)
    turn = _anthropic_turn(body)                       # unique ids across the conversation
    move = next_move(_anthropic_since_prompt(body), props, benchmark, script, handover_of(body))
    content: list[dict[str, Any]]
    if move is None:
        content = [{"type": "text", "text": f"script complete ({turn} calls)"}]
        stop_reason = "end_turn"
    else:
        tool, args = move
        content = [
            {"type": "text", "text": f"[fake] turn {turn}: {tool}"},
            {"type": "tool_use", "id": f"toolu_fake_{turn}", "name": full[tool], "input": args},
        ]
        stop_reason = "tool_use"
    return {
        "id": f"msg_fake_{turn}",
        "type": "message",
        "role": "assistant",
        "model": body.get("model", ""),
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


# ── openai-chat wire ──


def _chat_tools(body: dict[str, Any]) -> tuple[dict[str, str], dict[str, set[str]]]:
    full: dict[str, str] = {}
    props: dict[str, set[str]] = {}
    for tool in body.get("tools") or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        name = fn.get("name") if isinstance(fn, dict) else None
        if not name:
            continue
        short = short_name(name)
        full.setdefault(short, name)
        props.setdefault(short, set((fn.get("parameters") or {}).get("properties") or {}))
    return full, props


def _chat_turn(body: dict[str, Any]) -> int:
    return sum(
        1 for m in body.get("messages") or [] if isinstance(m, dict) and m.get("role") == "tool"
    )


def _chat_since_prompt(body: dict[str, Any]) -> int:
    """Tool results since the last user message (see ``_anthropic_since_prompt``)."""
    n = 0
    for m in body.get("messages") or []:
        if isinstance(m, dict) and m.get("role") == "tool":
            n += 1
        elif isinstance(m, dict) and m.get("role") == "user":
            n = 0
    return n


def chat_reply(body: dict[str, Any], benchmark: str, script: Script | None) -> dict[str, Any]:
    """One COMPLETE chat.completion for the request: the next scripted
    tool_call, or a stop text once the script is done."""
    full, props = _chat_tools(body)
    turn = _chat_turn(body)                            # unique ids across the conversation
    move = next_move(_chat_since_prompt(body), props, benchmark, script, handover_of(body))
    message: dict[str, Any] = {"role": "assistant", "content": None}
    if move is None:
        message["content"] = f"script complete ({turn} calls)"
        finish = "stop"
    else:
        tool, args = move
        message["content"] = f"[fake] turn {turn}: {tool}"
        message["tool_calls"] = [
            {
                "id": f"call_fake_{turn}",
                "type": "function",
                "function": {"name": full[tool], "arguments": json.dumps(args)},
            }
        ]
        finish = "tool_calls"
    return {
        "id": f"chatcmpl-fake-{turn}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", ""),
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def chat_sse(resp: dict[str, Any]) -> list[str]:
    """The completed chat.completion as a valid chunk stream: role, content,
    the tool call, the finish (with usage), ``[DONE]``."""
    choice = resp["choices"][0]
    message = choice["message"]
    base = {
        "id": resp["id"],
        "object": "chat.completion.chunk",
        "created": resp["created"],
        "model": resp["model"],
    }

    def chunk(delta: dict[str, Any], finish: str | None = None, **extra: Any) -> str:
        data = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}
        return f"data: {json.dumps(data)}\n\n"

    out = [chunk({"role": "assistant", "content": ""})]
    if message.get("content"):
        out.append(chunk({"content": message["content"]}))
    for i, call in enumerate(message.get("tool_calls") or []):
        out.append(chunk({"tool_calls": [{"index": i, **call}]}))
    out.append(chunk({}, choice["finish_reason"], usage=resp["usage"]))
    out.append("data: [DONE]\n\n")
    return out


# ── openai-responses wire (codex) ──
#
# codex 0.153+ is "code mode": MCP tools never reach the model as function
# tools. They are deferred — listed in the JS `exec` custom tool's ALL_TOOLS
# and called from JS (`await tools.mcp__env__step({...})`). So on this wire
# the script is played through exec: one discovery call first
# (`ALL_TOOLS` → the bridge's tool names), then one exec call per move, each
# forwarding the MCP result's text and image blocks. A request that still
# carries direct function tools (an older codex) takes the plain path.

DISCOVER_CALL = "call_fake_discover"
DISCOVER_JS = "text(JSON.stringify(ALL_TOOLS.map((t) => t.name)));"


def _walk_tools(tools: Any):
    """Every tool entry, namespaces flattened (codex nests its tools under
    ``{"type": "namespace", "tools": [...]}``)."""
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "namespace":
            yield from _walk_tools(tool.get("tools"))
        else:
            yield tool


def _responses_tool_entries(body: dict[str, Any]) -> list[dict[str, Any]]:
    """The request's tools: the top-level list (plain Responses API) plus the
    ``additional_tools`` input items codex sends instead."""
    entries = list(_walk_tools(body.get("tools")))
    items = body.get("input") if isinstance(body.get("input"), list) else []
    for item in items:
        if isinstance(item, dict) and item.get("type") == "additional_tools":
            entries += list(_walk_tools(item.get("tools")))
    return entries


def _responses_tools(body: dict[str, Any]) -> tuple[dict[str, str], dict[str, set[str]]]:
    """(short → full name, short → argument names) of the DIRECT function tools."""
    full: dict[str, str] = {}
    props: dict[str, set[str]] = {}
    for tool in _responses_tool_entries(body):
        if tool.get("type") != "function" or not tool.get("name"):
            continue
        short = short_name(tool["name"])
        full.setdefault(short, tool["name"])
        props.setdefault(short, set((tool.get("parameters") or {}).get("properties") or {}))
    return full, props


def _exec_present(body: dict[str, Any]) -> bool:
    return any(
        t.get("type") == "custom" and t.get("name") == "exec" for t in _responses_tool_entries(body)
    )


def _responses_items(body: dict[str, Any]) -> list[dict[str, Any]]:
    items = body.get("input")
    return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []


def _responses_turn(body: dict[str, Any]) -> int:
    """Tool results so far — function outputs, and exec outputs other than
    the discovery call's."""
    return sum(
        1
        for i in _responses_items(body)
        if i.get("type") == "function_call_output"
        or (i.get("type") == "custom_tool_call_output" and i.get("call_id") != DISCOVER_CALL)
    )


def _responses_since_prompt(body: dict[str, Any]) -> int:
    """Tool results since the last user message (see ``_anthropic_since_prompt``)."""
    n = 0
    for i in _responses_items(body):
        if i.get("type") == "function_call_output" or (
                i.get("type") == "custom_tool_call_output" and i.get("call_id") != DISCOVER_CALL):
            n += 1
        elif i.get("type", "message") == "message" and i.get("role") == "user":
            n = 0
    return n


def _discovered(body: dict[str, Any]) -> list[str] | None:
    """The bridge's tool names the discovery exec call returned, None before it."""
    for item in _responses_items(body):
        if item.get("type") == "custom_tool_call_output" and item.get("call_id") == DISCOVER_CALL:
            output = item.get("output")
            if isinstance(output, list):  # codex: input_text parts (a status line, then ours)
                output = "\n".join(
                    str(part.get("text", "")) for part in output if isinstance(part, dict)
                )
            return re.findall(r'"([A-Za-z0-9_]+)"', str(output))
    return None


def _exec_js(name: str, args: dict[str, Any]) -> str:
    """One move as exec source: call the nested tool, forward every text and
    image block of its MCP result. An ``answer`` refused for its argument
    name is retried with the other spelling (letter | text) — the schema is
    not visible on this wire."""
    lines = [
        "async function call(a) { try { return await tools[" + json.dumps(name) + "](a); } "
        'catch (e) { return { isError: true, content: [{ type: "text", text: String(e) }] }; } }',
        f"let r = await call({json.dumps(args)});",
    ]
    if short_name(name) == "answer":
        lines.append('if (r && r.isError) r = await call({ letter: "A" });')
    lines.append(
        "for (const c of (r && r.content) || []) { "
        'if (c.type === "image") image(c); else if (c.type === "text") text(c.text); '
        "else text(JSON.stringify(c)); }"
    )
    return "\n".join(lines)


def _custom_call(call_id: str, js: str) -> dict[str, Any]:
    return {
        "type": "custom_tool_call",
        "id": f"ctc_{call_id}",
        "call_id": call_id,
        "name": "exec",
        "input": js,
        "status": "completed",
    }


def responses_reply(body: dict[str, Any], benchmark: str, script: Script | None) -> dict[str, Any]:
    """One COMPLETE response object for the request: a message item plus the
    next scripted call (a function_call, or an exec custom_tool_call in code
    mode), or a message alone once the script is done."""
    full, props = _responses_tools(body)
    turn = _responses_turn(body)
    call: dict[str, Any] | None = None
    if next_move(0, props, benchmark, script) is not None:  # direct function tools
        move = next_move(_responses_since_prompt(body), props, benchmark, script, handover_of(body))
        if move is not None:
            tool, args = move
            call = {
                "type": "function_call",
                "id": f"fc_fake_{turn}",
                "call_id": f"call_fake_{turn}",
                "name": full[tool],
                "arguments": json.dumps(args),
                "status": "completed",
            }
            text = f"[fake] turn {turn}: {tool}"
        else:
            text = f"script complete ({turn} calls)"
    elif _exec_present(body):  # code mode
        names = _discovered(body)
        if names is None:
            call = _custom_call(DISCOVER_CALL, DISCOVER_JS)
            text = "[fake] discovering the bridge's tools (ALL_TOOLS)"
        else:
            nested = {short_name(n): n for n in names}
            move = next_move(_responses_since_prompt(body), {k: set() for k in nested}, benchmark, script,
                             handover_of(body))
            if move is not None:
                tool, args = move
                call = _custom_call(f"call_fake_{turn}", _exec_js(nested[tool], args))
                text = f"[fake] turn {turn}: {tool} (via exec)"
            else:
                text = f"script complete ({turn} calls)"
    else:
        text = f"script complete ({turn} calls)"
    output: list[dict[str, Any]] = [
        {
            "type": "message",
            "id": f"msg_fake_{turn}",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
    ]
    if call is not None:
        output.append(call)
    return {
        "id": f"resp_fake_{turn}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": body.get("model", ""),
        "output": output,
        "usage": {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        },
    }


def responses_sse(resp: dict[str, Any]) -> list[str]:
    """The completed response as a valid event stream: created, each output
    item added / (text delta) / done, completed."""

    def event(kind: str, **data: Any) -> str:
        return f"event: {kind}\ndata: {json.dumps({'type': kind, **data})}\n\n"

    started = {**resp, "status": "in_progress", "output": [], "usage": None}
    out = [event("response.created", response=started)]
    for i, item in enumerate(resp["output"]):
        out.append(event("response.output_item.added", output_index=i, item=item))
        if item["type"] == "message":
            for j, part in enumerate(item["content"]):
                out.append(
                    event(
                        "response.output_text.delta",
                        item_id=item["id"],
                        output_index=i,
                        content_index=j,
                        delta=part["text"],
                    )
                )
        out.append(event("response.output_item.done", output_index=i, item=item))
    out.append(event("response.completed", response=resp))
    return out


# ── server side (runs under `python -m core.api.fake_wire`) ──


def _build_app():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    benchmark = os.environ.get("AC_FAKE_TASK", "")
    script = parse_script(os.environ.get("AC_FAKE_SCRIPT", ""))
    key = os.environ.get("AC_FAKE_KEY", MASTER_KEY)
    app = FastAPI()

    def _authed(request: Request) -> bool:
        auth = request.headers.get("authorization", "")
        return key in (request.headers.get("x-api-key"), auth.removeprefix("Bearer ").strip())

    def _log(
        path: str, body: dict[str, Any], turn: int, calls: list[str], tools: list[str]
    ) -> None:
        print(
            f"[fake] {path} model={body.get('model', '')!r} turn={turn} "
            f"tools={tools} stream={bool(body.get('stream'))} -> {calls or 'end'}",
            flush=True,
        )

    n_dumped = 0

    def _dump(body: dict[str, Any]) -> None:
        """Every request on the codex wire, whole, numbered — codex changes
        its wire shape between releases (0.153 moved tools into an
        ``additional_tools`` input item and nests them under exec), and the
        parser has to follow. Only with AC_FAKE_DEBUG=1 (see FakeWireGateway)."""
        nonlocal n_dumped
        path = os.environ.get("AC_FAKE_DUMP")
        if path:
            n_dumped += 1
            target = Path(path).with_suffix(f".req{n_dumped}.json")
            target.write_text(json.dumps(body, indent=1))
            print(f"[fake]   request kept at {target}", flush=True)

    @app.get("/health/liveliness")
    async def liveliness():
        return {"status": "ok", "task": benchmark, "script": script or "default"}

    @app.post("/v1/messages")
    async def messages(request: Request):
        if not _authed(request):
            raise HTTPException(401, "bad fake key")
        body = await request.json()
        reply = anthropic_reply(body, benchmark, script)
        calls = [b["name"] for b in reply["content"] if b.get("type") == "tool_use"]
        _log("/v1/messages", body, _anthropic_turn(body), calls, list(_anthropic_tools(body)[0]))
        if not body.get("stream"):
            return JSONResponse(reply)
        return StreamingResponse(
            (
                f"event: {name}\ndata: {json.dumps(data)}\n\n"
                for name, data in synthesize_sse(reply)
            ),
            media_type="text/event-stream",
        )

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        if not _authed(request):
            raise HTTPException(401, "bad fake key")
        body = await request.json()
        n = sum(len(json.dumps(m)) // 4 for m in body.get("messages") or [])
        return JSONResponse({"input_tokens": int(n)})

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        if not _authed(request):
            raise HTTPException(401, "bad fake key")
        body = await request.json()
        reply = chat_reply(body, benchmark, script)
        calls = [
            c["function"]["name"] for c in reply["choices"][0]["message"].get("tool_calls") or []
        ]
        _log("/v1/chat/completions", body, _chat_turn(body), calls, list(_chat_tools(body)[0]))
        if not body.get("stream"):
            return JSONResponse(reply)
        return StreamingResponse(iter(chat_sse(reply)), media_type="text/event-stream")

    @app.get("/v1/models")
    async def models():
        # codex refreshes its model list from the provider at startup; an
        # empty list keeps it on the model it was told (a 404 is logged as
        # an ERROR in its stderr, nothing more)
        return {"models": []}  # codex's own shape (not the OpenAI list object)

    @app.post("/v1/responses")
    async def responses(request: Request):
        if not _authed(request):
            raise HTTPException(401, "bad fake key")
        body = await request.json()
        reply = responses_reply(body, benchmark, script)
        calls = [
            o["name"] if o["type"] == "function_call" else f"exec:{o['call_id']}"
            for o in reply["output"]
            if o["type"] in ("function_call", "custom_tool_call")
        ]
        tools = list(_responses_tools(body)[0]) or _discovered(body) or []
        _log("/v1/responses", body, _responses_turn(body), calls, tools)
        _dump(body)
        if not body.get("stream"):
            return JSONResponse(reply)
        return StreamingResponse(iter(responses_sse(reply)), media_type="text/event-stream")

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def unknown(path: str, request: Request):
        # an unsupported route is a finding, not a silent 404 — see the log
        print(f"[fake] unsupported {request.method} /{path}", flush=True)
        raise HTTPException(404, f"the fake endpoint has no /{path}")

    return app


def main() -> None:
    import uvicorn

    port = int(sys.argv[sys.argv.index("--port") + 1])
    uvicorn.run(_build_app(), host="127.0.0.1", port=port, log_level="warning")


# ── client side (constructed by core.api.fake_gateway_for) ──


class FakeWireGateway(LoopbackService):
    PORT_POOL = tuple(range(4120, 4130))
    START_TIMEOUT = 60.0
    POLL_INTERVAL = 0.5

    def __init__(self, benchmark: str, script: Any = None, tag: str = "fakeapi") -> None:
        super().__init__(tag)
        self.benchmark = benchmark
        self.script = parse_script(script)  # a malformed script dies here, before the run

    def command(self) -> list[str]:
        os.environ[CODEX_KEY_ENV] = MASTER_KEY  # codex resolves its provider key from env
        return [sys.executable, "-m", "core.api.fake_wire", "--port", str(self.port)]

    def spawn_env(self) -> dict[str, str]:
        return {
            **os.environ,
            "AC_FAKE_TASK": self.benchmark,
            "AC_FAKE_SCRIPT": json.dumps(self.script) if self.script else "",
            "AC_FAKE_KEY": MASTER_KEY,
            # AC_FAKE_DEBUG=1 in the launching shell: every codex request kept
            # beside the log, numbered (they carry the whole transcript, images
            # included — off by default)
            **(
                {"AC_FAKE_DUMP": str(self.log_path.with_suffix(".json"))}
                if os.environ.get("AC_FAKE_DEBUG")
                else {}
            ),
        }

    def spawn_cwd(self) -> str:
        return str(REPO_ROOT)

    # ── harness-facing projections ──

    def anthropic_env(self) -> dict[str, str]:
        """Env for the SDK harness: Claude Code speaks anthropic-messages to us."""
        return {"ANTHROPIC_BASE_URL": self.base_url, "ANTHROPIC_AUTH_TOKEN": MASTER_KEY}

    def codex_args(self, provider_id: str = "acfake") -> list[str]:
        """-c overrides for codex exec: a custom openai-responses provider."""
        return codex_provider_args(self.base_url, provider_id, "NavHarness fake endpoint")

    def litellm_kwargs(self) -> dict[str, Any]:
        """Completion kwargs for the mini harness: openai-chat here whatever
        the model id says — the provider is forced, not read off the name."""
        return {
            "api_base": f"{self.base_url}/v1",
            "api_key": MASTER_KEY,
            "custom_llm_provider": "openai",
        }

    def describe(self) -> dict:
        """Recorded into harness_inherent — the run must say it was scripted."""
        return {
            "backend": "fake-wire (scripted moves, no model)",
            "auth": "none (fake endpoint, no billing)",
            "port": self.port,
            "task": self.benchmark,
            "script": self.script or "default",
        }


if __name__ == "__main__":
    main()
