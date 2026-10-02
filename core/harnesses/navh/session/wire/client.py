"""The wire — one live Agent SDK client, driven by one task on the sessions' own event loop.

Every session of a chain is a ``ClaudeSDKClient`` that stays connected from
``open`` to ``close``; a ``hop`` is one ``query`` into it and everything the
CLI sends back until the ``ResultMessage``. The client keeps a task group
open for the whole connection and must be used from the task that connected
it, so each wire runs a single driver task that connects, serves hops from a
queue, and disconnects — and the orchestrator, which is synchronous, hands
it commands and waits.

The CLI caps every query at the ``max_turns`` the client was connected with.
A hop may be given a lower cap (what is left of a task's turn budget), which
no option can change mid-connection; the wire counts the hop's turns itself
and interrupts the query once that many are done — only between turns, when
every tool call of the last one has its result, as the CLI's own cap does.

``SessionLoop`` is the one event loop all the wires of a chain live on, on a
thread of its own; the chain's thread never runs a coroutine itself.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .blocks import Blocks


@dataclass
class Hop:
    """What one query into a session produced."""
    text: str = ""                                  # the assistant's text blocks, joined
    thinking: str = ""
    tool_uses: list[dict] = field(default_factory=list)      # {id, name, input}
    tool_results: list[dict] = field(default_factory=list)   # {tool_use_id, texts, images}
    num_turns: int = 0                              # assistant turns inside the hop
    subtype: str = "no_result"                      # success | error_max_turns | … | no_result
    usage: dict = field(default_factory=dict)
    session_id: str | None = None
    error: str | None = None
    latency_s: float = 0.0
    story: list[dict] = field(default_factory=list)  # the hop as text, in order, for the trace

    @property
    def tools_called(self) -> list[str]:
        return [t["name"] for t in self.tool_uses]


class SessionLoop:
    """One event loop on its own thread, shared by every session of a chain."""

    def __init__(self, name: str = "navh-sessions") -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coro) -> concurrent.futures.Future:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def call(self, fn: Callable, *args: Any) -> None:
        self.loop.call_soon_threadsafe(fn, *args)

    def stop(self) -> None:
        if self.loop.is_closed():
            return
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=10)
        if not self.thread.is_alive():
            self.loop.close()


class SdkWire:
    """One connected client. ``open`` connects (and waits for the MCP server
    when there is one), ``hop`` queries and collects, ``close`` disconnects."""

    connect_timeout_s = 120.0

    @staticmethod
    def options_for(session):
        """This wire's options, built by the session that owns it."""
        return session.options

    def __init__(self, loop: SessionLoop, options: Callable[[], Any], *,
                 on_message: Callable[[Any], None] | None = None) -> None:
        self.loop = loop
        self._options = options            # builds a fresh ClaudeAgentOptions per connection
        self.on_message = on_message       # sees every message as it streams, on the loop's thread
        self._cmds: asyncio.Queue | None = None
        self._driver: concurrent.futures.Future | None = None
        self._ready: concurrent.futures.Future | None = None
        self.info: dict = {}
        self.session_id: str | None = None
        self._cap: int | None = None       # the max_turns the client was connected with

    @property
    def is_open(self) -> bool:
        return self._driver is not None and not self._driver.done()

    # ── lifecycle, from the chain's thread ───────────────────────────

    def open(self) -> dict:
        if self.is_open:
            return self.info
        self._ready = concurrent.futures.Future()
        self._driver = self.loop.submit(self._drive())
        self.info = self._ready.result()   # raises what connect raised
        return self.info

    def hop(self, prompt: str | Blocks, max_turns: int | None = None) -> Hop:
        if not self.is_open or self._cmds is None:
            raise RuntimeError("session is not open")
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self.loop.call(self._cmds.put_nowait, ("hop", (prompt, max_turns), fut))
        return fut.result()

    def close(self) -> None:
        if self._driver is None:
            return
        if not self._driver.done() and self._cmds is not None:
            fut: concurrent.futures.Future = concurrent.futures.Future()
            self.loop.call(self._cmds.put_nowait, ("close", None, fut))
            with contextlib.suppress(Exception):
                fut.result(timeout=60)
        with contextlib.suppress(Exception):
            self._driver.result(timeout=60)
        self._driver = None
        self._cmds = None

    # ── the driver, on the loop ──────────────────────────────────────

    async def _drive(self) -> None:
        from claude_agent_sdk import ClaudeSDKClient
        assert self._ready is not None
        self._cmds = asyncio.Queue()
        options = self._options()
        self._cap = getattr(options, "max_turns", None)
        client = ClaudeSDKClient(options=options)
        try:
            await client.connect()
            info = await self._wait_for_mcp(client)
        except BaseException as e:
            self._ready.set_exception(e)
            return
        self._ready.set_result(info)
        try:
            while True:
                cmd, payload, fut = await self._cmds.get()
                if cmd == "close":
                    fut.set_result(True)
                    break
                try:
                    fut.set_result(await self._hop(client, *payload))
                except BaseException as e:
                    fut.set_exception(e)
        finally:
            with contextlib.suppress(Exception):
                await client.disconnect()

    async def _wait_for_mcp(self, client) -> dict:
        opts = client.options
        if not getattr(opts, "mcp_servers", None):
            return {"mcp": None}
        t0 = time.monotonic()
        status = None
        while True:
            try:
                got = await client.get_mcp_status()
            except Exception:
                got = None
            entries = got.get("mcpServers", []) if isinstance(got, dict) else []
            status = next((e.get("status") for e in entries if e.get("name") == "env"), None)
            if status in ("connected", "failed", "needs-auth", "disabled"):
                break
            if time.monotonic() - t0 > self.connect_timeout_s:
                break
            await asyncio.sleep(0.5)
        if status != "connected":
            await client.disconnect()
            raise RuntimeError(f"env bridge not connected: {status}")
        return {"mcp": status}

    async def _hop(self, client, prompt: str | Blocks, max_turns: int | None = None) -> Hop:
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            SystemMessage,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )
        t0 = time.time()
        hop = Hop()
        if isinstance(prompt, str):
            await client.query(prompt)
        else:
            async def stream():
                yield {"type": "user", "session_id": "default", "parent_tool_use_id": None,
                       "message": {"role": "user", "content": list(prompt)}}
            await client.query(stream())
        # the hop's own cap, enforced here only when it is below the connection's
        cap = max_turns if (max_turns is not None and self._cap and max_turns < self._cap) else None
        turns: list[str] = []              # the hop's assistant turns, by message id, in order
        in_turn = False                    # an assistant turn is streaming (no tool result since)
        open_calls: set[str] = set()       # tool calls still waiting for their result
        interrupted = False
        texts: list[str] = []
        thinking: list[str] = []
        result = None
        async for msg in client.receive_response():
            if self.on_message is not None:
                with contextlib.suppress(Exception):
                    self.on_message(msg)
            if isinstance(msg, AssistantMessage):
                # one model turn can arrive as several messages, one per block, under
                # one message id; without an id, a turn is what comes between results
                if getattr(msg, "parent_tool_use_id", None) is None:
                    mid = getattr(msg, "message_id", None)
                    if mid is not None and mid not in turns:
                        turns.append(mid)
                    elif mid is None and not in_turn:
                        turns.append(f"turn-{len(turns)}")
                    in_turn = True
                for b in msg.content:
                    if isinstance(b, ThinkingBlock):
                        thinking.append(getattr(b, "thinking", "") or "")
                    elif isinstance(b, ToolUseBlock):
                        open_calls.add(b.id)
                        hop.tool_uses.append({"id": b.id, "name": b.name, "input": b.input})
                        hop.story.append({"role": "assistant", "tool_call": {"name": b.name, "input": b.input}})
                    elif isinstance(b, TextBlock) and b.text.strip():
                        texts.append(b.text)
                        hop.story.append({"role": "assistant", "text": b.text})
            elif isinstance(msg, UserMessage):
                in_turn = False
                for b in (msg.content if isinstance(msg.content, list) else []):
                    if isinstance(b, ToolResultBlock):
                        open_calls.discard(b.tool_use_id)
                        parts, images = result_parts(b)
                        hop.tool_results.append({"tool_use_id": b.tool_use_id, "texts": parts, "images": images})
                        hop.story.append({"role": "tool", "id": b.tool_use_id, "texts": parts, "images": images})
                if cap is not None and not interrupted and len(turns) >= cap and not open_calls:
                    # the last allowed turn is done, tools and all: stop before the next one
                    try:
                        await client.interrupt()
                        interrupted = True
                    except Exception:
                        cap = None             # the CLI's own cap still ends the hop
            elif isinstance(msg, SystemMessage):
                if getattr(msg, "subtype", None) == "init":
                    data = getattr(msg, "data", {}) or {}
                    self.info["tools"] = data.get("tools")
                    self.info["model"] = data.get("model")
            elif isinstance(msg, ResultMessage):
                result = msg
        hop.text = "\n".join(texts)
        hop.thinking = "\n".join(thinking)
        hop.latency_s = round(time.time() - t0, 2)
        if result is not None:
            hop.session_id = getattr(result, "session_id", None)
            hop.num_turns = int(getattr(result, "num_turns", 0) or 0)
            hop.subtype = str(getattr(result, "subtype", "success") or "success")
            hop.usage = _plain(getattr(result, "usage", None) or {})
            if interrupted:
                # stopped at this hop's cap, as the CLI stops at its own: the
                # interrupted turn never acted, so it is not one of the hop's
                hop.num_turns = min(hop.num_turns or len(turns), cap or 0)
                hop.subtype = "error_max_turns"
            elif getattr(result, "is_error", False) and hop.subtype not in ("success", "error_max_turns"):
                hop.error = f"sdk result {hop.subtype}: {str(getattr(result, 'result', '') or '')[:200]}"
        self.session_id = hop.session_id or self.session_id
        return hop


def result_parts(block) -> tuple[list[str], int]:
    texts: list[str] = []
    images = 0
    c = getattr(block, "content", None)
    for part in (c if isinstance(c, list) else [c]):
        if isinstance(part, dict):
            if part.get("type") == "image":
                images += 1
            elif part.get("text"):
                texts.append(str(part["text"]))
            else:
                texts.append(f"<{part.get('type', 'blob')}>")
        elif isinstance(part, str) and part:
            texts.append(part)
    return texts, images


def _plain(obj: Any) -> Any:
    import json
    return json.loads(json.dumps(obj, default=str))
