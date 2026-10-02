"""BridgeToolSet — mini-swe-agent driving the arm's own MCP bridge.

The other two harnesses (Claude SDK, Codex CLI) spawn ``exp_workspace/<arm>/
mcp/bridge.py`` as an MCP stdio server and call its tools. This toolset
spawns the same bridge, under the same environment (EpisodeContext.bridge_env),
lists its tools as the model's menu and forwards every call over MCP — so
the surface mini sees is byte-for-byte the one the SDK and codex see, shaped
by the arm, never by the harness. A harness carries no tools of its own.

mini's loop is synchronous; the MCP client is asyncio. One background thread
owns the event loop, and ONE task on it owns the client session for the whole
episode: anyio's cancel scopes must be entered and exited by the same task,
so opening and closing cannot be two separate coroutines thrown at the loop
(anyio would raise "Attempted to exit cancel scope in a different task" at
close).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


@dataclass
class ToolResult:
    """One tool call's outcome.

    ``content`` is what the model sees (OpenAI content parts, images included);
    ``info`` is the machine-readable side for the environment (episode_over,
    end_reason, step counters) and the curated trajectory log.
    """

    content: list[dict[str, Any]] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)


class BridgeToolSet:
    """The arm's bridge as mini's toolset: {name, description, input_schema}
    schemas from ``list_tools``, ``execute`` = ``call_tool``. Exposes the
    counters the adapter reports (calls_by_tool / steps_taken / end_reason)."""

    def __init__(
        self,
        bridge_path: Path,
        env: dict[str, str],
        *,
        python: str = sys.executable,
        call_timeout_s: float = 600.0,
    ) -> None:
        self.bridge_path = Path(bridge_path)
        self.env = dict(env)
        self.python = python
        self.call_timeout_s = call_timeout_s
        self.calls_by_tool: dict[str, int] = {}
        self.steps_taken = 0
        self.end_reason: str | None = None
        self.episode_over = False
        self._schemas: list[dict[str, Any]] = []
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="mini-bridge", daemon=True)
        self._thread.start()
        self._session = None
        self._stop: asyncio.Event | None = None  # set on the loop by the owner task
        self._opened: concurrent.futures.Future = concurrent.futures.Future()
        self._owner = asyncio.run_coroutine_threadsafe(self._own(), self._loop)
        self._opened.result(timeout=self.call_timeout_s)  # the bridge's failure surfaces here

    # ── lifecycle ──

    def _run(self, coro):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=self.call_timeout_s)

    async def _own(self) -> None:
        """The one task holding the client contexts: open, list the tools,
        report ready, wait for close(), and exit the contexts itself."""
        from contextlib import AsyncExitStack

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._stop = asyncio.Event()
        params = StdioServerParameters(
            command=self.python,
            args=[str(self.bridge_path)],
            env={**os.environ, **self.env},
            cwd=str(self.bridge_path.parent),
        )
        try:
            async with AsyncExitStack() as stack:
                read, write = await stack.enter_async_context(stdio_client(params))
                self._session = await stack.enter_async_context(ClientSession(read, write))
                await self._session.initialize()
                listed = await self._session.list_tools()
                self._schemas = [
                    {"name": t.name, "description": t.description or "", "input_schema": t.inputSchema}
                    for t in listed.tools
                ]
                for s in self._schemas:
                    self.calls_by_tool.setdefault(s["name"], 0)
                self._opened.set_result(None)
                await self._stop.wait()
        except BaseException as exc:
            if not self._opened.done():
                self._opened.set_exception(exc)
            raise
        finally:
            self._session = None

    def close(self) -> None:
        """End the bridge process (one bridge = one episode)."""
        if self._loop.is_closed():
            return
        try:
            if self._stop is not None and not self._owner.done():
                self._loop.call_soon_threadsafe(self._stop.set)
            self._owner.result(timeout=self.call_timeout_s)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            self._loop.close()

    # ── the toolset contract ──

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [dict(s) for s in self._schemas]

    def tool_names(self) -> list[str]:
        return [s["name"] for s in self._schemas]

    def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        if name not in self.calls_by_tool:
            err = {"error": f"unknown tool '{name}'; available: {self.tool_names()}"}
            return ToolResult(content=[text_part(json.dumps(err))], info=err)
        self.calls_by_tool[name] += 1
        result = self._run(self._session.call_tool(name, args))
        return self._to_result(result)

    def _to_result(self, result: Any) -> ToolResult:
        """MCP content -> OpenAI content parts; the bridge's JSON status text
        (episode_over / end_reason / steps_taken_total …) -> ``info``."""
        content: list[dict[str, Any]] = []
        info: dict[str, Any] = {}
        for item in result.content or []:
            kind = getattr(item, "type", None)
            if kind == "text":
                text = item.text
                content.append(text_part(text))
                try:
                    parsed = json.loads(text)
                except (TypeError, ValueError):
                    continue
                if isinstance(parsed, dict):
                    info.update(parsed)
            elif kind == "image":
                mime = getattr(item, "mimeType", None) or "image/png"
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{item.data}"}}
                )
            else:  # resources and anything newer: keep a text trace
                content.append(text_part(str(item)))
        if getattr(result, "isError", False) and "error" not in info:
            info["error"] = "".join(p.get("text", "") for p in content) or "tool error"
        if "steps_taken_total" in info:
            self.steps_taken = int(info["steps_taken_total"])
        if info.get("episode_over"):
            self.episode_over = True
            self.end_reason = info.get("end_reason") or "episode_over"
        return ToolResult(content=content, info=info)
