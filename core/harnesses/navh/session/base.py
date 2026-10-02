"""BaseSession — what every kind of session shares: a live client, and the record of what it did.

A session is one conversation with the model, held by the CLI: a
``ClaudeSDKClient`` connected from ``open`` to ``close``. The CLASS says what
kind of conversation it is; the INSTANCE holds the wire it runs on and the
record of what happened. Three groups, in one object:

* the class attributes — ``kind`` (the name a hop is filed under), ``prompt``
  (which prompt in ``prompts.py`` it speaks), ``opens`` (the event that
  starts one), ``reads`` (what goes into its context), ``writes`` (where its
  output lands), ``lifetime`` (``call`` for one question, ``task`` / ``chain``
  for the agent's own);
* the wire — the model name, the options the client is opened with (system
  prompt, the MCP server that is its tools, the tool names, the turns one
  hop may take, and — the agent's session on Claude Code — the CLI's own
  ``Read`` / ``Glob`` / ``Grep`` fenced to the house notes, ``notes.py``),
  where the model's endpoint is (``endpoint``), and the wire itself while it
  is open (``SdkWire``, or the Codex or mini wire the ``transport`` names);
* the ``Run`` — the provider's session id, hops, turns, tokens, timing, how
  it ended; a new one each time the session is opened, frozen when closed.

``hop`` is one query into the conversation and everything the CLI answers
until its result. A one-question session is one hop between ``open`` and
``close``; the agent's is many. Every hop is accounted on the ``Run`` and,
when a ``Trace`` is attached, saved whole.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from .notes import READ_TOOLS, NotesFence
from .wire.blocks import Blocks, ctx_tokens, out_tokens, peak_tokens, subscription_env
from .wire.client import Hop, SdkWire, SessionLoop, result_parts
from .wire.trace import Trace


@dataclass
class Run:
    """What happened once a session was open. ``id`` is the CLI's session id."""
    seq: int                                  # the n-th time this session object was opened
    opened_at: float
    id: str | None = None
    hops: int = 0
    turns: int = 0                            # assistant turns, summed over hops
    prompt_tokens: int = 0                    # the last turn's context, summed over hops
    peak_prompt_tokens: int = 0               # the largest context any turn carried
    completion_tokens: int = 0
    closed_at: float | None = None
    subtype: str = ""                         # how the last hop ended (success | error_max_turns | …)
    error: str = ""                           # the last hop's error, if any
    trace: list[str] = field(default_factory=list)   # trace files written for this run

    def as_record(self) -> dict:
        return {"seq": self.seq, "id": self.id, "hops": self.hops, "turns": self.turns,
                "prompt_tokens": self.prompt_tokens, "peak_prompt_tokens": self.peak_prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "wall_s": round((self.closed_at or time.time()) - self.opened_at, 1),
                "subtype": self.subtype, "error": self.error}


class BaseSession:
    kind: ClassVar[str] = "call"
    prompt: ClassVar[str] = ""                # the prompt's name in prompts.py ("" = none of its own)
    opens: ClassVar[str] = ""                 # the event that opens one
    reads: ClassVar[tuple[str, ...]] = ()     # what its context is made of
    writes: ClassVar[str] = ""                # where its output lands
    lifetime: ClassVar[str] = "call"          # call | task | chain

    #: the wire a session opens; a test hands in a scripted one
    wire_cls: ClassVar[type] = SdkWire
    #: the house notes this session's CLI file tools are opened on (None: no built-in tools)
    notes: NotesFence | None = None

    def __init__(self, model: str, *, loop: SessionLoop, cwd: Path | str, trace: Trace | None = None,
                 rec: Any = None, meter: Callable[[str, dict], None] | None = None,
                 system_prompt: str | None = None, mcp: dict | None = None,
                 tools: tuple[str, ...] | list[str] = (), max_turns: int = 1,
                 effort: str | None = None, transport: str = "sdk-session",
                 endpoint: dict | None = None) -> None:
        self.model = model
        self.loop = loop
        self.cwd = str(cwd)
        self.trace = trace
        self.rec = rec                        # a Recorder: the hop's events, live
        self.meter = meter                    # billing: (model, usage) per hop
        self.system_prompt = system_prompt
        self.mcp = mcp                        # {"env": {...}} — the bridge, or None for a tool-less session
        self.tools = list(tools)              # short names; offered to the CLI as mcp__env__<name>
        self.max_turns = int(max_turns)       # agentic turns one hop may take
        self.effort = effort
        # Which CLI holds this conversation. One line, because everything above
        # the wire — the kinds, the gates, the ladder — does not care which.
        if transport not in ("sdk-session", "codex-session", "mini-session"):
            raise ValueError("session transport must be sdk-session | codex-session | mini-session, "
                             f"got {transport!r}")
        self.transport = transport
        if transport == "codex-session":
            from .wire.codex import CodexWire
            self.wire_cls = CodexWire
        elif transport == "mini-session":
            from .wire.mini import MiniWire
            self.wire_cls = MiniWire
        # What reaches this model's endpoint beyond its id, keyed by wire:
        # ``mini`` — the completion kwargs mini's model layer is built with
        # (api_base / api_key / custom_llm_provider); ``codex`` — the ``-c``
        # overrides that make an endpoint the Codex CLI's model provider.
        # Empty: the CLI's own login, or the provider the id itself names.
        # The SDK wire takes its endpoint from the process environment
        # (ANTHROPIC_BASE_URL).
        self.endpoint = dict(endpoint or {})
        self.wire: SdkWire | None = None
        self.run: Run | None = None
        self.runs = 0                         # how many times this session object was opened
        self.last_usage: dict = {}
        self.last_hop: Hop | None = None

    # ── the class, as a record ───────────────────────────────────────

    @classmethod
    def describe(cls) -> dict:
        return {"kind": cls.kind, "prompt": cls.prompt, "opens": cls.opens,
                "reads": list(cls.reads), "writes": cls.writes, "lifetime": cls.lifetime}

    @property
    def model_name(self) -> str | None:
        return self.model

    # ── the client's options ─────────────────────────────────────────

    def builtin_tools(self) -> list[str]:
        """The CLI's own tools this session is given: the notes reader's three
        when its notes are open (``notes``), none otherwise."""
        return list(READ_TOOLS) if self.notes is not None else []

    def options(self):
        """A fresh ``ClaudeAgentOptions`` for one connection: no settings from
        disk, only our MCP server, our system prompt, and no built-in tools —
        except, when the session's notes are open, ``Read`` / ``Glob`` /
        ``Grep``, which ``tools`` must list to exist at all and which a
        PreToolUse hook fences to the notes directory (the session's cwd)."""
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

        from .wire.home import ensure_relocated
        ensure_relocated()
        # the bridge is a process of this session's own: the CLI's environment, plus what the
        # orchestrator told it (the env, the state file, the workspace)
        mcp = {name: {**cfg, "env": subscription_env(cfg.get("env"))} for name, cfg in (self.mcp or {}).items()}
        builtin = self.builtin_tools()
        hooks = ({"PreToolUse": [HookMatcher(matcher=None, hooks=[self.notes.gate])]}
                 if self.notes is not None else None)
        return ClaudeAgentOptions(
            system_prompt=self.system_prompt,
            mcp_servers=mcp,
            tools=builtin, setting_sources=[], strict_mcp_config=True,
            allowed_tools=[f"mcp__env__{t}" for t in self.tools] + builtin, hooks=hooks,
            permission_mode="bypassPermissions", max_turns=self.max_turns, model=self.model,
            effort=None if self.effort in (None, "", "default") else self.effort,
            env=subscription_env(), max_buffer_size=32 * 1024 * 1024, cwd=self.cwd,
        )

    # ── lifecycle ────────────────────────────────────────────────────

    def open(self) -> Run:
        if self.wire is not None and self.wire.is_open:
            assert self.run is not None
            return self.run
        self.runs += 1
        self.run = Run(seq=self.runs, opened_at=time.time())
        # Each wire builds its own options: the SDK wants ClaudeAgentOptions,
        # Codex wants a CodexSpec. The wire class is the only thing that
        # knows which, so it is the one asked.
        build = getattr(self.wire_cls, "options_for", None)
        self.wire = self.wire_cls(self.loop, build(self) if build else self.options,
                                  on_message=self._on_message)
        try:
            self.wire.open()
        except BaseException as e:
            self.run.error = f"{type(e).__name__}: {str(e)[:400]}"
            self.run.closed_at = time.time()
            self.wire = None
            raise
        if self.trace is not None:
            # the system prompt travels as an SDK option and the CLI transcript
            # never keeps it: this file is the record of what the session was told
            rel = self.trace.record_open(purpose={"purpose": self.kind, "session": self.run.seq, "hop": 0},
                                         model=self.model, system_prompt=self.system_prompt,
                                         tools=self.tools + self.builtin_tools(), mcp=self.mcp,
                                         max_turns=self.max_turns,
                                         effort=self.effort, cwd=str(self.cwd))
            if rel:
                self.run.trace.append(rel)
        return self.run

    def close(self) -> Run | None:
        run = self.run
        if self.wire is not None:
            try:
                self.wire.close()
            finally:
                self.wire = None
        if run is not None and run.closed_at is None:
            run.closed_at = time.time()
        self.run = None
        return run

    @property
    def is_open(self) -> bool:
        return self.wire is not None and self.wire.is_open

    # ── one query inside the conversation ────────────────────────────

    def hop(self, prompt: str | Blocks, max_turns: int | None = None) -> Hop:
        """One query. ``max_turns`` lowers this hop's turn cap below the
        session's own ``max_turns`` — to what is left of a task's turn budget —
        and never raises it; every wire honours it."""
        limit = self.max_turns if max_turns is None else min(self.max_turns, int(max_turns))
        if limit < 1:
            raise ValueError(f"a hop needs at least one turn, got max_turns={max_turns}")
        if not self.is_open:
            self.open()
        assert self.wire is not None and self.run is not None
        t0 = time.time()
        hop: Hop | None = None
        err: str | None = None
        try:
            hop = self.wire.hop(prompt, max_turns=limit)
        except BaseException as e:
            err = f"{type(e).__name__}: {str(e)[:400]}"
            raise
        finally:
            self._account(hop, err, time.time() - t0, prompt)
        return hop

    # ── the stream, live ─────────────────────────────────────────────

    def _on_message(self, msg: Any) -> None:
        """Every message of a hop as it arrives: the record sees the agent's
        text, its thinking, each tool call and each result the moment they
        happen, not when the hop ends."""
        rec = self.rec
        if rec is None:
            return
        tagged = {} if self.kind == "agent" else {"purpose": self.kind}
        if isinstance(msg, dict):
            # the Codex wire speaks plain events; the SDK's classes are not
            # involved, and an isinstance ladder written for them sees nothing
            kind = msg.pop("kind", "")
            if kind:
                rec.emit(kind, **{**msg, **tagged})
            return
        from claude_agent_sdk import (
            AssistantMessage,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )
        tag = {} if self.kind == "agent" else {"purpose": self.kind}
        if isinstance(msg, AssistantMessage):
            for b in msg.content:
                if isinstance(b, ThinkingBlock):
                    t = getattr(b, "thinking", "") or ""
                    rec.emit("thinking", text=t[:2000], chars=len(t), **tag)
                elif isinstance(b, ToolUseBlock):
                    rec.emit("tool_use", id=b.id, name=b.name, input=b.input, **tag)
                elif isinstance(b, TextBlock) and b.text.strip():
                    rec.emit("assistant_text", text=b.text[:4000], **tag)
        elif isinstance(msg, UserMessage):
            for b in (msg.content if isinstance(msg.content, list) else []):
                if isinstance(b, ToolResultBlock):
                    texts, images = result_parts(b)
                    rec.emit("tool_result", tool_use_id=b.tool_use_id, texts=[t[:1500] for t in texts],
                             images=images, **tag)

    # ── accounting ───────────────────────────────────────────────────

    def _account(self, hop: Hop | None, err: str | None, latency_s: float, prompt: str | Blocks) -> None:
        run = self.run
        assert run is not None
        run.hops += 1
        usage = dict(hop.usage) if hop is not None else {}
        run.turns += int(hop.num_turns) if hop is not None else 0
        run.prompt_tokens += ctx_tokens(usage)
        run.peak_prompt_tokens = max(run.peak_prompt_tokens, peak_tokens(usage))
        run.completion_tokens += out_tokens(usage)
        run.subtype = "error" if err else (hop.subtype if hop is not None else "no_result")
        run.error = err or (hop.error or "" if hop is not None else "")
        if hop is not None and hop.session_id:
            run.id = hop.session_id
        self.last_usage = {"prompt_tokens": ctx_tokens(usage), "completion_tokens": out_tokens(usage)}
        self.last_hop = hop
        if self.meter is not None and usage:
            with contextlib.suppress(Exception):
                self.meter(self.model, usage)
        if self.trace is not None:
            rel = self.trace.record(purpose={"purpose": self.kind, "session": run.seq, "hop": run.hops},
                                    model=self.model, prompt=prompt, tools=self.tools, hop=hop,
                                    error=err, latency_s=latency_s, session_id=run.id)
            if rel:
                run.trace.append(rel)
