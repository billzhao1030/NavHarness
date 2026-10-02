"""The mini wire — mini-swe-agent as the reasoning core, over the same bridge.

The third CLI is not a CLI. Claude Code and Codex each hold a conversation of
their own and run the tool loop inside it; mini-swe-agent's loop is a Python
loop in this process. That is the one honest difference, and it is written here
rather than hidden: on this wire the HARNESS drives the model ↔ tool loop.
Everything else is the same — the same MCP bridge, the same workspace, the same
five session kinds, the same three gates, the same ladder.

Why it fits the wire interface anyway: a session is a conversation, and mini
keeps one (`agent.messages`). A hop is "advance that conversation by up to
`max_turns` model calls and hand control back", which is exactly what the other
two wires do; they merely ask a CLI to do the advancing.

    open()    spawns the bridge as an MCP stdio server (the same process the
              other two spawn), lists its tools as the model's menu, and seats
              the system prompt.
    hop()     appends the instruction and runs `agent.step()` until this hop's
              model-call budget is spent, the conversation exits, the model
              answers without a tool call, or the task is over.
              `DefaultAgent.run` is NOT used: it clears `messages` and loops to
              completion, which would make every hop a new session.
    close()   shuts the bridge down.

A reply with no tool call hands control back, as a CLI's turn ends when the
model stops calling tools: "reply DONE" after a close request ends the hop
there, so the request is answered where it was made. mini's own complaint
("every response MUST include a tool call") is kept for what it is for — a
call that did not parse, or a reply cut off at the output limit.

A SESSION WITHOUT TOOLS (the two completion checks, the place question) is
given no bridge. There is no tool loop to drive, so there is no agent either:
the conversation is a list of messages, and a hop is one completion through
mini's own model layer (`NavToolsModel.answer`) — the same retry, the same
message preparation, the same billing ledger as every tool-loop call.

THE FILE TOOLS COME FREE. The memory is read and written through the bridge's
`list_files` / `read_file` / `write_file`, and this wire takes the model's menu
straight from the bridge's `list_tools`. So a local model reaches the workspace
through the same door and the same `Workspace.resolve` fence as Claude and
Codex — nothing here grants it, and nothing here could widen it.

WHAT THIS WIRE CANNOT MATCH, recorded rather than hidden:

* The loop is ours. A CLI's own compaction, retries and prompt shaping are not
  in play; what the model sees is `agent.messages` and nothing else.
* Billing is a provider key, not a subscription: mini's model layer lives in
  `core/harnesses/mini/` and calls the provider directly. Nothing of that
  layer is imported into this package; this seat reaches a provider through
  mini, not through the harness. Where the provider lives beyond what the id
  names (an `api_base`, the scripted endpoint) arrives as the session's
  `endpoint["mini"]`: the kwargs mini's model is built with.
* mini's own limits (`step_limit`, `cost_limit`, the wall clock) are set wide
  here, because the orchestrator already caps turns per hop and per task; two
  budgets that disagree end a task for a reason the record cannot show. Its
  dollar estimate is not asked to succeed either (`cost_tracking`): the
  harness bills from the token ledger, and no local model has a list price,
  so a required estimate would refuse every call a local core makes.
"""

from __future__ import annotations

import contextlib
import sys
import uuid
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from .blocks import Blocks
from .client import Hop, SessionLoop

#: where mini's own modules live; they are imported by file, the way its
#: adapter does — the package they live in is not importable by name from here
_MINI = str(Path(__file__).resolve().parents[3] / "mini")


@dataclass
class MiniSpec:
    """What one hop needs. The mini counterpart of `ClaudeAgentOptions`."""
    model: str
    cwd: str
    system_prompt: str | None = None
    mcp: dict = field(default_factory=dict)      # {"env": {"command":…, "args":[…], "env": {…}}}
    max_turns: int = 20                          # model calls this hop may make
    effort: str | None = None
    model_kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_session(cls, s: Any) -> "MiniSpec":
        return cls(model=s.model, cwd=str(s.cwd), system_prompt=s.system_prompt,
                   mcp=dict(s.mcp or {}), max_turns=int(s.max_turns), effort=s.effort,
                   model_kwargs=dict((getattr(s, "endpoint", None) or {}).get("mini") or {}))

    @property
    def bridge(self) -> tuple[str, dict]:
        """The bridge script and the environment it is spawned under — the same
        two the other wires hand to their CLI."""
        server = (self.mcp or {}).get("env") or {}
        args = list(server.get("args") or [])
        return (args[0] if args else ""), dict(server.get("env") or {})


def _mini_modules():
    if _MINI not in sys.path:
        sys.path.insert(0, _MINI)


class MiniWire:
    """mini-swe-agent's loop, with the wire interface `BaseSession` drives."""

    def __init__(self, loop: SessionLoop, options: Callable[[], MiniSpec], *,
                 on_message: Callable[[Any], None] | None = None) -> None:
        # `loop` is unused: mini is synchronous and runs on the caller's thread.
        # It is accepted so the three wires are interchangeable.
        self.loop = loop
        self._options = options
        self.on_message = on_message
        self.info: dict = {}
        self.session_id: str | None = None
        self.env: Any = None
        self.agent: Any = None                    # the tool loop; None for a session without tools
        self.model: Any = None                    # mini's model, in both shapes of session
        self.messages: list[dict] = []            # the conversation of a session without tools
        self._seen = 0                            # messages already folded into a hop

    @staticmethod
    def options_for(session):
        return lambda: MiniSpec.from_session(session)

    @property
    def is_open(self) -> bool:
        return self.model is not None

    # ── lifecycle ────────────────────────────────────────────────────

    def open(self) -> dict:
        if self.is_open:
            return self.info
        spec = self._options()
        path, env = spec.bridge
        _mini_modules()
        from model import NavToolsModel            # noqa: E402  — see _MINI

        # The other two wires get a session id from the CLI that holds the
        # conversation. There is no CLI here, and `run.id` is what the record
        # and the trace index name a session by — so one is minted, per open,
        # and it means the same thing: this conversation, not another.
        session_id = f"mini-{uuid.uuid4().hex[:12]}"
        if not path:
            # no bridge: a question and its answer, nothing to loop on
            self.model = NavToolsModel(model_name=spec.model, tools=[], model_kwargs=dict(spec.model_kwargs),
                                       cost_tracking="ignore_errors")
            self.messages = ([{"role": "system", "content": spec.system_prompt}]
                             if spec.system_prompt else [])
            self.session_id = session_id
            self.info = {"model": spec.model, "transport": "mini-session", "session_id": session_id,
                         "tools": []}
            return self.info

        from env import BridgeEnvironment          # noqa: E402
        from nav_agent import NavAgent             # noqa: E402

        self.env = BridgeEnvironment(bridge_path=path, env={**env, "EH_EXECUTOR": "mini-swe"},
                                     server_url=env.get("NAVH_SERVER_URL", ""))
        model = NavToolsModel(model_name=spec.model, tools=self.env.toolset.tool_schemas(),
                              model_kwargs=dict(spec.model_kwargs), cost_tracking="ignore_errors")
        # The limits are the orchestrator's, not mini's: it caps turns per hop
        # and per task, and a second budget here would end a task for a reason
        # the record cannot show. `run` is never called, so `instance_template`
        # is unused — every instruction arrives through `hop`.
        self.agent = NavAgent(model, self.env, system_template="{% raw %}" + (spec.system_prompt or "")
                              + "{% endraw %}", instance_template="{% raw %}{% endraw %}",
                              step_limit=0, cost_limit=0.0, wall_time_limit_seconds=0,
                              event_hook=self._hook)
        self.agent.messages = []
        self.agent.add_messages(model.format_message(
            role="system", content=self.agent._render_template(self.agent.config.system_template)))
        self.model = model
        self._seen = len(self.agent.messages)
        self.session_id = session_id
        self.info = {"model": spec.model, "transport": "mini-session", "session_id": self.session_id,
                     "tools": self.env.toolset.tool_names()}
        return self.info

    def close(self) -> None:
        if self.env is not None:
            with contextlib.suppress(Exception):
                self.env.close()
        self.env = None
        self.agent = None
        self.model = None
        self.messages = []
        self.session_id = None
        self._seen = 0

    # ── one hop ──────────────────────────────────────────────────────

    def hop(self, prompt: str | Blocks, max_turns: int | None = None) -> Hop:
        if not self.is_open:
            raise RuntimeError("session is not open")
        if self.agent is None:
            return self._ask(prompt)
        from minisweagent.exceptions import FormatError, InterruptAgentFlow

        spec = self._options()
        if max_turns is not None:                  # this hop's cap: never above the session's own
            spec = replace(spec, max_turns=min(spec.max_turns, int(max_turns)))
        agent = self.agent
        t0 = time.time()
        hop = Hop()
        start_calls = int(getattr(agent, "n_calls", 0) or 0)
        billed = _billing(agent.model)
        agent.add_messages(_user_message(agent.model, prompt))
        error: str | None = None
        subtype = "success"
        while int(getattr(agent, "n_calls", 0) or 0) - start_calls < spec.max_turns:
            try:
                agent.step()
            except FormatError as e:
                said = _plain_reply(e)
                if said is not None:
                    # the model answered without calling a tool: its turn is
                    # over, as a CLI's is, and control goes back
                    agent.add_messages(said)
                    break
                # mini's own recovery: hand the complaint back and keep going,
                # exactly as its loop does.
                agent.add_messages(*e.messages)
            except InterruptAgentFlow as e:
                # Submitted / LimitsExceeded / TimeExceeded — the conversation is
                # finished for this task; the orchestrator reads the state file
                # and decides what that means.
                agent.add_messages(*e.messages)
                subtype = "exit"
                break
            except Exception as e:                 # noqa: BLE001 — a dead hop, not a dead chain
                error = f"{type(e).__name__}: {str(e)[:300]}"
                subtype = "error"
                break
            last = agent.messages[-1] if agent.messages else {}
            if last.get("role") == "exit":
                subtype = "exit"
                break
            if last.get("role") == "assistant" and not (last.get("extra") or {}).get("actions"):
                break                              # a turn that called nothing ends the hop

        self._fold(agent.messages[self._seen:], hop)
        self._seen = len(agent.messages)
        hop.num_turns = int(getattr(agent, "n_calls", 0) or 0) - start_calls
        hop.usage = _usage(billed, _billing(agent.model))
        hop.latency_s = round(time.time() - t0, 2)
        hop.error = error
        hop.subtype = subtype if (hop.num_turns or error) else "no_result"
        hop.session_id = self.session_id
        return hop

    def _ask(self, prompt: str | Blocks) -> Hop:
        """One hop of a session without tools: the message, one completion,
        the reply kept in the conversation (a second hop — the judge's
        "the verdict alone, please" — is asked in the same one)."""
        t0 = time.time()
        hop = Hop()
        billed = _billing(self.model)
        self.messages.append(_user_message(self.model, prompt))
        try:
            said = self.model.answer(self.messages)
        except Exception as e:                     # noqa: BLE001 — a failed answer, not a dead chain
            hop.error = f"{type(e).__name__}: {str(e)[:300]}"
            hop.subtype = "error"
            said = None
        if said is not None:
            text = _content_text(said.get("content"))
            thinking = str(said.get("reasoning_content") or "")
            self.messages.append({"role": "assistant", "content": text})
            hop.text, hop.thinking = text, thinking
            hop.num_turns, hop.subtype = 1, "success"
            if thinking:
                self._hook("thinking", {"text": thinking[:2000], "chars": len(thinking)})
            if text.strip():
                hop.story.append({"role": "assistant", "text": text})
                self._hook("assistant_text", {"text": text[:4000]})
        hop.usage = _usage(billed, _billing(self.model))
        hop.latency_s = round(time.time() - t0, 2)
        hop.session_id = self.session_id
        return hop

    # ── what happened, in the shape the other wires report ───────────

    def _fold(self, messages: list[dict], hop: Hop) -> None:
        texts: list[str] = []
        for m in messages:
            role = m.get("role")
            if role == "assistant":
                said = _content_text(m.get("content"))
                if said.strip():
                    texts.append(said)
                    hop.story.append({"role": "assistant", "text": said})
                # mini carries the parsed calls in `extra.actions` — tool /
                # args / tool_call_id — not in the provider's `tool_calls`.
                # Reading the wrong one records an empty hop on every turn.
                for act in ((m.get("extra") or {}).get("actions") or []):
                    use = {"id": act.get("tool_call_id"), "name": act.get("tool"),
                           "input": act.get("args")}
                    hop.tool_uses.append(use)
                    hop.story.append({"role": "assistant",
                                      "tool_call": {"name": use["name"], "input": use["input"]}})
            elif role in ("tool", "user") and m.get("tool_call_id"):
                out = _content_text(m.get("content"))
                hop.tool_results.append({"tool_use_id": m.get("tool_call_id"), "texts": [out], "images": 0})
                hop.story.append({"role": "tool", "id": m.get("tool_call_id"),
                                  "texts": [out], "images": 0})
        hop.text = "\n".join(texts)

    def _hook(self, kind: str, payload: dict) -> None:
        """NavAgent mirrors every appended message here in the event
        vocabulary the record already uses; pass it straight through."""
        if self.on_message is not None:
            with contextlib.suppress(Exception):
                self.on_message({"kind": kind, **(payload or {})})


def _user_message(model: Any, prompt: str | Blocks) -> dict:
    if isinstance(prompt, str):
        return model.format_message(role="user", content=prompt)
    return {"role": "user", "content": openai_parts(prompt)}


def openai_parts(blocks: Blocks) -> list[dict]:
    """A prompt's blocks in the shape mini's model layer sends. The harness
    builds Anthropic blocks (``session/wire/blocks.py``); the OpenAI chat
    shape mini speaks carries images only as ``image_url`` parts, and its
    Ollama route drops anything else — so an image left in the Anthropic
    shape reaches the model as no picture at all, and a judge would answer
    from the caption alone."""
    out: list[dict] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "image" and isinstance(b.get("source"), dict):
            src = b["source"]
            out.append({"type": "image_url", "image_url": {
                "url": f"data:{src.get('media_type') or 'image/png'};base64,{src.get('data') or ''}"}})
        else:
            out.append(b)
    return out


def _plain_reply(e: Any) -> dict | None:
    """The model's reply inside mini's "no tool call" complaint, when that is
    what the complaint is about: a finished reply (not one cut off at the
    output limit) that called nothing. ``None`` for every other complaint —
    a call that did not parse is mini's to recover from."""
    msgs = getattr(e, "messages", None) or ()
    first = msgs[0] if msgs and isinstance(msgs[0], dict) else {}
    response = (first.get("extra") or {}).get("response")
    if not isinstance(response, dict):
        return None
    choice = (response.get("choices") or [{}])[0] or {}
    if choice.get("finish_reason") in ("length", "tool_calls"):
        return None
    message = choice.get("message") or {}
    if message.get("tool_calls"):
        return None
    return {"role": "assistant", "content": _content_text(message.get("content")), "extra": {"actions": []}}


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text") or "") for p in content
                         if isinstance(p, dict) and p.get("type") == "text")
    return "" if content is None else str(content)


def _billing(model: Any) -> dict:
    """A snapshot of mini's running token ledger, flattened over models."""
    led = getattr(model, "billing", None)
    if not isinstance(led, dict):
        return {}
    total: dict[str, int] = {}
    for row in led.values():
        if isinstance(row, dict):
            for k, v in row.items():
                with contextlib.suppress(TypeError, ValueError):
                    total[k] = total.get(k, 0) + int(v)
    return total


def _usage(before: dict, after: dict) -> dict:
    """This hop's cost, in the shape `ctx_tokens` / `out_tokens` / the meter read.

    mini's model does NOT expose a per-call usage: it accumulates into
    `model.billing` through `add_litellm_usage`, whose rows are
    input / cache_read / cache_write_5m / output. So a hop's cost is the
    difference across it, mapped back to the Anthropic key names the rest of
    this harness speaks — `add_anthropic_usage` is the meter on every seat, and
    `in_tokens` sums exactly these three.
    """
    d = {k: max(0, after.get(k, 0) - before.get(k, 0)) for k in set(before) | set(after)}
    if not any(d.values()):
        return {}
    return {"input_tokens": d.get("input", 0),
            "cache_read_input_tokens": d.get("cache_read", 0),
            "cache_creation_input_tokens": d.get("cache_write_5m", 0) + d.get("cache_write_1h", 0),
            "output_tokens": d.get("output", 0)}


__all__ = ["MiniWire", "MiniSpec", "openai_parts"]
