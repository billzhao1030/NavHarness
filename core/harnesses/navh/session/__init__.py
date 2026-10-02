"""One conversation with the model — every kind of it, as a class.

A chain talks to the model through sessions, and each kind is a class here,
all of them ``BaseSession``: the class says what the session IS (``kind``,
the prompt it speaks, when it opens, what it reads, what it writes back, how
long it lives), an instance holds the wire — a live Agent SDK client, held
by Claude Code — and the ``Run``, what happened once it was open.

    agent.py        AgentSession — the agent's session over the bridge and the workspace
    judge.py        JudgeSession — PrecheckSession (the gate before the STOP) · CertifySession (after)
    wakeup.py       WakeupSession — which room is this, from four views and banked photos
    consolidate.py  ConsolidateSession — the house notes (index, overview, rooms, skills), rewritten at run end from the journal
    base.py         BaseSession · Run — what every kind is: open / hop / close, the accounting, the trace
    wire/           what base.py stands on: client.py (SessionLoop · SdkWire · Hop), blocks.py (content
                    blocks, usage, the CLI's environment), trace.py (every hop saved whole), home.py
                    (the CLI's config home)

λ is not a session of its own: it is the agent session's last hop before a
recovery that voids its context. Sessions connect through the workspace's
files, never through prompt text. Depth is calling depth: the orchestrator
opens the kinds; the kinds are BaseSession; BaseSession stands on wire/. What
the orchestrator needs by name — the loop and the trace it shares between
sessions, the block shape of a prompt, a hop — it takes from here.
"""

from __future__ import annotations

from .base import BaseSession, Run
from .wire import (
           Blocks,
           Hop,
           SessionLoop,
           Trace,
           ensure_relocated,
           image_block,
           prompt_text,
           text_block,
)

__all__ = ["BaseSession", "Blocks", "Hop", "Run", "SessionLoop", "Trace", "ensure_relocated", "image_block",
           "prompt_text", "text_block"]
