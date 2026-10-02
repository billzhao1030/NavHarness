"""NavHarness — a chain whose sessions connect through files.

An embodied harness whose every conversation with the model is a session
Claude Code holds: the agent's own, with the harness's MCP bridge as its
tools; the judge's two checks; the wake-up question; the day-end
consolidation. The orchestrator opens them, reads what their tools did off a
state file between hops, gates a close request, runs the recovery ladder
(``continue < attempt < place < room < floor < house``) and writes the
ledger; the sessions connect through a workspace of files they read and
write themselves.

    orchestrator/ Orchestrator (orchestrator.py) — a chain: seat, one task after another, the seam
                  between two, the task's hops and its gates; beside it what it calls (judge.py,
                  ledger.py, close.py, checkpoint.py) and under recovery/ the recovery ladder
    config.py     HarnessConfig — the switches, grouped by the layer that reads them; flat on the wire
    prompts.py    the words the harness says: the fixed system prompt, the instruction, the judge's brief
    record.py     the paper trail, written from every layer
    session/      one conversation with the model, every kind of it as a class over a live Agent SDK
                  client: the agent's own, precheck, certify, wake-up, consolidate; wire/ is the
                  client and the sessions' event loop, and wire/trace.py saves every hop under its kind
    bridge/       the tools, as the MCP server the CLI spawns beside a session: the env tools,
                  close_task, the ladder's proposal, the file tools; state.py the file it keeps
    memory/       what outlives a task: the workspace (files), the archive policy, the place graph
    env/          the env over HTTP (bench.py, this runner's bench_es.py), the bus behind the Recorder

Calls run downward — orchestrator → session → (the CLI) → bridge → env, and
orchestrator → env for the STOP, the map and the checkpoints — and memory
and record are read from every layer without reading any of them back.

The seat is ``core/harnesses/navharness.py`` (``harness=navh`` / ``navhcodex`` / ``navhmini``).
"""

from __future__ import annotations

from .config import HarnessConfig
from .orchestrator import Orchestrator

HARNESS_NAME = "navh"

__all__ = ["HARNESS_NAME", "HarnessConfig", "Orchestrator"]
