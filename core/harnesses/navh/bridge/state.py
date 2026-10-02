"""The state file — what the bridge knows about the running task, and how the orchestrator reads and edits it.

The bridge is a process of its own, spawned by the CLI as the session's MCP
server; the orchestrator never calls its tools. They meet in ONE JSON file:
the bridge rewrites it after every tool call (steps, frames, markers, the
body's readings, a close request, a recovery proposal, the files written);
the orchestrator reads it between hops and edits it between hops (a new task,
a request it has answered, what the session may write, the λ hop's
file-only mode). Hops are sequential, so there is never a concurrent writer,
and the bridge re-reads the file at the start of every tool call, so an edit
made between hops is in force on the next call.

``fresh_state`` is the shape; ``TaskState`` is the orchestrator's side of it.
The tool names are here too: they are the bridge's surface, and the record
and the seat describe that surface without importing the server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: the env tools every task is given (the spatial ones are withheld under instruments=0)
TOOL_NAMES = ("observe", "step", "get_goal", "get_map", "mark", "preview_path", "close_task")
#: the file tools, over the workspace
FILE_TOOL_NAMES = ("list_files", "read_file", "write_file")
#: the ladder's proposal, offered when recovery is on
RECOVERY_TOOL = "request_recovery"
#: what instruments=0 takes away: the agent can then look, walk, read the goal and close
SPATIAL_TOOLS = ("get_map", "mark", "preview_path")
#: the toolsets a bridge can register: all (the agent's under NavHarness), act (the
#: observation and action tools alone — a context-only session's), files (consolidation's)
TOOLSETS = ("all", "act", "files")


def tool_names(*, recovery: bool = False, instruments: int = 2,
               toolset: str = "all") -> list[str]:
    """The names a session is opened with, in the order the bridge registers them."""
    if toolset not in TOOLSETS:
        raise ValueError(f"toolset must be one of {' | '.join(TOOLSETS)}, got {toolset!r}")
    if toolset == "files":
        return list(FILE_TOOL_NAMES)
    if toolset == "act":
        return [n for n in TOOL_NAMES if n not in SPATIAL_TOOLS]
    names = [n for n in TOOL_NAMES if instruments != 0 or n not in SPATIAL_TOOLS]
    if recovery:
        names.append(RECOVERY_TOOL)
    return names + list(FILE_TOOL_NAMES)


def fresh_state(k: int, *, writable: list[str] | None = None) -> dict[str, Any]:
    return {
        "task": int(k),
        "steps_used": 0,
        "frames": [],                 # {id, path, step, kind} — the judge's evidence, in order
        "stop_frames": [],            # the look-around ids taken on a close request (or by the harness before
                                      #   a STOP it issues on a task that ended without one)
        "marks": [],                  # markers dropped this task
        "marker_photos": {},          # label -> banked views {id, path, step, heading, pose} — the orchestrator accumulates
        "body": {},                   # explored_m2 / floor / current_floor / slam / reanchors / floor_mismatch
        "calls": {},                  # tool -> count
        "close_request": None,        # {status, evidence} — pending until the orchestrator answers it
        "close_requests": 0,
        "closed": False,              # the orchestrator issued the STOP
        "budget_exhausted": False,    # the task's step budget is spent; the orchestrator closes it
        "task_over": False,           # the env ended it (budget) or the STOP went through
        "end_payload": None,
        "end_reason": "",
        "recovery_request": None,     # {scope, reason, confidence, steps_used}
        "env_dead": None,
        "writable": list(writable or ()),   # what write_file may write, relative to the workspace
        "written": [],                # what it wrote, in order
        "only_files": False,          # the λ hop: env tools refuse, file tools answer
    }


class TaskState:
    """The orchestrator's handle on the file the bridge keeps."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def read(self) -> dict[str, Any]:
        try:
            got = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            return {}
        return got if isinstance(got, dict) else {}

    def write(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, default=str), encoding="utf-8")
        tmp.replace(self.path)

    def reset(self, k: int, *, writable: list[str] | None = None) -> dict[str, Any]:
        """A new task: everything the bridge counts starts from zero."""
        st = fresh_state(k, writable=writable)
        self.write(st)
        return st

    def update(self, **fields: Any) -> dict[str, Any]:
        """Edit between hops: read, change these fields, write back."""
        st = self.read()
        st.update(fields)
        self.write(st)
        return st

    def allow(self, *paths: str) -> dict[str, Any]:
        st = self.read()
        have = list(st.get("writable") or [])
        for p in paths:
            if p not in have:
                have.append(p)
        return self.update(writable=have)


__all__ = ["FILE_TOOL_NAMES", "RECOVERY_TOOL", "SPATIAL_TOOLS", "TOOLSETS", "TOOL_NAMES",
           "TaskState", "fresh_state", "tool_names"]
