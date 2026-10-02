"""The bridge — the tools, as a process of the session's own.

A session's tools are an MCP server the CLI spawns beside it (``bridge.py``):
the six env tools, ``close_task``, the ladder's ``request_recovery``, and the three file tools over the
workspace — or, for a context-only session (toolset ``act``), the observation and action tools alone.
The orchestrator never calls a tool; it reads what the tools did from the
state file (``state.py``) between hops, and edits that file to answer a
close request, open a new task or put the λ hop into file-only mode.

Calls run CLI → bridge → env. Nothing here calls the orchestrator or a
session: a tool that opened a session would be a tool deciding a task's
outcome, and that decision is the orchestrator's.
"""

from __future__ import annotations

from .state import (
    FILE_TOOL_NAMES,
    RECOVERY_TOOL,
    SPATIAL_TOOLS,
    TOOL_NAMES,
    TOOLSETS,
    TaskState,
    fresh_state,
    tool_names,
)

#: the bridge, by module name — what an arm's own tool surface imports and wraps
BRIDGE_MODULE = __name__ + ".bridge"

__all__ = ["BRIDGE_MODULE", "FILE_TOOL_NAMES", "RECOVERY_TOOL", "SPATIAL_TOOLS",
           "TOOLSETS", "TOOL_NAMES", "TaskState", "fresh_state", "tool_names"]
