"""NotesFence — the house notes, opened to Claude Code's own file tools, and nothing else.

On the Claude Code seat the agent's session reads its notes the way a coding
agent reads a repository: it lists, it searches, it opens the index and
follows a line to the file it names, and it can look at a place's stored
photographs. Those are the CLI's own ``Read``, ``Glob`` and ``Grep``, so the
session is given those three and they are fenced to ONE directory, the
house's notes (the workspace's ``house/``: MEMORY.md, house.md, rooms/,
skills/, journal/ and the photo sets). The session's working directory is
that directory, so a ``Glob`` or ``Grep`` that names no path searches the
notes. Everything else stays shut: the run's records, the dataset, the
harness, the rest of the disk. The task records under ``run/`` are reached
through the bridge's own file tools.

The executor may read its notes but not write them. Only consolidation
writes the house notes; ``Write``, ``Edit``, ``Bash`` and every other
built-in are refused, whatever the CLI happens to offer.

The fence is a PreToolUse hook, default-deny. Two facts about the Agent SDK
decide that shape:

* ``allowed_tools`` permits a name, it does not provide the tool: with
  ``tools=[]`` the session has no ``Read`` at all, so ``tools`` must list
  the three;
* under ``permission_mode="bypassPermissions"`` every tool is approved and a
  ``can_use_tool`` callback is never consulted, so a fence built on it would
  be decorative. A hook is consulted, and its deny is final.

The bridge's ``mcp__env__*`` tools carry no file paths and are what the task
is made of; the hook lets them through untouched.
"""

from __future__ import annotations

from itertools import takewhile
from pathlib import Path
from typing import Any

#: what a reader needs and no more
READ_TOOLS = ("Read", "Glob", "Grep")

#: where a built-in names a path, under the argument names the CLI uses
PATH_ARGS = ("file_path", "path", "notebook_path")

#: arguments that are glob patterns, resolved against the search directory
PATTERN_ARGS = {"Glob": ("pattern",), "Grep": ("glob",)}

_WILD = "*?[{"


def _deny(reason: str) -> dict[str, Any]:
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                   "permissionDecision": "deny",
                                   "permissionDecisionReason": reason}}


def _inside(root: Path, value: Any) -> bool:
    """Is ``value`` a path inside ``root``? A relative path is taken against
    ``root`` (the session's working directory), so ``rooms/kitchen.md`` is
    inside and ``../run`` is not; links are resolved before judging."""
    text = str(value or "").strip()
    if not text:
        return True                       # no path named: the tool searches its cwd, the notes
    try:
        p = Path(text).expanduser()
        p = p if p.is_absolute() else root / p
        return p.resolve().is_relative_to(root)
    except (OSError, ValueError, RuntimeError):
        return False


def _pattern_inside(root: Path, value: Any) -> bool:
    """A glob pattern stays inside ``root``: no ``..`` part, and an absolute
    pattern's literal prefix (the parts before the first wildcard) is inside."""
    text = str(value or "").strip()
    if not text:
        return True
    parts = Path(text).expanduser().parts
    if ".." in parts:
        return False
    if not Path(text).expanduser().is_absolute():
        return True                       # relative to the search directory, judged on its own
    literal = list(takewhile(lambda part: not any(c in part for c in _WILD), parts))
    return _inside(root, Path(*literal)) if literal else False


class NotesFence:
    """One house's notes directory, opened for reading and fenced to itself."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def allows(self, tool: str, args: dict | None) -> tuple[bool, str]:
        """``(allowed, reason)`` for one tool call — the hook's whole decision."""
        if tool.startswith("mcp__"):
            return True, ""
        if tool not in READ_TOOLS:
            return False, (f"{tool} is not available here. Your house notes are read-only "
                           f"({', '.join(READ_TOOLS)}); only consolidation writes them.")
        args = args or {}
        named = [args[k] for k in PATH_ARGS if k in args]
        patterns = [args[k] for k in PATTERN_ARGS.get(tool, ()) if k in args]
        if all(_inside(self.root, v) for v in named) and all(_pattern_inside(self.root, v) for v in patterns):
            return True, ""
        return False, (f"outside the house notes ({self.root}). Read, Glob and Grep reach "
                       "that directory only; nothing else on this disk is readable.")

    async def gate(self, input_data: dict, tool_use_id: Any, context: Any) -> dict:
        """PreToolUse: a read inside the notes and the bridge's own tools pass;
        every other call is denied."""
        ok, reason = self.allows(str(input_data.get("tool_name") or ""), input_data.get("tool_input") or {})
        return {} if ok else _deny(reason)


__all__ = ["READ_TOOLS", "NotesFence"]
