"""AgentSession — the agent's own session: the one that acts.

Claude Code holds the conversation. Its tools are the bridge (the env tools,
``close_task``, the ladder's proposal, the file tools over the workspace),
its system prompt is fixed (``prompts.SYSTEM``), and each task arrives as ONE
instruction naming the goal and the files; nothing about earlier tasks is
pasted into the conversation — the session reads its files itself. A hop is
one instruction (or one answer from the orchestrator: a refusal, a
recovery, "continue") and up to ``chunk_turns`` of the CLI's own tool loop.

The lifetime is the orchestrator's to decide — one task under
``compact=boundary``, the chain otherwise — by when it calls ``open`` and
``close``; a rung that voids the context closes and reopens it, and the new
session's bridge picks the task up from the state file. λ is this session's
last hop before such a recovery: asked to write ``lambda.md``, with the env
tools refusing for that hop.

Under a context-only session policy (``session_mode`` independent | single)
the same class is opened with ``prompts.SESSION_SYSTEM`` and the bridge's
``act`` toolset — the observation and action tools, no files — and each task
arrives as ``prompts.SESSION_INSTRUCTION``, the goal alone.

Otherwise, on Claude Code (``sdk-session``) the session also reads its house
notes with the CLI's own ``Read`` / ``Glob`` / ``Grep``, fenced to the notes
directory, which is then its working directory (``notes.py``); the system
prompt gains one line saying where that directory is. Codex and mini read the
notes through the bridge's ``read_file`` alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from .. import prompts as P
from .base import BaseSession
from .notes import NotesFence


class AgentSession(BaseSession):
    kind: ClassVar[str] = "agent"
    prompt: ClassVar[str] = "SYSTEM"
    opens: ClassVar[str] = "task_open (kept across tasks unless compact=boundary); a recovery reopens"
    reads: ClassVar[tuple[str, ...]] = ("run/ledger.md", "run/tasks/<k-1>/handover.md", "house/MEMORY.md",
                                        "house/house.md", "whatever else it read_file's (or Read / Glob / Grep's "
                                        "under house/, on Claude Code)")
    writes: ClassVar[str] = "run/tasks/<k>/handover.md before close_task; run/tasks/<k>/lambda.md before a recovery"
    lifetime: ClassVar[str] = "task"

    def __init__(self, model: str, *, notes: Path | str | None = None, **kw: Any) -> None:
        super().__init__(model, **kw)
        if notes is not None and self.transport == "sdk-session":
            self.notes = NotesFence(notes)
            self.cwd = str(self.notes.root)
            self.system_prompt = (self.system_prompt or "") + P.NOTES_TOOLS.format(path=self.notes.root)
