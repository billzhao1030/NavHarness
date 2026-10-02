"""ConsolidateSession — the house notes, rewritten at run end from the run's records.

Opens when the chain ends: a session of its own whose bridge offers the
file tools only (``NAVH_TOOLSET=files``), told to read the run's ledger, every
task handover and recovery note, the notes the house already has and today's
journal (``house/journal/<day>.md``, written by the harness), and pointed at
the places' photo sets (``house/rooms/<room>/photos.json``); then to rewrite
the index, the house overview, the room notes and the skills
(``memory/longterm.py``). One hop; the CLI runs the reading and the writing
inside it. It is the only writer of those four: what was written is read off
the bridge's state file afterwards, never assumed from the answer, and the
harness adds nothing to them.
"""

from __future__ import annotations

from typing import Any, ClassVar

from .. import prompts as P
from .base import BaseSession


class ConsolidateSession(BaseSession):
    kind: ClassVar[str] = "consolidate"
    prompt: ClassVar[str] = "CONSOLIDATE"
    opens: ClassVar[str] = "run_end"
    lifetime: ClassVar[str] = "call"
    reads: ClassVar[tuple[str, ...]] = ("run/ledger.md", "run/tasks/*/handover.md", "run/tasks/*/lambda*.md",
                                        "house/MEMORY.md", "house/house.md", "house/rooms/*.md",
                                        "house/skills/*.md", "house/journal/<day>.md",
                                        "house/rooms/*/photos.json")
    writes: ClassVar[str] = "house/MEMORY.md, house/house.md, house/rooms/*.md, house/skills/*.md"

    def __init__(self, model: str, *, max_turns: int = 24, **kw: Any) -> None:
        super().__init__(model, max_turns=max_turns, **kw)

    def consolidate(self, ws: Any, state: Any, notes: Any) -> dict:
        """``ws`` the Workspace; ``state`` the TaskState of this session's own
        bridge, reset by the caller with the note patterns writable; ``notes``
        the house's HouseNotes. A session that cannot be opened costs this
        run's rewrite, not the run: no note is written (this session is their
        only writer), the error is reported, and the chain's journal and
        summary are written as always."""
        self.system_prompt = P.CONSOLIDATE_SYSTEM
        error = ""
        photos = [ws.rel(notes.dir / "rooms" / room / "photos.json") for room in notes.photo_index()]
        try:
            self.open()
            hop = self.hop(P.CONSOLIDATE.format(journal=ws.rel(notes.journal_today), ledger=ws.rel(ws.ledger),
                                                tasks=ws.rel(ws.tasks),
                                                photos=", ".join(photos) or "none yet"))
            error = hop.error or ""
            turns = hop.num_turns
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            turns = 0
        finally:
            run = self.close()
        written = list((state.read() or {}).get("written") or [])
        return {"model_written": any(w.startswith("house/") for w in written), "files": written,
                "turns": turns, "error": error, "day": notes.day,
                "prompt_tokens": run.prompt_tokens if run else 0,
                "completion_tokens": run.completion_tokens if run else 0}
