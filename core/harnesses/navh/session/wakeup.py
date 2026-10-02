"""WakeupSession — which room is this, from four views and the banked photos.

Opened by a recovery rung above ``attempt`` (``place``, ``room``, ``floor``,
``house`` — the rungs that cost steps) when the place graph holds photographs
to compare against: stage 1 of the coarse-to-fine funnel. A tool-less session
of its own; the conversation is one hop — what the robot sees now, then each
candidate place's photographs, then the question — and one JSON answer. The
candidates, the staging and the record are ``orchestrator/recovery/wakeup.py``'s.

A relocation between tasks opens none: the next task's own session is handed
the four look-around views with its instruction and works out where it is.
"""

from __future__ import annotations

from typing import Any, ClassVar

from .base import BaseSession
from .wire import Blocks


class WakeupSession(BaseSession):
    kind: ClassVar[str] = "wakeup"
    prompt: ClassVar[str] = "PLACE_QUESTION"
    opens: ClassVar[str] = "a recovery rung above attempt, when banked place photographs exist"
    reads: ClassVar[tuple[str, ...]] = ("four look-around views", "candidate places' banked photos")
    writes: ClassVar[str] = "Funnel.place_guess — the brief the next attempt opens with"
    lifetime: ClassVar[str] = "call"

    def __init__(self, model: str, **kw: Any) -> None:
        super().__init__(model, max_turns=1, **kw)

    def ask(self, content: Blocks) -> str:
        """One hop over one message; raises what the wire raises, and raises
        for a hop the wire reports as failed with nothing said — so the
        funnel records why the question went unanswered."""
        try:
            self.open()
            hop = self.hop(content)
        finally:
            self.close()
        if hop.error and not hop.text:
            raise RuntimeError(hop.error)
        return hop.text
