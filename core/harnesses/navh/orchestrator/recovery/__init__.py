"""Recovery — what ``Orchestrator._recover`` reaches for, through this one door.

    driver.py     RecoveryDriver — a recovery carried out: signals, the rung applied, λ carried, the record
      ladder.py     the ladder: the rungs as contracts (SCOPES) and the controller that decides one
      lam.py        λ — what crosses a recovery: six fields, a parser, the handover variants, the bank
      wakeup.py     waking up somewhere, coarse to fine (stage 1 asks a WakeupSession)

The orchestrator talks to the driver; the driver talks to the three below it.
``SCOPES`` (what a rung clears) and from λ ``Lambda``, ``Handover``,
``parse``, ``read_note`` and ``note_words`` are what the orchestrator also
needs by name, and they are re-exported here.
The config group that tunes all of this is ``ladder``.
"""

from __future__ import annotations

from .driver import RecoveryDriver
from .ladder import SCOPES
from .lam import Handover, Lambda, note_words, parse, read_note

__all__ = ["SCOPES", "Handover", "Lambda", "RecoveryDriver", "note_words", "parse", "read_note"]
