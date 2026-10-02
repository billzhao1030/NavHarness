"""What a task is, how one ends, and the one exception the env can raise.

Kept apart from ``bench.py`` (beside it) so that the judge, the sessions and
the bridge can name a ``Goal`` or a ``TaskEnd`` without importing the env.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Goal:
    index: int
    total: int
    text: str
    image_b64: str | None = None
    modality: str = "instruction"     # ivln: instruction; goat: object|description|image
    meta: dict = field(default_factory=dict)


@dataclass
class TaskEnd:
    """What the env said when a task closed."""
    end_reason: str                   # stop | budget | turns | inactive
    score: dict                       # env-truth metrics for THIS task
    chain_finished: bool
    next_goal: Goal | None = None
    wakeup: dict | None = None        # set when the body was put down at the next start, not walked
                                      # there (a deployment day's next episode, a teleported tour transit)


class EnvUnavailable(RuntimeError):
    """The environment stopped answering mid-chain.

    Distinct from a bad run: the data after this point is not the agent's
    performance, so the chain is abandoned WITHOUT a summary and the scheduler
    re-runs it from the ledger and the map sidecar."""
