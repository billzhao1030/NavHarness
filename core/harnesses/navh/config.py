"""The harness's switches, grouped by the organ that reads them.

Every seat, every yaml row and every record speaks the FLAT names —
``HarnessConfig.from_flat(**row)`` builds the config from them and
``HarnessConfig.flat()`` writes them back — and each group is handed to the
one organ that reads it:

    task      what one chain is (benchmark, budget, turn cap, instruments)
    session   the agent's session: the transport, whether it carries over a task boundary,
              how many turns one hop may take, the session policy
    judge     the two checks at the task boundary (the gate, the certification), and the core that holds them
    ladder    the recovery ladder and λ
    memory    the spatial memory, the place graph, the archive, the house's notes, which
              earlier records a session may read
    replay    resume, checkpoints and single-task replay

The session policy (``session_mode``) is the one switch that sets others.
``harness`` is NavHarness. ``independent`` and ``single`` are the two
context-only baselines: the session has the observation and action tools
and nothing else — no files, no map, no markers — so everything that reads
or writes that memory has no reader and is held off (``CONTEXT_ONLY``):
the certification, the recovery ladder, the place graph, the archive, the
house notes. ``independent`` opens a fresh session for every task;
``single`` keeps one across the chain and leaves its compaction to the
backend (``compact=token``). The pre-stop check answers inside the current
conversation, so it stays the ``precheck`` switch's. What the policy held
is kept on the config (``HarnessConfig.held``) and goes into the record.

The fixed-access memory control (``retrieval=bm25``) also sets one: the
harness, not the session, decides what earlier text the session sees, so the
file tools are scoped to the task's own files exactly as under
``ledger=none``, which it holds. A context-only session reads nothing, so
there ``retrieval`` is held at ``none``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class TaskConfig:
    benchmark: str                      # ivln | goat | es
    chain_id: int
    chain_ids: list = field(default_factory=list)   # GOAT: several episodes of one scene, concatenated
    split: str = "val_unseen"
    step_budget: int = 500              # env steps per task
    stop_views: int = 4                 # forced look-around at the stop position (judge evidence)
    max_turns_per_task: int = 200       # model turns per task (all its hops and attempts)
    instruments: int = 2
    rgb_resolution: str = "512"


@dataclass
class SessionConfig:
    transport: str = "sdk-session"      # sdk-session: Claude Code holds the session, the bridge is its tools
                                        # codex-session: the Codex CLI holds it, over the same bridge
                                        # mini-session: mini-swe-agent is the core; the loop is OURS
    compact: str = "boundary"           # boundary: one session per task | token / never: the session carries
                                        #   over the task boundary and compacts as the CLI compacts
    chunk_turns: int = 25               # agentic turns one hop may take before the orchestrator looks
    session_mode: str = "harness"       # harness: NavHarness | independent: a fresh context-only session per
                                        #   task (observation and action tools only) | single: the same tool
                                        #   surface, one session across tasks, compacted by the backend


@dataclass
class JudgeConfig:
    precheck: str = "model"             # none | model — the pre-stop verification (before the STOP)
    certify: str = "model"              # none | model — the post-stop certification (what the ledger records)
    judge_transport: str = ""           # who holds both checks' sessions: "" = the seat's own transport |
                                        #   sdk-session | codex-session | mini-session (a judge on another core)


@dataclass
class LadderConfig:
    """A long task is usually a lost task. Rather than let a turn cap kill it,
    take down what the attempt ruled out, void the smallest state scope
    consistent with the failure, and give the same task a fresh attempt.

    ``recovery`` is the master switch: False means no recovery of any kind,
    neither requested nor the ``reset_after_turns`` fallback, while the task
    keeps its whole turn and step budget. ``recovery_handover`` says what the
    next attempt is given to read:

        structured   the six-field note the attempt writes before it ends (λ)
        empty        nothing: the attempt writes no note and the next one reads none
        other        another task's structured note, in place of the attempt's own
        summary      an ordinary prose summary the attempt writes, at the mean
                     length of the structured notes in ``recovery_handover_bank``
    """
    recovery: bool = True               # False = no recovery at all: no requested and no fallback restarts
    recovery_ceiling: str = "house"     # highest rung the controller may apply
    recovery_min_turns: int = 12        # an attempt younger than this is young, not stuck
    reset_after_turns: int = 80         # T_f: turns on one attempt before the harness acts unasked (0 = off)
    max_resets_per_task: int = 1        # how many recoveries one task may have
    recovery_escalation: bool = True    # a repeat recovery at or below the last scope goes one rung up
    recovery_handover: str = "structured"   # structured | empty | other | summary — what the next attempt reads
    recovery_handover_bank: str = ""    # structured notes of an earlier run (its directory, a lessons.jsonl or
                                        #   a JSON list): `other` draws from them, `summary` matches their length


@dataclass
class MemoryConfig:
    persist_map: bool = True            # False = episodic: the map is rebuilt at every boundary
    place_graph: bool = True            # keep the place/room/floor/connector graph
    archive_policy: bool = False        # promotion, conflict, retrieval, forgetting
    notes: bool = True                  # the house notes (MEMORY.md, house.md, rooms/, skills/): consolidated at run end
    memory_root: str = ""               # where houses live (the workspace's house/ points there)
    map_state: str = ""                 # file the occupancy map + markers are saved to at every boundary
                                        # and restored from when a run opens in a house already mapped
    day: str = ""                       # the journal's day (ISO date; default today): a deployment's runs are its days
    ledger: str = "text"                # text: earlier tasks' records (ledger, handovers, task files, journal,
                                        #   notes) are required reading and readable | none: a session sees only
                                        #   its own task's files; the records are still written
    retrieval: str = "none"             # none: the session reads its records itself (the reference) | bm25: the
                                        #   fixed-access control — the harness retrieves historical text on a schedule
                                        #   (memory/retrieval.py) and the file tools are scoped as under ledger=none
    retrieval_top_k: int = 5            # bm25: excerpts per retrieval
    retrieval_budget_tokens: int = 2000 # bm25: tokens one retrieval may put into the conversation
    retrieval_chunk_tokens: int = 400   # bm25: records are cut into chunks this long ...
    retrieval_overlap_tokens: int = 50  # ... overlapping by this much
    retrieval_every_turns: int = 10     # bm25: navigation-model turns between scheduled retrievals


@dataclass
class ReplayConfig:
    resume_from: str = ""               # a previous run dir for this chain: replay its ledger, skip its tasks
    checkpoints: bool = False           # freeze pose+map+cursor at every boundary, so any task can be replayed
    replay_from: str = ""               # a run dir holding checkpoints/
    replay_task: int = -1               # which task to seat at and re-run (-1 = off)


_GROUPS: dict[str, type] = {
    "task": TaskConfig, "session": SessionConfig, "judge": JudgeConfig,
    "ladder": LadderConfig, "memory": MemoryConfig, "replay": ReplayConfig,
}
#: flat name -> group, derived from the dataclasses so the two can never disagree
GROUP_OF: dict[str, str] = {f.name: g for g, cls in _GROUPS.items() for f in fields(cls)}
#: the flat names, in group order — the order a summary's ``config`` block is written in
FLAT_ORDER: tuple[str, ...] = tuple(GROUP_OF)

SESSION_MODES: tuple[str, ...] = ("harness", "independent", "single")
LEDGER_MODES: tuple[str, ...] = ("text", "none")
RETRIEVAL_MODES: tuple[str, ...] = ("none", "bm25")

#: What a context-only session policy holds, whatever the row asks: the
#: certification's verdict reaches a later task only through the ledger, a
#: recovery note only through a file, the place graph only through a
#: recovery's brief, the archive and the house notes only through files — and
#: a context-only session reads none of them.
CONTEXT_ONLY: dict[str, Any] = {
    "certify": "none", "recovery": False, "reset_after_turns": 0, "max_resets_per_task": 0,
    "place_graph": False, "archive_policy": False, "notes": False, "ledger": "none", "retrieval": "none",
}


@dataclass
class HarnessConfig:
    task: TaskConfig
    session: SessionConfig = field(default_factory=SessionConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    ladder: LadderConfig = field(default_factory=LadderConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)

    def __post_init__(self) -> None:
        """The two policy switches are checked, and a context-only session
        policy holds what it rules out (``CONTEXT_ONLY``; ``compact`` follows
        the policy). ``held`` names every switch the policy changed, with the
        value it now has."""
        mode = self.session.session_mode
        if mode not in SESSION_MODES:
            raise ValueError(f"session_mode must be one of {' | '.join(SESSION_MODES)}, got {mode!r}")
        if self.memory.ledger not in LEDGER_MODES:
            raise ValueError(f"ledger must be one of {' | '.join(LEDGER_MODES)}, got {self.memory.ledger!r}")
        if self.memory.retrieval not in RETRIEVAL_MODES:
            raise ValueError(f"retrieval must be one of {' | '.join(RETRIEVAL_MODES)}, "
                             f"got {self.memory.retrieval!r}")
        held: dict[str, Any] = {}
        if self.context_only:
            policy = dict(CONTEXT_ONLY)
            # independent: a session per task; single: one session, compacted by its backend
            policy["compact"] = ("boundary" if mode == "independent"
                                 else "token" if self.session.compact == "boundary" else self.session.compact)
            for name, value in policy.items():
                obj = getattr(self, GROUP_OF[name])
                if getattr(obj, name) != value:
                    setattr(obj, name, value)
                    held[name] = value
        if self.memory.retrieval == "bm25" and self.memory.ledger != "none":
            # the records reach the session only as the harness's excerpts
            self.memory.ledger = "none"
            held["ledger"] = "none"
        self.held = held

    @property
    def context_only(self) -> bool:
        """A session policy without external memory: the session's context is all it has."""
        return self.session.session_mode != "harness"

    @classmethod
    def from_flat(cls, **row: Any) -> HarnessConfig:
        """The flat names in, one object per organ out. An unknown name is
        loud: a switch that is misspelt must not be silently dropped."""
        unknown = sorted(set(row) - set(FLAT_ORDER))
        if unknown:
            raise TypeError(f"HarnessConfig has no switch named {', '.join(unknown)}")
        by_group: dict[str, dict[str, Any]] = {g: {} for g in _GROUPS}
        for name, value in row.items():
            by_group[GROUP_OF[name]][name] = value
        return cls(**{g: c(**by_group[g]) for g, c in _GROUPS.items()})

    def flat(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for g, c in _GROUPS.items():
            obj = getattr(self, g)
            for f in fields(c):
                out[f.name] = getattr(obj, f.name)
        return out
