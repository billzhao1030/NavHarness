"""Orchestrator — a chain whose sessions connect through files.

Four questions, one method each:

    seat()             where the chain starts: replay / resume, the bench seated,
                       the workspace laid out, the code tiers mounted, the seat frozen
    task(k, goal)      one task's lifetime: run/tasks/<k>/goal.md written; the agent's
                       session opened (or kept); the bridge's state file reset; ONE
                       instruction naming the goal, the required reading and the file it
                       must write — the four look-around views along when the body was
                       moved (a tour boundary, or a run that opens on a restored map);
                       then the task's hops (``_run_task``): each hop is one query
                       into the session and up to ``chunk_turns`` of Claude Code's own tool
                       loop (fewer when the task's turn budget has fewer left, or the
                       attempt is fewer turns from the fallback threshold), after which
                       the orchestrator reads the state file and answers
                       — a close request through the gates, a stuck attempt up the
                       ladder, or "continue"; the env asked what comes next
    seam(k, goal, oc)  between two tasks: certify (the gate reused when the agent has
                       not moved) on the evidence taken before the task's STOP — the
                       stop views and the map with the route it drove; the handover
                       is the FILE the session wrote; the
                       ledger row; verdict.json and a ledger.md line; the code tiers'
                       bite (place graph, archive); the checkpoint; the record
    close(status)      the chain's end: the consolidation session rewrites the house
                       notes (MEMORY.md, house.md, rooms/, skills/) — their only writer;
                       the summary

Three roles: the orchestrator (this), the sessions (``session/``) and the
tools (``bridge/``, a process of the session's own). A close request is
gated here, between hops:

    file gate     the first close_task is refused, at no step cost, until
                  run/tasks/<k>/handover.md exists
    judge's gate  (``precheck: model``) the judge session may send the first
                  request back with counter-evidence; the second is honoured
    the STOP      only then an env action, issued by the orchestrator

A task the harness ends itself (a turn or step budget spent, a model that
has stopped acting) has no close request, so before its STOP the
orchestrator takes, for the certification, the look-around close_task
would have taken.

A refused request is kept in the task's record with the evaluator's truth at
the pose it was made from (``stop_truth``: would the STOP have succeeded
there), for scoring the gate; the agent never sees it.

The harness never summarises a task for the model, hands a transcript over
as text, distils λ from a story or asks a side model which room this is:
every one of those is something a session reads from, or writes to, the
workspace. λ is the agent session's last hop before a recovery that voids
its context. Checkpoints, resume and replay are ``checkpoint.py``; the code
tiers are ``memory/mount.py``.

What a session may reach is set by three switches (``config.py``):

    session_mode  harness: the above. independent / single: a context-only
                  session — the observation and action tools (bridge toolset
                  ``act``), its own system prompt, an instruction that names
                  the goal and nothing else; no required reading, no file
                  gate, no certification, no recovery, no consolidation (the
                  config holds them off). The pre-stop check runs if
                  ``precheck`` says so. independent opens a session per task;
                  single keeps one across the chain (``compact=token``).
    ledger        text: every earlier record can be read. none: no earlier
                  record is required reading and the file tools read only
                  the current task's directory (Claude Code's own Read /
                  Glob / Grep on the house notes are not opened either); the
                  harness writes them all.
    retrieval     none: the session reads its records itself. bm25: the
                  fixed-access control — the harness retrieves historical
                  text at task start, after a recovery and every
                  ``retrieval_every_turns`` turns (a hop takes no more turns
                  than are left until the next one), puts it into the next
                  message, and the file tools are scoped as under
                  ``ledger=none``, which it holds (``memory/retrieval.py``).

Every session meters its hops onto a per-task tally by kind (``Tally``): the
agent's hops, its recovery-note hops (λ or the summary), the two checks and
the wake-up land on the task's row; the consolidation on the run's summary.

``persist_map=false`` rebuilds the map and the markers at every boundary
(the env's ``reset_spatial``), and the place graph and the markers' banked
views go with them.
"""

from __future__ import annotations

import base64
import contextlib
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .. import prompts as P
from ..bridge import BRIDGE_MODULE, SPATIAL_TOOLS, TaskState, tool_names
from ..config import HarnessConfig
from ..env.bench import Bench, GOATBench, map_floor
from ..env.types import EnvUnavailable, Goal, TaskEnd
from ..memory.longterm import NOTE_PATTERNS, HouseNotes
from ..memory.mount import House
from ..memory.retrieval import FixedRetriever, historical_records, tokens
from ..memory.workspace import Workspace
from ..record import Recorder
from ..session import Blocks, Hop, SessionLoop, Trace, image_block, text_block
from ..session.agent import AgentSession
from ..session.consolidate import ConsolidateSession
from ..session.wakeup import WakeupSession
from ..session.wire.blocks import in_tokens, out_tokens, peak_tokens
from .checkpoint import Checkpoints, memory_root, seat_arg
from .close import (
    TERMINATIONS,
    ProviderUnavailable,
    TaskOutcome,
    check_provider,
    claim_status,
    end_from_state,
    force_stop,
    stop,
)
from .judge import Judge, Verdict, judge_gate
from .ledger import Ledger, TaskRow
from .recovery import SCOPES, Handover, RecoveryDriver, note_words, read_note
from .recovery.driver import _views_of

#: an absolute cap on tasks per run, against a misbehaving env (chains are at most a few hundred)
_TASK_CAP = 500


class Tally:
    """What the sessions consumed since the last ``take``, by kind of session.

    Every session is given a meter (``Orchestrator._meter``) that adds each
    hop's provider-reported usage here before it reaches the run's bill, so
    a task's row says what that task cost — the agent's hops across every
    session a recovery opened (``agent``), its recovery-note hops (``note``:
    λ or the summary), the judge's two checks and the wake-up — and the
    consolidation at run end is accounted on its own. Per kind: ``calls``
    (metered hops), ``prompt_tokens`` (every prompt token the provider
    reported: uncached, cache writes and cache reads), ``cached_tokens`` (the
    cache reads among them), ``completion_tokens``, and
    ``peak_prompt_tokens`` (the largest context one turn carried)."""

    def __init__(self) -> None:
        self.kinds: dict[str, dict[str, int]] = {}

    def add(self, kind: str, usage: dict) -> None:
        row = self.kinds.setdefault(kind, {"calls": 0, "prompt_tokens": 0, "cached_tokens": 0,
                                           "completion_tokens": 0, "peak_prompt_tokens": 0})
        row["calls"] += 1
        row["prompt_tokens"] += in_tokens(usage)
        row["cached_tokens"] += int(usage.get("cache_read_input_tokens") or usage.get("cached_input_tokens") or 0)
        row["completion_tokens"] += out_tokens(usage)
        row["peak_prompt_tokens"] = max(row["peak_prompt_tokens"], peak_tokens(usage))

    def get(self, kind: str, key: str) -> int:
        return int((self.kinds.get(kind) or {}).get(key, 0))

    def take(self) -> dict[str, dict[str, int]]:
        """Everything counted so far; the tally starts again from zero."""
        out, self.kinds = self.kinds, {}
        return out


class Orchestrator:
    harness_label = "navharness"
    max_idle_hops = 3               # hops in a row with no tool call before the task ends as inactive
    max_empty_hops = 3              # hops in a row that produced no turn at all: the provider, not the model

    def __init__(self, cfg: HarnessConfig, model: str, judge: str | None, bench: Bench, out_dir: Path, *,
                 bridge: Path | None = None, meter: Callable[[str, dict], None] | None = None,
                 effort: str | None = None, endpoint: dict | None = None,
                 judge_endpoint: dict | None = None) -> None:
        self.cfg = cfg
        self.model = model
        self.judge_model = judge or model
        # The two completion checks may sit on another core than the seat (the
        # cross-model judging comparison); by default they are the seat's own.
        self.judge_transport = cfg.judge.judge_transport or cfg.session.transport
        self.bench = bench
        self.out_dir = Path(out_dir)
        self.effort = effort
        frames_dir = self.out_dir / "live"
        # the bridge the sessions are given: the arm's own tool surface, or the package's
        self.bridge = Path(bridge) if bridge else Path(__file__).resolve().parents[1] / "bridge" / "bridge.py"
        # every conversation of this chain is a session, and every session's hops are traced
        self.trace = Trace(self.out_dir / "trace")
        self._rec = Recorder(self.out_dir, frames_dir)
        self.trace.rec = self._rec
        self.loop = SessionLoop()
        self.ledger = Ledger()
        self.house = House(cfg, bench, self.out_dir, self.rec, memory_root=memory_root(cfg), seat_arg=seat_arg(cfg))
        self.checkpoints = Checkpoints(cfg, bench, self.ledger, self.out_dir, self.rec)
        self.ws = Workspace(self.out_dir / "workspace", self.house.house_dir)
        self.notes = HouseNotes(self.ws.house, day=cfg.memory.day)
        self.state = TaskState(self.out_dir / "bridge_state.json")
        self.withheld: tuple[str, ...] = SPATIAL_TOOLS if int(cfg.task.instruments or 0) == 0 else ()
        self.tally = Tally()
        self._agent_kind = "agent"               # the agent's hops; "note" while it writes a recovery note
        common = {"loop": self.loop, "cwd": self.ws.root, "trace": self.trace, "rec": self.rec,
                  "effort": effort, "transport": cfg.session.transport, "endpoint": endpoint}
        # An effort word is one vendor's, so it reaches the judge only when the
        # judge sits on the seat's own core; another core's checks run at its
        # CLI's default. The endpoint is the seat's when the judge is the seat's
        # model on the seat's core, else whatever the caller routed for it. A
        # judge on another core is metered under its check all the same.
        own_core = self.judge_transport == cfg.session.transport
        judged = {**common, "transport": self.judge_transport, "effort": effort if own_core else None,
                  "endpoint": (judge_endpoint if judge_endpoint is not None
                               else endpoint if own_core and self.judge_model == model else None)}
        self.pre = Judge(cfg.judge.precheck, self.judge_model if cfg.judge.precheck == "model" else None,
                         kind="precheck", meter=self._meter("precheck", meter), **judged)
        self.cert = Judge(cfg.judge.certify, self.judge_model if cfg.judge.certify == "model" else None,
                          kind="certify", meter=self._meter("certify", meter), **judged)
        mem = cfg.memory
        self.retriever: FixedRetriever | None = (
            FixedRetriever(top_k=mem.retrieval_top_k, budget=mem.retrieval_budget_tokens,
                           chunk=mem.retrieval_chunk_tokens, overlap=mem.retrieval_overlap_tokens)
            if mem.retrieval == "bm25" else None)
        system = (P.SESSION_SYSTEM if cfg.context_only
                  else P.SYSTEM_FIXED_ACCESS if self.retriever is not None else P.SYSTEM)
        # On Claude Code the agent's session also reads the house notes with the CLI's own
        # file tools, fenced to that directory (session/notes.py); a context-only session
        # has no notes to read, and none are opened when earlier records are out of view
        # (ledger=none, which retrieval=bm25 holds)
        self.agent_session = AgentSession(model, system_prompt=system,
                                          mcp=self._mcp(self.state.path, "act" if cfg.context_only else "all"),
                                          tools=self.tool_names(), max_turns=cfg.session.chunk_turns,
                                          notes=None if cfg.context_only or mem.ledger == "none" else self.ws.house,
                                          meter=self._meter(lambda: self._agent_kind, meter), **common)
        # The judge model sits only in the two completion checks; the wake-up
        # and the consolidation are the navigating model's own sessions.
        self.driver = RecoveryDriver(rec=self.rec, out_dir=self.out_dir, ladder=cfg.ladder,
                                     wake=WakeupSession(model, meter=self._meter("wakeup", meter), **common))
        self.consolidate_state = TaskState(self.out_dir / "bridge_state_consolidate.json")
        self.consolidate_session = ConsolidateSession(
            model, mcp=self._mcp(self.consolidate_state.path, "files"),
            tools=tool_names(toolset="files"), meter=self._meter("consolidate", meter), **common)
        self._consolidations = 0
        self._after_wakeup = False               # the next row is the first task after a body move
        self._relocated: dict | None = None      # the env's wake report, spent on the next task
        # per task
        self.k = -1
        self.close_requests = 0
        self.precheck_verdict: Verdict | None = None
        self.steps_at_precheck = -1
        self.handover_prompts = 0
        self.refusals: list[dict] = []           # each refused close request, with the truth where it was made
        self._side_turns = 0                     # turns spent outside the hop loop (the λ hop)
        self._seam_map: str | None = None        # the task's map with its route, taken before its STOP
        self._described = ""                     # the session's latest description of what it sees
        self._retrievals = 0                     # this task's scheduled retrievals (retrieval: bm25)

    @property
    def rec(self) -> Recorder:
        return self._rec

    @rec.setter
    def rec(self, rec: Recorder) -> None:
        """A seat may swap the record after construction (the ES seat writes
        through the bus); everything that writes events follows."""
        self._rec = rec
        for holder in (self.trace, self.house, self.checkpoints, getattr(self, "driver", None),
                       getattr(self, "agent_session", None), getattr(self, "consolidate_session", None),
                       getattr(getattr(self, "pre", None), "session", None),
                       getattr(getattr(self, "cert", None), "session", None),
                       getattr(getattr(self, "driver", None), "wake", None)):
            if holder is not None and hasattr(holder, "rec"):
                holder.rec = rec

    def _meter(self, kind: str | Callable[[], str],
               bill: Callable[[str, dict], None] | None) -> Callable[[str, dict], None]:
        """The meter a session of ``kind`` is given (a callable names the kind at
        each hop): its usage is counted on this task's tally, then on the run's
        bill (``bill``, when there is one)."""
        def meter(model: str, usage: dict) -> None:
            self.tally.add(kind() if callable(kind) else kind, usage)
            if bill is not None:
                bill(model, usage)
        return meter

    # ── the tools, as the sessions are given them ────────────────────

    def tool_names(self) -> list[str]:
        return agent_tool_names(self.cfg)

    def _mcp(self, state_file: Path, toolset: str) -> dict:
        """The bridge as an MCP server config: a process of this session's own,
        told through its environment where the env, the frames, the state file
        and the workspace are (the session adds the CLI's own environment)."""
        import sys
        cfg = self.cfg
        # absolute: the agent's CLI may run in the notes directory, and the bridge it spawns with it
        out = self.out_dir.resolve()
        env = {
            "NAVH_SERVER_URL": self.bench.env.url,
            "NAVH_LIVE_DIR": str(out / "live"),
            "NAVH_MARK_DIR": str(out / "markers"),
            "NAVH_STATE_FILE": str(Path(state_file).resolve()),
            "NAVH_WORKSPACE": str(self.ws.root.resolve()),
            "NAVH_HOUSE_DIR": str(self.ws.house.resolve()),
            "NAVH_STEP_BUDGET": str(cfg.task.step_budget),
            "NAVH_STOP_VIEWS": str(cfg.task.stop_views),
            # Every bench on this stack reads the goal in force off the env
            # (the ES benches, IR2R-CE tours included, subclass GOATBench);
            # the tour-shaped branch re-resets an env that has no such verb.
            "NAVH_BENCH": "goat" if isinstance(self.bench, GOATBench) else "ivln",
            "NAVH_RECOVERY": "1" if cfg.ladder.recovery else "0",
            "NAVH_INSTRUMENTS": "0" if self.withheld else str(cfg.task.instruments),
            "NAVH_TOOLSET": toolset,
            # the agent's file tools follow ``ledger``; consolidation reads every record
            "NAVH_LEDGER": cfg.memory.ledger if toolset != "files" else "text",
            "NAVH_BRIDGE_MODULE": BRIDGE_MODULE,
        }
        return {"env": {"type": "stdio", "command": sys.executable, "args": [str(self.bridge)], "env": env}}

    # ── the chain ────────────────────────────────────────────────────

    def run(self) -> dict:
        try:
            k, goal, limit = self.seat()
            status = "chain_incomplete"           # only a chain that runs out of goals is complete
            while goal is not None and k < limit:
                outcome, end = self.task(k, goal)
                self.seam(k, goal, outcome)
                if goal.meta.get("slot_mismatch"):
                    self.rec.emit("warning", kind_detail="slot_mismatch", **goal.meta["slot_mismatch"])
                k += 1
                if end.chain_finished or end.next_goal is None:
                    status = "chain_complete"
                    break
                goal = end.next_goal
            return self.close(status)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Every session closed, the sessions' loop stopped — also on the way out of an error."""
        for s in (self.agent_session, self.consolidate_session):
            with contextlib.suppress(Exception):
                s.close()
        self.loop.stop()

    # ── seat ─────────────────────────────────────────────────────────

    def seat(self) -> tuple[int, Goal | None, int]:
        """Where the chain starts: ``(k, first goal, task limit)``."""
        cfg = self.cfg
        replay = self.checkpoints.replay_task()    # a single task re-run from a frozen boundary
        done = replay if replay is not None else self.checkpoints.replay_ledger()
        goal = self.bench.seat(seat_arg(cfg), cfg.memory.persist_map, skip=done)
        if replay is not None:
            self.house.places = self.checkpoints.restore(replay, self.house.places)
        elif cfg.memory.map_state and Path(cfg.memory.map_state).exists():
            # The map outlives the process, and not only when a cut-short chain
            # is resumed: a run that opens in a house already mapped (the next
            # day of a deployment) starts from the map it left. A house does
            # not change overnight; the ledger, which is this run's, does not
            # cross — consolidation has already read it into the notes.
            out = self.bench.env.call(self.bench.fn("load_spatial"), {"path": cfg.memory.map_state})
            self.rec.emit("map_restored", path=cfg.memory.map_state,
                          **(out if isinstance(out, dict) else {"out": str(out)}))
            if not (isinstance(out, dict) and out.get("error")):
                # The body was put down at this run's start, not walked there from
                # where the map was saved: a prescribed relocation. The first task
                # opens as any task after one does — the four views and the
                # instruction to work out which room this is. The restored map is
                # not re-registered to the new tracker's frame: loading a map after
                # a tracker restart does not recover their alignment, and
                # recognising the room is the session's to do.
                self._relocated = {"cause": "map_restored", "map_state": cfg.memory.map_state}
                self._after_wakeup = True
                self.rec.emit("relocated", index=-1, cause="map_restored")
        # The restore above runs before the graph exists, so it cannot load it;
        # the graph is mounted here, from the replay's checkpoint when there is one.
        self.house.mount_places(self.checkpoints.replay_graph(replay))
        if self.house.places is not None:
            self.driver.places = self.house.places
            self.driver.first_visit = not self.house.places.places and self.notes.is_new()
        # The places marked on earlier runs keep their views in the house
        # (rooms/<room>/photos.json); a wake-up compares against them as it does
        # against the views banked in this run, which replace them label by label.
        labels = list(self.house.places.places) if self.house.places is not None else []
        self.driver.marker_photos.update(self.notes.photo_views(labels))
        self.rec.emit("longterm", house=self.ws.house.name, day=self.notes.day,
                      first_visit=self.notes.is_new(), notes=self.notes.list_notes())
        # The boundary checkpoints are written as each task CLOSES, which leaves
        # the first task with no starting snapshot — and task 0 is where a chain
        # is most often lost, since nothing has been mapped yet. So freeze the
        # seat itself, before anything has run.
        self.checkpoints.freeze(done, self.house.places, self.house.places_path)
        self.house.mount_archive()
        self.rec.emit("workspace", root=str(self.ws.root), house=str(self.ws.house),
                      notes=self.notes.index.exists(), ledger_lines=_lines(self.ws.ledger),
                      handovers=len(self.ws.handovers()), bridge=str(self.bridge))
        if self.withheld:
            self.rec.emit("instruments", level=0, withheld=list(self.withheld),
                          note="bare: no map, no markers, no route preview")
        if cfg.context_only or cfg.memory.ledger != "text":
            self.rec.emit("session_policy", session_mode=cfg.session.session_mode, ledger=cfg.memory.ledger,
                          tools=self.tool_names(), held=dict(cfg.held))
        # A chain runs every task it has. A replay re-runs the single task it
        # was seated at; the cap is a guard against a misbehaving env only.
        limit = done + 1 if cfg.replay.replay_task >= 0 else _TASK_CAP + done
        return done, goal, limit

    # ── one task ─────────────────────────────────────────────────────

    def task(self, k: int, goal: Goal) -> tuple[TaskOutcome, TaskEnd]:
        self.trace.open_task(k)
        self.rec.open_task(k, {"episode_id": goal.meta.get("episode_id", goal.index),
                                 "instruction": goal.text, "modality": goal.modality,
                                 "errand_index": goal.index, "errand_total": goal.total})
        self.ws.write_goal(k, goal.text, goal.index, goal.total, goal.modality)
        self.state.reset(k, writable=[] if self.cfg.context_only else [self.ws.rel(self.ws.handover_path(k))])
        self._retrievals = 0
        self._open_agent_session()
        relocated, self._relocated = self._relocated, None
        if self.cfg.context_only:
            relocated = None                     # the instruction names the goal and nothing else
        views: list[str] = []
        if relocated is not None:
            views, _floor = self.driver._look(self.bench, "orient_look_around")
        instruction = self._instruction(k, goal, relocated=relocated is not None)
        self.rec.emit("instruction", index=k, text=instruction, views=len(views),
                      relocated=relocated is not None, required=self._required_reading(k))
        # The certification's map is taken before the task's STOP
        # (``_stop_evidence``): the boundary below may move the body or rebuild
        # the map, and the certification is of THIS task.
        self._seam_map = None
        outcome = self._run_task(k, goal, _with_views(instruction, views))
        end = self.bench.after_end(outcome.end, self.cfg.memory.persist_map)
        outcome.end = end
        if end.wakeup:
            # a tour boundary: the body will be somewhere it did not walk to.
            # No side session recognises the room; the next task's session
            # does, from the views its instruction carries.
            self._relocated = dict(end.wakeup)
            self.rec.emit("relocated", index=k, **{key: v for key, v in end.wakeup.items()
                                                    if key != "brief"})
        return outcome, end

    def _open_agent_session(self) -> None:
        """The agent's session carries over a task boundary unless ``compact``
        is ``boundary``; then this task opens a fresh one."""
        session = self.agent_session
        carries = self.cfg.session.compact != "boundary"
        if not (carries and session.is_open):
            session.close()
            session.open()

    def _required_reading(self, k: int) -> list[str]:
        if self.cfg.context_only or self.cfg.memory.ledger == "none":
            return []                            # no earlier record is in view
        req = []
        if k > 0 and self.ws.handover_path(k - 1).exists():
            req.append(self.ws.rel(self.ws.handover_path(k - 1)))
        if self.ws.ledger.exists():
            req.append(self.ws.rel(self.ws.ledger))
        for p in (self.notes.index, self.notes.overview):     # the house notes: index, then overview
            if p.exists():
                req.append(self.ws.rel(p))
        return req

    def _instruction(self, k: int, goal: Goal, *, relocated: bool) -> str:
        photo = P.PHOTO_LINE if goal.image_b64 else ""
        if self.cfg.context_only:
            return P.SESSION_INSTRUCTION.format(k=goal.index + 1, n=goal.total, goal=goal.text, photo=photo)
        req = self._required_reading(k)
        required = ("\n".join(f"- {r}" for r in req) if req
                    else P.RECORDS_WITHHELD if self.cfg.memory.ledger == "none" and k > 0
                    else "- (nothing yet: this is the first task in this house)")
        retrieved = ""
        if self.retriever is not None:
            required = P.FIXED_ACCESS_REQUIRED
            retrieved = "\n" + self._retrieve(k, goal, "task_start") + "\n"
        text = P.INSTRUCTION.format(
            k=goal.index + 1, n=goal.total, goal=goal.text, photo=photo,
            required=required, handover=self.ws.rel(self.ws.handover_path(k)), recovered=retrieved)
        if relocated:
            text = text.replace("\nBegin.", P.RELOCATED + "\n\nBegin.")
        return text

    def _retrieve(self, k: int, goal: Goal, why: str) -> str:
        """One retrieval of the fixed-access policy: the query is the goal and
        the session's latest description of what it sees; the block is what
        the next message carries."""
        assert self.retriever is not None
        query = goal.text + ("\n" + self._described if self._described else "")
        records = historical_records(self.ws, k, day=self.notes.day)
        block, held = self.retriever.render(records, query)
        self._retrievals += 1
        self.rec.emit("retrieval", index=k, why=why, query=query[:2000], records=len(records),
                      excerpts=held, tokens=tokens(block))
        return block

    # ── the hops of a task ───────────────────────────────────────────

    def _run_task(self, k: int, goal: Goal, first: str | Blocks) -> TaskOutcome:
        """The task's hops: the session runs its own tool loop for one hop; the
        orchestrator reads the state file, answers a close request through
        the gates, sends a stuck attempt up the ladder, or says continue."""
        self.k = k
        self.close_requests, self.precheck_verdict, self.steps_at_precheck = 0, None, -1
        self.handover_prompts = 0
        self.refusals = []
        session = self.agent_session
        max_turns = self.cfg.task.max_turns_per_task
        end: TaskEnd | None = None
        turns = 0
        attempt_turns = 0
        idle = 0
        empty = 0
        resets: list[dict] = []
        prompt: str | Blocks = first
        since_retrieval = 0                      # the opening instruction carried one (retrieval: bm25)
        every = max(1, self.cfg.memory.retrieval_every_turns)
        # ΔA is newly explored area: the first checkpoint measures it from the
        # map this task starts on, not from an empty one
        self.driver.new_task(area=self.driver.explored(self.bench) if self.cfg.ladder.recovery else 0.0)
        while end is None and turns < max_turns:
            # a hop may take what is left of the task's turns, up to the session's own cap;
            # while the fallback can still fire, no more than are left until its threshold
            # on this attempt; under fixed access, no more than are left until the next
            # scheduled retrieval
            cap = max_turns - turns
            to_fallback = self.driver.ctrl.turns_to_fallback(attempt_turns, len(resets))
            if to_fallback is not None:
                cap = min(cap, to_fallback)
            if self.retriever is not None:
                if since_retrieval >= every:
                    prompt = _with_text(prompt, self._retrieve(k, goal, "schedule"))
                    since_retrieval = 0
                cap = min(cap, every - since_retrieval)
            hop = session.hop(prompt, max_turns=cap)
            turns += hop.num_turns
            attempt_turns += hop.num_turns
            since_retrieval += hop.num_turns
            self._described = _latest_description(hop) or self._described
            self.rec.emit("usage", prompt_tokens=session.last_usage.get("prompt_tokens", 0),
                          completion_tokens=session.last_usage.get("completion_tokens", 0),
                          sdk_turns=hop.num_turns, subtype=hop.subtype, latency_s=hop.latency_s)
            check_provider(hop)
            # A hop that produced no turn at all is the provider refusing, not
            # the model thinking; a hop full of turns and no tool call is a
            # model that has stopped acting. Neither is allowed to burn the
            # turn budget and be written down as a failed task.
            empty = empty + 1 if hop.num_turns == 0 else 0
            if empty >= self.max_empty_hops:
                raise ProviderUnavailable(f"{empty} hops in a row produced no turn (subtype={hop.subtype})")
            idle = idle + 1 if not hop.tool_uses else 0
            st = self.state.read()
            self.driver.marker_photos.update(st.get("marker_photos") or {})
            if st.get("env_dead"):
                raise EnvUnavailable(json.dumps(st["env_dead"])[:300])
            if st.get("task_over") and not st.get("closed"):
                # the env ended it: the budget ran out. The body has not moved
                # since, so the stop views and the map are still the task's.
                self._stop_evidence(views=True)
                end = end_from_state(self.bench, st)
                break
            if st.get("budget_exhausted") and not st.get("closed"):
                self._stop_evidence(views=True)
                end = force_stop(self.bench, "budget")    # the task's step budget is spent
                self.state.update(closed=True, task_over=True, end_reason="budget")
                break
            req = st.get("close_request")
            if req:
                self.state.update(close_request=None)
                answer, end = self._close(k, goal, req, st, turns)
                if end is not None:
                    break
                prompt = answer
                continue
            if idle >= self.max_idle_hops:
                # the model has stopped acting: the task ends as inactive, not as a spent budget
                self.rec.emit("warning", kind_detail="no_tool_calls", hops=idle)
                self._stop_evidence(views=True)
                end = force_stop(self.bench, "inactive")
                self.state.update(closed=True, task_over=True, end_reason="inactive")
                break
            asked = st.get("recovery_request")
            if asked is not None:
                self.state.update(recovery_request=None)
            recovered = len(resets)
            nxt = self._recover(k, goal, asked, attempt_turns, turns, resets, st)
            turns += self._side_turns
            since_retrieval += self._side_turns
            self._side_turns = 0
            if nxt is not None:
                attempt_turns = 0
                prompt = nxt
                if self.retriever is not None and len(resets) > recovered:
                    # the fresh attempt opens with a retrieval
                    prompt = _with_text(prompt, self._retrieve(k, goal, "recovery"))
                    since_retrieval = 0
            else:
                prompt = P.CONTINUE
        if end is None:
            self._stop_evidence(views=True)
            end = force_stop(self.bench)
            self.state.update(closed=True, task_over=True, end_reason="turns")
        st = self.state.read()
        claim = st.get("close_request") or st.get("claim") or {"status": "stop", "evidence": ""}
        if end.end_reason in TERMINATIONS and not st.get("claim"):
            # ended by the harness, not by a closure: the record says why and claims nothing
            claim = {"status": "stop", "evidence": TERMINATIONS[end.end_reason]}
        return TaskOutcome(end=end, claim={"status": claim.get("status", "stop"), "evidence": claim.get("evidence", "")},
                           frames=list(st.get("frames") or []), steps_used=int(st.get("steps_used") or 0),
                           marks=list(st.get("marks") or []), tool_calls=dict(st.get("calls") or {}),
                           # every agent hop of this task, its recovery notes included, in whichever
                           # session it ran: a recovery that voids the context opens a new one mid-task
                           turns=turns, peak_prompt_tokens=max(self.tally.get("agent", "peak_prompt_tokens"),
                                                               self.tally.get("note", "peak_prompt_tokens")),
                           meta={"transport": self.cfg.session.transport, "close_requests": self.close_requests,
                                 "precheck_verdict": (self.precheck_verdict.verdict
                                                      if self.precheck_verdict else None),
                                 "precheck_full": (asdict(self.precheck_verdict)
                                                   if self.precheck_verdict else None),
                                 "precheck_steps": self.steps_at_precheck,
                                 "refusals": list(self.refusals),
                                 "resets": resets,
                                 "handover_written": self.ws.handover_path(k).exists(),
                                 "handover_prompts": self.handover_prompts,
                                 "files_written": list(st.get("written") or []),
                                 "session_id": session.run.id if session.run else None,
                                 "hops": session.run.hops if session.run else 0,
                                 "retrievals": self._retrievals})

    # ── the close, and the gates before it ───────────────────────────

    def _close(self, k: int, goal: Goal, req: dict, st: dict, turn: int) -> tuple[str, TaskEnd | None]:
        """The agent asked to close (the bridge took the look-around and
        recorded the claim): the file gate, then the judge's gate, and only
        then the STOP. Returns ``(answer for the next hop, end)``."""
        claim = {"status": claim_status(req.get("status", "complete")), "evidence": str(req.get("evidence", ""))[:400]}
        self.close_requests += 1
        if (not self.cfg.context_only and self.handover_prompts == 0
                and not self.ws.handover_path(k).exists()):
            # the file gate: no handover file, no closure — once, at no step cost
            self.handover_prompts += 1
            path = self.ws.rel(self.ws.handover_path(k))
            truth = self._refused(turn, "handover", st, claim)
            self.rec.emit("handover_prompted", index=k, turn=turn, path=path, truth=truth)
            return P.HANDOVER_REQUIRED.format(path=path), None
        if self.close_requests - self.handover_prompts == 1:     # the judge's first look at this closure
            v, refusal = judge_gate(self.pre, self.bench, goal, claim,
                                    frames=list(st.get("frames") or []), record=self._record(goal, st))
            if v is not None:
                self.precheck_verdict = v
                self.steps_at_precheck = int(st.get("steps_used") or 0)
            if refusal is not None:
                truth = self._refused(turn, "judge", st, claim, verdict=v.verdict if v else None)
                self.rec.emit("precheck_refused", index=k, turn=turn, verdict=v.verdict if v else None,
                              truth=truth)
                return refusal, None
        self._stop_evidence(views=False)              # close_task took the stop views
        end = stop(self.bench, claim["status"])
        self.state.update(closed=True, task_over=True, end_reason="stop", claim=claim)
        return "", end

    def _refused(self, turn: int, by: str, st: dict, claim: dict, verdict: str | None = None) -> dict:
        """Keep a refused close request (``by``: ``handover`` — the file gate,
        ``judge`` — the judge's gate) with the evaluator's truth where it was
        made, so a refused stop can be scored as if it had gone through."""
        truth = self._truth_here()
        self.refusals.append({"by": by, "turn": turn, "steps_used": int(st.get("steps_used") or 0),
                              "claim": claim["status"], "verdict": verdict, "truth": truth})
        return truth

    def _truth_here(self) -> dict:
        """The evaluator's truth at the body's pose, read without ending the
        task (the env's ``stop_truth``): whether the closing move would have
        succeeded here, and the distance to the goal, as the env reports them.
        Scoring only — it goes into the task's record and the event stream,
        never into a prompt, the workspace or the state file the bridge reads.
        An env without the verb leaves the reason instead of a number."""
        try:
            out = self.bench.env.call(self.bench.fn("stop_truth"), {})
        except Exception as e:
            return {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        if not isinstance(out, dict) or out.get("error"):
            return {"error": str(out.get("error") if isinstance(out, dict) else out)[:200]}
        return {key: out[key] for key in ("success", "distance_to_goal", "success_distance") if key in out}

    def _record(self, goal: Goal, st: dict) -> dict:
        """What the judge gets: mechanical facts only, NO claim."""
        return {"task": goal.index + 1, "of": goal.total,
                "steps_used": int(st.get("steps_used") or 0), "step_budget": self.cfg.task.step_budget,
                "frames_taken": len(st.get("frames") or []),
                "markers_dropped_this_task": list(st.get("marks") or [])}

    # ── the ladder, with λ as a file ─────────────────────────────────

    def _recover(self, k: int, goal: Goal, asked: dict | None, attempt_turns: int, turns: int,
                 resets: list[dict], st: dict) -> str | Blocks | None:
        """One checkpoint of the ladder. Returns the instruction of the attempt
        that follows, or None to carry on with the one that is running.

        ``recovery: false`` switches every restart off, the fallback included.
        A task whose turns are spent is not recovered either: the note and the
        new attempt would both have to come out of a budget that is gone."""
        d = self.driver
        max_turns = self.cfg.task.max_turns_per_task
        if not self.cfg.ladder.recovery or turns >= max_turns:
            return None
        sig = d.signals(body=dict(st.get("body") or {}), frames=len(st.get("frames") or []),
                        marks=len(st.get("marks") or []), steps_used=int(st.get("steps_used") or 0),
                        step_budget=self.cfg.task.step_budget, attempt_turns=attempt_turns, turns=turns,
                        max_turns=max_turns, resets=resets)
        proposal = None
        if asked is not None:
            proposal = {"scope": str(asked.get("scope") or "attempt"), "reason": asked.get("reason", ""),
                        "confidence": asked.get("confidence")}
        dec = d.ctrl.decide(proposal, sig)
        if not dec.acts():
            if proposal is not None:
                d.refuse(k, dec, attempt_turns)
            return None
        steps_now = int(st.get("steps_used") or 0)
        attempt = len(resets) + 1
        path = self.ws.lambda_path(k, attempt)
        written = (self._note_hop(k, goal, attempt_turns, steps_now, dec.scope, path, turns_left=max_turns - turns)
                   if d.writes_note() else "")
        note = d.carry(goal.text, written, attempt)
        self._hand_over(k, path, written, note)
        applied = d.apply_scope(dec.scope, self.bench)
        views = applied.pop("views", None) or []
        d.record(k, goal, dec=dec, sig=sig, note=note, applied=applied,
                 turns=turns, attempt_turns=attempt_turns, steps_now=steps_now,
                 reason=str((proposal or {}).get("reason", "")), resets=resets)
        if "context" in SCOPES[dec.scope].invalidates:
            # the conversation is void: a new session, whose bridge picks the
            # task up from the state file, reads the note the old one wrote
            self.agent_session.close()
            self.agent_session.open()
        brief = ("\n\n" + applied["brief"]) if applied.get("brief") else ""
        if note.source == "summary":
            template = P.RECOVERED_SUMMARY
        elif note.source in ("empty", "other-unavailable", "summary-empty"):
            template = P.RECOVERED_NO_NOTE
        else:
            template = P.RECOVERED
        text = template.format(scope=dec.scope, goal=goal.text, what=P.WHAT_WAS_CLEARED.get(
            dec.scope, "your conversation was discarded"), brief=brief, path=self.ws.rel(path))
        if goal.image_b64:
            text += " Call get_goal first to see the goal photograph again."
        return _with_views(text, views)

    def _note_hop(self, k: int, goal: Goal, turns: int, steps: int, scope: str, path: Path, *,
                  turns_left: int) -> str:
        """The session's last hop before the recovery: the file tools only,
        asked to write its note to ``path`` — λ, or under the ``summary``
        handover a prose summary of the attempt. The hop's turns are the
        task's, so it may take no more than ``turns_left``. Returns the note
        as written ("" when there is none)."""
        rel = self.ws.rel(path)
        self.state.allow(rel)
        self.state.update(only_files=True)
        what = P.WHAT_WAS_CLEARED.get(scope, "your conversation will be discarded")
        summary = self.driver.handover == "summary"
        if summary:
            ask = P.SUMMARY_ASK.format(turns=turns, steps=steps, goal=goal.text, scope=scope, what=what,
                                       path=rel, words=self.driver.summary_words)
        else:
            ask = P.LAMBDA_ASK.format(turns=turns, steps=steps, goal=goal.text, scope=scope, what=what,
                                      path=rel)
        self.rec.emit("lambda_ask", index=k, scope=scope, path=rel, handover=self.driver.handover)
        hop: Hop | None = None
        self._agent_kind = "note"                     # its usage is the note's, on the task's tally
        try:
            hop = self.agent_session.hop(ask, max_turns=turns_left)
            self._side_turns += hop.num_turns         # the note's turns are the task's
        except Exception as e:
            self.rec.emit("warning", kind_detail="lambda_failed", error=str(e)[:200])
        finally:
            self._agent_kind = "agent"
            self.state.update(only_files=False)
        raw = path.read_text(encoding="utf-8").strip() if path.exists() else ""
        if not raw and hop is not None and hop.text.strip():
            # the note in the reply rather than through the tool: keep it, as the file
            self.ws.session_write(rel, hop.text.strip(), {rel})
            raw = path.read_text(encoding="utf-8").strip() if path.exists() else ""
        if not raw:
            self.rec.emit("warning", kind_detail="lambda_empty", path=rel)
            return ""
        self.rec.emit("lambda_file", index=k, path=rel, words=note_words(raw),
                      parsed=None if summary else read_note(raw).as_record())
        return raw

    def _hand_over(self, k: int, path: Path, written: str, note: Handover) -> None:
        """Leave at ``path`` exactly what the next attempt is to read: the note
        as its attempt wrote it, a borrowed or shortened one in its place, or
        no file at all."""
        if note.text == written:
            return
        if note.text:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(note.text, encoding="utf-8")
        else:
            path.unlink(missing_ok=True)
        self.rec.emit("lambda_handed_over", index=k, path=self.ws.rel(path), source=note.source,
                      words=note.words)

    # ── the seam ─────────────────────────────────────────────────────

    def _map_png(self) -> str | None:
        """The top-down map, for the judge: where the robot went, not what it saw."""
        try:
            out = self.bench.env.call(self.bench.fn("get_map"), {})
            return out.get("map") if isinstance(out, dict) else None
        except Exception:
            return None

    def _stop_evidence(self, *, views: bool) -> None:
        """What the certification is given of where the task ended, taken
        while the body still stands there and the map still holds the route
        it drove this task: the STOP that follows starts the next sub-goal's
        route on GOAT, and an IR2R-CE boundary moves the body to the next
        start. ``views``: the task ends without a close request (a budget, or
        a model that stopped acting), so the look-around ``close_task`` takes
        at the stopping point is taken here instead. Only a model
        certification reads either, so nothing is taken without one."""
        if self.cfg.judge.certify != "model":
            return
        if views:
            self._stop_views()
        self._seam_map = self._map_png()

    def _stop_views(self) -> None:
        """The look-around at the stopping point, ``stop_views`` of its four
        views, filed as the bridge files the ones ``close_task`` takes: frames
        of kind ``stop`` in the state file, which the judge's frame sample
        always keeps. Best-effort: a view that cannot be taken costs the
        certification that view, not the chain."""
        try:
            got = self.bench.env.call(self.bench.fn("look_around"), {}) or {}
        except Exception as e:
            self.rec.emit("warning", kind_detail="stop_views", error=str(e)[:200])
            return
        views = _views_of(got if isinstance(got, dict) else {})[:max(1, int(self.cfg.task.stop_views))]
        if not views:
            self.rec.emit("warning", kind_detail="stop_views",
                          error=str((got or {}).get("error") if isinstance(got, dict) else got)[:200])
            return
        st = self.state.read()
        frames = list(st.get("frames") or [])
        live = (self.out_dir / "live").resolve()
        live.mkdir(parents=True, exist_ok=True)
        ids: list[int] = []
        for png in views:
            fid = len(frames) + 1
            path = live / f"obs_{fid:04d}.png"
            try:
                path.write_bytes(base64.b64decode(png))
            except (OSError, ValueError):
                break
            frames.append({"id": fid, "path": str(path), "step": int(st.get("steps_used") or 0), "kind": "stop"})
            ids.append(fid)
        self.state.update(frames=frames, stop_frames=ids)
        self.rec.emit("stop_views", index=self.k, frames=ids)

    def seam(self, k: int, goal: Goal, oc: TaskOutcome) -> None:
        end = oc.end
        claim = oc.claim or {"status": "stop", "evidence": ""}
        record = {"task": goal.index + 1, "of": goal.total, "steps_used": oc.steps_used,
                  "step_budget": self.cfg.task.step_budget, "frames_taken": len(oc.frames),
                  "markers_dropped_this_task": list(oc.marks)}
        # gate → certification reuse: if the gate judged this closure and the agent has not moved
        # since, the certification is the same judgement and is not paid for twice
        seam_map, self._seam_map = self._seam_map, None
        pre_full = oc.meta.get("precheck_full") or None
        pre_steps = int(oc.meta.get("precheck_steps", -1) if oc.meta.get("precheck_steps") is not None else -1)
        gate_reused = bool(pre_full and self.cfg.judge.certify == "model" and self.cfg.judge.precheck == "model"
                           and pre_steps == oc.steps_used and pre_full.get("judge") == "model")
        if gate_reused:
            verdict = Verdict(**{key: v for key, v in pre_full.items() if key in Verdict.__dataclass_fields__})
            verdict.judge = "gate-reused"
        else:
            verdict = self.cert.run(goal, oc.frames, record, claim=claim, map_b64=seam_map)
        judge_calls = (1 if pre_full else 0) + (0 if gate_reused or self.cfg.judge.certify != "model" else 1)
        judge_pt = int(((pre_full or {}).get("usage") or {}).get("prompt_tokens", 0) or 0) \
            + (0 if gate_reused else int((verdict.usage or {}).get("prompt_tokens", 0) or 0))
        judge_ct = int(((pre_full or {}).get("usage") or {}).get("completion_tokens", 0) or 0) \
            + (0 if gate_reused else int((verdict.usage or {}).get("completion_tokens", 0) or 0))
        # the handover is the file the session wrote, not a summary the harness asked for
        summary = self.ws.read_handover(k)
        row = self.ledger.add(TaskRow(
            index=goal.index, goal=goal.text, modality=goal.modality,
            claim=claim["status"], claim_evidence=claim.get("evidence", ""),
            verdict=verdict.verdict, verdict_evidence=verdict.evidence,
            searched=list(verdict.searched), markers=list(oc.marks),
            summary=summary, steps=oc.steps_used, end_reason=end.end_reason,
            precheck_verdict=(oc.meta.get("precheck_verdict") or ""), close_requests=int(oc.meta.get("close_requests", 1)),
            truth_success=end.score.get("success"), truth_distance=end.score.get("distance_to_goal"),
        ))
        self.ws.write_verdict(k, asdict(verdict))
        self.ws.append_ledger(row.render_line())
        # the house's journal: the task record that outlives this run, and what
        # consolidation reads at run end; the photo sets of the places marked
        label = {"complete": "CERTIFIED", "incomplete": "NOT DONE",
                 "unknown": "UNRESOLVED"}.get(verdict.verdict, "CLAIMED")
        if end.end_reason in TERMINATIONS and not oc.claim.get("evidence"):
            label += f" ({TERMINATIONS[end.end_reason]})"
        self.notes.journal(goal.index, goal.text, label, oc.steps_used, list(oc.marks), summary)
        for lab in oc.marks:
            with contextlib.suppress(OSError):
                self.notes.attach_photos(lab, list(self.driver.marker_photos.get(lab) or []), goal.index)
        self.rec.emit("files", index=k, handover=self.ws.handover_path(k).exists(),
                      handover_prompts=int(oc.meta.get("handover_prompts") or 0),
                      written=list(oc.meta.get("files_written") or []),
                      ledger=self.ws.rel(self.ws.ledger), verdict=self.ws.rel(self.ws.verdict_path(k)))
        if self.cfg.memory.map_state:
            try:
                self.bench.env.call(self.bench.fn("dump_spatial"), {"path": self.cfg.memory.map_state})
            except Exception as e:
                self.rec.emit("warning", kind_detail="dump_spatial", error=str(e)[:200])
        if self.house.places is not None:
            # The floor is read once, here, rather than at mark time: a mark is
            # on the hot path and this is not.
            floor = 0
            with contextlib.suppress(Exception):
                floor = map_floor(self.bench.env.call(self.bench.fn("get_map"), {})) or 0
            self.house.absorb_places(k, list(oc.marks), floor, self.driver.marker_photos)
        if not self.cfg.memory.persist_map:
            # The map and the markers were rebuilt at this boundary, so the
            # places named on them and the views banked there go too: the
            # next task starts with no inherited spatial memory at all.
            self.house.forget_places()
            self.driver.marker_photos.clear()
        self.house.absorb_archive(k, goal, summary, verdict.verdict, list(oc.marks))
        used = self.tally.take()                    # this task's sessions, the certification included
        # Frozen only now, after the place graph and the archive have taken this
        # task in: the next task starts from them, so its checkpoint must hold them.
        self.checkpoints.freeze(k + 1, self.house.places, self.house.places_path)
        run = self.agent_session.run
        self.rec.emit("boundary", index=goal.index, claim=claim, verdict=asdict(verdict),
                      summary=summary, ledger_rows=len(self.ledger))
        self.rec.add_row({
            "index": goal.index, "goal": goal.text, "modality": goal.modality,
            "end_reason": end.end_reason, "steps": oc.steps_used, "frames": len(oc.frames),
            "turns": oc.turns, "claim": claim, "verdict": asdict(verdict),
            # the judge's gate on this task: its verdict, and every refused close
            # request with the evaluator's truth where it was made (``refusals``;
            # ``refused_once`` counts the judge's refusals, not the file gate's)
            "precheck": {"mode": self.cfg.judge.precheck, "verdict": oc.meta.get("precheck_verdict"),
                         "close_requests": oc.meta.get("close_requests", 1),
                         "refused_once": any(r.get("by") == "judge" for r in oc.meta.get("refusals") or []),
                         "refusals": list(oc.meta.get("refusals") or [])},
            "resets": list(oc.meta.get("resets") or []),  # recoveries inside this task
            "wakeup": end.wakeup,                       # boundary event: the body is moved AFTER this task
            "after_wakeup": self._after_wakeup,         # this task is the first one after a body move
            "truth": end.score,                         # env metrics, never shown to the agent
            "ledger_row": row.render(),
            "context": {"compact": self.cfg.session.compact,
                        "peak_prompt_tokens": oc.peak_prompt_tokens,
                        "session_seq": run.seq if run else None, "session_hops": run.hops if run else 0,
                        "session_turns": run.turns if run else 0,
                        "judge_calls": judge_calls, "judge_prompt_tokens": judge_pt,
                        "judge_completion_tokens": judge_ct, "gate_reused": gate_reused,
                        "aux_calls": int((used.get("wakeup") or {}).get("calls", 0)),
                        "aux_prompt_tokens": int((used.get("wakeup") or {}).get("prompt_tokens", 0)),
                        "tokens": used,
                        **{key: v for key, v in oc.meta.items() if key not in ("precheck_full", "refusals")}},
            "tool_calls": oc.tool_calls,
            "episode_id": goal.meta.get("episode_id"),  # the env episode this task belongs to
            "subtask": goal.meta.get("subtask"),        # its sub-goal index in that episode
        })
        self._after_wakeup = bool(end.wakeup)

    # ── the end ──────────────────────────────────────────────────────

    def close(self, status: str) -> dict:
        # Close the task stream FIRST: every event after this point belongs to
        # the run, not to the task that happened to be last.
        self.rec.close_stream()
        self.agent_session.close()
        self.tally.take()                           # nothing after the last row belongs to a task
        notes_report = None
        if self.cfg.memory.notes:
            self.consolidate_state.reset(-1, writable=list(NOTE_PATTERNS))
            notes_report = self.consolidate_session.consolidate(self.ws, self.consolidate_state, self.notes)
            self._consolidations += 1
            self.rec.emit("notes_consolidated", at="run_end", house=self.ws.house.name, **notes_report)
        self.rec.emit("exit", exit_status=status, tasks=len(self.rec.rows))
        header = {"harness": self.harness_label, "exit_status": status,
                  "transport": self.cfg.session.transport, "config": self.cfg.flat(),
                  "model": self.model, "judge": self.cert.model_name or self.pre.model_name,
                  "judge_transport": self.judge_transport,
                  "persist_map": self.cfg.memory.persist_map,
                  "session_policy": {"session_mode": self.cfg.session.session_mode,
                                     "ledger": self.cfg.memory.ledger, "held": dict(self.cfg.held)},
                  "tools": self.tool_names(), "bridge": str(self.bridge),
                  "ledger_redactions": self.ledger.redactions, "notes": notes_report,
                  "consolidations": self._consolidations,
                  "workspace": {"root": str(self.ws.root), "house": str(self.ws.house),
                                "handovers": len(self.ws.handovers()),
                                "ledger_lines": _lines(self.ws.ledger),
                                "notes": self.notes.list_notes()},
                  "sessions": self._sessions_report(), "trace": self.trace.stats(),
                  "run_tokens": self.tally.take()}        # the run-end consolidation's usage
        return self.rec.write_summary(header, self.bench.chain_report())

    def _sessions_report(self) -> dict:
        """The kinds of session this chain opened, by class, with how many times
        each was opened and the model and core that held it — a judge on
        another core is a different run, and the record says so."""
        kinds = [self.agent_session, self.pre.session, self.cert.session, self.driver.wake,
                 self.consolidate_session]
        out: dict[str, Any] = {}
        for s in kinds:
            if s is None:
                continue
            out[s.kind] = {**type(s).describe(), "class": type(s).__name__, "opened": s.runs,
                           "model": s.model, "transport": s.transport}
        return out


def agent_tool_names(cfg: HarnessConfig) -> list[str]:
    """The tools the agent's session is opened with under ``cfg``: what its
    bridge registers and what the CLI is allowed to call."""
    if cfg.context_only:
        return tool_names(toolset="act")
    return tool_names(recovery=bool(cfg.ladder.recovery), instruments=int(cfg.task.instruments or 0))


def _with_views(text: str, views: list[str]) -> str | Blocks:
    """The instruction as one message: the text, then the look-around views when the body was moved."""
    if not views:
        return text
    blocks: Blocks = [text_block(text)]
    blocks += [image_block(b) for b in views]
    blocks.append(text_block(f"[{len(views)} look-around view(s): front, right, back, left]"))
    return blocks


def _with_text(prompt: str | Blocks, text: str) -> str | Blocks:
    """The message with ``text`` after it: a string grows, a block list gains a text block."""
    if isinstance(prompt, str):
        return f"{prompt}\n\n{text}" if prompt else text
    return [*prompt, text_block(text)]


def _latest_description(hop: Hop) -> str:
    """The session's latest words in a hop: what it last said about what it
    sees (the protocol word DONE is not a description)."""
    for item in reversed(hop.story or []):
        text = str(item.get("text") or "").strip() if item.get("role") == "assistant" else ""
        if text and text.upper() != "DONE":
            return text
    text = (hop.text or "").strip()
    return "" if text.upper() == "DONE" else text


def _lines(p: Path) -> int:
    try:
        return sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0
