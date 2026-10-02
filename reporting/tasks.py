"""Per-task outcomes of a run, keyed so that runs pair task by task.

The paper's scores are over TASKS (a GOAT-Bench sub-goal, an IR2R-CE tour
episode) and SEQUENCES (a GOAT-Bench episode, an IR2R-CE tour); the runner's
record is per sequence. This module turns a run directory into its tasks,
reading the per-task truth where the run wrote it:

``summary.json`` ``episodes[i].metrics``, the env's own scoring of sequence ``i``
    GOAT-Bench  ``subtask_success`` / ``spl_by_subtask``: one entry per sub-goal
                of the episode, 0 for a sub-goal never closed
    IR2R-CE     ``tour_episodes[*].metrics``: one row per episode the tour
                reached (an episode never reached is a failure up to the
                tour's size, ``agent.tour_size``); ``t_ndtw`` is the tour's
                t-nDTW (x100)
``navh_<i>/summary.json`` ``tasks[*]``, the harness's per-task rows
    ``truth`` is the env's score of that task (the event stream carries the
    same row as ``metrics.truth``). It is the only per-task record of a chain
    over several env episodes (a deployment day), and the rows carry the
    per-task accounting: turns, steps, peak prompt, tokens by session,
    recoveries, the claim and the verdict.

A task's key is ``(scene, sequence, slot)`` — the scene's stem, the GOAT
episode id or the IR2R-CE tour id, the sub-goal index or the position in the
tour — the same in every run of a benchmark. A sequence the runner does not
score (``runner.is_scored``: a crash, a rate-limit casualty, a wall-clock
stub that holds no metrics) has no task set and is listed in
``Replicate.unscored``, never silently dropped.

A REPLICATE is what one repetition of a configuration produced: one run
directory, or a directory of run directories (a deployment's ``runs/``, one
run per house per day) taken together.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.scores import task_scores  # noqa: E402
from runner import is_scored  # noqa: E402 — the runner's own rule for what counts


@dataclass
class Task:
    scene: str
    sequence: str               # GOAT episode id | IR2R-CE tour id
    slot: int                   # sub-goal index | position in the tour
    success: float
    spl: float
    run: str                    # the run directory's name
    index: int                  # the runner's episode index (the sequence's record)
    row: dict | None = None     # the harness's per-task row, when the task ran

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.scene, self.sequence, self.slot)


@dataclass
class Sequence:
    scene: str
    sequence: str
    run: str
    index: int
    tasks: int
    t_ndtw: float | None = None         # IR2R-CE: the tour's t-nDTW (x100); None when unscored
    t_ndtw_prefix: bool = False         # the value covers only the episodes the tour finished
    error: str | None = None
    wall_sec: float = 0.0               # the session through evaluation (consolidation included)
    setup_sec: float = 0.0              # placement and briefing before it
    run_tokens: dict = field(default_factory=dict)   # the run-end consolidation's usage
    recorded: bool = False              # the harness wrote its summary (navh_<i>/summary.json)


@dataclass
class Replicate:
    name: str
    runs: list[Path]
    tasks: list[Task] = field(default_factory=list)
    sequences: list[Sequence] = field(default_factory=list)
    unscored: list[dict] = field(default_factory=list)   # {run, index, error}
    notes: list[str] = field(default_factory=list)


def run_dirs(path: Path) -> list[Path]:
    """The run directories one argument names: itself, or the runs below it."""
    path = Path(path)
    if (path / "summary.json").is_file():
        return [path]
    if path.is_dir():
        found = sorted(p for p in path.iterdir() if (p / "summary.json").is_file())
        if found:
            return found
    raise FileNotFoundError(f"{path}: no summary.json, and no run directories below it")


def load(path: Path | str) -> Replicate:
    """One replicate: every scored sequence's tasks, keyed."""
    path = Path(path)
    rep = Replicate(name=path.name, runs=run_dirs(path))
    for run_dir in rep.runs:
        summary = json.loads((run_dir / "summary.json").read_text())
        if not isinstance(summary.get("episodes"), list):
            raise ValueError(f"{run_dir}/summary.json is not a run's summary (no episode records)")
        for rec in summary["episodes"]:
            index = int(rec.get("index", -1))
            if not is_scored(rec):
                rep.unscored.append({"run": run_dir.name, "index": index, "error": rec.get("error")})
                continue
            navh = _navh_summary(run_dir, rec)
            tasks, seq = _sequence(run_dir, rec, navh, rep.notes)
            rep.tasks += tasks
            rep.sequences.append(seq)
    seen: set[tuple[str, str, int]] = set()
    for t in rep.tasks:
        if t.key in seen:
            rep.notes.append(f"task {t.key} appears twice in {rep.name}; each run should hold it once")
        seen.add(t.key)
    return rep


# ── one sequence ─────────────────────────────────────────────────────


def scene_of(scene_id: Any) -> str:
    """A scene path's stem (``…/00800-TEEsavR23oF/TEEsavR23oF.basis.glb`` -> TEEsavR23oF)."""
    return Path(str(scene_id or "")).name.split(".")[0] or "?"


def _navh_summary(run_dir: Path, rec: dict) -> dict | None:
    index = rec.get("index")
    for d in (run_dir / f"navh_{index}", Path(str((rec.get("agent") or {}).get("navh_dir") or ""))):
        p = d / "summary.json"
        if str(d) and p.is_file():
            try:
                return json.loads(p.read_text())
            except ValueError:
                return None
    return None


def _num(v: Any) -> float:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0


def _row_slot(row: dict) -> int:
    """The sub-goal a harness row scored: written on the row, else the env's own index in its truth."""
    if row.get("subtask") is not None:
        return int(row["subtask"])
    truth = row.get("truth") or {}
    if isinstance(truth.get("index"), int):
        return int(truth["index"])
    return int(row.get("index", 0))


def _row_episode(run_dir: Path, index: int, row: dict) -> str:
    """The env episode a harness row belongs to: on the row, else in the task's own event stream."""
    if row.get("episode_id") is not None:
        return str(row["episode_id"])
    p = run_dir / f"navh_{index}" / f"episode_{row.get('index')}.jsonl"
    try:
        with p.open() as fh:
            meta = json.loads(fh.readline())
        return str(meta.get("episode_id"))
    except (OSError, ValueError):
        return "?"


def _sequence(run_dir: Path, rec: dict, navh: dict | None, notes: list[str]) -> tuple[list[Task], Sequence]:
    m = rec.get("metrics") or {}
    agent = rec.get("agent") or {}
    index = int(rec.get("index", -1))
    scene = scene_of(rec.get("scene_id"))
    rows = list((navh or {}).get("tasks") or [])
    chain = list(((navh or {}).get("config") or {}).get("chain_ids") or [])
    run = run_dir.name
    seq = Sequence(scene, str(rec.get("episode_id")), run, index, 0, error=rec.get("error"),
                   wall_sec=_num(rec.get("wall_sec")), setup_sec=_num(rec.get("setup_sec")),
                   run_tokens=dict((navh or {}).get("run_tokens") or {}), recorded=navh is not None)
    tasks: list[Task] = []
    if len(chain) > 1:
        # several env episodes in one chain (a deployment day): the harness's rows
        per_ep: dict[str, int] = {}
        for r in rows:
            truth = r.get("truth") or {}
            ep, slot = _row_episode(run_dir, index, r), _row_slot(r)
            ok, spl = _num(truth.get("success")), _num(truth.get("spl"))
            if isinstance(truth.get("subtask_success"), list):
                # a task the episode's budget ended carries the episode's lists, not its own score
                succ, spls = truth["subtask_success"], list(truth.get("spl_by_subtask") or [])
                ok = _num(succ[slot]) if slot < len(succ) else 0.0
                spl = _num(spls[slot]) if slot < len(spls) else 0.0
            tasks.append(Task(scene, ep, slot, ok, spl, run, index, r))
            per_ep[ep] = max(per_ep.get(ep, 0), int(truth.get("n_subtasks") or 0))
        have = {(t.sequence, t.slot) for t in tasks}
        for ep, n in per_ep.items():
            tasks += [Task(scene, ep, s, 0.0, 0.0, run, index) for s in range(n) if (ep, s) not in have]
        if len(per_ep) < len(chain):
            notes.append(f"{run}: the chain reached {len(per_ep)} of its {len(chain)} episodes; the tasks "
                         "of the others are unknown here and not counted")
    else:
        # the env's own scoring of the sequence: a task per sub-goal, or per tour episode
        tour = agent.get("task_shape") == "tour" or "tour_episodes" in m
        rows_by = {int(r.get("index", n)) if tour else _row_slot(r): r for n, r in enumerate(rows)}
        if tour:
            seq.sequence = str(m.get("tour_id") if m.get("tour_id") is not None else rec.get("episode_id"))
            seq.t_ndtw = m.get("t_ndtw") if isinstance(m.get("t_ndtw"), (int, float)) else None
        for s, (ok, spl) in enumerate(task_scores(m, agent) or []):
            tasks.append(Task(scene, seq.sequence, s, ok, spl, run, index, rows_by.get(s)))
        if tour:
            reached = len(m.get("tour_episodes") or [])
            seq.t_ndtw_prefix = reached < len(tasks)
            if reached < len(tasks) and "tour_size" not in m:
                notes.append(f"{run} tour {seq.sequence}: {reached} of {len(tasks)} episodes reached; "
                             "the rest count as failures")
    seq.tasks = len(tasks)
    return tasks, seq


__all__ = ["Replicate", "Sequence", "Task", "load", "run_dirs", "scene_of"]
