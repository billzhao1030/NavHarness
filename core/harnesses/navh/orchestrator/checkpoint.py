"""Checkpoints — freeze the world between two tasks; resume and replay from what was frozen.

Three things, none of which touch the model:

* ``freeze(k)`` — before task ``k`` starts: map + markers (the env's own dump),
  the body pose (simulator truth, read only by a replay), the subtask cursor,
  the place graph, and a snapshot of the house's long-term memory.
* ``replay_task()`` / ``restore(k)`` — a faithful re-run of ONE task from its
  frozen boundary (``replay_from`` + ``replay_task``): the ledger up to ``k``,
  the memory as it stood, the map and the body put back in that order.
* ``replay_ledger()`` — a run a wall clock cut short (``resume_from``): the
  ledger rows are rebuilt from the previous run's records; the map is not,
  it lived in the env process and is rebuilt by walking.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ..config import HarnessConfig
from ..env.bench import Bench
from ..memory.mount import house_slug
from ..memory.placegraph import PlaceGraph
from .ledger import Ledger, TaskRow


class Checkpoints:
    def __init__(self, cfg: HarnessConfig, bench: Bench, ledger: Ledger, out_dir: Path, rec: Any) -> None:
        self.cfg = cfg
        self.bench = bench
        self.ledger = ledger
        self.out_dir = out_dir
        self.rec = rec

    # ── replay one task ──────────────────────────────────────────────

    def replay_task(self) -> int | None:
        """Set up a faithful re-run of ONE task, or return None if not replaying.

        Returns the number of tasks to seat past, so the caller treats it the
        way it treats a resume. The ledger is replayed only up to that task:
        rows the original run wrote AFTER it describe a future this attempt has
        not had, and letting them into the context would be handing the agent
        the answer."""
        if self.cfg.replay.replay_task < 0 or not self.cfg.replay.replay_from:
            return None
        k = int(self.cfg.replay.replay_task)
        cp = Path(self.cfg.replay.replay_from) / "checkpoints" / f"t{k}.json"
        if not cp.exists():
            raise SystemExit(f"no checkpoint for task {k} in {self.cfg.replay.replay_from} "
                             f"(the run must have been made with --checkpoints)")
        if self.cfg.memory.notes or self.cfg.memory.archive_policy:
            # The house's memory as it stood at this boundary, never the live
            # directory: by now that holds notes written after task k, and a
            # replay reading them would be told how its own task went.
            snap = cp.with_name(f"t{k}.memory")
            if not snap.is_dir():
                raise SystemExit(f"checkpoint t{k} carries no long-term memory snapshot; replaying it "
                                 f"with notes / archive_policy on would read the memory written after it")
            root = self.out_dir / "_longterm_replay"
            if root.exists():
                shutil.rmtree(root)
            shutil.copytree(snap, root)
            self.cfg.memory.memory_root = str(root)
            self.rec.emit("memory_restored", source=str(snap), houses=sorted(p.name for p in root.iterdir()))
        prev = Path(self.cfg.replay.replay_from) / "summary.json"
        rows = json.loads(prev.read_text()).get("tasks", []) if prev.exists() else []
        for r in sorted((r for r in rows if int(r.get("index", -1)) < k),
                        key=lambda r: int(r["index"])):
            v, c = r.get("verdict") or {}, r.get("claim") or {}
            self.ledger.add(TaskRow(
                index=int(r["index"]), goal=str(r.get("goal", "")), modality=str(r.get("modality", "")),
                claim=str(c.get("status", "stop")), claim_evidence=str(c.get("evidence", "")),
                verdict=str(v.get("verdict", "none")), verdict_evidence=str(v.get("evidence", "")),
                searched=list(v.get("searched") or []), markers=[],
                summary=str(r.get("summary", "")), steps=int(r.get("steps", 0) or 0),
                end_reason=str(r.get("end_reason", "stop")), extra={"replayed": True}))
            self.rec.rows.append({**r, "replayed": True})
        self.rec.emit("replay", source=str(self.cfg.replay.replay_from), task=k,
                      ledger_rows=len([r for r in rows if int(r.get("index", -1)) < k]))
        return k

    def replay_graph(self, k: int | None) -> Path | None:
        """The place graph as it stood at the replayed boundary, if any."""
        if k is None:
            return None
        p = Path(self.cfg.replay.replay_from) / "checkpoints" / f"t{k}.places.json"
        return p if p.exists() else None

    def restore(self, k: int, places: PlaceGraph | None) -> PlaceGraph | None:
        """Put the world back: map and markers first, then the body.

        Order matters. ``load_spatial`` restores the harness's spatial memory
        without moving anything; ``set_body_pose`` moves the body and re-anchors
        SLAM there. Doing it the other way round would anchor the tracker at the
        old pose and then swap the map under it. Returns the place graph to use
        from here (the frozen one when the checkpoint carries it)."""
        d = Path(self.cfg.replay.replay_from) / "checkpoints"
        blob = json.loads((d / f"t{k}.json").read_text())
        out = self.bench.env.call(self.bench.fn("load_spatial"), {"path": str(d / f"t{k}.pkl")})
        self.rec.emit("map_restored", **(out if isinstance(out, dict) else {"out": str(out)}))
        pose = blob.get("pose") or {}
        if pose.get("position"):
            placed = self.bench.env.call(self.bench.fn("set_body_pose"),
                                         {"position": pose["position"],
                                          "rotation_xyzw": pose.get("rotation_xyzw") or [0, 0, 0, 1]})
            self.rec.emit("body_restored", **(placed if isinstance(placed, dict) else {}))
        else:
            self.rec.emit("warning", kind_detail="replay_no_pose",
                          error="checkpoint carries no pose; the body starts where reset put it")
        gp = d / f"t{k}.places.json"
        if gp.exists() and places is not None:
            return PlaceGraph.load(gp)
        return places

    # ── freeze ───────────────────────────────────────────────────────

    def freeze(self, before_task: int, places: PlaceGraph | None, places_path: Path | None) -> None:
        """Freeze the world as it stands between two tasks.

        A finished chain answers "did subtask 4 succeed" and nothing else. The
        question worth asking afterwards is the counterfactual — would it have
        succeeded with a different note, a different rung, a different budget —
        and that question is only answerable if task 4 can be started again from
        the state it actually started from. Otherwise a rerun is a different
        experiment wearing the same index: a different pose, a map that has seen
        the whole house, and a ledger that already knows how the story ends.

        So the checkpoint holds the three things that make a task's starting
        position what it was:

            map + markers   the occupancy dump, which the env already knows how
                            to write and read back
            body pose       simulator ground truth. It is written HERE and read
                            only by a replay; it never enters a prompt, a note
                            or the ledger, exactly as the map file's geometry
                            does not. A checkpoint is infrastructure, not
                            something the agent is told.
            the cursor      which subtask is next, and how much budget is spent

        The ledger prefix is not copied — the rows are already in the records,
        and a replay reads the ones before its task.

        One thing cannot be frozen: the SLAM tracker's own map lives inside the
        ORB-SLAM3 process. On replay it re-anchors at the restored pose, the way
        it does after a wake-up. So a replay reproduces the BODY and the
        harness's spatial memory exactly, and the tracker's internal state only
        approximately — which is worth saying out loud whenever a replayed
        number is compared with its original."""
        if not self.cfg.replay.checkpoints:
            return
        d = self.out_dir / "checkpoints"
        try:
            d.mkdir(parents=True, exist_ok=True)
            self.bench.env.call(self.bench.fn("dump_spatial"), {"path": str(d / f"t{before_task}.pkl")})
            # The node answers {"pose": {...}} — its output PORT is named `pose`,
            # so the position is read from inside it, not off the envelope;
            # a checkpoint without the body would restore everything except
            # where the robot was standing.
            out = self.bench.env.call(self.bench.fn("get_body_pose"), {}) or {}
            pose = out.get("pose") if isinstance(out.get("pose"), dict) else out
            if not (pose or {}).get("position"):
                self.rec.emit("warning", kind_detail="checkpoint_no_pose",
                              error=str(out)[:200])
            cur = {}
            try:
                c0 = self.bench.env.call(self.bench.fn("subtask_cursor"), {}) or {}
                cur = c0.get("cursor") if isinstance(c0.get("cursor"), dict) else c0
            except Exception:
                pass
            (d / f"t{before_task}.json").write_text(json.dumps({
                "before_task": before_task, "benchmark": self.cfg.task.benchmark,
                "chain_id": self.cfg.task.chain_id, "split": self.cfg.task.split,
                "pose": pose, "cursor": cur,
                "places": str(places_path) if places is not None else "",
            }, indent=2))
            if places is not None:
                places.save(d / f"t{before_task}.places.json")
            self._snapshot_memory(d / f"t{before_task}.memory")
            self.rec.emit("checkpoint", before_task=before_task)
        except Exception as e:
            self.rec.emit("warning", kind_detail="checkpoint", error=str(e)[:200])

    def _snapshot_memory(self, dst: Path) -> None:
        """Freeze this house's long-term notes and archive with the checkpoint.

        An empty directory is a snapshot too: at the seat, before task 0, the
        house has no memory yet, and a replay of task 0 must start with none."""
        if not (self.cfg.memory.notes or self.cfg.memory.archive_policy):
            return
        root = memory_root(self.cfg)
        src = root / house_slug(self.bench.house_id or f"{self.cfg.task.benchmark}-{seat_arg(self.cfg)}")
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        if src.is_dir():
            shutil.copytree(src, dst / src.name)

    # ── resume a cut-short run ───────────────────────────────────────

    def replay_ledger(self) -> int:
        """Rebuild the ledger of a run that a wall clock cut short.

        What survives a killed job is what was written down: the ledger rows in
        the previous run's records. The map does not — it lived in the env
        process — so the resumed run rebuilds it by walking. That asymmetry is
        the point: the task state is durable because the harness externalised
        it, the spatial state only as long as the process lives."""
        if not self.cfg.replay.resume_from:
            return 0
        by_index: dict[int, dict] = {}
        for d in str(self.cfg.replay.resume_from).split(","):
            d = d.strip()
            if not d:
                continue
            prev = Path(d)
            found: list[dict] = []
            summ = prev / "summary.json"
            if summ.exists():
                found = json.loads(summ.read_text()).get("tasks", [])
            else:
                # Killed before the summary was written. The per-task rows were
                # still emitted as they closed, so recover the FULL row from the
                # episode_metrics event rather than a stub: a resumed run has to
                # be able to report the whole tour, not just its own share of it.
                for f in sorted(prev.glob("episode_*.jsonl"), key=lambda q: int(q.stem.split("_")[-1])):
                    for line in f.read_text().splitlines():
                        if not line.strip():
                            continue
                        try:
                            e = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if e.get("kind") == "episode_metrics" and isinstance(e.get("metrics"), dict):
                            found.append(dict(e["metrics"]))
            for r in found:                    # a later attempt wins for the same task
                by_index[int(r.get("index", 0))] = r
        rows = [by_index[i] for i in sorted(by_index)]
        for r in rows:
            v = r.get("verdict") or {}
            c = r.get("claim") or {}
            self.ledger.add(TaskRow(
                index=int(r.get("index", 0)), goal=str(r.get("goal", "")), modality=str(r.get("modality", "")),
                claim=str(c.get("status", "stop")), claim_evidence=str(c.get("evidence", "")),
                verdict=str(v.get("verdict", "none")), verdict_evidence=str(v.get("evidence", "")),
                searched=list(v.get("searched") or []), markers=[],
                summary=str(r.get("summary", "")), steps=int(r.get("steps", 0) or 0),
                end_reason=str(r.get("end_reason", "stop")),
                extra={"replayed": True}))
        # The replayed tasks belong to this tour's results too. Without this the
        # summary of a resumed run covers only the tasks it happened to run
        # itself, and every aggregate silently describes a fragment.
        for r in rows:
            self.rec.rows.append({**r, "replayed": True})
        seat_at = (max(by_index) + 1) if by_index else 0
        self.rec.emit("resume", from_dirs=str(self.cfg.replay.resume_from),
                      tasks_replayed=len(rows), seat_at=seat_at)
        return seat_at


def seat_arg(cfg: HarnessConfig):
    """What the bench is seated with: the chain ids of a tour, else the chain id."""
    return cfg.task.chain_ids if cfg.task.chain_ids else cfg.task.chain_id


def memory_root(cfg: HarnessConfig) -> Path:
    """Where the house's long-term memory lives: the run's own root, or the package's."""
    return (Path(cfg.memory.memory_root) if cfg.memory.memory_root
            else Path(__file__).resolve().parents[2] / "longterm")   # core/harnesses/longterm
