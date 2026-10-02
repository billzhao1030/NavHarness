"""How NavHarness meets this runner's environment.

``env/bench.py`` drives an env through two objects: ``EnvClient``
(``call(fn, inputs, config)`` over HTTP) and ``Bench`` (how a chain is
seated, how a task ends, what the next goal is). Both speak ``auto_host``
servers (``/call/<nodeset>__<fn>``, GOAT / IVLN task switching inside the
server). This module supplies the two objects for ``core.envserver`` envs —
one verb per public method, ``POST /<verb>`` with the kwargs as JSON, the
runner having already placed the episode — so the rest of the harness is
unchanged.

Semantics on the EmbodiedScore-envs stack (the lhnav_es arm's env):

* a runner episode is one chain. A GOAT episode (``n_goals`` > 1) is N
  tasks, one per sub-goal, closed with SUBTASK_STOP (action 6); a
  single-goal line (R2R, ObjectNav …) is a chain of one task, closed with
  STOP (action 0). The harness always issues ``step {"action": 0}`` to
  close (``close_task`` / the turn-budget force-stop); ``EsEnvClient``
  rewrites it to the line's closing action.
* the next goal is read off ``goal_spec`` (the env's current sub-goal);
  the statement is worded exactly as ``core.shapes.es.es_goal`` words it
  for the cc / codex / mini seats, so every harness reads the same goal.
* ``persist_map=False`` (episodic memory) asks the env for
  ``reset_spatial`` at each boundary: the SLAM session, map and markers are
  rebuilt in place, the body does not move.
* checkpoints, resume and replay work: the arm's env answers ``dump_spatial``
  / ``load_spatial`` / ``get_body_pose`` / ``set_body_pose`` /
  ``subtask_cursor`` / ``skip_subtask``. An ES batch episode runs once and
  needs none of it; a NavHarness chain is many tasks long, a wall clock can cut
  one short, and a replayed task is only the same task if it starts from the
  state it actually started from. A resume restores the STORY (the ledger and
  the cursor) and a checkpointed replay also restores the HOUSE (map, markers,
  body); a resume without checkpoints meets a fresh map, and says so.
* multi-episode chains (``chain_ids``: several GOAT episodes of one house, a
  deployment day) are placed one after another with the map kept; resuming
  one part-way is refused, not silently skipped.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests

from .bench import GOATBench
from .types import EnvUnavailable, Goal, TaskEnd

# the harness's short verbs → this runner's env verbs (core.envserver: one
# public method each). Anything not listed goes through under its own name
# and, on an env that lacks it, fails like any other HTTP error.
VERBS = {
    "observe": "observe",
    "step": "step",
    "get_map": "get_map",
    "mark": "mark",
    "preview_path": "preview_path",
    "look_around": "look_around",
    "get_goal": "goal_spec",
    "evaluate": "evaluate",
    # the evaluator's truth where a close request was refused, without the STOP
    # (scoring only; the orchestrator keeps it in the task's record)
    "stop_truth": "stop_truth",
    "reset_spatial": "reset_spatial",
    # checkpoints / resume / replay: the harness's chain is many tasks long, so
    # unlike an ES batch episode it needs to be able to freeze a boundary and
    # start again from it. The arm's env answers all five under their own names.
    "dump_spatial": "dump_spatial",
    "load_spatial": "load_spatial",
    "get_body_pose": "get_body_pose",
    "set_body_pose": "set_body_pose",
    "subtask_cursor": "subtask_cursor",
    "skip_subtask": "skip_subtask",
    # IVLN-CE tours: the chain walks the tour, one task per episode
    "advance_tour": "advance_tour",
    "tour_report": "tour_report",
    "place_next": "place_next",
}
STOP, SUBTASK_STOP = 0, 6

#: what a relocation's record keeps of the env's transit report: how the
#: tracker came through the jump (evaluation figures, never shown to the agent)
TRANSIT_KEYS = ("slam", "reloc_error_m", "pose_jump_m", "lost_ticks", "spin_ticks", "epochs", "level",
                "explored_m2")


def relocation(episode: Any, persist_map: bool, transit: dict[str, Any]) -> dict[str, Any]:
    """``TaskEnd.wakeup`` for a body put down at the next start rather than
    walked there: where it was put down, whether the map came with it, and the
    transit report's figures. The orchestrator opens the next task with the
    four look-around views and the instruction to work out which room this
    is."""
    return {"episode": episode, "map_carried_over": bool(persist_map),
            **{k: v for k, v in transit.items() if k in TRANSIT_KEYS}}


class EsEnvClient:
    """``bench.EnvClient``'s surface over ``core.envserver``.

    ``call`` posts ``inputs`` as the verb's kwargs and lifts the reply's
    ``info`` / ``result`` / ``goal`` dicts to the top level exactly as
    ``EnvClient`` does (the tools read ``out["explored_m2"]`` whether the env
    nested it or not). ``panel`` / ``action`` are env-panel writes; the
    runner seats the episode, so they are no-ops here."""

    DEAD_AFTER = int(os.environ.get("NAVH_ENV_DEAD_AFTER", "8"))  # consecutive failures = dead

    def __init__(self, url: str, timeout: int = 900) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._fails = 0
        self.stop_action = STOP  # the line's closing move; GOATEsBench sets it at seat()
        requests.get(f"{self.url}/health", timeout=10).raise_for_status()

    def call(self, fn: str, inputs: dict | None = None, config: dict | None = None) -> dict:
        kwargs = dict(inputs or {})
        if fn == "step" and int(kwargs.get("action", -1)) == STOP:
            kwargs["action"] = self.stop_action  # close_task / force-stop → the line's closing move
        try:
            r = requests.post(f"{self.url}/{fn}", json=kwargs, timeout=self.timeout)
            r.raise_for_status()
            out = r.json()
        except Exception as exc:
            self._fails += 1
            if self._fails >= self.DEAD_AFTER:
                raise EnvUnavailable(
                    f"{self._fails} consecutive failures, last on {fn}: {type(exc).__name__}: {exc}"[
                        :300
                    ]
                ) from exc
            raise
        self._fails = 0
        if isinstance(out, dict):
            if fn == "look_around" and out.get("view_front") is not None:
                # the slam arms answer four named ports; the tools read a ``views`` list
                out.setdefault(
                    "views",
                    [out[k] for k in ("view_front", "view_right", "view_back", "view_left")],
                )
            for sub in ("info", "result", "goal"):
                inner = out.get(sub)
                if isinstance(inner, dict):
                    for k, v in inner.items():
                        out.setdefault(k, v)
        return out

    def panel(self, name: str, value: Any) -> None:  # the runner placed the episode
        return None

    def action(self, name: str) -> None:
        return None


def goal_statement(spec: dict[str, Any]) -> tuple[str, str]:
    """(statement, modality) of the goal in force — worded as
    ``core.shapes.es.es_goal`` words it for the other seats, so the
    harnesses differ in their loop, not in the goal they were given."""
    goal = spec.get("goal") or {}
    kind = str(goal.get("kind") or "none")
    text = str(spec.get("task") or "").strip()
    if kind == "object":
        return (
            f'find a "{str(goal.get("category") or "").replace("_", " ")}" (any instance counts)',
            "object",
        )
    if kind == "description":
        return f'find the object this describes — "{goal.get("description")}"', "description"
    if kind == "image":
        return (
            "find the specific object shown in the photo (that very instance, not just anything of its kind)",
            "image",
        )
    return text, "instruction"


class GOATEsBench(GOATBench):
    """One runner episode = one chain, on an env the runner already placed.

    Subclasses ``GOATBench`` (rather than ``Bench``) because the tools branch
    on that class for ``get_goal`` — the branch that reads the current goal
    off the env instead of re-resetting it, which is what a placed episode
    needs on every line, GOAT or not; the orchestrator words the bridge's
    benchmark from ``isinstance(bench, GOATBench)``, so the IR2R-CE tour bench
    below reads its goal the same way."""

    NODESET = ""

    def __init__(
        self, env: EsEnvClient, *, episode_id: str = "", scene_id: str = "", **kw: Any
    ) -> None:
        super().__init__(env, **kw)
        self.episode_id = episode_id
        self.scene_id = scene_id
        self.n_goals = 1
        # The house is known before the seat: the memory directory is mounted
        # when the chain is built, and it must be the HOUSE's (the scene's stem,
        # the name every other record of it uses), not this chain's number —
        # a deployment's days are different chains in the same house.
        if scene_id:
            self.house_id = Path(str(scene_id)).name.split(".")[0]

    def fn(self, name: str) -> str:
        return VERBS.get(name, name)

    # ── seating: the runner did it; read what is in force ──

    def seat(self, chain_id: int | list[int], persist_map: bool, skip: int = 0) -> Goal:
        """Read the episode the runner placed. A chain is one episode, or —
        a deployment day — several episodes of the same house (``chain_ids``,
        the first being the one the runner placed): when one episode's last
        sub-goal closes, the next is placed with the map kept
        (``_seat_next_episode``)."""
        self.chain_ids = [int(c) for c in (chain_id if isinstance(chain_id, (list, tuple)) else [chain_id])]
        self.chain_pos = 0
        self.offset = 0
        self.episode_index = self.chain_ids[0]
        spec = self.env.call(self.fn("get_goal"), {})
        if "error" in spec:
            raise RuntimeError(f"goal_spec: {spec['error']}")
        self.n_goals = int(spec.get("n_goals") or 1)
        self.env.stop_action = SUBTASK_STOP if self.n_goals > 1 else STOP
        # The house a chain's memory belongs to. The scene arrives as a PATH
        # (data/scene_datasets/hm3d/val/<nnnnn>-<scene>/<scene>.basis.glb);
        # its stem is the name every other record of this house already uses, so
        # a memory directory written here is the one a later run reads.
        scene = str(self.scene_id or spec.get("scene_id") or "")
        stem = Path(scene).name.split(".")[0] if scene else ""
        self.house_id = stem or scene or f"episode-{self.episode_index}"
        if skip:
            if len(self.chain_ids) > 1:
                raise NotImplementedError("resuming a multi-episode chain is not supported; "
                                          "re-run the day")
            spec = self._skip_to(int(skip), spec)
        return self._goal_of(spec)

    def _skip_to(self, skip: int, spec: dict[str, Any]) -> dict[str, Any]:
        """Walk the cursor past the tasks a resumed ledger already holds.

        What survives a killed job is what was written down — the ledger rows in
        the previous run's records. The env does not survive it, so a resumed
        chain meets a fresh episode whose cursor is at sub-goal 0 while the
        ledger says K are done. Without this the env replays those K, every
        index is written twice and the walking is redone for nothing.

        The map does not come back either; only a checkpointed run can restore
        that (``replay_from``). A resume continues the story, not the house."""
        left = min(int(skip), max(0, self.n_goals - int(spec.get("goal_index") or 0)))
        for _ in range(left):
            out = self.env.call(self.fn("skip_subtask"), {})
            if isinstance(out, dict) and out.get("error"):
                raise RuntimeError(f"skip_subtask: {out['error']}")
            if isinstance(out, dict) and out.get("episode_over"):
                break
        return self.env.call(self.fn("get_goal"), {})

    def _seat_next_episode(self, persist_map: bool) -> tuple[Goal, dict]:
        """Place the next episode of the chain in the same house; return
        (its first goal, the relocation report).

        The body is put down at the episode's prescribed start: a relocation,
        not a walk. With ``persist_map`` the map, the markers and the world
        frame carry over and the tracker re-finds the body; without it the
        episode starts on a fresh map."""
        self.offset += self.total
        self.chain_pos += 1
        eid = self.chain_ids[self.chain_pos]
        verb = "place_next" if persist_map else "place"
        meta = self.env.call(self.fn(verb), {"index": eid})
        if not isinstance(meta, dict) or meta.get("error"):
            raise EnvUnavailable(f"{verb}({eid}): {(meta or {}).get('error') if isinstance(meta, dict) else meta}")
        self.episode_index = eid
        self.episode_id = str(meta.get("episode_id") or "")
        wake = relocation(eid, persist_map, dict(meta.get("transit") or {}))
        spec = self.env.call(self.fn("get_goal"), {})
        if "error" in spec:
            raise RuntimeError(f"goal_spec: {spec['error']}")
        self.n_goals = int(spec.get("n_goals") or 1)
        return self._goal_of(spec), wake

    def _goal_of(self, spec: dict[str, Any]) -> Goal:
        self.sub = int(spec.get("goal_index") or 0)
        self.total = int(spec.get("n_goals") or 1)
        # numbered across the whole chain: a deployment day's second episode
        # continues the first's numbering (the record keeps one row per index)
        self.chain_total = self.offset + self.total
        text, modality = goal_statement(spec)
        return Goal(
            index=self.offset + self.sub,
            total=self.chain_total,
            text=text,
            image_b64=spec.get("goal_image"),
            modality=modality,
            meta={
                "category": (spec.get("goal") or {}).get("category"),
                "episode": self.episode_index,
                "episode_id": self.episode_id or spec.get("episode_id"),
                "subtask": self.sub,
                "goal_kind": (spec.get("goal") or {}).get("kind"),
            },
        )

    def current_goal(self) -> Goal:
        spec = self.env.call(self.fn("get_goal"), {})
        if "error" in spec:
            return Goal(index=self.offset + self.sub, total=self.chain_total, text="",
                        meta={"error": spec["error"]})
        if spec.get("episode_over"):
            return Goal(index=self.offset + self.total, total=self.chain_total, text="",
                        meta={"episode_over": True})
        return self._goal_of(spec)

    # ── how a task ends ──

    def parse_step(self, out: dict, claimed: str | None) -> tuple[dict, TaskEnd | None]:
        info = out.get("info") or {}
        terminated = bool(out.get("terminated"))
        truncated = bool(out.get("truncated"))
        if info.get("subtask_closed"):
            # the sub-goal just closed, scored by the package at the closing pose
            score = dict(info.get("subtask_result") or {})
            end = TaskEnd(end_reason="stop", score=score, chain_finished=terminated or truncated)
            if not end.chain_finished:
                self.sub = int(info.get("goal_index", self.sub + 1))
            return info, end
        if terminated or truncated:
            score = dict(info.get("metrics") or out.get("metrics") or {})
            return info, TaskEnd(
                end_reason="budget" if truncated else "stop", score=score, chain_finished=True
            )
        return info, None

    def after_end(self, end: TaskEnd, persist_map: bool) -> TaskEnd:
        if end.chain_finished:
            # this episode is done; more episodes in the chain (a deployment day)?
            if self.chain_pos + 1 < len(self.chain_ids):
                end.chain_finished = False
                end.next_goal, end.wakeup = self._seat_next_episode(persist_map)
            return end
        if not persist_map:
            # episodic memory: SLAM session, map and markers rebuilt in place
            out = self.env.call(self.fn("reset_spatial"), {})
            if isinstance(out, dict) and out.get("error"):
                raise RuntimeError(f"reset_spatial: {out['error']}")
        nxt = self.current_goal()
        if self._exhausted(nxt):
            if self.chain_pos + 1 < len(self.chain_ids):
                end.next_goal, end.wakeup = self._seat_next_episode(persist_map)
                return end
            end.chain_finished = True
            end.next_goal = None
            return end
        end.next_goal = nxt
        return end

    def chain_report(self) -> dict:
        # the line's metrics, driver-side (the runner scores the episode the
        # same way after the session; this copy lands in the harness's own record)
        out = self.env.call(self.fn("evaluate"), {"trigger": "harness"})
        return out if isinstance(out, dict) else {}


class TourEsBench(GOATEsBench):
    """One IVLN-CE tour = one chain; a task is one of its episodes.

    What a tour asks is what carries between its episodes, so it is the chain
    and not the episode that has to be the unit: the ledger, the place graph
    and the house memory then span the tour, which is the thing being measured.
    The boundary is the package's own oracle transit — the body is walked to
    the next start, not teleported, and the map, the markers and the world
    frame go with it (``advance_tour``). When the oracle cannot reach the
    start it puts the body there (the transit report's ``teleported``): that
    boundary is a relocation, recorded as a deployment day's next episode is.

    Subclasses ``GOATEsBench`` for its goal reading and its step parsing, which
    are the same here; what differs is the boundary. A single-goal episode ends
    on STOP, so ``parse_step`` already reports the task finished — this class
    says whether the CHAIN is finished, which it is only when the tour runs out.
    """

    def seat(self, chain_id: int | list[int], persist_map: bool, skip: int = 0) -> Goal:
        if isinstance(chain_id, (list, tuple)):
            chain_id = int(chain_id[0])
        # the runner placed the tour (core.shapes es.tour); this reads what is
        # in force, exactly as the episode-major bench does
        goal = super().seat(chain_id, persist_map, skip=0)
        if skip:
            for _ in range(int(skip)):
                out = self.env.call(self.fn("advance_tour"), {}) or {}
                if out.get("error"):
                    raise RuntimeError(f"advance_tour: {out['error']}")
                if out.get("tour_over"):
                    break
            goal = self._goal_of(self.env.call(self.fn("get_goal"), {}))
        self.slot = int(skip)
        return goal

    def _goal_of(self, spec: dict[str, Any]) -> Goal:
        """A task is an EPISODE of the tour, so it is numbered by the slot.

        The episode-major bench numbers a task by its sub-goal index, which on
        a single-goal line is 0 for every episode — and the record keeps one
        row per index, so a whole tour's rows would collapse onto one."""
        goal = super()._goal_of(spec)
        self.slot = int(spec.get("tour_slot", getattr(self, "slot", 0)))
        self.total = int(spec.get("tour_size") or 1)
        goal.index, goal.total = self.slot, self.total
        goal.meta = {**goal.meta, "tour_slot": self.slot, "tour_size": self.total}
        return goal

    def parse_step(self, out: dict, claimed: str | None) -> tuple[dict, TaskEnd | None]:
        """An episode ending ends the TASK; only a finished tour ends the chain.

        The episode-major bench reads terminated/truncated as the end of
        everything, which is right when the chain is one episode long. Here it
        is the boundary — ``after_end`` walks to the next episode and only
        closes the chain when the tour runs out."""
        info, end = super().parse_step(out, claimed)
        if end is not None:
            end.chain_finished = False
        return info, end

    def after_end(self, end: TaskEnd, persist_map: bool) -> TaskEnd:
        """A task ended; walk the tour to its next episode, or close the chain.

        ``persist_map=False`` (episodic memory) means what it means elsewhere —
        the SLAM session and the map are rebuilt in place — and is applied
        before the walk. A transit the oracle ended by placing the body sets
        ``end.wakeup``; a walked one is no relocation."""
        if not persist_map:
            out = self.env.call(self.fn("reset_spatial"), {})
            if isinstance(out, dict) and out.get("error"):
                raise RuntimeError(f"reset_spatial: {out['error']}")
        out = self.env.call(self.fn("advance_tour"), {}) or {}
        if out.get("error"):
            raise EnvUnavailable(f"advance_tour: {out['error']}")
        if out.get("tour_over"):
            end.chain_finished = True
            end.next_goal = None
            return end
        self.slot = int(out.get("tour_slot", self.slot + 1))
        end.chain_finished = False
        end.next_goal = self._goal_of(self.env.call(self.fn("get_goal"), {}))
        transit = dict(out.get("transit") or {})
        if transit.get("teleported"):
            end.wakeup = {**relocation(self.slot, persist_map, transit), "teleported": True}
        return end

    def chain_report(self) -> dict:
        out = self.env.call(self.fn("tour_report"), {})
        return out if isinstance(out, dict) else {}


__all__ = [
    "STOP",
    "SUBTASK_STOP",
    "TRANSIT_KEYS",
    "VERBS",
    "EsEnvClient",
    "GOATEsBench",
    "TourEsBench",
    "goal_statement",
    "relocation",
]
