"""A sequence the wall-clock backstop cuts short is scored, not dropped.

Its tasks stay in the denominators: what the env scored before the cut
stands, every task it has not scored counts as failed, and a run is scored
over the benchmark's full task set. Checked against canned envs, no server.
"""

from __future__ import annotations

import asyncio
from typing import Any

from core.shapes.base import Goal
from core.shapes.es import EsGoat
from exp_workspace.lhnav_es.mcp.shapes import EsTour
from runner import Runner, _stub_record, aggregate, is_scored


class _Env:
    url = "http://canned"

    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers

    async def call(self, name: str, **kwargs: Any) -> dict[str, Any]:
        out = self.answers[name]
        if isinstance(out, Exception):
            raise out
        return out


def _runner(module: Any, goal: Goal, index: int = 4) -> Runner:
    r = Runner.__new__(Runner)
    r._inflight = {index: {"module": module, "goal": goal}}
    return r


def _goat(n: int = 5) -> Goal:
    return Goal(text="find a bed", task="goat", n_goals=n,
                ep={"episode_id": "17", "scene_id": "hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"})


def test_the_env_scores_what_it_closed_and_the_rest_are_failures() -> None:
    env = _Env({"evaluate": {"success": 0.4, "spl": 0.3, "subtask_success": [1.0, 1.0, 0.0, 0.0, 0.0],
                             "spl_by_subtask": [0.8, 0.7, 0.0, 0.0, 0.0], "n_subtasks": 5}})
    goal = _goat()
    rec = asyncio.run(_runner(EsGoat(env, {}, None), goal).timeout_record(4, 8600.0))
    assert rec["error"] == "timeout" and rec["episode_id"] == "17" and rec["scene_id"] == goal.scene_id
    m = rec["metrics"]
    assert m["subtask_success"] == [1.0, 1.0, 0.0, 0.0, 0.0] and m["n_subtasks"] == 5
    assert m["success"] == 0.4 and abs(m["spl"] - 0.3) < 1e-9 and m["composite_success"] == 0.0
    assert rec["agent"]["timeout_scored_by"] == "env" and rec["wall_sec"] == 8600.0
    assert is_scored(rec), "a cut-short sequence counts"


def test_an_env_that_cannot_answer_leaves_every_task_failed() -> None:
    env = _Env({"evaluate": RuntimeError("env hung")})
    rec = asyncio.run(_runner(EsGoat(env, {}, None), _goat(7)).timeout_record(4, 1.0))
    m = rec["metrics"]
    assert m["subtask_success"] == [0.0] * 7 and m["spl_by_subtask"] == [0.0] * 7 and m["success"] == 0.0
    assert rec["agent"]["timeout_scored_by"] == "placement" and is_scored(rec)


def test_a_sequence_never_placed_stays_an_unscored_stub() -> None:
    r = Runner.__new__(Runner)
    r._inflight = {}
    rec = asyncio.run(r.timeout_record(9, 1.0))
    assert rec == _stub_record(9, "timeout") and not is_scored(rec)


def test_a_cut_short_tour_scores_its_whole_size_and_leaves_t_ndtw_unscored() -> None:
    rows = [{"slot": 0, "episode_id": "a", "metrics": {"success": 1.0, "spl": 0.9}},
            {"slot": 1, "episode_id": "b", "metrics": {"success": 0.0, "spl": 0.0}}]
    env = _Env({"tour_report": {"tour_id": "t7", "episodes": rows, "n_episodes": 2, "success": 0.5,
                                "t_ndtw": 61.2}})
    goal = Goal(text="walk", task="nav", ep={"episode_id": "a", "scene_id": "mp3d/x.glb", "tour_size": 4})
    rec = asyncio.run(_runner(EsTour(env, {}, None), goal).timeout_record(4, 1.0))
    m = rec["metrics"]
    assert m["tour_size"] == 4 and m["success"] == 0.25 and abs(m["spl"] - 0.225) < 1e-9
    assert m["t_ndtw"] is None and m["t_ndtw_prefix"] == 61.2 and len(m["tour_episodes"]) == 2
    assert rec["agent"]["task_shape"] == "tour" and rec["agent"]["tour_size"] == 4 and is_scored(rec)


def test_other_failures_keep_the_rules_they_had() -> None:
    crash = _stub_record(1, "RuntimeError('boom')")
    throttled = {**_stub_record(2, "rate_limited"), "metrics": {"success": 0.0}}
    exhausted = {"index": 3, "error": "max turns", "metrics": {"success": 0.0},
                 "agent": {"env_steps": 40, "called_stop": False}}
    assert not is_scored(crash) and not is_scored(throttled) and is_scored(exhausted)
    cut = {**_stub_record(4, "timeout"), "metrics": {"success": 0.0, "spl": 0.0}}
    assert aggregate([e for e in (crash, cut, exhausted) if is_scored(e)])["episode_count"] == 2
