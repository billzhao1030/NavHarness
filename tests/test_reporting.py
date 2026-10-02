"""The reporters against synthetic run directories: per-task truth, the
paper's score definitions, the accounting, and the paired bootstrap.

A run directory here is what the runner and the harness write — the
runner's ``summary.json`` (one record per sequence) and each sequence's
``navh_<i>/summary.json`` (the harness's per-task rows) — written by hand,
so every expected number can be worked out on paper.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from reporting import paired_ci, summarize
from reporting.tasks import load, run_dirs

SCENE = "data/scene_datasets/hm3d/val/00800-{s}/{s}.basis.glb"


def _row(slot: int, ok: float, *, turns: int = 40, peak: int = 20000, modality: str = "object",
         claim: str = "complete", verdict: str = "complete", resets: list | None = None,
         episode: str | None = None, marks: int = 1) -> dict[str, Any]:
    return {"index": slot, "modality": modality, "turns": turns, "steps": 100 + slot,
            "claim": {"status": claim}, "verdict": {"verdict": verdict}, "resets": resets or [],
            "truth": {"index": slot, "success": ok, "spl": 0.5 * ok, "n_subtasks": 3},
            "context": {"peak_prompt_tokens": peak,
                        "tokens": {"agent": {"calls": 4, "prompt_tokens": 1000, "cached_tokens": 600,
                                             "completion_tokens": 50, "peak_prompt_tokens": peak},
                                   "certify": {"calls": 1, "prompt_tokens": 300, "cached_tokens": 0,
                                               "completion_tokens": 20, "peak_prompt_tokens": 300}}},
            "tool_calls": {"mark": marks}, "episode_id": episode, "subtask": slot}


def _goat_record(index: int, scene: str, ep: str, succ: list[float], *, wall: float = 100.0,
                 error: str | None = None) -> dict[str, Any]:
    rec = {"index": index, "episode_id": ep, "scene_id": SCENE.format(s=scene),
           "metrics": {"success": sum(succ) / len(succ), "spl": 0.5 * sum(succ) / len(succ),
                       "subtask_success": succ, "spl_by_subtask": [0.5 * s for s in succ],
                       "n_subtasks": len(succ), "composite_success": float(all(succ))},
           "agent": {"env_steps": 300, "task_shape": "goat"}, "wall_sec": wall, "setup_sec": 10.0}
    if error:
        rec["error"] = error
    return rec


def _write_run(root: Path, name: str, records: list[dict], rows: dict[int, list[dict]],
               config: dict | None = None, run_tokens: dict | None = None) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"run_name": name, "episodes": records}))
    for i, rs in rows.items():
        (d / f"navh_{i}").mkdir()
        (d / f"navh_{i}" / "summary.json").write_text(json.dumps(
            {"config": config or {"chain_ids": []}, "tasks": rs, "run_tokens": run_tokens or {}}))
    return d


def _goat_run(root: Path, name: str = "goat") -> Path:
    """Two scenes: an episode of 3 sub-goals (2 succeed), one of 1 (succeeds),
    a cut-short episode of 2 (none scored), and a crash that is not scored."""
    records = [
        _goat_record(0, "A", "10", [1.0, 0.0, 1.0]),
        _goat_record(1, "B", "11", [1.0]),
        {**_goat_record(2, "B", "12", [0.0, 0.0], wall=8600.0, error="timeout"),
         "agent": {"env_steps": 0, "task_shape": "goat", "end_reason": "wall_clock"}},
        {"index": 3, "error": "RuntimeError('boom')", "metrics": {}, "agent": {"env_steps": 0}},
    ]
    cap = {"trigger": "cap", "attempt_turns": 80}
    mine = {"trigger": "self", "attempt_turns": 30}
    rows = {0: [_row(0, 1.0, resets=[mine]), _row(1, 0.0, verdict="incomplete", resets=[cap], modality="image"),
                _row(2, 1.0, claim="incomplete", verdict="incomplete")],
            1: [_row(0, 1.0, turns=10, peak=10000, verdict="incomplete")]}
    return _write_run(root, name, records, rows, run_tokens={"consolidate": {"calls": 1, "prompt_tokens": 5000}})


# ── the scores ──


def test_task_success_is_pooled_and_every_task_of_a_scored_sequence_counts(tmp_path: Path) -> None:
    rep = load(_goat_run(tmp_path))
    assert len(rep.tasks) == 6, "3 + 1 + the cut-short episode's 2; the crash has no task set"
    assert rep.unscored == [{"run": "goat", "index": 3, "error": "RuntimeError('boom')"}]
    s = summarize.scores(rep)
    assert s["s_sr"] == pytest.approx(100 * 3 / 6), "pooled over tasks, not a mean of episode means (0.5)"
    assert s["spl"] == pytest.approx(100 * 1.5 / 6)
    assert s["e_sr"] == pytest.approx(100 / 3) and s["sequences"] == 3
    assert {t.key for t in rep.tasks if t.sequence == "12"} == {("B", "12", 0), ("B", "12", 1)}


def test_the_accounting_per_task(tmp_path: Path) -> None:
    rep = load(_goat_run(tmp_path))
    a = summarize.accounting(rep)
    assert a["tasks_with_record"] == 4
    assert a["turns_per_task"] == pytest.approx((40 * 3 + 10) / 4)
    assert a["peak_prompt_tokens_per_task"] == pytest.approx((20000 * 3 + 10000) / 4)
    assert a["tokens_per_task"]["agent"]["prompt_tokens"] == 1000 and a["tokens_per_task"]["certify"]["calls"] == 1
    assert a["consolidation_per_sequence"]["prompt_tokens"] == 5000
    assert a["wall_sec_per_task"] == pytest.approx((110 + 110 + 8610) / 6), "every sequence's time over every task"
    rc = a["recoveries"]
    assert rc["events"] == 2 and rc["tasks_with_recovery_pct"] == 50.0 and rc["events_leading_to_success_pct"] == 50.0
    assert rc["requested"]["share_pct"] == 50.0 and rc["requested"]["median_attempt_turns"] == 30
    assert rc["requested"]["success_pct"] == 100.0 and rc["fallback"]["success_pct"] == 0.0
    cert = a["certification"]
    # claimed completions: (ok, certified) (fail, rejected) (ok, rejected)
    assert cert["claimed_completions"] == 3 and cert["accuracy_pct"] == pytest.approx(200 / 3)
    assert cert["accept_all_accuracy_pct"] == pytest.approx(200 / 3)
    nxt = a["successor"]
    # episode 10: slot 0 (true record) -> slot 1 fails; slot 1 (rejected, failed: true) -> slot 2 succeeds
    assert nxt["after_true_record_n"] == 2 and nxt["after_true_record_pct"] == 50.0
    assert nxt["after_false_record_n"] == 0
    b = summarize.breakdowns(rep)
    assert b["by_modality"]["image"]["s_sr"] == 0.0 and b["by_position"][1]["tasks"] == 3


def _tour_record(index: int, tour: str, eps: list[float], size: int, t_ndtw: float | None) -> dict:
    rows = [{"slot": s, "episode_id": f"{tour}-{s}", "metrics": {"success": ok, "spl": 0.8 * ok}}
            for s, ok in enumerate(eps)]
    return {"index": index, "episode_id": f"{tour}-0", "scene_id": "mp3d/S1/S1.glb",
            "metrics": {"tour_id": tour, "tour_episodes": rows, "n_episodes": len(rows), "t_ndtw": t_ndtw,
                        "success": sum(eps) / len(eps)},
            "agent": {"env_steps": 50, "task_shape": "tour", "tour_size": size}, "wall_sec": 10.0}


def test_ir2r_tours_score_every_episode_and_t_ndtw_per_tour(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "ivln", [_tour_record(0, "t1", [1.0, 1.0], 2, 60.0),
                                        _tour_record(1, "t2", [1.0, 0.0], 4, 40.0)], {})
    rep = load(run)
    s = summarize.scores(rep)
    assert s["tasks"] == 6 and s["s_sr"] == pytest.approx(100 * 3 / 6), "two episodes of t2 never reached: failures"
    assert s["e_sr"] == 50.0 and s["t_ndtw"] == 50.0 and s["t_ndtw_tours"] == 2
    assert any("2 of 4 episodes reached" in n for n in rep.notes)


def test_a_deployment_is_one_replicate_over_its_day_runs(tmp_path: Path) -> None:
    root = tmp_path / "deploy" / "runs"
    for day, (a, b) in enumerate(((1.0, 0.0), (1.0, 1.0)), 1):
        rows = [_row(0, a, episode=f"e{day}a"), _row(1, b, episode=f"e{day}a"),
                {**_row(0, 1.0, episode=f"e{day}b"), "index": 2}]
        for r in rows:
            r["truth"]["n_subtasks"] = 2
        rec = _goat_record(day * 10, "H", f"e{day}b", [1.0, 0.0])        # the env's lists: last episode only
        _write_run(root, f"deploy_full_H_d{day}", [rec], {day * 10: rows}, config={"chain_ids": [day * 10, day * 10 + 1]})
    assert len(run_dirs(root)) == 2
    rep = load(root)
    keys = {t.key: t.success for t in rep.tasks}
    assert len(keys) == 8, "two days x two episodes x two sub-goals; e?b's second sub-goal never closed"
    assert keys[("H", "e1b", 1)] == 0.0 and keys[("H", "e2a", 1)] == 1.0
    assert summarize.scores(rep)["s_sr"] == pytest.approx(100 * 5 / 8)


def test_across_seeds_the_sample_standard_deviation() -> None:
    m, sd = summarize.mean_sd([100 * 0 / 36, 100 * 1 / 36, 100 * 2 / 36])
    assert m == pytest.approx(2.78, abs=0.01) and sd == pytest.approx(2.78, abs=0.01)
    assert summarize.mean_sd([None, None]) == (None, None) and summarize.mean_sd([5.0])[1] == 0.0


def test_the_cli_prints_the_runs_and_their_mean(tmp_path: Path, capsys) -> None:
    a, b = _goat_run(tmp_path / "1", "goat"), _goat_run(tmp_path / "2", "goat_s2")
    summarize.main([str(a), str(b), "--json", str(tmp_path / "out.json")])
    out = capsys.readouterr().out
    assert "mean ± s.d. (2 runs)" in out and "50.0 ± 0.0" in out and "not scored" in out
    data = json.loads((tmp_path / "out.json").read_text())
    assert data["across"]["s_sr"]["mean"] == 50.0 and len(data["replicates"]) == 2


def test_a_summary_that_is_not_a_run_is_refused(tmp_path: Path) -> None:
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "summary.json").write_text(json.dumps({"tasks": 3}))
    with pytest.raises(ValueError):
        load(tmp_path / "x")


# ── the paired bootstrap ──


def _scene_outcomes(per_scene: dict[str, list[float]]) -> dict:
    return {(s, "e", i): v for s, vs in per_scene.items() for i, v in enumerate(vs)}


def test_the_change_is_compare_minus_base_over_shared_tasks() -> None:
    base = [_scene_outcomes({"A": [1, 1], "B": [1, 1, 1, 1]})] * 2
    comp = [_scene_outcomes({"A": [0, 1], "B": [1, 1, 1, 0]}), _scene_outcomes({"A": [1, 1], "B": [1, 1, 1, 1]})]
    r = paired_ci.paired_change(base, comp, resamples=2000, seed=1)
    assert r["base"] == 100.0 and r["compare"] == pytest.approx(100 * 10 / 12)
    assert r["change"] == pytest.approx(-100 * 2 / 12) and r["shared_tasks"] == 6 and r["scenes"] == 2
    lo, hi = r["ci"]
    assert lo <= r["change"] <= hi <= 0.0


def test_identical_runs_change_nothing_and_unshared_tasks_are_left_out() -> None:
    one = _scene_outcomes({"A": [1, 0], "B": [1]})
    extra = {**one, ("C", "e", 0): 1.0}
    r = paired_ci.paired_change([one], [extra], resamples=500)
    assert r["change"] == 0.0 and r["ci"] == [0.0, 0.0] and r["dropped"] == {"base": [0], "compare": [1]}
    with pytest.raises(ValueError):
        paired_ci.paired_change([one], [{("Z", "e", 0): 1.0}])


def test_the_resampling_unit_is_the_scene() -> None:
    """All of the change sits in one of ten scenes: resamples that miss that
    scene (about a third of them) see no change, so the interval reaches 0."""
    scenes = {f"S{i}": [1.0] * 10 for i in range(10)}
    base = [_scene_outcomes(scenes)]
    comp = [_scene_outcomes({**scenes, "S0": [0.0] * 10})]
    r = paired_ci.paired_change(base, comp, resamples=5000, seed=0)
    assert r["change"] == pytest.approx(-10.0)
    lo, hi = r["ci"]
    assert hi == 0.0 and lo < -10.0
    again = paired_ci.paired_change(base, comp, resamples=5000, seed=0)
    assert again["ci"] == r["ci"], "a fixed seed reproduces the interval"


def test_the_cli_pairs_run_directories(tmp_path: Path, capsys) -> None:
    a = _goat_run(tmp_path / "1", "full")
    b = _goat_run(tmp_path / "2", "ablation")
    paired_ci.main(["--base", str(a), "--compare", str(b), "--resamples", "200", "--json", str(tmp_path / "ci.json")])
    out = capsys.readouterr().out
    assert "change +0.0 pp" in out and "95% CI" in out and "6 shared tasks" in out
    data = json.loads((tmp_path / "ci.json").read_text())
    assert data["s_sr"]["resamples"] == 200 and not math.isnan(data["spl"]["change"])


# ── the pooled scores in the run's own summary ──


def test_the_run_summary_adds_the_scores_over_tasks() -> None:
    from runner import aggregate, is_scored

    records = [_goat_record(0, "A", "10", [1.0, 0.0, 1.0]), _goat_record(1, "B", "11", [1.0]),
               _tour_record(2, "t2", [1.0, 0.0], 4, 40.0)]
    agg = aggregate([r for r in records if is_scored(r)])
    assert agg["success"] == pytest.approx((2 / 3 + 1 + 0.5) / 3, abs=1e-4), "the per-sequence mean stays as it was"
    assert agg["tasks"] == 8 and agg["s_sr"] == pytest.approx(4 / 8)
    assert agg["e_sr"] == pytest.approx(1 / 3, abs=1e-4) and agg["task_spl"] == pytest.approx((1.0 + 0.5 + 0.8) / 8, abs=1e-4)


def test_the_pre_stop_gate_is_scored_where_it_refused(tmp_path: Path, capsys) -> None:
    """y_stop is read at the judge's refusal; a rescue is (0, 1), a harm (1, 0); the file gate's
    refusals and a refusal whose truth was not read are not guessed at."""
    def refused(y_stop: float | None, *, handover: bool = False) -> dict:
        out = [{"by": "handover", "truth": {"success": 0.0}}] if handover else []
        truth = {"error": "no verb stop_truth"} if y_stop is None else {"success": y_stop, "distance_to_goal": 2.0}
        return {"refused_once": True, "refusals": [*out, {"by": "judge", "verdict": "incomplete", "truth": truth}]}

    rows = [_row(0, 1.0), _row(1, 0.0), _row(2, 1.0), _row(3, 0.0), _row(4, 1.0)]
    for r, gate in zip(rows, (refused(0.0, handover=True), refused(1.0), refused(0.0), refused(None),
                              {"refused_once": False, "refusals": [{"by": "handover", "truth": {}}]}), strict=True):
        r["precheck"] = gate
    run = _write_run(tmp_path, "gate", [_goat_record(0, "A", "10", [1.0, 0.0, 1.0, 0.0, 1.0])], {0: rows})
    gate = summarize.summarize(load(run))["accounting"]["gate"]
    assert (gate["refused_tasks"], gate["truth_read"]) == (4, 3)
    assert gate["refused_first_stop_pct"] == pytest.approx(80.0)
    assert gate["rejection_precision_pct"] == pytest.approx(200 / 3)
    assert gate["rescue_among_correct_pct"] == pytest.approx(100.0)
    assert (gate["rescues_pct"], gate["harms_pct"]) == (pytest.approx(40.0), pytest.approx(20.0))
    assert gate["net_local_change_pp"] == pytest.approx(20.0)
    summarize.main([str(run)])
    assert "pre-stop gate: refused first stop 80.0% of tasks (n=4)" in capsys.readouterr().out
