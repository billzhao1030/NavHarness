"""Paired changes between two configurations, with the paper's bootstrap intervals.

    python reporting/paired_ci.py --base <run> [<run> ...] --compare <run> [<run> ...]
                                  [--metric s_sr spl] [--resamples 5000] [--level 0.95]
                                  [--seed 0] [--json <out.json>]

Each ``<run>`` is one replicate (``reporting/tasks.py``): a run directory, or a
directory of run directories taken together (a deployment's ``runs/``). The
change reported is COMPARE minus BASE, in percentage points:

pairing   tasks are paired by ``(scene, sequence, slot)`` — the scene, the GOAT
          episode or IR2R-CE tour, the sub-goal or tour position. Only tasks
          present in every replicate of both configurations count, each once
          per replicate; how many each replicate loses to this is reported.
estimate  each configuration's outcomes are pooled over its replicates and the
          shared tasks; the change is the difference of the pooled scores.
interval  a scene-clustered bootstrap: ``--resamples`` times (5,000), draw the
          scenes with replacement and recompute the change over the drawn
          scenes' tasks, every replicate of both configurations included, so
          a task's paired outcomes stay together. The ``--level`` (95%)
          interval is the percentile interval of those changes. These are
          pointwise intervals, without multiple-comparison adjustment.

The paper's comparisons:

    # a component (Table 3): the change from the full reference, three seeds each
    python reporting/paired_ci.py --base outputs/navharness/std_goat_slam_es_navh_opus-5_default_lhnav{,_s2,_s3} \\
        --compare outputs/navharness/std_goat_slam_es_navh_opus-5_default_lhnav_<ablation>{,_s2,_s3}
    # fixed-access memory: NavHarness minus fixed access
    python reporting/paired_ci.py --base outputs/navharness/std_goat_slam_es_navh_opus-5_default_lhnav_fixed{,_s2,_s3} \\
        --compare outputs/navharness/std_goat_slam_es_navh_opus-5_default_lhnav{,_s2,_s3}
    # continuous deployment: full minus no consolidation, every house's days pooled;
    # a house is its scene, so the bootstrap resamples houses
    python reporting/paired_ci.py --base outputs/deploy/nocons/runs --compare outputs/deploy/full/runs

``s_sr`` scores a task 1 when it succeeded, ``spl`` scores its SPL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reporting.tasks import Replicate, load  # noqa: E402

METRICS = ("s_sr", "spl")
RESAMPLES = 5000
LEVEL = 0.95

Key = tuple[str, str, int]


def outcomes(rep: Replicate, metric: str) -> dict[Key, float]:
    """One value per task of a replicate (the first, if a task were recorded twice)."""
    out: dict[Key, float] = {}
    for t in rep.tasks:
        if t.key not in out:
            out[t.key] = (1.0 if t.success >= 1.0 else 0.0) if metric == "s_sr" else float(t.spl)
    return out


def paired_change(base: list[dict[Key, float]], compare: list[dict[Key, float]], *,
                  resamples: int = RESAMPLES, level: float = LEVEL, seed: int = 0) -> dict[str, Any]:
    """The paired change compare - base over the shared tasks, and its
    scene-clustered bootstrap interval (both in percentage points)."""
    if not base or not compare:
        raise ValueError("both configurations need at least one replicate")
    shared = set.intersection(*(set(o) for o in [*base, *compare]))
    if not shared:
        raise ValueError("the two configurations share no task")
    scenes = sorted({k[0] for k in shared})
    at = {s: i for i, s in enumerate(scenes)}
    keys = sorted(shared)
    where = np.array([at[k[0]] for k in keys])

    def per_scene(group: list[dict[Key, float]]) -> tuple[np.ndarray, np.ndarray]:
        sums = np.zeros(len(scenes))
        counts = np.zeros(len(scenes))
        for o in group:
            np.add.at(sums, where, np.array([o[k] for k in keys]))
            np.add.at(counts, where, 1.0)
        return sums, counts

    sb, nb = per_scene(base)
    sc, nc = per_scene(compare)
    score_b, score_c = sb.sum() / nb.sum(), sc.sum() / nc.sum()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(scenes), size=(int(resamples), len(scenes)))
    boot = sc[draws].sum(1) / nc[draws].sum(1) - sb[draws].sum(1) / nb[draws].sum(1)
    tail = 100.0 * (1.0 - level) / 2.0
    lo, hi = np.percentile(boot, [tail, 100.0 - tail])
    return {
        "base": 100.0 * score_b,
        "compare": 100.0 * score_c,
        "change": 100.0 * (score_c - score_b),
        "ci": [100.0 * float(lo), 100.0 * float(hi)],
        "level": level,
        "resamples": int(resamples),
        "shared_tasks": len(shared),
        "scenes": len(scenes),
        "dropped": {"base": [len(o) - len(shared) for o in base],
                    "compare": [len(o) - len(shared) for o in compare]},
    }


def compare_runs(base: list[Path], compare: list[Path], metrics: tuple[str, ...] = METRICS, *,
                 resamples: int = RESAMPLES, level: float = LEVEL, seed: int = 0) -> dict[str, Any]:
    reps_b, reps_c = [load(p) for p in base], [load(p) for p in compare]
    out: dict[str, Any] = {"base": [r.name for r in reps_b], "compare": [r.name for r in reps_c]}
    for metric in metrics:
        out[metric] = paired_change([outcomes(r, metric) for r in reps_b], [outcomes(r, metric) for r in reps_c],
                                    resamples=resamples, level=level, seed=seed)
    out["unscored"] = {r.name: r.unscored for r in [*reps_b, *reps_c] if r.unscored}
    return out


def render(res: dict[str, Any]) -> str:
    lines = [f"base:    {', '.join(res['base'])}", f"compare: {', '.join(res['compare'])}"]
    for metric in METRICS:
        r = res.get(metric)
        if not r:
            continue
        lo, hi = r["ci"]
        lines.append(f"{metric:<5} base {r['base']:.1f}  compare {r['compare']:.1f}  "
                     f"change {r['change']:+.1f} pp  {round(100 * r['level'])}% CI [{lo:+.1f}, {hi:+.1f}]  "
                     f"({r['shared_tasks']} shared tasks, {r['scenes']} scenes, {r['resamples']} resamples)")
        dropped = [d for d in r["dropped"]["base"] + r["dropped"]["compare"] if d]
        if dropped:
            lines.append(f"      tasks not shared by every replicate, left out: base {r['dropped']['base']}, "
                         f"compare {r['dropped']['compare']}")
    for name, unscored in (res.get("unscored") or {}).items():
        lines.append(f"WARNING {name}: {len(unscored)} sequence(s) not scored and not paired")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", nargs="+", type=Path, required=True, help="the reference configuration's replicates")
    ap.add_argument("--compare", nargs="+", type=Path, required=True, help="the other configuration's replicates")
    ap.add_argument("--metric", nargs="+", choices=METRICS, default=list(METRICS))
    ap.add_argument("--resamples", type=int, default=RESAMPLES)
    ap.add_argument("--level", type=float, default=LEVEL)
    ap.add_argument("--seed", type=int, default=0, help="the bootstrap's random seed")
    ap.add_argument("--json", type=Path, default=None, help="also write the numbers to this file")
    args = ap.parse_args(argv)
    res = compare_runs(args.base, args.compare, tuple(args.metric), resamples=args.resamples,
                       level=args.level, seed=args.seed)
    print(render(res))
    if args.json:
        args.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
