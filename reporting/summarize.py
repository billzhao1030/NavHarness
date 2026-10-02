"""The paper's scores of one or more runs, from what the runs wrote.

    python reporting/summarize.py <run> [<run> ...] [--json <out.json>]

Each ``<run>`` is one replicate: a run directory (``outputs/navharness/<run
name>``), or a directory of run directories taken together (a deployment's
``runs/``). Given several — the three seeds of one configuration, e.g.
``std_goat_slam_es_navh_opus-5_default_lhnav{,_s2,_s3}`` — it prints each and
then the mean and the sample standard deviation across them, the form of the
paper's tables. The definitions (per-task truth as ``reporting/tasks.py``
reads it):

    s-SR      individual-task success pooled over every task of the run
    SPL       per-task SPL pooled over every task (a failed task scores 0)
    e-SR      the fraction of sequences (a GOAT-Bench episode, an IR2R-CE
              tour) in which every task succeeds
    t-nDTW    IR2R-CE: the tour's t-nDTW (the benchmark's, x100), averaged
              over tours; a tour cut short has none and is counted apart

all in percent. Every task of a scored sequence counts — an unattempted one
as a failure — and a sequence the runner could not score is listed, not
dropped silently.

The accounting beside the scores, per task over the tasks that ran: model
turns, executed steps, markers dropped, the peak prompt (the largest context
any of the task's turns carried, across all of its sessions — the paper's
"mean per-task peak prompt"), and tokens by kind of session (the agent, its
recovery-note hops, pre-stop verification, certification, wake-up), with the
run-end consolidation per sequence; the amortized wall time per task (every
sequence's elapsed time, placement included, over the number of tasks);
recoveries by trigger (requested by the model, or the turn fallback); the
certification of claimed completions against the env's truth, and the
success of the task that follows a false or a true record; the pre-stop
gate scored where it refused (below); and s-SR by goal modality and by
position in the sequence.

The pre-stop gate: for a task whose first stopping request the judge sent
back, ``y_stop`` is the evaluator's success had STOP been issued at that
state (the harness reads it at the refusal, ``precheck.refusals[].truth``)
and ``y_final`` the task's own. A rescue is (0, 1), a harm (1, 0); a correct
refusal has ``y_stop`` = 0. Reported: tasks with a refused first stop, the
rejection precision (correct refusals over refusals whose truth was read),
rescues among correct refusals, rescues and harms over all tasks, and the
net local change (rescues - harms) / tasks, in points. A refusal whose truth
was not recorded is counted as refused and left out of the rest.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reporting.tasks import Replicate, load  # noqa: E402

TOKEN_KEYS = ("calls", "prompt_tokens", "cached_tokens", "completion_tokens")


def _pct(xs: list[float]) -> float | None:
    return 100.0 * sum(xs) / len(xs) if xs else None


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


# ── the scores ───────────────────────────────────────────────────────


def scores(rep: Replicate) -> dict[str, Any]:
    tasks = rep.tasks
    by_seq: dict[tuple[str, str], list[float]] = defaultdict(list)
    for t in tasks:
        by_seq[(t.scene, t.sequence)].append(t.success)
    tours = [s for s in rep.sequences if s.t_ndtw is not None]
    out: dict[str, Any] = {
        "tasks": len(tasks),
        "sequences": len(by_seq),
        "s_sr": _pct([1.0 if t.success >= 1.0 else 0.0 for t in tasks]),
        "spl": _pct([t.spl for t in tasks]),
        "e_sr": _pct([1.0 if all(s >= 1.0 for s in v) else 0.0 for v in by_seq.values()]),
        "t_ndtw": _mean([float(s.t_ndtw) for s in tours]) if tours else None,
        "t_ndtw_tours": len(tours),
        "t_ndtw_unscored": sum(1 for s in rep.sequences if s.t_ndtw is None and s.t_ndtw_prefix),
        "unscored_sequences": len(rep.unscored),
    }
    return out


# ── the accounting ───────────────────────────────────────────────────


def _tokens(row: dict) -> dict[str, dict[str, int]]:
    """A row's usage by kind of session; a row from before the per-session
    tally carries only the judge's prompt and completion tokens."""
    ctx = row.get("context") or {}
    if ctx.get("tokens") is not None:
        return dict(ctx["tokens"])
    judge = {"calls": int(ctx.get("judge_calls") or 0), "prompt_tokens": int(ctx.get("judge_prompt_tokens") or 0),
             "completion_tokens": int(ctx.get("judge_completion_tokens") or 0)}
    return {"judge": judge} if judge["calls"] else {}


def accounting(rep: Replicate) -> dict[str, Any]:
    ran = [t for t in rep.tasks if t.row is not None]
    rows = [t.row for t in ran]
    out: dict[str, Any] = {"tasks_with_record": len(ran)}
    out["turns_per_task"] = _mean([float(r.get("turns") or 0) for r in rows])
    out["steps_per_task"] = _mean([float(r.get("steps") or 0) for r in rows])
    out["markers_per_task"] = _mean([float((r.get("tool_calls") or {}).get("mark", 0)) for r in rows])
    out["peak_prompt_tokens_per_task"] = _mean(
        [float((r.get("context") or {}).get("peak_prompt_tokens") or 0) for r in rows])
    usage = [_tokens(r) for r in rows]
    out["tokens_per_task"] = {
        kind: {key: _mean([float((u.get(kind) or {}).get(key, 0) or 0) for u in usage]) for key in TOKEN_KEYS}
        for kind in sorted({k for u in usage for k in u})}
    cons = [s.run_tokens.get("consolidate") or {} for s in rep.sequences if s.recorded]
    out["consolidation_per_sequence"] = {
        key: _mean([float(c.get(key, 0) or 0) for c in cons]) for key in TOKEN_KEYS} if any(cons) else None
    wall = sum(s.wall_sec + s.setup_sec for s in rep.sequences)
    out["wall_sec_per_task"] = wall / len(rep.tasks) if rep.tasks else None
    out["recoveries"] = _recoveries(ran)
    out["certification"] = _certification(ran)
    out["successor"] = _successor(ran)
    out["gate"] = _gate(ran, len(rep.tasks))
    return out


def _gate(ran: list, n_tasks: int) -> dict[str, Any] | None:
    """The pre-stop gate at the states it refused (module docstring)."""
    pairs: list[tuple[float | None, bool]] = []
    for t in ran:
        first = next((x for x in ((t.row.get("precheck") or {}).get("refusals") or [])
                      if x.get("by") == "judge"), None)
        if first is not None:
            y_stop = (first.get("truth") or {}).get("success")
            pairs.append((None if y_stop is None else float(y_stop), t.success >= 1.0))
    if not pairs or not n_tasks:
        return None
    known = [(y, f) for y, f in pairs if y is not None]
    correct = [(y, f) for y, f in known if y < 1.0]
    rescues = sum(1 for _, f in correct if f)
    harms = sum(1 for y, f in known if y >= 1.0 and not f)
    return {
        "refused_tasks": len(pairs),
        "truth_read": len(known),
        "refused_first_stop_pct": 100.0 * len(pairs) / n_tasks,
        "rejection_precision_pct": 100.0 * len(correct) / len(known) if known else None,
        "rescue_among_correct_pct": 100.0 * rescues / len(correct) if correct else None,
        "rescues_pct": 100.0 * rescues / n_tasks,
        "harms_pct": 100.0 * harms / n_tasks,
        "net_local_change_pp": 100.0 * (rescues - harms) / n_tasks,
    }


def _recoveries(ran: list) -> dict[str, Any] | None:
    events = [(x, t) for t in ran for x in (t.row.get("resets") or [])]
    if not ran:
        return None
    out: dict[str, Any] = {
        "tasks_with_recovery_pct": _pct([1.0 if t.row.get("resets") else 0.0 for t in ran]),
        "events": len(events),
        "events_leading_to_success_pct": _pct([1.0 if t.success >= 1.0 else 0.0 for _, t in events]),
    }
    for trigger, name in (("self", "requested"), ("cap", "fallback")):
        mine = [(x, t) for x, t in events if (x.get("trigger") == "self") == (trigger == "self")]
        turns = [float(x.get("attempt_turns", x.get("at_turn", 0)) or 0) for x, _ in mine]
        out[name] = {
            "share_pct": 100.0 * len(mine) / len(events) if events else None,
            "median_attempt_turns": statistics.median(turns) if turns else None,
            "success_pct": _pct([1.0 if t.success >= 1.0 else 0.0 for _, t in mine]),
        }
    return out


def _certification(ran: list) -> dict[str, Any] | None:
    """Post-stop certification over the claimed completions: certified
    (verdict complete) or not, against the env's truth; the accuracy of
    the verdicts, the accuracy of accepting every claim, and Cohen's kappa."""
    claimed = [t for t in ran if ((t.row.get("claim") or {}).get("status")) == "complete"
               and ((t.row.get("verdict") or {}).get("verdict") not in (None, "none"))]
    if not claimed:
        return None
    n = len(claimed)
    cert = [((t.row.get("verdict") or {}).get("verdict")) == "complete" for t in claimed]
    true = [t.success >= 1.0 for t in claimed]
    cells = {
        "certified_true": sum(c and y for c, y in zip(cert, true, strict=True)),
        "certified_false": sum(c and not y for c, y in zip(cert, true, strict=True)),
        "rejected_true": sum((not c) and y for c, y in zip(cert, true, strict=True)),
        "rejected_false": sum((not c) and (not y) for c, y in zip(cert, true, strict=True)),
    }
    acc = (cells["certified_true"] + cells["rejected_false"]) / n
    p_cert, p_true = sum(cert) / n, sum(true) / n
    chance = p_cert * p_true + (1 - p_cert) * (1 - p_true)
    return {
        "claimed_completions": n,
        **{f"{k}_pct": 100.0 * v / n for k, v in cells.items()},
        "accuracy_pct": 100.0 * acc,
        "accept_all_accuracy_pct": 100.0 * p_true,
        "kappa": (acc - chance) / (1 - chance) if chance < 1 else None,
    }


def _says_complete(row: dict) -> bool:
    """What the task's record says: the certification's verdict when one ran, else the claim."""
    verdict = (row.get("verdict") or {}).get("verdict")
    if verdict not in (None, "none"):
        return verdict == "complete"
    return (row.get("claim") or {}).get("status") == "complete"


def _successor(ran: list) -> dict[str, Any] | None:
    """Success of the task that follows a false record (the record says
    otherwise than the env's truth) and of the one that follows a true one,
    within a sequence."""
    by_seq: dict[tuple[str, str], dict[int, Any]] = defaultdict(dict)
    for t in ran:
        by_seq[(t.scene, t.sequence)][t.slot] = t
    after: dict[bool, list[float]] = {True: [], False: []}
    for tasks in by_seq.values():
        for slot, t in tasks.items():
            nxt = tasks.get(slot + 1)
            if nxt is not None:
                after[_says_complete(t.row) == (t.success >= 1.0)].append(1.0 if nxt.success >= 1.0 else 0.0)
    if not after[True] and not after[False]:
        return None
    return {"after_false_record_pct": _pct(after[False]), "after_false_record_n": len(after[False]),
            "after_true_record_pct": _pct(after[True]), "after_true_record_n": len(after[True])}


def breakdowns(rep: Replicate) -> dict[str, Any]:
    by_mod: dict[str, list] = defaultdict(list)
    by_pos: dict[int, list] = defaultdict(list)
    for t in rep.tasks:
        if t.row is not None and t.row.get("modality"):
            by_mod[str(t.row["modality"])].append(t)
        by_pos[t.slot + 1].append(t)
    return {
        "by_modality": {m: {"tasks": len(ts), "s_sr": _pct([1.0 if t.success >= 1.0 else 0.0 for t in ts]),
                            "spl": _pct([t.spl for t in ts])} for m, ts in sorted(by_mod.items())},
        "by_position": {p: {"tasks": len(ts), "s_sr": _pct([1.0 if t.success >= 1.0 else 0.0 for t in ts])}
                        for p, ts in sorted(by_pos.items())},
    }


def summarize(rep: Replicate) -> dict[str, Any]:
    return {"replicate": rep.name, "runs": [str(r) for r in rep.runs], **scores(rep),
            "accounting": accounting(rep), **breakdowns(rep), "unscored": rep.unscored, "notes": rep.notes}


def mean_sd(values: list[float | None]) -> tuple[float | None, float | None]:
    """Mean and sample standard deviation (n - 1) across replicates."""
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None, None
    return sum(xs) / len(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


# ── printing ─────────────────────────────────────────────────────────


def _f(v: Any, nd: int = 1) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def render(results: list[dict[str, Any]]) -> str:
    keys = (("s_sr", "s-SR"), ("spl", "SPL"), ("e_sr", "e-SR"), ("t_ndtw", "t-nDTW"))
    width = max([len(r["replicate"]) for r in results] + [24])
    lines = [f"{'replicate':<{width}}  {'tasks':>6} {'seqs':>5}  " + "  ".join(f"{h:>11}" for _, h in keys)]
    for r in results:
        lines.append(f"{r['replicate']:<{width}}  {r['tasks']:>6} {r['sequences']:>5}  "
                     + "  ".join(f"{_f(r[k]):>11}" for k, _ in keys))
    if len(results) > 1:
        cells = []
        for k, _ in keys:
            m, sd = mean_sd([r[k] for r in results])
            cells.append(f"{_f(m)} ± {_f(sd)}" if m is not None else "-")
        lines.append(f"{f'mean ± s.d. ({len(results)} runs)':<{width}}  {'':>6} {'':>5}  "
                     + "  ".join(f"{c:>11}" for c in cells))
    for r in results:
        a = r["accounting"]
        lines.append("")
        lines.append(f"{r['replicate']}: per task ({a['tasks_with_record']} with a record) — "
                     f"turns {_f(a['turns_per_task'])} · steps {_f(a['steps_per_task'])} · "
                     f"markers {_f(a['markers_per_task'], 2)} · peak prompt {_f(a['peak_prompt_tokens_per_task'], 0)} "
                     f"tokens · wall {_f(a['wall_sec_per_task'])} s")
        for kind, v in a["tokens_per_task"].items():
            lines.append(f"  tokens/task {kind:<9} prompt {_f(v.get('prompt_tokens'), 0)} "
                         f"(cached {_f(v.get('cached_tokens'), 0)}) · completion {_f(v.get('completion_tokens'), 0)} "
                         f"· calls {_f(v.get('calls'), 2)}")
        if a["consolidation_per_sequence"]:
            c = a["consolidation_per_sequence"]
            lines.append(f"  consolidation/sequence prompt {_f(c['prompt_tokens'], 0)} · "
                         f"completion {_f(c['completion_tokens'], 0)}")
        rc = a["recoveries"]
        if rc and rc["events"]:
            lines.append(f"  recoveries: {_f(rc['tasks_with_recovery_pct'])}% of tasks, {rc['events']} events, "
                         f"{_f(rc['events_leading_to_success_pct'])}% lead to success; requested "
                         f"{_f(rc['requested']['share_pct'])}% (median {_f(rc['requested']['median_attempt_turns'])} "
                         f"turns, {_f(rc['requested']['success_pct'])}% success), fallback "
                         f"{_f(rc['fallback']['share_pct'])}% (median {_f(rc['fallback']['median_attempt_turns'])} "
                         f"turns, {_f(rc['fallback']['success_pct'])}% success)")
        cert = a["certification"]
        if cert:
            lines.append(f"  certification of {cert['claimed_completions']} claimed completions: accuracy "
                         f"{_f(cert['accuracy_pct'])}% (accept-all {_f(cert['accept_all_accuracy_pct'])}%), "
                         f"kappa {_f(cert['kappa'], 2)}")
        nxt = a["successor"]
        if nxt:
            lines.append(f"  the next task succeeds {_f(nxt['after_false_record_pct'])}% after a false record "
                         f"(n={nxt['after_false_record_n']}), {_f(nxt['after_true_record_pct'])}% after a true one "
                         f"(n={nxt['after_true_record_n']})")
        gate = a["gate"]
        if gate:
            lines.append(f"  pre-stop gate: refused first stop {_f(gate['refused_first_stop_pct'])}% of tasks "
                         f"(n={gate['refused_tasks']}) · rejection precision {_f(gate['rejection_precision_pct'])}% · "
                         f"rescue among correct refusals {_f(gate['rescue_among_correct_pct'])}% · rescues "
                         f"{_f(gate['rescues_pct'])}% · harms {_f(gate['harms_pct'])}% · net local change "
                         f"{_f(gate['net_local_change_pp'])} pp (y_stop read at {gate['truth_read']} of "
                         f"{gate['refused_tasks']} refusals)")
        if r["by_modality"]:
            lines.append("  by modality: " + " · ".join(f"{m} {_f(v['s_sr'])} / {_f(v['spl'])} (n={v['tasks']})"
                                                       for m, v in r["by_modality"].items()))
        if r["t_ndtw_unscored"]:
            lines.append(f"  {r['t_ndtw_unscored']} tour(s) cut short: no t-nDTW, left out of its mean")
        if r["unscored"]:
            lines.append(f"  WARNING {len(r['unscored'])} sequence(s) not scored (no task set; re-run them): "
                         + ", ".join(f"{u['run']}#{u['index']} ({u['error']})" for u in r["unscored"][:8]))
        for note in r["notes"][:8]:
            lines.append(f"  note: {note}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="run directories (or directories of runs), one per replicate")
    ap.add_argument("--json", type=Path, default=None, help="also write every number to this file")
    args = ap.parse_args(argv)
    results = [summarize(load(p)) for p in args.runs]
    print(render(results))
    if args.json:
        across = {k: dict(zip(("mean", "sd"), mean_sd([r[k] for r in results]), strict=True))
                  for k in ("s_sr", "spl", "e_sr", "t_ndtw")}
        args.json.write_text(json.dumps({"replicates": results, "across": across}, indent=1, default=str))


if __name__ == "__main__":
    main()
