"""Record — the run's paper trail, in the shape the monitor already reads.

* ``episode_{k}.jsonl``  one event stream per task (tool_use / tool_result /
                          thinking / assistant_text / episode_meta /
                          episode_metrics / boundary / exit)
* ``live_{k}/``           that task's frames (moved from ``live/`` at the
                          boundary)
* ``summary.json``        harness config, per-task rows (env truth + claim +
                          verdict + context accounting), chain aggregates

Two scoping rules:
  - every per-task number is sliced to the task (steps, frames, folds);
  - chain-level numbers are computed from the rows, never from a running
    env counter.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def _dedupe(rows: list[dict]) -> list[dict]:
    """One row per task index, last writer wins.

    A resumed run seeds the replayed rows and then runs the rest; if the env
    re-seats at the start of the tour instead of skipping what was done, the
    same index is written twice and every aggregate is computed over an
    inflated denominator. Keeping the newest row per index is correct either
    way: it is the attempt that actually finished the task."""
    by: dict = {}
    for r in rows:
        by[int(r.get("index", len(by)))] = r
    return [by[i] for i in sorted(by)]


class Recorder:
    def __init__(self, out_dir: Path, frames_dir: Path) -> None:
        self.out_dir = out_dir
        self.frames_dir = frames_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self._fh = None
        self.k = -1
        self.rows: list[dict] = []
        self.preamble: list[dict] = []   # events outside any task: seat, resume, map_restored

    # ── event stream ─────────────────────────────────────────────────

    def open_task(self, k: int, meta: dict) -> None:
        self.close_stream()
        self.k = k
        self._fh = (self.out_dir / f"episode_{k}.jsonl").open("w")
        self.emit("episode_meta", index=k, **meta)

    def emit(self, kind: str, **payload: Any) -> None:
        rec = {"t": round(time.time() - self.t0, 1), "kind": kind, **payload}
        if self._fh is None:
            # before the first task opens (seat, resume, map_restored) and after
            # the last one closes: keep it in the run's own log, not a task's
            self.preamble.append(rec)
            with (self.out_dir / "run.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
            return
        self._fh.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close_stream(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self.k >= 0:
            dst = self.out_dir / f"live_{self.k}"
            dst.mkdir(exist_ok=True)
            for f in sorted(self.frames_dir.glob("obs_*.png")):
                f.rename(dst / f.name)

    # ── per-task row ───────────────────────────────────────────────

    def add_row(self, row: dict) -> None:
        self.rows.append(row)
        self.emit("episode_metrics", index=row.get("index"), metrics=row)

    # ── summary ──────────────────────────────────────────────────────

    def write_summary(self, header: dict, chain_report: dict | None = None) -> dict:
        rows = _dedupe(self.rows)
        n = len(rows)
        succ = [float(r.get("truth", {}).get("success", 0.0) or 0.0) for r in rows]
        claims = [r.get("claim", {}).get("status", "stop") for r in rows]
        verdicts = [r.get("verdict", {}).get("verdict", "none") for r in rows]

        false_complete = sum(1 for s, c in zip(succ, claims, strict=True) if c == "complete" and s < 1.0)
        n_claim_complete = sum(1 for c in claims if c == "complete")
        # contamination gap: P(fail_{k+1} | fail_k) - P(fail_{k+1} | success_k)
        after_fail = [1.0 - succ[i + 1] for i in range(n - 1) if succ[i] < 1.0]
        after_ok = [1.0 - succ[i + 1] for i in range(n - 1) if succ[i] >= 1.0]
        gap = ((sum(after_fail) / len(after_fail)) if after_fail else None,
               (sum(after_ok) / len(after_ok)) if after_ok else None)

        refused = sum(1 for r in rows if (r.get("precheck") or {}).get("refused_once"))
        insisted_and_failed = sum(1 for r, s in zip(rows, succ, strict=True)
                                  if (r.get("precheck") or {}).get("refused_once") and s < 1.0)
        wakeups = [r["wakeup"] for r in rows if r.get("wakeup")]
        # successor success conditioned on the certification the predecessor got
        succ_after = {}
        for i in range(n - 1):
            v = verdicts[i]
            succ_after.setdefault(v, []).append(succ[i + 1])
        agg = {
            "tasks": n,
            "precheck_refusals": refused,
            "precheck_refused_then_failed": insisted_and_failed,
            "wakeups": len(wakeups),
            "wakeup_reanchors": sum(int((w or {}).get("reanchors") or 0) for w in wakeups),
            "s_sr_after_wakeup": (lambda xs: sum(xs) / len(xs) if xs else None)(
                [s for r, s in zip(rows, succ, strict=True) if r.get("after_wakeup")]),
            "successor_success_by_verdict": {v: sum(x) / len(x) for v, x in succ_after.items()},
            "e_sr": float(n > 0 and all(s >= 1.0 for s in succ)),
            "s_sr": (sum(succ) / n) if n else 0.0,
            "spl": (sum(float(r.get("truth", {}).get("spl", 0.0) or 0.0) for r in rows) / n) if n else 0.0,
            "false_completion_rate": (false_complete / n_claim_complete) if n_claim_complete else None,
            "claims": {c: claims.count(c) for c in set(claims)},
            "verdicts": {v: verdicts.count(v) for v in set(verdicts)},
            "verdict_vs_truth": _confusion(verdicts, succ),
            "p_fail_after_fail": gap[0], "p_fail_after_success": gap[1],
            "mid_task_folds": sum(int(r.get("context", {}).get("mid_task_folds", 0)) for r in rows),
            "rollovers": max((int(r.get("context", {}).get("rollovers", 0)) for r in rows), default=0),
            "tasks_per_session": (round(len(rows) / (1 + max((int(r.get("context", {}).get("rollovers", 0))
                                                              for r in rows), default=0)), 2) if rows else 0),
            "judge_calls": sum(int(r.get("context", {}).get("judge_calls", 0)) for r in rows),
            "judge_prompt_tokens": sum(int(r.get("context", {}).get("judge_prompt_tokens", 0)) for r in rows),
            "aux_calls": sum(int(r.get("context", {}).get("aux_calls", 0)) for r in rows),
            "aux_prompt_tokens": sum(int(r.get("context", {}).get("aux_prompt_tokens", 0)) for r in rows),
            "gate_reused": sum(1 for r in rows if r.get("context", {}).get("gate_reused")),
            "resets": sum(len(r.get("resets") or []) for r in rows),
            "tasks_with_reset": sum(1 for r in rows if r.get("resets")),
            "s_sr_after_reset": (lambda xs: sum(xs) / len(xs) if xs else None)(
                [float(r.get("truth", {}).get("success", 0) or 0) for r in rows if r.get("resets")]),
            **_reset_diagnostics(rows),
            **_ladder_diagnostics(rows),
            **_gate_diagnostics(rows),
            "peak_prompt_tokens": max((int(r.get("context", {}).get("peak_prompt_tokens", 0)) for r in rows), default=0),
            "steps": sum(int(r.get("truth", {}).get("steps_taken", r.get("steps", 0)) or 0) for r in rows),
            "tokens": _token_totals(rows),
        }
        summary = {**header, "run_events": self.preamble, "aggregate": agg, "tasks": rows,
                   "chain_report": chain_report or {}, "wall_sec": round(time.time() - self.t0, 1)}
        (self.out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str, ensure_ascii=False))
        return summary


def _token_totals(rows: list[dict]) -> dict:
    """The tasks' session usage (``context.tokens``), summed by kind of
    session; the peak is the largest any task reached."""
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        for kind, row in ((r.get("context") or {}).get("tokens") or {}).items():
            tot = out.setdefault(kind, {})
            for key, v in (row or {}).items():
                if key == "peak_prompt_tokens":
                    tot[key] = max(tot.get(key, 0), int(v or 0))
                else:
                    tot[key] = tot.get(key, 0) + int(v or 0)
    return out


def _reset_diagnostics(rows: list[dict], cap: int = 80, early: int = 25) -> dict:
    """What the two restart triggers actually did.

    The turn cap is one trigger; the model's own request is the other, and the
    question that matters is whether it fires EARLIER than the cap would have
    and still recovers the task."""
    rs = [x for r in rows for x in (r.get("resets") or [])]
    if not rs:
        return {}
    def _t(x):                       # a row without `attempt_turns` falls back to `at_turn`
        return int(x.get("attempt_turns", x.get("at_turn", 0)) or 0)
    mine = [x for x in rs if x.get("trigger") == "self"]
    capped = [x for x in rs if x.get("trigger") != "self"]
    recovered = [float(r.get("truth", {}).get("success", 0) or 0)
                 for r in rows if any(x.get("trigger") == "self" for x in (r.get("resets") or []))]
    return {
        "resets_self": len(mine),
        "resets_cap": len(capped),
        "cap_share": len(capped) / len(rs),
        "self_mean_turns": (sum(_t(x) for x in mine) / len(mine)) if mine else None,
        "self_turns_saved_vs_cap": (cap - sum(_t(x) for x in mine) / len(mine)) if mine else None,
        "self_premature_rate": (sum(1 for x in mine if _t(x) <= early) / len(mine)) if mine else None,
        "s_sr_after_self_reset": (sum(recovered) / len(recovered)) if recovered else None,
    }


def _ladder_diagnostics(rows: list[dict]) -> dict:
    """What the ladder has to answer.

    Two questions, and neither is "did SR go up". First, does the ladder pick
    the SMALLER scope — the whole claim is that recovery should be selective,
    so a run that always climbs to `house` has disproved itself. Second, the
    recovery funnel: a wake-up can fail at "which room is this" or at "the task
    was hard", and reporting only the task outcome cannot tell them apart."""
    rs = [x for r in rows for x in (r.get("resets") or []) if x.get("scope")]
    if not rs:
        return {}
    by_scope: dict[str, int] = {}
    for x in rs:
        by_scope[x["scope"]] = by_scope.get(x["scope"], 0) + 1
    dec = [x.get("decision") or {} for x in rs]
    funnels = [(x.get("applied") or {}).get("funnel") for x in rs]
    funnels = [f for f in funnels if f]
    stage = {"place": 0, "pose": 0, "state": 0, "task": 0}
    for f in funnels:
        if not f.get("place_recognised"):
            stage["place"] += 1
        elif not f.get("pose_relocalised"):
            stage["pose"] += 1
        elif not f.get("state_reused"):
            stage["state"] += 1
        else:
            stage["task"] += 1
    recovered = [float(r.get("truth", {}).get("success", 0) or 0)
                 for r in rows if r.get("resets")]
    return {
        "recoveries": len(rs),
        "recoveries_by_scope": by_scope,
        "scope_share_attempt": by_scope.get("attempt", 0) / len(rs),
        "harness_overrode": sum(1 for d in dec if d.get("adjusted")),
        "harness_escalated": sum(1 for d in dec if d.get("adjusted") == "escalated"),
        "harness_downgraded": sum(1 for d in dec if d.get("adjusted") == "downgraded"),
        "model_proposed": sum(1 for d in dec if d.get("proposed")),
        "mean_confidence": (sum(float(d.get("confidence") or 0) for d in dec) / len(dec)),
        "funnel_deepest_stage": stage,
        "s_sr_after_recovery": (sum(recovered) / len(recovered)) if recovered else None,
    }


def _gate_diagnostics(rows: list[dict]) -> dict:
    """The judge's gate, scored at the states it refused.

    For a task whose first close request the judge sent back, ``y_stop`` is
    the evaluator's success had the STOP been issued there (read at the
    refusal: ``precheck.refusals[].truth``) and ``y_final`` the task's own. A
    rescue is (0, 1), a harm (1, 0); a correct refusal has ``y_stop`` = 0. All
    counts, so chains add up: rejection precision is correct / truth_read,
    the rescue rate among correct refusals is rescues / correct, and the net
    local change is (rescues - harms) / tasks. A refusal whose truth could not
    be read is counted as refused and left out of the rest, never guessed."""
    refused = known = correct = rescues = harms = 0
    for r in rows:
        first = next((x for x in ((r.get("precheck") or {}).get("refusals") or []) if x.get("by") == "judge"), None)
        if first is None:
            continue
        refused += 1
        y_stop = (first.get("truth") or {}).get("success")
        if y_stop is None:
            continue
        known += 1
        y_final = float(r.get("truth", {}).get("success", 0) or 0) >= 1.0
        if float(y_stop) < 1.0:
            correct += 1
            rescues += int(y_final)
        else:
            harms += int(not y_final)
    if not refused:
        return {}
    return {"gate_refused_tasks": refused, "gate_truth_read": known, "gate_correct_refusals": correct,
            "gate_rescues": rescues, "gate_harms": harms}


def _confusion(verdicts: list[str], succ: list[float]) -> dict:
    c: dict[str, int] = {}
    for v, s in zip(verdicts, succ, strict=True):
        key = f"{v}|{'success' if s >= 1.0 else 'fail'}"
        c[key] = c.get(key, 0) + 1
    return c
