"""A sequence's tasks as the env scored them, and the scores pooled over tasks.

The runner keeps one record per sequence — a GOAT-Bench episode, an IR2R-CE
tour — and the env's metrics for it are means over the sequence's tasks
(GOAT-Bench's ``success`` is the episode's partial success). Averaging those
means over sequences weighs a task by the length of its sequence. The paper's
scores are over tasks instead:

    s_sr       individual-task success, pooled over every task
    task_spl   per-task SPL, pooled over every task
    e_sr       the fraction of sequences in which every task succeeds

``task_scores`` reads a sequence's tasks off its metrics — GOAT-Bench's
per-sub-goal lists (0 for a sub-goal never closed), an IR2R-CE tour's
episode rows (an episode never reached is a failure, up to the tour's size)
— and ``pooled`` is what ``runner.aggregate`` adds to a run's summary. A
chain over several env episodes (a deployment day) carries the env's lists
of its last episode only; ``reporting/tasks.py`` reads such a chain off the
harness's per-task rows.
"""

from __future__ import annotations

from typing import Any


def _num(v: Any) -> float:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0


def task_scores(metrics: dict[str, Any], agent: dict[str, Any] | None = None) -> list[tuple[float, float]] | None:
    """``(success, spl)`` of every task of a sequence, in order; None when
    the metrics score no task."""
    m, agent = metrics or {}, agent or {}
    if isinstance(m.get("subtask_success"), list):
        spl = list(m.get("spl_by_subtask") or [])
        return [(_num(ok), _num(spl[i]) if i < len(spl) else 0.0) for i, ok in enumerate(m["subtask_success"])]
    if agent.get("task_shape") == "tour" or "tour_episodes" in m:
        eps = list(m.get("tour_episodes") or [])
        size = int(m.get("tour_size") or agent.get("tour_size") or m.get("n_episodes") or len(eps) or 1)
        by_slot = {int(e.get("slot", n)): (e.get("metrics") or {}) for n, e in enumerate(eps)}
        return [(_num(by_slot.get(s, {}).get("success")), _num(by_slot.get(s, {}).get("spl")))
                for s in range(max(size, len(eps)))]
    if isinstance(m.get("success"), (int, float)):
        return [(_num(m.get("success")), _num(m.get("spl")))]
    return None


def pooled(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The scores over tasks of the given (scored) sequence records, as fractions."""
    succ: list[float] = []
    spl: list[float] = []
    whole: list[float] = []
    for rec in records:
        tasks = task_scores(rec.get("metrics") or {}, rec.get("agent") or {})
        if not tasks:
            continue
        succ += [1.0 if ok >= 1.0 else 0.0 for ok, _ in tasks]
        spl += [s for _, s in tasks]
        whole.append(1.0 if all(ok >= 1.0 for ok, _ in tasks) else 0.0)
    if not succ:
        return {"tasks": 0}
    return {
        "tasks": len(succ),
        "s_sr": round(sum(succ) / len(succ), 4),
        "task_spl": round(sum(spl) / len(spl), 4),
        "e_sr": round(sum(whole) / len(whole), 4),
    }


__all__ = ["pooled", "task_scores"]
