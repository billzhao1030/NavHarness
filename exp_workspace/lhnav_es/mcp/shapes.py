"""This arm's task shapes beyond the built-ins — one: an IVLN-CE TOUR as the
unit of a run.

A shape answers what an episode IS to the loop: how index `i` is placed, what
the briefing embeds, how it is evaluated, what the record carries. Nothing in
the loop says index `i` has to be one env episode — so `i` here is one TOUR,
and its record is the tour's.

It lives here rather than in the runner because it is a statement about this
benchmark, not about the framework. IVLN's question is what carries ACROSS a
tour's episodes: the body is walked to the next start rather than teleported,
and the map, the markers and what the agent worked out about the house go with
it. Scored one episode at a time, the harness would get a fresh ledger and a
fresh place graph at every episode and the benchmark's own question would never
be asked. One tour, one chain, one record.
"""

from __future__ import annotations

from typing import Any

from core.shapes.base import Goal, Placement
from core.shapes.es import EsNav


class EsTour(EsNav):
    """One IVLN-CE tour: placed as a tour, scored as a tour.

    `place(i)` seats tour `i` at its first episode with a fresh map; the
    harness walks the rest through `advance_tour` (see
    `core.harnesses.navh.env.bench_es.TourEsBench`), which is where the oracle
    transit and the carried-over map happen. `evaluate` reads the tour's own
    ruler — every episode's metrics, their means, and t-nDTW, the one number an
    episode cannot give.
    """

    shape = "es.tour"
    task_word = "nav"

    async def place(self, index: int) -> Placement:
        ep = await self.env.call("place_tour", index=index)
        if "error" in ep:
            raise RuntimeError(f"{self.env.url} place_tour({index}): {ep['error']}")
        goal = self.goal_of(ep, index)
        # The tour's facts ride on the goal so the record and the briefing can
        # both say which tour this is and how long it runs.
        goal.ep.setdefault("tour_size", (ep.get("tour") or {}).get("tour_size"))
        return Placement(goal, self.caps_of(goal))

    async def evaluate(self, sink: Any, goal: Goal) -> dict[str, Any]:
        try:
            out = await self.env.call("tour_report")
            if "error" in out:
                raise RuntimeError(out["error"])
        except Exception as exc:  # noqa: BLE001 — a lost report is not a lost tour
            sink.emit("driver_error", {"error": f"tour_report failed: {exc!r}"})
            return {}
        # The episode rows stay in the record (a tour that went wrong went wrong
        # somewhere in particular), the means and t-nDTW are the tour's score.
        return {k: v for k, v in out.items() if k != "episodes"} | {
            "tour_episodes": out.get("episodes") or []
        }

    def record_extras(self, sink: Any, goal: Goal) -> dict[str, Any]:
        return {"task_shape": "tour", "tour_size": goal.ep.get("tour_size")}

    def unfinished_metrics(self, goal: Goal, metrics: dict[str, Any]) -> dict[str, Any]:
        """A tour cut short: every one of its episodes is scored. The rows the
        tour's ruler returned keep the package's scores (an episode still in
        progress has no STOP, so it already reads as a failure); an episode
        never reached is a failure. The means run over the whole tour.
        t-nDTW compares the tour's driven path with the WHOLE reference
        tour, which a cut-short tour has not driven, so it is left unscored;
        the package's value over the finished prefix is kept beside it."""
        rows = list(metrics.get("tour_episodes") or [])
        size = max(int(goal.ep.get("tour_size") or 0), len(rows), 1)

        def score(key: str) -> list[float]:
            vals = [(r.get("metrics") or {}).get(key) for r in rows]
            return [float(v) if isinstance(v, (int, float)) else 0.0 for v in vals] + [0.0] * (size - len(rows))

        succ, spl = score("success"), score("spl")
        return {
            **metrics,
            "tour_episodes": rows,
            "n_episodes": len(rows),
            "tour_size": size,
            "success": sum(succ) / size,
            "spl": sum(spl) / size,
            "t_ndtw": None,
            "t_ndtw_prefix": metrics.get("t_ndtw"),
        }


__all__ = ["EsTour"]
