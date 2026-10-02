"""es.* — the navigation task shapes over EmbodiedScore-envs.

One placement flow for every line: reset returns the dataset's task
statement (``task``), the goal in force (``goal`` — a GOAT sequence's first
sub-goal) and ``n_goals``; the task word the briefing gets is read off that
payload. The declared shape (es.nav / es.objnav / es.goat) names the line's
expected word; a payload that derives differently is reported, not silently
re-shaped. GOAT scales the per-episode caps to N sub-goals x the per-goal
budget, under the episode cap.
"""

from __future__ import annotations

from typing import Any

from core.shapes.base import FIRST_PROMPT, Goal, TaskModule


def es_goal(ep: dict[str, Any]) -> tuple[str, str, int]:
    """(task word, briefing statement, sub-goal count) from what the env's
    reset verb returned. point -> nav (the instruction); object -> objnav
    (the category, dataset underscores rendered as spaces); GOAT -> goat,
    the FIRST sub-goal only (the rest are revealed one at a time), and an
    image sub-goal's statement never names the category (the photo rides
    the tool channel)."""
    goal = ep.get("goal") or {}
    kind = str(goal.get("kind") or "none")
    n_goals = int(ep.get("n_goals") or 1)
    task_text = str(ep.get("task") or "").strip()
    if n_goals > 1 or kind in ("image", "description"):
        if kind == "object":
            stmt = f'find a "{str(goal.get("category") or "").replace("_", " ")}" (any instance counts)'
        elif kind == "description":
            stmt = f'find the object this describes — "{goal.get("description")}"'
        elif kind == "image":
            stmt = (
                "find the specific object shown in the photo attached to your first observe() "
                "(that very instance, not just anything of its kind)"
            )
        else:
            stmt = task_text
        return "goat", stmt, n_goals
    if kind == "object":
        return "objnav", task_text.replace("_", " "), 1
    return "nav", task_text, 1


class EsTask(TaskModule):
    """Common behaviour of the es.* shapes; ``task_word`` is the declared
    expectation, the goal carries the derived word."""

    shape = "es.nav"
    task_word = "nav"

    def goal_of(self, ep: dict[str, Any], index: int) -> Goal:
        if not ep.get("episode_id"):
            raise RuntimeError(
                f"{self.env.env_name} reset returned no episode (index {index}) "
                "— episode placement failed?"
            )
        task, statement, n_goals = es_goal(ep)
        if not statement:
            raise RuntimeError(
                f"{self.env.env_name} reset returned an empty task statement "
                f"(episode {index}, goal {ep.get('goal')!r})"
            )
        if task != self.task_word:
            print(
                f"[std] WARNING {self.env.env_name} episode {index}: profile "
                f"declares {self.shape} but reset derives {task!r}"
            )
        return Goal(text=statement, task=task, n_goals=n_goals, ep=ep)

    def n_actions(self, goal: Goal) -> int:
        """The line's action-table size: a declared ``n_actions``, else 7 with
        SUBTASK_STOP on GOAT, else the standard 6."""
        declared = self.cfg.get("n_actions")
        if declared:
            return int(declared)
        return 7 if goal.task == "goat" else 6

    def caps_of(self, goal: Goal) -> dict[str, Any]:
        if goal.task != "goat":
            return {}
        caps: dict[str, Any] = {}
        if self.cfg.get("max_turns_per_goal"):
            caps["max_turns"] = min(
                int(self.cfg["max_turns"]), int(self.cfg["max_turns_per_goal"]) * goal.n_goals
            )
        if self.cfg.get("episode_timeout_per_goal"):
            caps["episode_timeout"] = min(
                int(self.cfg["episode_timeout"]),
                int(self.cfg["episode_timeout_per_goal"]) * goal.n_goals,
            )
        return caps

    def briefing_kwargs(self, goal: Goal) -> dict[str, Any]:
        return {**super().briefing_kwargs(goal), "n_actions": self.n_actions(goal)}

    def first_prompt(self, goal: Goal) -> str:
        return FIRST_PROMPT

    def bridge_knobs(self, goal: Goal) -> dict[str, Any]:
        return {"es_task": goal.task, "es_actions": self.n_actions(goal)}

    async def evaluate(self, sink: Any, goal: Goal) -> dict[str, Any]:
        return await self.call_evaluate(sink, {"trigger": "driver"})

    def unfinished_metrics(self, goal: Goal, metrics: dict[str, Any]) -> dict[str, Any]:
        """A GOAT episode cut short: every one of its ``n_goals`` sub-goals is
        scored. The package's per-sub-goal lists already hold 0 for a sub-goal
        it has not closed; without them (the env could not answer) every
        sub-goal is a failure. The episode's means are recomputed over all
        of them."""
        if goal.task != "goat":
            return super().unfinished_metrics(goal, metrics)
        n = max(1, int(goal.n_goals or 1), len(metrics.get("subtask_success") or []))
        succ = [float(x or 0.0) for x in (metrics.get("subtask_success") or [])]
        spl = [float(x or 0.0) for x in (metrics.get("spl_by_subtask") or [])]
        succ += [0.0] * (n - len(succ))
        spl += [0.0] * (n - len(spl))
        return {
            **metrics,
            "subtask_success": succ,
            "spl_by_subtask": spl,
            "n_subtasks": n,
            "success": sum(succ) / n,
            "spl": sum(spl) / n,
            "composite_success": float(all(s >= 1.0 for s in succ)),
        }


class EsNav(EsTask):
    shape = "es.nav"
    task_word = "nav"


class EsObjNav(EsTask):
    shape = "es.objnav"
    task_word = "objnav"


class EsGoat(EsTask):
    shape = "es.goat"
    task_word = "goat"

