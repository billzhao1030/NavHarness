"""Task shapes — the per-episode semantics between the loop and the harness.

A shape answers five questions the loop cannot: how an episode is PLACED
(which env-panel pushes, what reset returns, what the briefing embeds),
what the arm's briefing builder needs to know, what the FIRST prompt is,
how the episode is EVALUATED (which verb, which inputs), and what extra
fields the episode record carries. The
loop (runner.Runner.run_episode) calls these in order and never tests a benchmark
name; the env picks its shape per task family with one config literal
(``env.shapes.<family>: pkg.mod.Class`` in the experiment file).

Every shape is a subclass of ``TaskModule`` with a registry name (its
``shape`` attribute; ``core.shapes.SHAPES``). Built-ins: es.nav · es.objnav
· es.goat for the EmbodiedScore-envs stack (core.shapes.es). An env that
needs more than a built-in names its own class in ``shapes``.

Removing a shape removes the envs that declare it — nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.envclient import EnvClient

# The first user message of a session, per shape family. An arm may
# override it by defining FIRST_PROMPT in its prompts.py (the runner asks
# the arm first — core.arm.Arm.first_prompt).
FIRST_PROMPT = "Begin navigating. Call observe() first to see where you are."
EXPLORE_FIRST_PROMPT = "Begin exploring. Call observe() first to see where you are."


@dataclass
class Goal:
    """What the episode asks for, as the briefing embeds it. ``text`` is the
    recorded ``instruction`` (an instruction, a goal category, a formatted
    question …); ``task`` is the word handed to briefing builders that take
    a ``task=`` kwarg (nav / objnav / goat …)."""

    text: str
    task: str = "nav"
    n_goals: int = 1
    ep: dict[str, Any] = field(default_factory=dict)  # the reset payload

    @property
    def episode_id(self) -> Any:
        return self.ep.get("episode_id")

    @property
    def scene_id(self) -> Any:
        return self.ep.get("scene_id")


@dataclass
class Placement:
    goal: Goal
    caps: dict[str, Any] = field(default_factory=dict)  # per-episode cfg
    # overrides (GOAT: N sub-goals x the per-goal budget, under the episode cap)


class TaskModule:
    """One task shape. Instantiated per episode with the served env, the
    flat run config (runner.run_cfg) and the agent's arm; every method
    has a default for instruction-following navigation, subclasses
    override what differs for their env family."""

    shape: str = "nav"  # registry name (core.shapes.SHAPES)
    task_word: str = "nav"  # `task=` kwarg for builders that name it

    def __init__(self, env: EnvClient, cfg: dict[str, Any], arm: Any) -> None:
        self.env = env
        self.cfg = cfg
        self.arm = arm

    # ── run start: server preparation (once per server, before episode 0) ──

    @classmethod
    def check_health(cls, url: str, payload: dict[str, Any], env_name: str, run_name: str) -> None:
        """Refuse a server that serves another nodeset — a mismatched server
        would place the WRONG episodes, so fail loud rather than degrade."""
        server_name = payload["name"]
        print(f"[std] {url} healthy: {server_name}")
        if server_name != env_name:
            raise RuntimeError(
                f"{url} serves {server_name!r} but run {run_name} needs {env_name!r}"
            )

    @classmethod
    async def setup_server(cls, env: EnvClient, cfg: dict[str, Any]) -> None:
        """One ``configure`` that places the whole run: the task's line and
        split, plus any env-declared run-level switches (``env_fields``,
        e.g. the slam envs' ``instruments`` — pushed BEFORE any placement
        so the first seat already builds the right arm)."""
        extra = {name: cfg[name] for name in (cfg.get("env_fields") or ()) if name in cfg}
        out = await env.call("configure", line=cfg["dataset"], split=cfg["split"], **extra)
        if "error" in out:
            raise RuntimeError(f"{env.url} configure({cfg['dataset']}, {cfg['split']}): {out['error']}")

    # ── per episode ──

    async def place(self, index: int) -> Placement:
        """``place(index)`` on the env; the goal is read off the task facts
        it returned (``goal_of``)."""
        ep = await self.env.call("place", index=index)
        if "error" in ep:
            raise RuntimeError(f"{self.env.url} place({index}): {ep['error']}")
        goal = self.goal_of(ep, index)
        return Placement(goal, self.caps_of(goal))

    def goal_of(self, ep: dict[str, Any], index: int) -> Goal:
        raise NotImplementedError

    def caps_of(self, goal: Goal) -> dict[str, Any]:
        return {}

    def briefing_kwargs(self, goal: Goal) -> dict[str, Any]:
        """Everything an arm's ``build_briefing`` MAY take beyond
        (instruction, step_budget). The loop passes a key ONLY when the
        builder's signature names it, so a two-arg builder is called with
        exactly (instruction, step_budget)."""
        return {
            # any arm switch by name
            **self.arm.switches(),
            "task": goal.task,
            "n_goals": goal.n_goals,
            # slam arms: toolface switch (twotool default) and the tilt
            # mask; wp arms: the decision-step cap
            "face": str(self.cfg.get("toolface") or "twotool"),
            "tilt": bool(self.cfg.get("tilt_actions", True)),
            "wp_max_moves": self.cfg.get("wp_max_moves", 30),
        }

    def first_prompt(self, goal: Goal) -> str:
        return FIRST_PROMPT

    def bridge_knobs(self, goal: Goal) -> dict[str, Any]:
        """Shape-derived EpisodeContext fields the arm's surface may render
        into the bridge's environment (slam_task / slam_tilt / es_task /
        es_actions)."""
        return {}

    @staticmethod
    def answer_of(sink: Any) -> str:
        """The agent's answer, if it gave one: it rides the tool-result
        channel (the bridges' answer() result carries steps_taken_total, so
        it lands as the sink's last parsed step result)."""
        return str((sink.last_step_result or {}).get("answer") or "")

    async def evaluate(self, sink: Any, goal: Goal) -> dict[str, Any]:
        """The line's own ruler, driver-triggered; the agent never sees it."""
        return await self.call_evaluate(sink, {})

    async def call_evaluate(self, sink: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            metrics = await self.env.call("evaluate", **inputs)
            if "error" in metrics:
                raise RuntimeError(metrics["error"])
            return metrics
        except Exception as exc:
            sink.emit("driver_error", {"error": f"evaluate failed: {exc!r}"})
            return {}

    def record_extras(self, sink: Any, goal: Goal) -> dict[str, Any]:
        """Extra ``agent`` fields of the episode record."""
        return {"task_shape": goal.task}

    def unfinished_metrics(self, goal: Goal, metrics: dict[str, Any]) -> dict[str, Any]:
        """The metrics of an episode the wall-clock backstop cut short
        (``runner.Runner.timeout_record``): what the env scored stands, a
        task it has not scored counts as failed. ``metrics`` is what the
        env's ruler answered after the cut ({} when it could not answer).
        A single-goal episode the env has not scored is a failure."""
        out = dict(metrics)
        for key in ("success", "spl"):
            if not isinstance(out.get(key), (int, float)):
                out[key] = 0.0
        return out


def format_choices(question: str, choices: list[str]) -> str:
    """The multiple-choice question as the briefing's {question} slot needs
    it, mirroring explore-eqa's own formatting ("\\nA. <choice>")."""
    return question + "".join(
        f"\n{letter}. {c}" for letter, c in zip("ABCD", choices, strict=False)
    )
