"""The per-episode and per-run contexts every part of the runner shares.

- ``EpisodeContext``: everything a harness adapter needs for ONE clean
  session (briefing, prompts, budgets, the bridge to spawn and its
  environment). Built by the runner after the task shape placed the episode.
- ``SessionOutcome`` / ``HarnessAdapter``: the adapter contract.
- ``ModelSpec``: the model as the adapters' ``prepare()`` sees it (id + API
  params — the ``agent.model`` block of the four-layer config).
- ``RunContext``: the run as the callbacks see it (name, the full config,
  the flat run cfg, run_dir, the records so far, run-level stats and
  provenance).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from core.bus import EventBus as EventSink


@dataclass(frozen=True)
class ModelSpec:
    """``agent.model`` as the harness adapters read it once per run: the
    harness-facing id, the run's effort (the top-level ``effort`` key — the
    vendor's own word, ``default`` = send nothing), and the API routing
    params beside the id (api_base / api_gateway / gateway_route …)."""

    model_id: str
    effort: str = "default"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def extra_dict(self) -> dict[str, Any]:
        return dict(self.extra)

    @property
    def is_local(self) -> bool:
        """Served on our own GPU — no meter, no rate limit."""
        return self.model_id.startswith(("ollama", "hosted_vllm/"))


@dataclass
class EpisodeContext:
    index: int
    instruction: str
    briefing: str  # rendered system prompt / user-message briefing
    first_prompt: str
    server_url: str
    bare: bool
    model: str
    max_turns: int
    step_budget: int
    episode_timeout: int
    workdir: Path
    live_dir: Path
    raw_dir: Path
    env_name: str = ""  # the env object's class name (env._target_)
    effort: str = "default"  # the run's reasoning effort, the vendor's word; "default" = the API's own
    wp_server_url: str = ""  # waypoint-predictor auto_host (wp / hybrid arms)
    wp_max_moves: int = 30  # decision-step cap enforced by wp_bridge (wp only)
    benchmark: str = "r2r"  # the task key (task_key)
    hmeqa_tilt: bool = True  # camera-tilt actions 4/5 on the toolface
    # (task.tilt_actions; false masks them — bridge +
    # briefing follow together)
    instruments: int = 0  # slam arms: env-panel switch (0 bare / 1 map v1
    # / 2 map v2); the toolface + briefing come from the
    # arm folder, not this flag
    toolface: str = "twotool"  # slam arms: "twotool" (observe / step) |
    # "singlestep" (step(actions) returns the view);
    # rendered as SLAMB_FACE + into the briefing
    arm: Any = None  # the experiment's Arm (core.arm): bridge.py / prompts.py / surface
    max_budget_usd: float | None = None  # per-episode USD fuse (where the
    # harness supports one; the Claude Code CLI's --max-budget-usd)
    # auto-observe: step()/goto() carry the resulting view (observe() first-look
    # only), saving the observe() turn after every move.
    auto_observe: bool = False
    # toolbox surface: an arm's extended tool set, read by its bridge
    toolbox: bool = False
    # toolbox ground-truth switch: False = the vision-only variant
    toolbox_gt: bool = True
    # shape-derived bridge knobs (TaskModule.bridge_knobs): the slam arms'
    # task surface + tilt mask; the es task shape + action count (6
    # standard, 7 GOAT). None keeps a key out of the bridge's environment.
    slam_task: str | None = None
    slam_tilt: bool | None = None
    es_task: str | None = None
    es_actions: int | None = None
    es_arms: str | None = None  # the robot's arm names, comma-separated (None for navigation)
    extra: dict[str, Any] = field(default_factory=dict)  # harness-specific knobs
    run_dir: Path | None = None  # the run this episode belongs to
    bus: Any = None  # the episode's EventBus (set by the runner)

    def knob(self, name: str, default=None):
        """One arm switch (agent.arm.*: toolface / instruments …),
        ``default`` when the arm does not declare it."""
        if self.arm is None:
            return default
        return self.arm.switch(name, default)

    @property
    def turn_budget(self) -> int:
        """Bridge broadcast/STOP-gate budget: off (0) in the bare condition."""
        return 0 if self.bare else self.max_turns

    @property
    def bridge_path(self) -> Path:
        """The stdio bridge the session talks to: the owning arm's
        bridge.py, or the file its surface names (``surface.bridge``,
        relative to the folder)."""
        if self.arm is None:
            raise RuntimeError(
                f"cell for benchmark {self.benchmark!r} has no arm — there is no bridge to run"
            )
        return self.arm.bridge_path

    def bridge_env(self) -> dict[str, str]:
        """Env for the harnesses that spawn the stdio bridge, rendered by the
        arm from its own surface (``env_map``) — the driver knows no arm's
        knobs; a surface entry may name any attribute of this context or
        any switch of the arm."""
        if self.arm is None:
            raise RuntimeError(f"cell for benchmark {self.benchmark!r} has no arm")
        return self.arm.render_env(self)


@dataclass
class SessionOutcome:
    usage: dict[str, Any] | None = None
    cost_usd: float | None = None
    turns: int | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class HarnessAdapter(Protocol):
    name: str  # the harness's recorded name (e.g. "navh")
    inherent: dict  # recorded harness-inherent facts (thinking, auth, caps)

    def prepare(self, spec: ModelSpec) -> None:
        """Once per run, BEFORE any episode: auth guards, version pins into
        self.inherent, and any serving stack the harness owns (e.g. ollama
        for a local model). Raise to abort the run: a misconfigured server
        that silently degrades is worse than no run."""

    def describe(self, ctx: EpisodeContext) -> dict[str, Any]:
        """Harness-specific block merged into the session_inputs event."""

    async def run(self, ctx: EpisodeContext, sink: EventSink) -> SessionOutcome:
        """Run ONE clean session; emit events through ``sink.emit`` only and
        hand every harness-native record to ``sink.raw`` (core.bus.EventBus)."""

    def finalize(self, run_dir: Path) -> dict[str, Any]:
        """Optional. Once per run, AFTER the last episode: whatever audit the
        harness alone can produce (e.g. exact per-request prompt-token counts
        sliced from a local serve log). Merged into summary.run_stats."""


@dataclass
class RunContext:
    """One run, as the callbacks see it."""

    name: str  # run.name
    app: dict[str, Any]  # the full four-layer config, resolved (task · env · agent · run)
    cfg: dict[str, Any]  # the flat run cfg (runner.run_cfg)
    run_dir: Path
    servers: list[str]
    adapter: Any
    arm: Any  # the agent's Arm (core.arm)
    module_cls: type  # the TaskModule class serving it
    env_name: str = ""  # env._target_ class name
    episodes_spec: str | None = None
    prior_episodes: dict[int, dict[str, Any]] = field(default_factory=dict)  # resume
    episodes: dict[int, dict[str, Any]] = field(default_factory=dict)  # records so far
    run_stats: dict[str, Any] = field(default_factory=dict)  # summary.run_stats
    provenance: dict[str, Any] = field(default_factory=dict)  # summary.provenance
    summary_extra: dict[str, Any] = field(default_factory=dict)  # run-level keys callbacks add
    callback_errors: dict[str, str] = field(default_factory=dict)
    callbacks: Any = None  # the CallbackSet (set by the runner)
    stopped_by: str | None = None  # the callback that raised StopRun
    drain: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "summary.json"
