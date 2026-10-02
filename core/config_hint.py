"""The four-layer experiment config, as a dataclass tree — the IDE hint and
the structured schema for exp_workspace/<arm>/configs/<run name>.yaml.
One file = one experiment, every
value written out. Nothing here composes or validates at runtime — runner.py
composes with Hydra and hands the runner a plain dict — and it is not a
strict schema either: four blocks are open by design (an arm's switches, a
benchmark's own task keys, a harness's constants, a model's API params), so
``OmegaConf.structured(AppCfg)`` would reject every real file. Read it as
the field-by-field meaning of the four layers.

    task    what is measured: data + the benchmark's own budgets
    env     where: simulator stack + shape per task family + body deviations
    agent   who: arm (interface + switches) · harness · model
    run     this run: name (assembled) · dir · episodes · servers · recorders
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING


@dataclass
class TaskCfg:
    """One benchmark protocol: the data and the budgets the benchmark itself
    fixes. Extra keys are benchmark-specific (n_actions, max_turns_per_goal,
    episode_timeout_per_goal, max_budget_usd, tilt_actions …)."""

    family: str = (
        MISSING  # nav | objnav | goat | tour … -> env.shapes[family]
    )
    dataset: Optional[str] = None  # noqa: UP045 — OmegaConf structured configs need typing.Optional  # the env panel's dataset selector (None: single-dataset panels)
    split: str = MISSING
    episodes: str = MISSING  # the full set, "0-99"
    max_turns: int = MISSING  # harness turn cap
    step_budget: int = MISSING  # env action budget
    episode_timeout: int = MISSING  # seconds
    rgb_resolution: Any = None  # the benchmark's sensor (512 | "480x640" | [480, 640]); reaches the env's
    # configure() only when env.env_fields lists it
    require: list[str] = field(default_factory=list)  # data files that must exist


@dataclass
class EnvCfg:
    """One world: the env object (``_target_`` + its kwargs, e.g. ``variant``),
    served by core.envserver under ``python`` on ``port`` — the runner spawns it
    when ``run.servers`` is empty — and the TaskModule per task family."""

    _target_: str = MISSING  # exp_workspace.<arm>.mcp.env.<Class>
    python: Optional[str] = None  # noqa: UP045  # interpreter: conda env name or path (None: this one)
    port: int = 9200
    shapes: dict[str, str] = MISSING  # family -> core.shapes.<mod>.<Class>
    env_fields: list[str] = field(default_factory=list)  # cfg keys passed to configure() (run-level env switches)
    # the env class's own kwargs beyond these — what its __init__ takes, e.g.:
    #   body — Body overrides on the preset (forward_step_m, turn_deg, tilt_deg, camera_pitch_deg,
    #     agent_height_m, rgb: {width, height, hfov_deg, position}, depth: null …); {} = the preset as is
    #   slam — the SLAM stack's knobs (stack.SLAM_KEYS): cell_size_m, map_size_m, max_coarse_steps,
    #     frontier_match_m, start_pitch_deg, map_mode (v1 | v2 | null = v2 for instruments=2 else v1),
    #     mapper (the map class's kwargs), render (render_annotated_map's)
    body: dict[str, Any] = field(default_factory=dict)
    slam: Optional[dict[str, Any]] = None  # noqa: UP045


@dataclass
class SurfaceCfg:
    env_prefix: str = MISSING
    env_map: dict[str, str] = MISSING  # <ENV SUFFIX>: <EpisodeContext attr | arm switch>
    bridge: Optional[str] = None  # noqa: UP045 — OmegaConf structured configs need typing.Optional  # bridge path relative to the arm folder (default bridge.py)
    allowed_tools: Optional[list[str]] = None  # noqa: UP045
    needs_wp_server: bool = False


@dataclass
class ArmCfg:
    """The agent's interface: a code folder + its switches. Any extra key is
    a switch (bare / toolface / instruments / auto_observe / wp_max_moves …)
    — read through Arm.switch() / ctx.knob()."""

    _target_: str = "core.arm.Arm"
    dir: str = MISSING  # exp_workspace/<arm>
    word: str = MISSING  # the {arm} segment of run.name
    prompts: str = "prompts.py"
    surface: SurfaceCfg = MISSING


@dataclass
class HarnessCfg:
    """The adapter class + its constants (constructor kwargs); ``root`` and
    ``output_dir`` name the run and its output root. NavHarness's constants:
    transport, persist_map, precheck, certify, compact, chunk_turns,
    stop_views, max_turns_per_task, step_budget_per_task, the recovery ladder
    (recovery, recovery_ceiling, reset_after_turns, max_resets_per_task …),
    the memory tiers (place_graph, notes …), judge_model and the session
    policy (session_mode, ledger) — see configs/harness/<name>.yaml."""

    _target_: str = MISSING
    root: str = MISSING  # navh | navhcodex | navhmini — the {harness} segment of run.name
    output_dir: str = MISSING  # outputs/<output_dir>


@dataclass
class ModelCfg:
    """One model for one harness: the harness-facing id, the API routing
    params beside it (api_base / api_gateway / gateway_route …) and its
    sampling ``params`` (temperature / max_tokens / top_p …) — consumed by
    the mini-swe-agent core (litellm kwargs); Claude Code and the Codex CLI
    refuse them. Effort is not here — it is the top-level ``effort`` key."""

    id: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryCfg:
    """The rate-limit re-run policy (core.retry.RetryPolicy)."""

    max_attempts: int = 6
    base_backoff_s: float = 30.0  # exponential per attempt
    max_backoff_s: float = 300.0  # per-backoff cap


@dataclass
class RunCfg:
    name: str = (
        MISSING  # assembled: {prefix}_{task}_{env}_{harness root}_{model_key}_{effort}_{arm word}{suffix}
    )
    dir: str = MISSING
    episodes: str = MISSING  # the indices THIS launch runs (default: the task's full set)
    servers: Optional[list[str]] = None  # noqa: UP045  # attach to these env servers; None: spawn run.workers from env
    workers: int = 1  # +run.workers=N on the command line (not a yaml key): run.servers empty -> spawn N env servers on env.port, env.port+1, …; one worker per server
    wp_server: Optional[str] = None  # noqa: UP045 — OmegaConf structured configs need typing.Optional
    resume: bool = False
    dry_run: bool = False
    budget_usd: Optional[float] = None  # noqa: UP045  # run-level cost fuse (callbacks.CostLogger)
    note: Optional[str] = None  # noqa: UP045 — OmegaConf structured configs need typing.Optional
    retry: RetryCfg = field(default_factory=RetryCfg)  # rate-limit re-runs; absent = core.retry.DEFAULT_RETRY
    server_timeout_s: float = 600.0  # a spawned env server: wait for /health this long (cold scene loads)
    client_timeout_s: float = 600.0  # one env verb (place / step / evaluate …) over HTTP
    callbacks: list[Any] = MISSING  # [{_target_: core.callbacks.<mod>.<Class>, …}]


@dataclass
class ApiCfg:
    """The endpoint the harness talks to — the experiment's api/ group,
    picked with ``api=<native | fake>``; every key lands beside the model id
    in ModelSpec.extra. native: nothing (the vendor's own endpoint). fake:
    the scripted endpoint (core/api/fake_wire.py) — a real harness, no
    model; the run is renamed fakeapi_… by runner.apply_fake_api."""

    api_gateway: Optional[str] = None  # noqa: UP045  # null | fake (| litellm, from a models-table row)
    gateway_task: Optional[str] = None  # noqa: UP045  # fake: the task key the script closes on (${task_key})
    gateway_script: Optional[Any] = None  # noqa: UP045  # fake: [[tool, args], …] replacing the default walk


@dataclass
class AgentCfg:
    arm: ArmCfg = MISSING
    harness: HarnessCfg = MISSING
    model: ModelCfg = MISSING
    api: ApiCfg = field(default_factory=ApiCfg)


@dataclass
class AppCfg:
    # the five-tuple the run name is made of (recorded; run.name is the identity)
    task_key: str = MISSING
    env_key: str = MISSING
    harness_key: str = MISSING  # derived: ${harness.root} — the harness chosen with `harness=<navh | navhcodex | navhmini>`
    model: str = MISSING  # the model chosen with `model=<name>`: a row of models/models.yaml
    model_key: str = MISSING  # derived: ${model}
    effort: str = "default"  # reasoning effort, the vendor's own word: Claude low|medium|high|max,
    # OpenAI low|medium|high|xhigh; "default" = the API's own default, nothing sent. An explicit
    # parameter of every harness adapter (EpisodeContext.effort), and a segment of run.name.
    prefix: str = "std"
    suffix: str = ""
    harness: HarnessCfg = MISSING  # the Hydra group beside the file, configs/harness/<name>.yaml (defaults: - harness: ???); agent.harness = ${harness}
    models: dict[str, dict[str, Any]] = MISSING  # configs/models/models.yaml beside the file (defaults: - models: models): model -> harness -> ModelCfg; agent.model = ${models[${model}][${harness.root}]}
    task: TaskCfg = MISSING
    env: EnvCfg = MISSING
    agent: AgentCfg = MISSING
    run: RunCfg = MISSING
