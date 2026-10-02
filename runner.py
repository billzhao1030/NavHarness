"""runner — the single entry point: run one experiment from one fully written
config.

    python runner.py <experiment.yaml | run name> [key=value …]
    python runner.py --config-name=<run name> [key=value …]        # Hydra's spelling

An experiment is ONE file, ``exp_workspace/<arm>/configs/<name>.yaml``,
four layers and nothing else, every value written out:

    task    what is measured: data + the benchmark's own budgets and body
    env     where: the nodeset + the shape (TaskModule) per task family
    agent   who: arm (dir · word · switches · surface) · harness (adapter + constants) · model (id + API params)
    run     this run: name · dir · episodes · servers · wp_server · recorders

Nothing is looked up elsewhere: a different value is a different experiment
(copy the file, rename it). Along the harness and model axes one file is
many seats. The harness is a Hydra config group beside the file —
``exp_workspace/<arm>/configs/harness/{navh,navhcodex,navhmini}.yaml``,
NavHarness over its three interchangeable reasoning cores (Claude Code, the
Codex CLI, mini-swe-agent) — that the file requires
(``defaults: - harness: ???``) and the command line picks
(``harness=navhmini``, ``harness.chunk_turns=30`` to touch one of its constants).
The model is one row of the sibling table ``models/models.yaml`` (model ->
harness -> the harness-facing id), picked with ``model=opus-5``.
``effort`` (top-level, default ``default``) is the run's reasoning effort in
the vendor's own word — an explicit parameter of every harness adapter, never
an ``agent.model`` extra. ``api`` is the endpoint, a third group beside the
file (``api/{native,fake}.yaml``, default native): fake = the scripted
endpoint, the real harness with no model. All three complete the run name
(``std_goat_slam_es_navh_opus-5_default_lhnav``); a judge on another model
or core (``harness.judge_model`` / ``harness.judge_transport``) adds itself
(``…_lhnav_judge-claude-opus-5-sdk``). Command-line overrides are
dotted paths into the same four layers (``run.episodes=0-9``,
``task.max_turns=300``). The resolved config is saved as ``<run.dir>/config.yaml``.

    python runner.py std_goat_slam_es_lhnav harness=navh model=opus-5
    python runner.py std_goat_slam_es_lhnav harness=navhcodex model=gpt-6 effort=medium
    python runner.py std_goat_slam_es_lhnav harness=navhmini model=qwen3.8-27b
    python runner.py std_ivlnce_tour_slam_es_lhnav harness=navh model=opus-5
    python runner.py std_goat_slam_es_lhnav harness=navh model=opus-5 run.dry_run=true          # print the resolved config, run nothing
    python runner.py std_goat_slam_es_lhnav harness=navh model=opus-5 api=fake run.episodes=0   # wiring check: the real harness on the scripted endpoint, no model
    python runner.py std_goat_slam_es_lhnav harness=navh model=opus-5 run.episodes=0-9          # part of the episodes -> a test_… run
    python runner.py std_goat_slam_es_lhnav harness=navh model=opus-5 run.episodes=3,7 run.resume=true   # re-run episodes inside the full run

The file is the loop too (region Runner): per run, let the harness prepare,
check every server through the task shape, snapshot and start the callbacks
(on_run_start), prepare every server for the whole run (dataset / split),
then one worker per server pulls episode indices from a queue until it is
empty or the run drains. Per episode: the shape places it, the arm's
prompts.py renders the briefing, the harness runs one clean session through
an EventBus, the shape evaluates, and the record is built — the callbacks
see every step (core.callbacks.base for the hooks). A rate-limited session
is re-run in place (core.retry), earlier attempts archived as
``*.attempt{k}*``. summary.json is flushed after every episode;
``summary.provenance`` names the code, env, shape, harness and callbacks
that produced the run, ``summary.writers`` maps every callback-owned key /
file to its class.

Graceful drain: while a run is live, ``touch <run.dir>/DRAIN`` (or
``kill -USR1 <pid>``) makes every worker finish its current episode, flush,
and exit WITHOUT starting a new one; the runner prints the un-run indices
as a ready-to-paste ``run.resume=true run.episodes=…`` for the next launch.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
import yaml
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException, MissingConfigException
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationKeyError, MissingMandatoryValue

from core import envserver as env_server
from core.bus import EventBus
from core.callbacks import Callback, CallbackSet, StopRun
from core.envclient import EnvClient
from core.episode import EpisodeContext, HarnessAdapter, ModelSpec, RunContext
from core.panel import bridge_tool_schemas, json_safe
from core.paths import EXP_WORKSPACE, REPO_ROOT
from core.retry import DEFAULT_RETRY, RetryPolicy, is_rate_limited
from core.scores import pooled
from core.shapes import TaskModule

OmegaConf.register_new_resolver("eval", eval, replace=True)

# harness keys that name the run / its output root, not adapter kwargs
HARNESS_META = ("root", "output_dir")

# Env clients are built through this hook so a no-server check can hand the
# loop a canned env instead of monkeypatching HTTP verbs.
env_client_factory = EnvClient

# How long the env is given to report its metrics once more after the
# wall-clock backstop has cut a sequence short (Runner.timeout_record).
TIMEOUT_EVALUATE_S = 120.0

# ---------------------------------------------------------------
# region 0. Tools — config, instantiation


EXPERIMENTS = EXP_WORKSPACE  # every arm's configs/*.yaml


def resolve_experiment(token: str) -> Path:
    """The experiment file a token names: a path to a .yaml, or a run name
    found under exp_workspace/*/configs/."""
    path = Path(token)
    if token.endswith(".yaml") or path.is_file():
        if not path.is_file():
            sys.exit(f"[std] no experiment file {path}")
        return path.resolve()
    hits = sorted(EXPERIMENTS.glob(f"*/configs/{token}.yaml"))
    if len(hits) != 1:
        sys.exit(
            f"[std] {token!r}: {'no' if not hits else len(hits)} experiment file(s) under "
            f"{EXPERIMENTS} — pass a path"
        )
    return hits[0]


def compose_app(experiment: Path | str, overrides: list[str]) -> dict[str, Any]:
    """The resolved four-layer config of one experiment file (+ overrides)
    as a plain dict."""
    path = resolve_experiment(str(experiment))
    # the file's folder is the Hydra config dir; its harness/ subfolder is the
    # experiment's own group (`defaults: - harness: ???` + `harness=navh`), its
    # models/models.yaml the table `model=opus-5` picks a row of
    try:
        with initialize_config_dir(config_dir=str(path.parent), version_base=None):
            cfg: DictConfig = compose(config_name=path.stem, overrides=overrides)
    except (ConfigCompositionException, MissingConfigException) as exc:  # no / unknown harness= or model=
        sys.exit(
            f"[std] {path.name}: {str(exc).splitlines()[0]} — harness=<one of: {', '.join(harness_names(path))}> "
            f"model=<one of: {', '.join(model_names(path))}> api=<one of: {', '.join(api_names(path))}>"
        )
    try:  # the chosen model must have an entry under the chosen harness
        OmegaConf.resolve(cfg)
        app = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    except (MissingMandatoryValue, InterpolationKeyError) as exc:
        chosen = str(cfg.harness.root) if "harness" in cfg else None
        models = [m for m in model_names(path) if chosen is None or chosen in model_spec(path, m)]
        sys.exit(
            f"[std] {path.name}: {str(exc).splitlines()[0]} — pass "
            f"model=<one of{' (under ' + chosen + ')' if chosen else ''}: {', '.join(models)}>"
        )
    assert isinstance(app, dict)
    # run.dir is written relative in every experiment file (outputs/<harness dir>/<run name>);
    # relative paths in a config are repo-root relative, never cwd — the same rule as agent.arm.dir
    run_dir = Path(app["run"]["dir"])
    if not run_dir.is_absolute():
        app["run"]["dir"] = str(REPO_ROOT / run_dir)
    return app


def apply_test_rule(app: dict[str, Any]) -> dict[str, Any]:
    """A ``std_`` run over only part of the task's episodes is renamed
    ``test_…`` so it is never mistaken for a full run. Re-running episodes
    inside an existing full run takes BOTH ``run.resume=true`` and a run dir
    that already holds records (intent + evidence)."""
    run = app["run"]
    full = set(parse_episodes(str(app["task"]["episodes"])))
    asked = set(parse_episodes(str(run["episodes"])))
    name: str = run["name"]
    if asked == full or not name.startswith("std_"):
        return app
    run_dir = Path(run["dir"])
    has_board = (run_dir / "summary.json").is_file()
    if run.get("resume"):
        if not has_board:
            sys.exit(
                f"[std] run.resume=true but no full run at {run_dir} — "
                "nothing to fill (drop run.resume for a test_ run)"
            )
        return app  # fill / re-run episodes of the existing full run
    new_name = "test_" + name.removeprefix("std_")
    run["name"] = new_name
    run["dir"] = str(run_dir.with_name(new_name))
    hint = (
        " (the full run is untouched — add run.resume=true to fill it)"
        if has_board
        else ""
    )
    print(f"[std] partial episodes -> test run {new_name}{hint}")
    return app


def apply_judge_name(app: dict[str, Any]) -> dict[str, Any]:
    """A run whose two completion checks are another model's, or sit on
    another core (``harness.judge_model`` / ``harness.judge_transport``, the
    cross-model judging comparison), is named for its judge —
    ``…_judge-<model>-<core>`` — so it never lands in the run of the same
    seat with its own judge, whose episodes it would replace."""
    harness = app["agent"]["harness"]
    seat = str(app["agent"]["model"]["id"])
    judge = str(harness.get("judge_model") or seat)
    core = str(harness.get("judge_transport") or harness.get("transport") or "")
    if judge == seat and core == str(harness.get("transport") or ""):
        return app
    run = app["run"]
    tag = "".join(c if c.isalnum() or c in ".-" else "-" for c in judge).strip("-")
    tag = f"_judge-{tag}-{core.split('-')[0]}"
    if tag not in run["name"]:            # ``suffix`` (a repeat's, say) stays; the judge is named once
        run["name"] = f"{run['name']}{tag}"
        run["dir"] = str(Path(run["dir"]).with_name(run["name"]))
    return app


def apply_fake_api(app: dict[str, Any]) -> dict[str, Any]:
    """``api=fake`` (the experiment's api/ group): the REAL harness (Claude
    Code / Codex CLI / mini-swe-agent core) against the scripted endpoint
    (core.api.fake_wire) instead of a vendor — the adapter, its loop, the
    tool schemas it registers and the bridge all exercised, no tokens. The
    block itself reaches the adapter through ModelSpec.extra (api_gateway:
    fake + gateway_task, optionally gateway_script); this rule only renames
    the run ``fakeapi_…`` under the harness's own root, so it is never
    mistaken for a model's run."""
    run = app["run"]
    if (app["agent"].get("api") or {}).get("api_gateway") != "fake":
        return app
    name = run["name"]
    for prefix in ("std_", "test_", "ui_", "xapi_"):
        name = name.removeprefix(prefix)
    run["name"] = f"fakeapi_{name}"
    run["dir"] = str(Path(run["dir"]).with_name(run["name"]))
    print(f"[std] fake endpoint -> run {run['name']}")
    return app


def build(app: dict[str, Any]) -> tuple[Any, Any, type, list[Callback]]:
    """Instantiate the agent's parts from the config: the harness adapter,
    the arm, the task shape the env names for the task's family, and the
    recorders."""
    harness = {k: v for k, v in app["agent"]["harness"].items() if k not in HARNESS_META}
    adapter = instantiate(harness, _convert_="all")
    arm = instantiate(app["agent"]["arm"], _convert_="all")
    family = app["task"]["family"]
    shapes = app["env"]["shapes"]
    if family not in shapes:
        raise KeyError(
            f"env {app['env_key']!r} declares no shape for task family {family!r} "
            f"(shapes: {sorted(shapes)})"
        )
    module_cls = get_class(shapes[family])
    callbacks: list[Callback] = []
    for entry in app["run"]["callbacks"]:
        cb = instantiate(entry, _convert_="all")
        if not isinstance(cb, Callback):
            raise TypeError(f"{entry.get('_target_')!r} is not a Callback subclass")
        callbacks.append(cb)
    return adapter, arm, module_cls, callbacks


def show(app: dict[str, Any], arm: Any) -> None:
    """The resolved config plus the flat run cfg the loop derives from it."""
    view = dict(app)
    view["run_cfg"] = run_cfg(app, arm)
    print(yaml.safe_dump(view, sort_keys=False, allow_unicode=True), end="")


# endregion
# ---------------------------------------------------------------
# region 1. Records and the flat run cfg


# ── records ────────────────────────────────────────────────────────────────


def is_scored(rec: dict[str, Any]) -> bool:
    """True if the episode is a real, scored navigation attempt that counts
    toward SR.

    INCLUDES a turn-exhausted episode (hit the SDK ``max_turns`` cap without
    calling STOP): it navigated, was evaluated, ``success`` is 0.0 — a
    legitimate FAILURE, not an error to hide (dropping it inflates SR).

    INCLUDES a sequence the wall-clock backstop cut short (``error ==
    "timeout"`` with metrics, ``Runner.timeout_record``): its tasks stay in
    the denominators, the ones the env had not scored counted as failed, so
    a run is scored over the benchmark's full task set.

    EXCLUDES two kinds of non-attempts:
    - genuine infra failures (a crash, or a backstop before the sequence was
      placed) that produced no metrics (``metrics == {}`` → ``success is None``);
    - rate-limit / 'limit exceeded' casualties — errored WITHOUT taking a
      single navigation step (``error`` set + ``env_steps`` 0), so the model
      never really attempted the task. (A genuine turn-exhausted run has
      ``env_steps`` > 0 and stays counted.)
    """
    if (rec.get("metrics") or {}).get("success") is None:
        return False
    if rec.get("error") == "timeout":
        return True
    return not (rec.get("error") and not ((rec.get("agent") or {}).get("env_steps") or 0))


def aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """The run's numbers over its scored sequences: every numeric metric
    averaged over sequences (the env's per-sequence means, as recorded), the
    stop rate, and the scores over tasks the paper reports — ``tasks``,
    ``s_sr``, ``task_spl``, ``e_sr`` (``core.scores.pooled``)."""
    agg: dict[str, Any] = {"episode_count": len(episodes)}
    numeric: dict[str, list[float]] = {}
    for rec in episodes:
        for key, value in (rec.get("metrics") or {}).items():
            if isinstance(value, bool):
                value = float(value)
            if isinstance(value, (int, float)):
                numeric.setdefault(key, []).append(float(value))
        numeric.setdefault("env_steps", []).append(float(rec["agent"].get("env_steps", 0)))
    for key, values in numeric.items():
        if values:
            agg[key] = round(sum(values) / len(values), 4)
    agg["stop_rate"] = round(
        sum(1 for r in episodes if r["agent"].get("called_stop")) / max(1, len(episodes)), 4
    )
    agg.update(pooled(episodes))
    return agg


def format_episodes(indices: list[int]) -> str:
    """Inverse of parse_episodes: [7,8,9,14,44] -> '7-9,14,44'."""
    xs = sorted(set(indices))
    if not xs:
        return ""
    out: list[str] = []
    start = prev = xs[0]
    for x in xs[1:]:
        if x == prev + 1:
            prev = x
            continue
        out.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = x
    out.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(out)


def parse_episodes(spec: str) -> list[int]:
    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.extend(range(int(lo), int(hi) + 1))
        elif part:
            indices.append(int(part))
    return indices


def archive_attempt(run_dir: Path, index: int, attempt: int) -> None:
    """Before attempt k+1 re-runs an episode, move attempt k's files aside so
    every attempt is kept. The final attempt keeps the plain names, which is
    all run_stats reads."""
    suffix = f".attempt{attempt}"
    moves = [
        (run_dir / f"episode_{index}.jsonl", run_dir / f"episode_{index}{suffix}.jsonl"),
        (
            run_dir / "raw" / f"episode_{index}.jsonl",
            run_dir / "raw" / f"episode_{index}{suffix}.jsonl",
        ),
        (
            run_dir / "raw" / f"episode_{index}.stderr.log",
            run_dir / "raw" / f"episode_{index}{suffix}.stderr.log",
        ),
        (
            run_dir / "raw" / f"context_manifest_{index}.json",
            run_dir / "raw" / f"context_manifest_{index}{suffix}.json",
        ),
        (run_dir / f"live_{index}", run_dir / f"live_{index}{suffix}"),
    ]
    for src, dst in moves:
        if src.exists():
            src.rename(dst)


def _stub_record(index: int, error: str) -> dict[str, Any]:
    return {
        "index": index,
        "error": error,
        "metrics": {},
        "agent": {"env_steps": 0, "called_stop": False, "tool_calls": {}},
    }


class _NoSink:
    """A sink for the env calls made after an episode's bus has closed."""

    last_step_result: dict[str, Any] | None = None

    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        pass

    def raw(self, rec: Any) -> None:
        pass


# ── the flat run cfg ───────────────────────────────────────────────────────

# env keys the shapes read by name (run-level env switches; the task's own
# keys — rgb_resolution among them — are already in cfg, and env_fields names
# which of them the env's configure() takes)
ENV_CFG_KEYS = (
    "tilt_actions",
    "env_fields",
)
# arm switches the loop / the shapes / the bridge read by name (the rest of
# the arm's switches stay on the Arm and are reached through ctx.knob())
ARM_CFG_KEYS = (
    "auto_observe",
    "wp_max_moves",
    "toolface",
    "instruments",
    "toolbox",
    "toolbox_gt",
)


def model_extra(model: dict[str, Any], api: dict[str, Any] | None = None) -> dict[str, Any]:
    """The API routing params beside the model id (api_base / api_gateway /
    gateway_route …) and the model's sampling ``params`` (temperature /
    max_tokens …, consumed by harnesses that expose them — the mini-swe-agent
    core; Claude Code and the Codex CLI refuse them): everything in
    ``agent.model`` but ``id``, then the experiment's api/ block
    (``agent.api``, picked with ``api=<name>``) on top, unset (null) values
    left out. Effort is NOT one of them — it is
    the top-level ``effort`` key, an explicit parameter of every harness
    adapter."""
    if "effort" in model:
        sys.exit("[std] agent.model.effort is not a knob — set the top-level `effort` key (effort=max …)")
    extra = {k: v for k, v in model.items() if k != "id" and v is not None}
    extra.update({k: v for k, v in (api or {}).items() if v is not None})
    return extra


def run_cfg(app: dict[str, Any], arm: Any) -> dict[str, Any]:
    """The flat run cfg from the four layers: the task's data + budgets,
    the env's stack facts, the arm's switches, the run's fuse and the
    model's params — the dict the shapes (``cfg["dataset"]``), the episode
    context and the recorders read. No defaults of its own: every key is
    present exactly when the config declares it."""
    task, env, run = app["task"], app["env"], app["run"]
    cfg: dict[str, Any] = {k: v for k, v in task.items() if k != "family"}
    for key in ENV_CFG_KEYS:
        if key in env:
            cfg[key] = env[key]
    switches = arm.switches()
    for key in ARM_CFG_KEYS:
        if key in switches:
            cfg[key] = switches[key]
    cfg["wp_server"] = run.get("wp_server") if arm.needs_wp_server else None
    if run.get("budget_usd") is not None:  # run-level fuse, see callbacks.CostLogger
        cfg["budget_usd"] = float(run["budget_usd"])
    cfg["extra"] = model_extra(app["agent"]["model"], app["agent"].get("api"))
    if run.get("instruction"):  # a free-form instruction: no dataset to read one from
        cfg["extra"]["instruction"] = str(run["instruction"])
    return cfg


# endregion
# ---------------------------------------------------------------
# region 2. Runner


class Runner:
    def __init__(
        self,
        adapter: HarnessAdapter,
        arm: Any,
        module_cls: type[TaskModule],
        app: dict[str, Any],
        callbacks: list[Callback],
        retry: RetryPolicy = DEFAULT_RETRY,
    ) -> None:
        self.adapter = adapter
        self.arm = arm
        self.module_cls = module_cls
        self.app = app
        self.name: str = app["run"]["name"]
        self.servers = list(app["run"].get("servers") or [])  # empty: the runner spawns one
        self.env_name: str = str(app["env"]["_target_"]).rpartition(".")[2]  # the env object's class
        self._spawned: list[env_server.ServerHandle] = []
        self.task_key: str = app["task_key"]
        self.model = ModelSpec(
            app["agent"]["model"]["id"],
            effort=str(app.get("effort") or "default"),
            extra=model_extra(app["agent"]["model"], app["agent"].get("api")),
        )
        self.cfg = run_cfg(app, arm)
        self._callbacks = list(callbacks)
        self.retry = retry
        # run.client_timeout_s: one env verb over HTTP (place / step / evaluate …)
        self.client_timeout_s = float(app["run"].get("client_timeout_s") or 600.0)
        self.run_ctx: RunContext | None = None  # the run in progress
        # index -> {module, goal} of an episode in flight, from its placement on:
        # what timeout_record needs to score a sequence the backstop cut short
        self._inflight: dict[int, dict[str, Any]] = {}

    # ── one episode ──

    async def run_episode(self, cfg: dict[str, Any], url: str, index: int) -> dict[str, Any]:
        run = self.run_ctx
        assert run is not None
        started = time.monotonic()
        env = env_client_factory(url, timeout_s=self.client_timeout_s)
        module = self.module_cls(env, cfg, self.arm)

        placement = await module.place(index)
        goal = placement.goal
        self._inflight[index] = {"module": module, "goal": goal}
        if placement.caps:
            cfg = {**cfg, **placement.caps}  # per-episode caps (GOAT); run cfg untouched

        briefing = self.arm.briefing(goal, cfg["step_budget"], module.briefing_kwargs(goal))

        workdir = run.run_dir / f"workdir_{index}"
        workdir.mkdir(parents=True, exist_ok=True)
        raw_dir = run.run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        ctx = EpisodeContext(
            index=index,
            instruction=goal.text,
            briefing=briefing,
            first_prompt=self.arm.first_prompt(goal, module),
            server_url=url,
            bare=self.arm.bare,
            model=self.model.model_id,
            effort=self.model.effort,
            max_turns=cfg["max_turns"],
            step_budget=cfg["step_budget"],
            episode_timeout=cfg["episode_timeout"],
            workdir=workdir,
            live_dir=run.run_dir / f"live_{index}",
            raw_dir=raw_dir,
            env_name=self.env_name,
            benchmark=self.task_key,
            hmeqa_tilt=bool(cfg.get("tilt_actions", True)),
            instruments=int(cfg.get("instruments") or 0),
            toolface=str(cfg.get("toolface") or "twotool"),
            arm=self.arm,
            max_budget_usd=cfg.get("max_budget_usd"),
            auto_observe=bool(cfg.get("auto_observe", False)),
            toolbox=bool(cfg.get("toolbox", False)),
            toolbox_gt=bool(cfg.get("toolbox_gt", True)),
            wp_server_url=cfg.get("wp_server") or "",
            wp_max_moves=cfg.get("wp_max_moves", 30),
            extra=dict(cfg.get("extra") or {}),
            run_dir=run.run_dir,
            **module.bridge_knobs(goal),
        )

        callbacks = run.callbacks
        setup = time.monotonic() - started  # placement and briefing, before the session
        bus = EventBus(ctx, callbacks)
        ctx.bus = bus
        record: dict[str, Any] | None = None
        callbacks.fire("on_episode_start", ctx)
        try:
            bus.emit(
                "episode_meta",
                {
                    "index": index,
                    "episode_id": goal.episode_id,
                    "scene_id": goal.scene_id,
                    "instruction": goal.text,
                },
            )
            bus.emit(
                "session_inputs",
                {
                    "cell": self.name,
                    "harness": self.adapter.name,
                    "model": self.model.model_id,
                    "system_prompt": briefing,
                    "first_prompt": ctx.first_prompt,
                    # The recorded surface is the one the agent gets: same bridge,
                    # same environment the session hands it.
                    "tool_schemas": await bridge_tool_schemas(ctx.bridge_path, ctx.bridge_env()),
                    **json_safe(self.adapter.describe(ctx)),
                },
            )

            outcome = await self.adapter.run(ctx, bus)

            bus.emit(
                "result",
                {
                    "result": json_safe(
                        {
                            "usage": outcome.usage,
                            "cost_usd": outcome.cost_usd,
                            "turns": outcome.turns,
                            "error": outcome.error,
                            **outcome.extra,
                        }
                    )
                },
            )

            # Evaluate while the trajectory is still open so the metrics land
            # in the log itself. Driver-side; the agent never sees it.
            metrics = await module.evaluate(bus, goal)
            bus.emit("episode_metrics", {"metrics": metrics})
            wall = bus.elapsed

            last = bus.last_step_result or {}
            record = {
                "index": index,
                "episode_id": goal.episode_id,
                "scene_id": goal.scene_id,
                "instruction": goal.text,
                "metrics": metrics,
                "agent": {
                    "tool_calls": bus.tool_calls,
                    "env_steps": last.get("steps_taken_total", 0),
                    "end_reason": last.get("end_reason"),
                    # deliberate termination: STOP, answer() or closing the last
                    # sub-goal — the same "the agent chose to end" semantics
                    "called_stop": last.get("end_reason")
                    in ("stop_called", "answer_submitted", "subtasks_closed"),
                    "num_turns": outcome.turns,
                    "usage": outcome.usage,
                    "total_cost_usd": outcome.cost_usd,
                    **json_safe(outcome.extra),
                    **module.record_extras(bus, goal),
                },
                "wall_sec": round(wall, 1),
                "setup_sec": round(setup, 1),
            }
            if outcome.error:
                record["error"] = outcome.error
        finally:
            try:
                callbacks.fire("on_episode_end", ctx, record)
            except StopRun as stop:
                self._stop(stop)
        return record

    async def timeout_record(self, index: int, wall: float) -> dict[str, Any]:
        """The record of a sequence the wall-clock backstop cut short.

        It is scored, not dropped: every task of the sequence stays in the
        denominators, so a run is scored over the benchmark's full task set.
        What the env scored before the cut stands, and every task it has not
        scored counts as failed (the shape's ``unfinished_metrics``). The env
        is asked once more for its metrics under a short timeout; when it
        cannot answer, every task of the sequence is a failure. A sequence
        that was never placed has no known task set and stays an unscored
        stub."""
        rec = _stub_record(index, "timeout")
        live = self._inflight.get(index)
        if live is None:
            return rec
        module, goal = live["module"], live["goal"]
        sink = _NoSink()
        metrics: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            metrics = await asyncio.wait_for(module.evaluate(sink, goal), timeout=TIMEOUT_EVALUATE_S)
        rec.update(
            {
                "episode_id": goal.episode_id,
                "scene_id": goal.scene_id,
                "instruction": goal.text,
                "metrics": module.unfinished_metrics(goal, metrics or {}),
                "wall_sec": round(wall, 1),
            }
        )
        rec["agent"].update(
            {
                **module.record_extras(sink, goal),
                "end_reason": "wall_clock",
                "timeout_scored_by": "env" if metrics else "placement",
            }
        )
        return rec

    def _stop(self, stop: StopRun) -> None:
        run = self.run_ctx
        assert run is not None
        if run.stopped_by is None:
            run.stopped_by = str(stop)
            print(f"[std] STOP requested by a callback: {stop} — draining")
        run.drain.set()

    # ── the run ──

    async def run(self, run_dir: Path) -> dict[str, Any]:
        """Run (or resume) one experiment into ``run_dir``. Existing episode
        records in the run dir's summary are kept; requested indices
        (``run.episodes``) are re-run and replace their records."""
        app, env_name = self.app, self.env_name

        # May raise (bad auth, unpinnable serving context) — that is the point.
        self.adapter.prepare(self.model)

        # the env: attach to run.servers, or spawn run.workers servers from the
        # env block (own interpreter, env.python; ports env.port, env.port+1, …)
        # and stop them when the run ends
        workers = int(app["run"].get("workers") or 1)
        if self.servers:
            if workers != 1 and workers != len(self.servers):
                raise ValueError(f"run.workers={workers} but run.servers lists {len(self.servers)} — give one or the other")
        else:
            base = int(app["env"].get("port") or 9200)
            # n=1: env_server.log; n>1: one log per port
            handles = [
                env_server.launch(app["env"], base + i, run_dir / (f"env_server_{base + i}.log" if workers > 1 else "env_server.log"))
                for i in range(workers)
            ]
            self._spawned.extend(handles)
            # run.server_timeout_s: how long /health may take (cold scene loads)
            env_server.wait_healthy(handles, timeout_s=float(app["run"].get("server_timeout_s") or 600.0))
            self.servers = [h.url for h in handles]
            print(f"[std] {workers} env server(s) {env_name} spawned on {base}-{base + workers - 1} (logs: {run_dir})")
        servers = self.servers
        try:
            return await self._run(run_dir, servers)
        finally:
            for handle in self._spawned:
                handle.stop()
            self._spawned.clear()

    async def _run(self, run_dir: Path, servers: list[str]) -> dict[str, Any]:
        adapter, cfg, app = self.adapter, self.cfg, self.app
        run_name, env_name = self.name, self.env_name
        episodes_spec: str = str(app["run"]["episodes"])
        wp_server = app["run"].get("wp_server")

        for url in servers:
            health = requests.get(f"{url}/health", timeout=10)
            health.raise_for_status()
            self.module_cls.check_health(url, health.json(), env_name, run_name)

        if self.arm.needs_wp_server:
            # the wp surface is wired to the SmartWay waypoint predictor
            # (habitat-trained, clockwise pano numbering). An env with its own
            # predictor and a counter-clockwise pano would degrade silently
            # (mirrored headings, wrong-domain candidates) — refuse instead.
            if env_name == "env_vlnverse":
                raise RuntimeError(
                    f"run {run_name}: wp/hybrid is wired to the SmartWay "
                    "predictor (habitat-trained, CW pano); this env has its own "
                    "predictor and pano convention. "
                    "Use a bare experiment."
                )
            # a wp/hybrid run without its predictor would silently degrade — refuse
            if not wp_server:
                raise RuntimeError(
                    "wp/hybrid runs need run.wp_server (waypoint-predictor auto_host)"
                )
            health = requests.get(f"{wp_server}/health", timeout=10)
            health.raise_for_status()
            print(f"[std] {wp_server} healthy: {health.json()['name']} (waypoint predictor)")

        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "summary.json"
        # the ONE full record of what this launch runs on — the resolved
        # four-layer config — plus the flat run cfg the loop derived from it
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(
                {**json_safe(app), "run_cfg": json_safe(cfg)}, sort_keys=False, allow_unicode=True
            )
        )

        # resume: keep prior records for indices not being re-run
        prior: dict[int, dict[str, Any]] = {}
        if summary_path.exists():
            try:
                for rec in json.loads(summary_path.read_text()).get("episodes", []):
                    prior[int(rec["index"])] = rec
            except Exception:
                pass

        run = RunContext(
            name=run_name,
            app=app,
            cfg=cfg,
            run_dir=run_dir,
            servers=servers,
            adapter=adapter,
            arm=self.arm,
            module_cls=self.module_cls,
            env_name=env_name,
            episodes_spec=episodes_spec,
            prior_episodes=prior,
            episodes=dict(prior),
        )
        callbacks = CallbackSet(self._callbacks, run)
        run.callbacks = callbacks
        run.provenance.update(
            {
                "task": self.task_key,
                "env_key": app["env_key"],
                "arm": self.arm.name,
                "env": env_name,
                "shape": self.module_cls.shape,
                "module": f"{self.module_cls.__module__}:{self.module_cls.__name__}",
                "harness": adapter.name,
                "callbacks": callbacks.names(),
            }
        )
        self.run_ctx = run
        callbacks.fire("on_run_start", run)

        # Graceful drain:
        # `touch <run_dir>/DRAIN` (or SIGUSR1) asks
        # every worker to finish its current episode, flush, and exit WITHOUT
        # pulling a new one — in-flight work is never cut, un-pulled indices stay
        # pending for the next resume. A callback's StopRun takes the same path.
        drain = run.drain
        drain_sentinel = run_dir / "DRAIN"
        if drain_sentinel.exists():
            drain_sentinel.unlink()  # clear a stale sentinel from a prior run
        with contextlib.suppress(
            NotImplementedError, RuntimeError
        ):  # unavailable on this platform / thread
            asyncio.get_running_loop().add_signal_handler(signal.SIGUSR1, drain.set)

        indices = parse_episodes(episodes_spec)
        print(
            f"[std] run={run_name} model={self.model.model_id} eps={indices[0]}-{indices[-1]} "
            f"workers={len(servers)} -> {run_dir}"
        )
        # every server placed for the whole run (dataset / split / body / arm
        # switches) — what the shape's env family needs, nothing per benchmark
        for url in servers:
            await self.module_cls.setup_server(
                env_client_factory(url, timeout_s=self.client_timeout_s), cfg
            )

        queue: asyncio.Queue[int] = asyncio.Queue()
        for index in indices:
            queue.put_nowait(index)

        episodes = run.episodes
        write_lock = asyncio.Lock()

        def build_summary() -> dict[str, Any]:
            ordered = [episodes[i] for i in sorted(episodes)]
            return {
                "run_name": run_name,
                "cell": run_name,
                "harness": adapter.name,
                "harness_inherent": adapter.inherent,
                "config": {
                    **{k: v for k, v in cfg.items() if k != "extra"},
                    "model": self.model.model_id,
                    "effort": self.model.effort,
                    "bare": self.arm.bare,
                    "extra": json_safe(cfg["extra"]),
                },
                "servers": servers,
                "run_stats": run.run_stats,
                "aggregate": aggregate([e for e in ordered if is_scored(e)]),
                "episodes": ordered,
                **run.summary_extra,
                "provenance": run.provenance,
                "writers": callbacks.writers(),
            }

        async def flush_summary(summary: dict[str, Any] | None = None) -> None:
            async with write_lock:
                summary_path.write_text(json.dumps(summary or build_summary(), indent=2))

        async def worker(position: int, url: str) -> None:
            await asyncio.sleep(position * 2)  # stagger cold-server scene loads
            while True:
                if drain.is_set() or drain_sentinel.exists():
                    print(f"[std] worker {position} ({url}) draining — no new episode pulled")
                    return
                try:
                    index = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                print(f"[std] episode {index} starting on {url}")
                # Rate-limit retry (without it a throttled episode lands as a
                # contaminated partial record instead of re-running): back off
                # OUTSIDE the timed scope and re-run the episode fresh, so the
                # wait never eats the episode's wall-clock budget.
                episode: dict[str, Any] = {}
                for attempt in range(1, self.retry.max_attempts + 1):
                    if attempt > 1:
                        archive_attempt(run_dir, index, attempt - 1)
                    started = time.monotonic()
                    try:
                        episode = await asyncio.wait_for(
                            self.run_episode(cfg, url, index),
                            timeout=cfg["episode_timeout"] + 600,  # backstop over in-session caps
                        )
                    except asyncio.TimeoutError:
                        print(f"[std] episode {index} TIMED OUT (backstop) — its unfinished tasks score as failures")
                        episode = await self.timeout_record(index, time.monotonic() - started)
                    except Exception as exc:
                        tag = "rate_limited" if is_rate_limited(repr(exc)) else repr(exc)
                        print(f"[std] episode {index} FAILED: {exc!r}")
                        episode = _stub_record(index, tag)
                    if episode.get("error") != "rate_limited" or attempt == self.retry.max_attempts:
                        break
                    backoff = self.retry.backoff(attempt)
                    print(
                        f"[std] episode {index} RATE-LIMITED (attempt {attempt}/"
                        f"{self.retry.max_attempts}) — backing off {backoff}s "
                        f"(episode countdown paused) then retrying"
                    )
                    await asyncio.sleep(backoff)  # OUTSIDE wait_for: excluded from episode_timeout
                self._inflight.pop(index, None)
                episodes[index] = episode
                await flush_summary()
                m = episode.get("metrics") or {}
                # steps come from the ENV's own metrics (the agent-side counter
                # is only the fallback); display only
                _dtg = m.get("distance_to_goal")
                _steps = m.get("steps_taken") or episode["agent"].get("env_steps")
                print(
                    f"[std] episode {index} done: success={m.get('success')} "
                    f"spl={m.get('spl')} "
                    f"dtg={round(_dtg, 2) if isinstance(_dtg, (int, float)) else None} "
                    f"steps={_steps}"
                )

        try:
            await asyncio.gather(*(worker(i, url) for i, url in enumerate(servers)))
        finally:
            summary = build_summary()
            with contextlib.suppress(StopRun):  # the run is over either way
                callbacks.fire("on_run_end", run, summary)
            if run.stopped_by:
                run.run_stats["stopped_by"] = run.stopped_by
            if run.callback_errors:
                run.run_stats["callback_errors"] = dict(run.callback_errors)
            finalize = getattr(adapter, "finalize", None)
            if callable(finalize):
                try:
                    run.run_stats.update(finalize(run_dir) or {})
                except Exception as exc:
                    run.run_stats["finalize_error"] = repr(exc)
            # on_run_end may have added run-level keys; rebuild so run_stats /
            # aggregate / episodes reflect everything before the final write
            for key, value in summary.items():
                if key not in (
                    "run_name",
                    "cell",
                    "harness",
                    "harness_inherent",
                    "config",
                    "servers",
                    "run_stats",
                    "aggregate",
                    "episodes",
                    "provenance",
                    "writers",
                ):
                    run.summary_extra[key] = value
            await flush_summary()

        if drain.is_set() or drain_sentinel.exists():
            left: list[int] = []
            while not queue.empty():
                left.append(queue.get_nowait())
            print(
                f"[std] DRAINED — in-flight episodes finished; "
                f"{len(left)} un-run, resume with --resume --episodes {format_episodes(left)}"
                if left
                else "[std] DRAINED — queue already empty"
            )
        if drain_sentinel.exists():
            drain_sentinel.unlink()

        final = aggregate([e for e in episodes.values() if is_scored(e)])
        print(f"[std] run complete -> {summary_path}")
        if run.run_stats:
            print(f"[std] run stats: {json.dumps(run.run_stats)}")
        print(json.dumps(final, indent=2))
        callbacks.fire("on_run_written", run, summary_path)
        return final



# endregion
# ---------------------------------------------------------------
# region 3. Main


def run(app: dict[str, Any]) -> None:
    app = apply_fake_api(apply_test_rule(apply_judge_name(app)))
    adapter, arm, module_cls, callbacks = build(app)
    if app["run"].get("dry_run"):
        show(app, arm)
        return
    run_dir = Path(app["run"]["dir"])
    # run.retry: the rate-limit re-run policy (core.retry); absent = DEFAULT_RETRY
    retry = RetryPolicy(**app["run"]["retry"]) if app["run"].get("retry") else DEFAULT_RETRY
    asyncio.run(Runner(adapter, arm, module_cls, app, callbacks, retry=retry).run(run_dir))


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = list(sys.argv[1:] if argv is None else argv)
    experiment: str | None = None
    overrides: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--config-name="):
            experiment = a.split("=", 1)[1]
        elif a in ("--config-name", "-cn"):
            i += 1
            experiment = args[i]
        elif a in ("-h", "--help"):
            print(__doc__)
            return
        elif experiment is None and "=" not in a:
            experiment = a
        else:
            overrides.append(a)
        i += 1
    if not experiment:
        sys.exit(f"[std] which experiment? a path or a run name under {EXPERIMENTS} (see --help)")
    run(compose_app(experiment, overrides))


def experiments() -> list[Path]:
    return sorted(EXPERIMENTS.glob("*/configs/*.yaml"))


def harness_names(experiment: Path) -> list[str]:
    """The experiment's harness/ group: one yaml per loop beside the file."""
    return sorted(p.stem for p in (experiment.parent / "harness").glob("*.yaml"))


def api_names(experiment: Path) -> list[str]:
    """The experiment's api/ group: one yaml per endpoint beside the file."""
    return sorted(p.stem for p in (experiment.parent / "api").glob("*.yaml"))


def models_table(experiment: Path) -> dict[str, dict[str, Any]]:
    """The experiment's models/models.yaml beside the file: model -> harness -> spec."""
    with open(experiment.parent / "models" / "models.yaml") as fh:
        return yaml.safe_load(fh) or {}


def model_names(experiment: Path) -> list[str]:
    return sorted(models_table(experiment))


def model_spec(experiment: Path, name: str) -> dict[str, Any]:
    return models_table(experiment).get(name) or {}


def seats() -> list[tuple[Path, list[str]]]:
    """Every seat: an experiment file plus the overrides that pick it —
    ``harness=<h> model=<m>`` for every harness of its folder's harness/
    group and every row of its models/models.yaml with an entry under it."""
    out: list[tuple[Path, list[str]]] = []
    for path in experiments():
        for h in harness_names(path):
            for m in model_names(path):
                if h in model_spec(path, m):
                    out.append((path, [f"harness={h}", f"model={m}"]))
    return out


# endregion
# ---------------------------------------------------------------

if __name__ == "__main__":
    main()
