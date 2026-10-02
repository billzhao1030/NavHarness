"""Layer L2 — the benchmark declarations, and ``make()`` that turns one into a
Gymnasium stack. Each member module exposes a ``BENCHMARKS`` tuple; nothing
else in the package knows their names.

Every line declares two variants: ``<line>`` (the default — EmbodiedScore's shared
body, ``presets.bodies.STANDARD``) and ``<line>-upstream`` (the line's own
evaluator: its rig, action table and depth post-processing). Task semantics —
loader, goal, success distance, budget, metric keys — are the line's and do not
change between variants. Every line here runs on habitat-sim, in process."""

from __future__ import annotations

import importlib
from typing import Any

import gymnasium as gym

from .env import Benchmark, DepthClip, DynamicTimeLimit, HabitatEnv

_MEMBERS = ("vlnce", "ivlnce", "goat")

BENCHMARKS: dict[str, Benchmark] = {}
for _m in _MEMBERS:
    try:
        _mod = importlib.import_module(f"{__name__}.{_m}")
    except ModuleNotFoundError as _e:   # a member not written yet
        if _e.name != f"{__name__}.{_m}":
            raise
        continue
    for _b in _mod.BENCHMARKS:
        if _b.name in BENCHMARKS:
            raise RuntimeError(f"duplicate benchmark name {_b.name}")
        BENCHMARKS[_b.name] = _b
del _m, _mod, _b


_SUFFIX = "-upstream"


def resolve(name: str, variant: str | None = None) -> str:
    """The declaration name for ``name`` under ``variant``: ``None`` keeps the
    name as given, ``"standard"`` strips ``-upstream``, ``"upstream"`` appends it."""
    if variant is None:
        return name
    if variant not in ("standard", "upstream"):
        raise ValueError(f"variant must be 'standard' or 'upstream', got {variant!r}")
    line = name[: -len(_SUFFIX)] if name.endswith(_SUFFIX) else name
    return line + _SUFFIX if variant == "upstream" else line


def benchmark(name: str, variant: str | None = None) -> Benchmark:
    key = resolve(name, variant)
    try:
        return BENCHMARKS[key]
    except KeyError:
        raise KeyError(f"unknown benchmark {key!r}; known: {sorted(BENCHMARKS)}") from None


def build_env(benchmark_name: str, split: str, data_root: str | None = None, scene_root: str | None = None,
              body: Any = None, actions: Any = None, gpu_id: int = 0, sim_seed: int = 42,
              render_mode: str | None = None, **loader_kwargs: Any) -> gym.Env:
    """The bare body for a benchmark (what the gym ids point at). ``body`` /
    ``actions`` override the declaration (a different rig or action table on
    the same episodes and measures — how the legacy nodesets' per-graph
    sensor knobs and upstream RxR's body are served). ``sim_seed`` is habitat's."""
    b = benchmark(benchmark_name)
    episodes = b.episodes(split, data_root=data_root, scene_root=scene_root, **loader_kwargs)
    if not episodes:
        raise ValueError(f"{benchmark_name}/{split}: no episodes")
    body = body or b.body
    common = dict(episodes=episodes, body=body, budget=b.budget, dtg_policy=b.dtg_policy, gpu_id=gpu_id,
                  sim_seed=sim_seed, render_mode=render_mode)
    return HabitatEnv(actions=actions or b.actions, **common)


def make(name: str, split: str, *, variant: str | None = None, metrics: bool = True, depth: bool = True,
         depth_spec: Any = None, metric_overrides: dict[str, Any] | None = None, **kwargs: Any) -> gym.Env:
    """The standard stack: gym.make (PassiveEnvChecker / OrderEnforcing / TimeLimit
    when the benchmark has a static budget) -> the benchmark's metric wrapper ->
    DepthClip -> DynamicTimeLimit (per-episode budgets). ``variant`` picks the
    declaration (``"upstream"`` == ``make(f"{name}-upstream", ...)``; ``None``
    takes ``name`` literally, so a bare line name is its standard variant).
    ``kwargs`` go to :func:`build_env` (data_root, scene_root, body, actions,
    gpu_id, loader options); ``depth_spec`` replaces the declared depth
    post-processing; ``metric_overrides`` (e.g. ``success_distance``) reach the
    metric factory."""
    b = benchmark(name, variant)
    env = gym.make(b.gym_id, split=split, **kwargs)
    if metrics and b.metrics is not None:
        env = b.metrics(env, **(metric_overrides or {}))
    spec = depth_spec if depth_spec is not None else b.depth
    body = kwargs.get("body") or b.body
    if depth and spec is not None and body.depth is not None:
        env = DepthClip(env, spec.min_m, spec.max_m, spec.normalize)
    if b.budget is not None and b.truncate_at_budget:
        env = DynamicTimeLimit(env)
    return env


__all__ = ["BENCHMARKS", "benchmark", "build_env", "make", "resolve"]
