"""Observation / budget wrappers (layer L1)."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DepthClip(gym.ObservationWrapper):
    """clip(depth, lo, hi), optionally scaled to [0, 1] — habitat-lab's
    HabitatSimDepthSensor post-processing (min_depth / max_depth / normalize_depth)."""

    def __init__(self, env: gym.Env, min_m: float = 0.0, max_m: float = 10.0, normalize: bool = True) -> None:
        super().__init__(env)
        self.lo, self.hi, self.normalize = float(min_m), float(max_m), bool(normalize)
        d = env.observation_space["depth"]
        new = dict(env.observation_space.spaces)
        new["depth"] = spaces.Box(0.0, 1.0, shape=d.shape, dtype=np.float32) if normalize \
            else spaces.Box(self.lo, self.hi, shape=d.shape, dtype=np.float32)
        self.observation_space = spaces.Dict(new)

    def observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        d = np.clip(obs["depth"], self.lo, self.hi)
        if self.normalize:
            d = (d - self.lo) / (self.hi - self.lo)
        obs = dict(obs)
        obs["depth"] = d.astype(np.float32)
        return obs


class DynamicTimeLimit(gym.Wrapper):
    """``truncated`` after ``info["step_budget"]`` steps — for benchmarks whose
    budget is per episode (``Benchmark.budget``)."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._budget: int | None = None
        self._n = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        b = info.get("step_budget")
        self._budget = int(b) if b is not None else None
        self._n = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._n += 1
        if self._budget is not None and self._n >= self._budget:
            truncated = True
        return obs, reward, terminated, truncated, info
