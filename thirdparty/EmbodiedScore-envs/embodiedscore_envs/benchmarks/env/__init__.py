"""Layer L1 — what every benchmark is built on: the environment body (the
habitat engine's), the episode / goal vocabulary, the metric and observation
wrappers, and the simulator facade underneath (``env.sim``).

``SimWorld`` / ``Follower`` / ``FollowerError`` are the habitat_sim importers
and resolve lazily (``sim.__getattr__``); everything else imports without a
simulator installed.
"""

from __future__ import annotations

from .habitat_env import HabitatEnv
from .metrics import NAV_KEYS, VLN_KEYS, NavMetrics, SequenceNavMetrics
from .schema import (ENGINES, Act, Benchmark, DepthSpec, Episode, Goal, GoalSequence, ImageGoal, ObjectGoal,
                     ObjectInstance, PointGoal, TextGoal, action_prefix, data_root, scene_root,
                     targets_of, to_dict)
from .sim import Body, CameraSpec, NavMesh, SceneRef
from .wrappers import DepthClip, DynamicTimeLimit

_SIM_LAZY = ("SimWorld", "Follower", "FollowerError")


def __getattr__(name: str):
    if name in _SIM_LAZY:
        from . import sim
        return getattr(sim, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HabitatEnv",
    "NavMetrics", "SequenceNavMetrics",
    "NAV_KEYS", "VLN_KEYS",
    "ENGINES", "Act", "Benchmark", "DepthSpec", "Episode", "Goal", "GoalSequence", "ImageGoal",
    "ObjectGoal", "ObjectInstance", "PointGoal", "TextGoal", "action_prefix", "data_root", "scene_root",
    "targets_of", "to_dict",
    "Body", "CameraSpec", "NavMesh", "SceneRef", "SimWorld", "Follower", "FollowerError",
    "DepthClip", "DynamicTimeLimit",
]
