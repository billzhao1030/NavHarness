"""Layer L0 — the simulator facade.

``habitat/``  habitat-sim 0.3.3 in process: Body · SceneRef · SimWorld (+ Follower).

The pure-data types are importable without the simulator installed;
``SimWorld`` and its companions (the habitat_sim importers) resolve lazily, so
``import embodiedscore_envs`` works on a host without habitat-sim — a line then
fails at construction with the ImportError that names the build to install.
"""

from __future__ import annotations

from .habitat import Body, CameraSpec, NavMesh, SceneRef

_HABITAT_LAZY = ("SimWorld", "Follower", "FollowerError", "FORWARD", "LEFT", "RIGHT", "LOOK_UP", "LOOK_DOWN")


def __getattr__(name: str):
    if name in _HABITAT_LAZY:
        from .habitat import world
        return getattr(world, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Body", "CameraSpec", "NavMesh", "SceneRef", "SimWorld", "Follower", "FollowerError",
           "FORWARD", "LEFT", "RIGHT", "LOOK_UP", "LOOK_DOWN"]
