"""Embodiment parameters — pure data (layer L0, no habitat_sim import).

A :class:`Body` says what the agent is: step / turn / tilt amounts, the agent
cylinder, wall sliding, the camera rig, and how the navmesh is obtained
(shipped file or recomputed for this body). Every number SimWorld uses comes
from here; benchmarks declare their Body in ``benchmarks/<name>.py``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraSpec:
    width: int
    height: int
    hfov_deg: float = 90.0
    position: tuple[float, float, float] = (0.0, 1.25, 0.0)   # metres, agent frame (y up)


@dataclass(frozen=True)
class NavMesh:
    """How the navmesh is obtained when a scene loads.

    ``kind="file"``: load the ``.navmesh`` next to the scene (habitat-lab 0.1.7
    behaviour; the file is baked for a 0.1 m / 1.5 m agent). ``kind="recompute"``:
    rebuild it from the scene geometry with habitat_sim defaults except the
    four agent knobs (what habitat-lab 0.2.4 does whenever the baked settings
    differ from the agent's). ``fallback`` is used when the file is missing.
    """
    kind: str = "file"
    agent_radius: float = 0.1
    agent_height: float = 1.5
    agent_max_climb: float = 0.2
    cell_height: float = 0.2
    fallback: "NavMesh | None" = None

    @classmethod
    def file(cls, fallback: "NavMesh | None" = None) -> "NavMesh":
        return cls(kind="file", fallback=fallback)

    @classmethod
    def recompute(cls, agent_radius: float, agent_height: float,
                  agent_max_climb: float = 0.2, cell_height: float = 0.2) -> "NavMesh":
        return cls(kind="recompute", agent_radius=agent_radius, agent_height=agent_height,
                   agent_max_climb=agent_max_climb, cell_height=cell_height)


@dataclass(frozen=True)
class Body:
    forward_step_m: float = 0.25
    turn_deg: float = 15.0
    tilt_deg: float | None = None          # None = no LOOK_UP / LOOK_DOWN actions
    tilt_limit_deg: float | None = None    # |pitch| clamp for LOOK actions; None = unlimited
    camera_pitch_deg: float = 0.0          # sensor pitch right after place(); negative looks down
    pitch_moves_camera: bool = False       # True: the rig swings with pitch about the agent's feet (explore-eqa
                                           # pitches the whole agent, so a -30° camera sits 0.75 m ahead and lower)
    locomotion: str = "sim_step"           # "sim_step": habitat's agent.act + step filter (habitat-lab stacks);
                                           # "teleport": explore-eqa's arithmetic — yaw kept as a float64 angle,
                                           # forward = pathfinder.try_step(pos, pos + step·(-sin θ, 0, -cos θ)),
                                           # then agent.set_state; collided = moved < 0.9·step
    agent_height_m: float = 1.5
    agent_radius_m: float = 0.1
    allow_sliding: bool = True
    rgb: CameraSpec = CameraSpec(512, 512)
    depth: CameraSpec | None = CameraSpec(256, 256)
    navmesh: NavMesh = NavMesh.file()

    @property
    def has_tilt(self) -> bool:
        return self.tilt_deg is not None
