"""Embodiment presets — every number an upstream config or the EmbodiedScore
protocol fixes, nothing derived.

STANDARD  EmbodiedScore's shared body: the VLN-CE R2R rig (0.25 m / 15°, 1.5 m /
          0.1 m agent, sliding on, RGB 512² + depth 256² at hfov 90 and 1.25 m,
          scene navmesh file) plus a 30° tilt clamped to ±60° so every line has
          LOOK_UP / LOOK_DOWN.
VLNCE     VLN-CE ``habitat_extensions/config/vlnce_task.yaml`` (R2R-CE; IVLN-CE
          reuses it): as STANDARD but RGB 224² and no tilt.
RXR_CE    VLN-CE ``rxr_vlnce_english_task.yaml``: the LoCoBot rig on habitat-lab
          0.1.7 — 0.25 m / 30°, tilt 30° unlimited, 0.88 m / 0.18 m, sliding off,
          RGB-D 640x480 hfov 79 at 0.88 m, scene navmesh file (0.1.7 never
          recomputes).
STRETCH   goat-bench ``goat_stretch_hm3d.yaml``:
          0.25 m / 30°, tilt 30° unlimited, 1.41 m / 0.17 m, sliding off, RGB
          360x640 (portrait) hfov 42 at 1.31 m, no depth sensor, navmesh
          recomputed with climb 0.1 / cell 0.05 (every released experiment config).
"""

from __future__ import annotations

from ..env import Body, CameraSpec, NavMesh

STANDARD = Body(forward_step_m=0.25, turn_deg=15.0, tilt_deg=30.0, tilt_limit_deg=60.0,
                agent_height_m=1.5, agent_radius_m=0.1, allow_sliding=True,
                rgb=CameraSpec(512, 512, 90.0, (0.0, 1.25, 0.0)), depth=CameraSpec(256, 256, 90.0, (0.0, 1.25, 0.0)),
                navmesh=NavMesh.file())

VLNCE = Body(forward_step_m=0.25, turn_deg=15.0, agent_height_m=1.5, agent_radius_m=0.1, allow_sliding=True,
             rgb=CameraSpec(224, 224, 90.0, (0.0, 1.25, 0.0)), depth=CameraSpec(256, 256, 90.0, (0.0, 1.25, 0.0)),
             navmesh=NavMesh.file())

RXR_CE = Body(forward_step_m=0.25, turn_deg=30.0, tilt_deg=30.0, tilt_limit_deg=None,
              agent_height_m=0.88, agent_radius_m=0.18, allow_sliding=False,
              rgb=CameraSpec(640, 480, 79.0, (0.0, 0.88, 0.0)), depth=CameraSpec(640, 480, 79.0, (0.0, 0.88, 0.0)),
              navmesh=NavMesh.file())

STRETCH = Body(forward_step_m=0.25, turn_deg=30.0, tilt_deg=30.0, tilt_limit_deg=None,
               agent_height_m=1.41, agent_radius_m=0.17, allow_sliding=False,
               rgb=CameraSpec(360, 640, 42.0, (0.0, 1.31, 0.0)), depth=None,
               navmesh=NavMesh.recompute(agent_radius=0.17, agent_height=1.41, agent_max_climb=0.1, cell_height=0.05))

__all__ = ["STANDARD", "VLNCE", "RXR_CE", "STRETCH"]
