"""WorldBody — the SLAM middleware's view of the EmbodiedScore-envs body.

The simulator belongs to the package's ``HabitatEnv`` (built by
``stack.build`` on the FINE body: the STANDARD rig at micro quanta). This
adapter exposes exactly the surface ``SlamEnv`` consumes — ``step`` /
``observe`` / ``intrinsics`` / the two ground-truth camera poses in OpenCV
convention / ``pitch_deg`` / ``camera_height`` / ``collisions`` — over
``base.step`` and ``base.world``:

* ``step(a)`` is a FINE gym step of the package env (``HabitatEnv.step``,
  0.05 m / 3° / 6° on the FINE body); its return tuple is kept in ``last``
  so the coarse wrapper can hand the last micro-step's ``info`` outward.
  The collision verdict is a forward that moved the body by less than
  1e-4 m — not habitat's ``collided`` flag (which is also raised by a slide
  along a wall): the SLAM stack aborts the remaining micro-steps and marks an
  obstacle on a *blocked* forward only.
* ``render_headings`` renders the current pose at yaw offsets through
  ``SimWorld.render_at`` — the body does not move (a closed in-place spin
  has zero net motion, so this renders the same views without the 120
  micro-turns).
* ``close`` is a no-op: the simulator outlives the episode (the package
  reconfigures it scene to scene).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

FORWARD, LEFT, RIGHT, LOOK_UP, LOOK_DOWN = 1, 2, 3, 4, 5
_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenGL camera axes <-> OpenCV


def camera_intrinsics(hfov_deg: float, width: int, height: int) -> dict:
    """Pinhole intrinsics of a habitat camera; fx == fy for any aspect
    (habitat derives the vertical FOV from hfov and the aspect ratio)."""
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return {"fx": float(f), "fy": float(f), "cx": width / 2.0, "cy": height / 2.0,
            "width": int(width), "height": int(height)}


def _pose_matrix(position: Any, rotation_xyzw: Any) -> np.ndarray:
    import quaternion  # numpy-quaternion

    r = np.asarray(rotation_xyzw, dtype=np.float64)
    q = np.quaternion(r[3], r[0], r[1], r[2])
    T = np.eye(4)
    T[:3, :3] = quaternion.as_rotation_matrix(q)
    T[:3, 3] = np.asarray(position, dtype=np.float64)
    return T


class WorldBody:
    def __init__(self, base: Any) -> None:
        self._base = base            # the package HabitatEnv (unwrapped)
        self._world = base.world
        body = base.body
        self.hfov_deg = float(body.rgb.hfov_deg)
        self.img_w, self.img_h = int(body.rgb.width), int(body.rgb.height)
        self.camera_height = float(body.rgb.position[1])
        self.tilt_angle_deg = body.tilt_deg      # the FINE micro-tilt quantum, None = no LOOK actions
        self.lever_arm = bool(body.pitch_moves_camera)
        self.collisions = 0
        self.last: tuple | None = None           # (obs, reward, terminated, truncated, info) of the last fine step

    # ── episode control ──

    def reset(self, pitch_deg: float = 0.0) -> None:
        """The package env has already placed the body (level, at the
        episode start); apply a non-zero start pitch in tilt quanta."""
        self.collisions = 0
        self.last = None
        if pitch_deg:
            if self.tilt_angle_deg is None:
                raise ValueError("pitch_deg needs a Body with tilt_deg (LOOK actions)")
            n = round(abs(pitch_deg) / self.tilt_angle_deg)
            for _ in range(n):
                self._world.move(LOOK_UP if pitch_deg > 0 else LOOK_DOWN)

    # ── motion (one FINE gym step of the package env) ──

    def step(self, action: int) -> bool:
        """Apply one micro action through ``HabitatEnv.step``; returns True
        when a forward left the body in place (the collision rule above)."""
        a = int(action)
        if a in (LOOK_UP, LOOK_DOWN) and self.tilt_angle_deg is None:
            raise ValueError(f"action {a} (camera tilt) is not registered on this body")
        prev, _ = self._world.pose()
        self.last = self._base.step(a)
        pos, _ = self._world.pose()
        collided = a == FORWARD and float(np.linalg.norm(pos - prev)) < 1e-4
        if collided:
            self.collisions += 1
        return collided

    # ── perception ──

    def observe(self) -> dict:
        obs = self._world.observe()
        return {"rgb": np.asarray(obs["rgb"], dtype=np.uint8),
                "depth": np.asarray(obs["depth"], dtype=np.float32).squeeze()}

    def render_headings(self, offsets_deg: list) -> list:
        """RGB views at yaw offsets from the current heading (positive =
        right, matching get_pose), rendered without moving the body."""
        import quaternion  # noqa: F401 — enables np.quaternion

        pos, rot = self._world.pose()
        q = np.quaternion(rot[3], rot[0], rot[1], rot[2])
        views = []
        for deg in offsets_deg:
            half = math.radians(-float(deg)) / 2.0      # habitat yaw is about +y, positive = left
            r = q * np.quaternion(math.cos(half), 0.0, math.sin(half), 0.0)
            views.append(np.asarray(self._world.render_at(pos, [r.x, r.y, r.z, r.w])["rgb"], dtype=np.uint8))
        return views

    def intrinsics(self) -> dict:
        return camera_intrinsics(self.hfov_deg, self.img_w, self.img_h)

    @property
    def pitch_deg(self) -> float:
        return float(self._world.pitch())

    def agent_position(self) -> list:
        return [float(v) for v in self._world.pose()[0]]

    def gt_camera_pose_cv(self) -> np.ndarray:
        """Ground-truth 4x4 world-from-camera pose in OpenCV convention
        (habitat's OpenGL camera right-multiplied by diag(1,-1,-1,1))."""
        pos, rot = self._world.camera_pose("rgb")
        return _pose_matrix(pos, rot) @ _FLIP

    def gt_level_pose_cv(self) -> np.ndarray:
        """Ground-truth world-from-EYE pose: the body's rotation at the level
        camera position (body + (0, camera_height, 0)); coincides with
        gt_camera_pose_cv at pitch 0 on a fixed mount."""
        pos, rot = self._world.pose()
        T = _pose_matrix(pos, rot)
        T[:3, 3] = np.asarray(pos, dtype=np.float64) + np.array([0.0, self.camera_height, 0.0])
        return T @ _FLIP

    def close(self) -> None:
        """The simulator is the package env's; nothing to close here."""
