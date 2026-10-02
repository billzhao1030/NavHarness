"""Pose-frame geometry helpers shared by the SLAM middleware.

Yaw is measured in the mapping world frame (OpenCV camera axes, y down):
camera forward = pose[:3, 2], yaw = atan2(forward_x, forward_z).
"""

from __future__ import annotations

import numpy as np


def yaw_of(pose_wc: np.ndarray) -> float:
    fwd = pose_wc[:3, 2]
    return float(np.arctan2(fwd[0], fwd[2]))


def angle_diff(a: float, b: float) -> float:
    """Smallest signed difference a-b in (-pi, pi]."""
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return float(d)


def relative_cv_pose(T_cv: np.ndarray, T0_inv: np.ndarray) -> np.ndarray:
    return T0_inv @ T_cv


_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenGL camera axes <-> OpenCV


def mount_transform_cv(pitch_deg: float, camera_height: float) -> np.ndarray:
    """Camera-from-eye mount kinematics (OpenCV convention) for a camera
    pitched on a lever arm: a body whose pitch swings the camera, mounted
    ``camera_height`` above the body origin, about that origin — at pitch
    p it sits at body + (0, h cos p, h sin p) (habitat axes, -z forward)
    looking p degrees up. The EYE frame is where the camera would be at
    pitch 0 (body + (0, h, 0), level); this returns T_eye_cam so a camera
    pose (what SLAM tracks) converts to the eye pose (what get_pose /
    trajectory / markers report): eye = cam @ inv(T_eye_cam). Pure
    geometry of the mount — not localization."""
    p = np.deg2rad(pitch_deg)
    c, s = float(np.cos(p)), float(np.sin(p))
    T_gl = np.eye(4)
    T_gl[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    T_gl[:3, 3] = [0.0, camera_height * c - camera_height, camera_height * s]
    return _FLIP @ T_gl @ _FLIP
