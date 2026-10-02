"""Depth post-processing presets (``DepthClip`` parameters = habitat-lab's
``min_depth`` / ``max_depth`` / ``normalize_depth``). ``None`` in a declaration
means the depth frame is left in metres."""

from __future__ import annotations

from ..env import DepthSpec

STANDARD = DepthSpec(0.0, 10.0, normalize=True)     # habitat-lab default; VLN-CE, IVLN-CE
LOCOBOT = DepthSpec(0.5, 5.0, normalize=True)       # RxR-CE LoCoBot config

__all__ = ["STANDARD", "LOCOBOT"]
