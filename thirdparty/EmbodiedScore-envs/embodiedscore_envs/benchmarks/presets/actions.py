"""Action-table presets: prefixes of the global ``Act`` sequence
(0 STOP · 1 FORWARD · 2 LEFT · 3 RIGHT · 4 LOOK_UP · 5 LOOK_DOWN · 6 SUBTASK_STOP)."""

from __future__ import annotations

from ..env import Act

NAV = (Act.STOP, Act.FORWARD, Act.LEFT, Act.RIGHT)                    # habitat-lab action space v0 (VLN-CE R2R)
NAV_LOOK = NAV + (Act.LOOK_UP, Act.LOOK_DOWN)                          # action space v1 (RxR-CE)
GOAT = NAV_LOOK + (Act.SUBTASK_STOP,)                                  # goat-bench: v1 plus the sub-task stop
STANDARD = NAV_LOOK                                                    # EmbodiedScore's shared table

__all__ = ["NAV", "NAV_LOOK", "GOAT", "STANDARD"]
