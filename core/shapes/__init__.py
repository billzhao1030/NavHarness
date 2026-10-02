"""Task shapes — ``env.shapes.<family>`` → TaskModule class.

The es.* shapes (EsNav · EsObjNav · EsGoat) speak EmbodiedScore-envs' reset /
evaluate. runner.py resolves the class with hydra.utils.get_class and
instantiates it per episode with (env, cfg, arm). See core.shapes.base for
the contract.
"""

from __future__ import annotations

from core.shapes.base import Goal, Placement, TaskModule, format_choices
from core.shapes.es import EsGoat, EsNav, EsObjNav, es_goal

SHAPES: dict[str, type[TaskModule]] = {
    cls.shape: cls
    for cls in (EsNav, EsObjNav, EsGoat)
}

__all__ = [
    "SHAPES",
    "Goal",
    "Placement",
    "TaskModule",
    "es_goal",
    "format_choices",
]
