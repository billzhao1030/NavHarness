"""Declaration presets (layer L2, pure data): the embodiments, action tables and
depth post-processing the benchmark declarations are assembled from.

``bodies.STANDARD`` / ``actions.STANDARD`` / ``depth.STANDARD`` are EmbodiedScore's
shared protocol — every benchmark's default variant runs on them. The other
constants are the upstream evaluators' own rigs, used by the ``-upstream``
variants. Members of ``benchmarks/`` import from here; this package imports only
from ``benchmarks.env``.
"""

from . import actions, bodies, depth

__all__ = ["actions", "bodies", "depth"]
