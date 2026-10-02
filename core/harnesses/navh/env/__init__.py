"""The env — over HTTP, from the orchestrator's side.

    types.py     Goal · TaskEnd · EnvUnavailable
    bench.py     EnvClient, and how a chain is seated and a task begins and ends on each
                 benchmark (IVLNBench, GOATBench); map_floor
    bench_es.py  this runner's env behind the same Bench (EsEnvClient, GOATEsBench, TourEsBench)
    record_sink.py  the runner's bus behind the harness's Recorder

The agent's tools reach the same env from the bridge (``bridge/``), a
process of the session's own; this is the orchestrator's door — the seat,
the STOP, the map for the judge, the look-around of a wake-up, the
checkpoints.
"""

from __future__ import annotations
