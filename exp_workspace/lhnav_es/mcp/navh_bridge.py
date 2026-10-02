"""This arm's tool surface for the NavHarness seats — the experiment's, not the harness's.

The navh session seats hand their agent an MCP server and the harness reads
its state file at every hop, so the tools have to be a process of this
experiment's, speaking this experiment's env.

What it does NOT do is define tools of its own. The harness ships its own tool
set — observe / step / get_map / mark / look_around / get_goal / close_task /
request_recovery and the file tools, the state file, the forced look-around at
the stop position and the pre-close judge — in the module `NAVH_BRIDGE_MODULE`
names (`core.harnesses.navh.bridge.bridge`). That bridge reaches its env as
`POST /call/<prefix>__<verb>` (`step_discrete`, `observe_egocentric`); this
arm's env is `core.envserver` (`POST /<verb>`, one public method each). So
this module imports that bridge, keeps every tool exactly as it is, and
replaces the single function that reaches the env.

Two things beyond the URL differ and are translated here rather than there:
the goal (the env answers `goal_spec`, worded by `bench_es.goal_statement`;
the bridge's `get_goal` reads its own field names), and the closing move (the
harness always sends STOP, and a GOAT sub-goal closes with SUBTASK_STOP — read
off the env's own `n_goals`, not configured, so no run can be seated with the
wrong one).

The verb map and the goal wording come from `core.harnesses.navh.env.bench_es`,
through which the orchestrator drives this same env: one environment worded
one way, whichever side is asking.

    NAVH_SERVER_URL     the env server (core.envserver)
    (everything else — state file, live dir, bench, gate, recovery — is read by
     the harness's bridge from its own environment variables)

The experiment names this file to the navh seats through its Arm surface
(`surface.navh_bridge`), and the session executors start it as
`sys.executable <this file>` — a script, not `-m` — so the repo root goes on
the path here before `core.` is imported.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# the harness's bridge module, as the orchestrator names it (NAVH_BRIDGE_MODULE)
B = importlib.import_module(os.environ.get("NAVH_BRIDGE_MODULE", "core.harnesses.navh.bridge.bridge"))
from core.harnesses.navh.env.bench_es import STOP, SUBTASK_STOP, VERBS, goal_statement  # noqa: E402

URL = B.SERVER_URL.rstrip("/")
TIMEOUT = int(os.environ.get("NAVH_ENV_TIMEOUT", "900"))

_close_action: int | None = None


def _post(verb: str, kwargs: dict) -> Any:
    r = requests.post(f"{URL}/{verb}", json=kwargs, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _closing_move() -> int:
    """Which action ends the task on this line, asked of the env once.

    A GOAT sub-goal closes with SUBTASK_STOP and leaves the episode running;
    every single-goal line closes with STOP. Getting it backwards does not
    raise — it ends the whole episode on sub-goal 0, or refuses to end it at
    all — so it is read off the env's own ``n_goals`` rather than configured.
    """
    global _close_action
    if _close_action is None:
        spec = _post(VERBS["get_goal"], {})
        n = int((spec or {}).get("n_goals") or 1) if isinstance(spec, dict) else 1
        _close_action = SUBTASK_STOP if n > 1 else STOP
    return _close_action


def _call(fn: str, inputs: dict | None = None) -> dict:
    """One call to a `core.envserver` env, with the harness bridge's own
    dead-environment detector kept intact.

    An env that has lost its SLAM subprocess answers /health and then 500s on
    every step. Left alone the agent keeps "working" against a wall and the
    chain records a screen of zeros that reads exactly like a very bad agent —
    so consecutive failures are counted here as they are counted there, and the
    state file says the environment died.
    """
    verb = VERBS.get(fn, fn)
    kwargs = dict(inputs or {})
    try:
        if verb == "step" and int(kwargs.get("action", -1)) == STOP:
            kwargs["action"] = _closing_move()   # close_task / force-stop → the line's closing move
        out = _post(verb, kwargs)
    except Exception as e:
        B._env_fails += 1
        if B._env_fails >= B.ENV_DEAD_AFTER and not B._state.get("env_dead"):
            B._state["env_dead"] = {"fn": verb, "fails": B._env_fails, "error": str(e)[:200],
                                    "steps_used": B._state["steps_used"]}
            B._save()
        raise
    B._env_fails = 0
    return _lift(verb, out)


def _lift(verb: str, out: Any) -> dict:
    """The env answers in ports; the tools read a flat dict — the same lifting
    `EsEnvClient` does, so the bridge and the orchestrator read one environment."""
    if not isinstance(out, dict):
        return {"result": out}
    if verb == "look_around" and out.get("view_front") is not None:
        out.setdefault("views", [out[k] for k in ("view_front", "view_right", "view_back", "view_left")])
    for sub in ("info", "result", "goal"):
        inner = out.get(sub)
        if isinstance(inner, dict):
            for k, v in inner.items():
                out.setdefault(k, v)
    if verb == VERBS["get_goal"] and "error" not in out:
        out.update(_goal(out))
    return out


def _goal(spec: dict) -> dict:
    """``goal_spec`` in the words the bridge's ``get_goal`` reads.

    The statement itself comes from ``bench_es.goal_statement`` — the same
    sentence the harness's own ledger and judge see. Only the field names are
    this function's business."""
    text, modality = goal_statement(spec)
    # an IR2R-CE tour numbers its tasks by the episode's slot in the tour
    tour = spec.get("tour_size") is not None
    return {
        "subtask_index": int(spec.get("tour_slot") or 0) if tour else int(spec.get("goal_index") or 0),
        "subtask_total": int(spec.get("tour_size") or 1) if tour else int(spec.get("n_goals") or 1),
        "text": text,
        "modality": modality,
        "image": spec.get("goal_image"),
    }


B._call = _call

# instruments=0 is a BARE run: no map, no markers, no route preview, no pose.
# The three spatial tools leave the tool surface rather than switching to a
# ground-truth pose — that would remove the drift rather than the instrument and
# hand a bare run perfect localisation — so the session's tool list says the
# same thing the schemas do.
if os.environ.get("NAVH_INSTRUMENTS", "2") == "0":
    for _name in ("get_map", "mark", "preview_path"):
        with contextlib.suppress(Exception):     # a tool this bridge never had
            B.mcp._tool_manager.remove_tool(_name)


if __name__ == "__main__":
    B._save()
    B.mcp.run(transport="stdio")
