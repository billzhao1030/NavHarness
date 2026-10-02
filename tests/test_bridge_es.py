"""The lhnav arm's NavHarness tool surface, end to end on a stub server.

`exp_workspace/lhnav_es/mcp/navh_bridge.py` is the harness's own bridge
(core/harnesses/navh/bridge) with one function replaced — the call that
reaches the env — and the session transports reach the world through nothing
else, so the tools are exercised here against an HTTP server that answers the way
`core.envserver` answers: `POST /<verb>`, images as base64 PNG, the verb's own
fields under `info` / `result` / `goal`. No simulator, no model, no MCP client:
the tool functions are called directly, which is what the MCP layer does.
"""

from __future__ import annotations

import base64
import importlib
import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BRIDGE = REPO / "exp_workspace" / "lhnav_es" / "mcp" / "navh_bridge.py"


def _load():
    """The arm's bridge, loaded by PATH the way the executor starts it."""
    spec = importlib.util.spec_from_file_location("lhnav_navh_bridge", BRIDGE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

PNG = base64.b64encode(
    base64.b64decode(  # 1x1 white png
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
).decode()

GOAL_SPEC = {  # exactly what exp_workspace/lhnav_es/mcp/env.py::goal_spec answers
    "task": "find the sofa",
    "goal": {"kind": "object", "category": "sofa"},
    "goal_kind": "object",
    "goal_index": 2,
    "n_goals": 6,
    "episode_over": False,
    "episode_id": "ep-7",
    "scene_id": "hm3d/00800",
    "step_count": 12,
    "step_budget": 5000,
}


class _Stub:
    """One episode of an ES env, in the shapes the real server sends."""

    def __init__(self) -> None:
        self.calls: list = []
        self.spec = dict(GOAL_SPEC)
        self.step_info: dict = {}
        self.map_info: dict = {}

    def __call__(self, verb: str, kwargs: dict) -> dict:
        self.calls.append((verb, dict(kwargs)))
        if verb == "goal_spec":
            return dict(self.spec)
        if verb == "observe":
            return {"rgb": PNG, "depth": None, "pose": {}, "instruction_text": "find the sofa"}
        if verb == "step":
            info = {"action": kwargs["action"], "collision": False, "slam": "ok", "step_count": 13,
                    **self.step_info}
            if int(kwargs["action"]) == 6:
                info["subtask_closed"] = True
                info["subtask_result"] = {"index": 2, "success": 1.0, "spl": 0.7}
            return {"reward": 0.0, "terminated": False, "truncated": False, "info": info}
        if verb == "get_map":
            return {"map": PNG, "map_local": None, "frontiers": [],
                    "info": {"floor": 1, "explored_m2": 31.5, "markers": ["kitchen"], **self.map_info}}
        if verb == "look_around":
            return {"view_front": PNG, "view_right": PNG, "view_back": PNG, "view_left": PNG,
                    "info": {"turns": 4, "headings_deg": [0, 90, 180, -90]}}
        if verb == "mark":
            return {"result": {"marked": {"label": kwargs["label"], "x": 1.5, "z": -2.0, "yaw_deg": 100.0},
                               "n_markers": 1, "current_floor": 1}}
        if verb == "preview_path":
            return {"map": PNG, "result": {"reachable": True, "distance_m": 4.2}}
        raise KeyError(verb)


@pytest.fixture(params=["core.harnesses.navh.bridge.bridge"])
def bridge(tmp_path, monkeypatch, request):
    """bridge_es imported against a live stub server, its state in tmp_path —
    reloaded per test so each gets the stub server this test started."""
    monkeypatch.setenv("NAVH_BRIDGE_MODULE", request.param)
    stub = _Stub()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_POST(self):
            verb = self.path.strip("/")
            n = int(self.headers.get("Content-Length") or 0)
            kwargs = json.loads(self.rfile.read(n) or b"{}")
            try:
                body = json.dumps(stub(verb, kwargs)).encode()
            except KeyError:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("NAVH_SERVER_URL", f"http://127.0.0.1:{srv.server_port}")
    monkeypatch.setenv("NAVH_LIVE_DIR", str(tmp_path / "live"))
    monkeypatch.setenv("NAVH_STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setenv("NAVH_STEP_BUDGET", "5000")
    monkeypatch.setenv("NAVH_PRECHECK", "none")
    # what SdkSessionExecutor.run_task passes, derived from the bench class name
    # (GOATEsBench): on this runner the goal in force is always read off the
    # env's goal_spec, whatever the line — the runner placed the episode
    monkeypatch.setenv("NAVH_BENCH", "goat")
    for mod in ("core.harnesses.navh.bridge.bridge", "core.harnesses.navh.bridge", "lhnav_navh_bridge"):
        sys.modules.pop(mod, None)
    B = _load()
    yield B, stub
    srv.shutdown()
    for mod in ("core.harnesses.navh.bridge.bridge", "core.harnesses.navh.bridge", "lhnav_navh_bridge"):
        sys.modules.pop(mod, None)


def test_the_goal_is_the_one_every_other_seat_is_given(bridge) -> None:
    """The env answers `goal_spec` in EmbodiedScore's words; the bridge's tool
    reads the harness's own goal format. The sentence in between is
    `bench_es.goal_statement` — the same one the harness's ledger and judge
    see, on every transport, so the reasoning cores differ in their loop and
    not in the task."""
    B, _ = bridge
    out = B.B.get_goal()
    body = json.loads(out[0])
    assert body["goal"] == 'find a "sofa" (any instance counts)'
    assert body["modality"] == "object"
    assert (body["task"], body["of"]) == (3, 6)  # sub-goal 2 of 6, 1-based for the model


def test_close_task_records_the_claim_and_never_ends_the_task(bridge) -> None:
    """A tool that decided a task's outcome would be a tool doing the
    orchestrator's job. `close_task` records the claim and takes the
    look-around at the stopping point — evidence gathered BEFORE anyone has
    judged — and hands the turn back. The STOP is an env action the
    orchestrator issues itself, after the file gate and the judge's gate.
    """
    B, stub = bridge
    out = B.B.close_task("complete", "at the sofa by the window")
    assert out.get("task_closed") is not True, out
    assert ("look_around", {}) in stub.calls
    assert not [c for c in stub.calls if c[0] == "step"], stub.calls


def test_the_closing_move_is_read_off_the_env_not_configured(bridge) -> None:
    """When the orchestrator does issue it, a GOAT sub-goal closes with
    SUBTASK_STOP and a single-goal line with STOP. Getting it backwards does
    not raise — it ends the whole episode on sub-goal 0, or refuses to end it
    at all — so it is read off the env's own `n_goals`."""
    B, stub = bridge
    assert B._closing_move() == B.SUBTASK_STOP          # this line has 6 goals
    B._close_action = None
    stub.spec = {**GOAL_SPEC, "n_goals": 1, "goal_index": 0,
                 "goal": {"kind": "none"}, "task": "walk to the kitchen"}
    assert B._closing_move() == B.STOP


def test_the_walking_tools_read_the_env_the_tools_expect(bridge) -> None:
    B, _ = bridge
    obs = B.B.observe()
    assert json.loads(obs[1])["steps_used"] == 0
    res = B.B.step([1, 1, 2])
    assert res["executed"] == 3 and res["steps_used"] == 3 and res["slam"] == "ok"
    m = json.loads(B.B.get_map()[-1])
    assert m["floor"] == 1 and m["explored_m2"] == 31.5  # lifted out of `info`
    marked = json.loads(B.B.mark("kitchen"))
    # marking photographs the spot through the env's own look_around — the four
    # views a later wake-up compares against, so a place named is a place
    # recognisable
    assert marked["marked"]["label"] == "kitchen" and marked["views_banked"] == 4
    assert json.loads(B.B.preview_path(1.0, 2.0)[-1])["distance_m"] == 4.2


def test_the_controllers_readings_come_off_every_step_not_off_get_map(bridge) -> None:
    """ΔA is measured on the whole map's explored area, which every step
    reply carries; a get_map reply's figure is the area of the floor it draws
    and must not replace it. A floor mismatch is noted from either."""
    B, stub = bridge
    stub.step_info = {"explored_m2": 12.0, "current_floor": 0, "floor_mismatch": False}
    B.B.step([1])
    assert B.B._state["body"]["explored_m2"] == 12.0 and B.B._state["body"]["floor_mismatch"] is False
    stub.map_info = {"floor_mismatch": True, "current_floor": 0}
    m = json.loads(B.B.get_map(floor=2)[-1])
    assert m["explored_m2"] == 31.5, "the model is shown the area of the floor drawn"
    body = B.B._state["body"]
    assert body["explored_m2"] == 12.0 and body["floor_mismatch"] is True
    stub.step_info = {"explored_m2": 14.5, "current_floor": 1, "floor_mismatch": False}
    B.B.step([1])
    assert B.B._state["body"]["explored_m2"] == 14.5 and B.B._state["body"]["current_floor"] == 1


def test_a_marked_place_banks_four_views_with_their_pose_estimates(bridge) -> None:
    """A place is a marker and at most four views bucketed by heading, each
    with the navigator's pose estimate: the marker's planar pose, turned."""
    B, _ = bridge
    marked = json.loads(B.B.mark("kitchen"))
    assert "current_floor" not in marked, "the layer is the record's; the model reads floors from get_map"
    views = B.B._state["marker_photos"]["kitchen"]
    assert [v["heading"] for v in views] == ["front", "right", "back", "left"]
    assert [v["pose"]["yaw_deg"] for v in views] == [100.0, -170.0, -80.0, 10.0]
    assert all(v["pose"]["x"] == 1.5 and v["pose"]["z"] == -2.0 and v["pose"]["floor"] == 1 for v in views)
    assert all(Path(v["path"]).is_file() for v in views)


def test_an_env_that_stops_answering_is_declared_dead(bridge) -> None:
    """Not an unlucky call: an env whose SLAM subprocess is gone answers
    /health and 500s every step, and a chain that keeps walking into it records
    a screen of zeros that reads exactly like a very bad agent."""
    B, stub = bridge
    stub.spec = {**GOAL_SPEC}
    for _ in range(B.B.ENV_DEAD_AFTER):
        with pytest.raises(Exception):
            B.B._call("no_such_verb")
    assert B.B._state["env_dead"]["fails"] >= B.B.ENV_DEAD_AFTER
    assert "error" in B.B.step([1]), "every tool owes the model one refusal once the env has gone"


def test_it_is_the_harness_bridge_with_one_function_replaced(bridge) -> None:
    """Every tool, the state file, the stop views and the pre-close gate are
    the harness's own; only the call that reaches the env is ours."""
    B, _ = bridge
    assert B.B._call is B._call
    assert B.B.mcp.name == "navharness-env"
    src = BRIDGE.read_text()
    assert "@mcp.tool" not in src, "a tool defined here would drift from the harness's own"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
