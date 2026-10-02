"""The bridge — the session's tools as a process — end to end on a stub env, in-process.

No simulator, no model, no MCP client: the tool functions are called directly, which is what the
MCP layer does, against an HTTP server that answers the way an ``auto_host`` nodeset answers
(``POST /call/<nodeset>__<fn>`` → ``{"outputs": …}``). What is checked is the contract the
orchestrator relies on: frames and steps land in the state file; ``close_task`` takes the look-around
and records the request WITHOUT issuing the STOP; the file tools write only what the state file says
is writable; the λ hop's file-only mode; the orchestrator's edits are in force on the next call.
"""

from __future__ import annotations

import base64
import importlib
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from core.harnesses.navh.bridge import TaskState

PNG = base64.b64encode(base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)).decode()
MODULE = "core.harnesses.navh.bridge.bridge"


class _Stub:
    def __init__(self) -> None:
        self.calls: list = []
        self.replies: dict = {}                    # fn -> the reply a test sets

    def __call__(self, fn: str, inputs: dict) -> dict:
        self.calls.append((fn, dict(inputs)))
        if fn in self.replies:
            return self.replies[fn]
        if fn == "observe_egocentric":
            return {"rgb": PNG, "explored_m2": 2.0}
        if fn == "step_discrete":
            return {"info": {"collision": False, "slam": "ok"}, "terminated": False, "truncated": False}
        if fn == "look_around":
            return {"views": [PNG, PNG, PNG, PNG]}
        if fn == "get_map":
            return {"map": PNG, "floor": 1, "explored_m2": 3.5}
        if fn == "mark":
            return {"marked": {"label": inputs["label"]}, "n_markers": 1}
        raise KeyError(fn)


@pytest.fixture
def bridge(tmp_path: Path, monkeypatch):
    stub = _Stub()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            fn = self.path.rsplit("__", 1)[-1]
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            try:
                out = json.dumps({"outputs": stub(fn, body.get("inputs") or {})}).encode()
            except KeyError:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    ws = tmp_path / "workspace"
    house = tmp_path / "house"
    state = tmp_path / "state.json"
    monkeypatch.setenv("NAVH_SERVER_URL", f"http://127.0.0.1:{srv.server_port}")
    monkeypatch.setenv("NAVH_LIVE_DIR", str(tmp_path / "live"))
    monkeypatch.setenv("NAVH_MARK_DIR", str(tmp_path / "markers"))
    monkeypatch.setenv("NAVH_STATE_FILE", str(state))
    monkeypatch.setenv("NAVH_WORKSPACE", str(ws))
    monkeypatch.setenv("NAVH_HOUSE_DIR", str(house))
    monkeypatch.setenv("NAVH_STEP_BUDGET", "50")
    monkeypatch.setenv("NAVH_STOP_VIEWS", "4")
    monkeypatch.setenv("NAVH_BENCH", "goat")
    monkeypatch.setenv("NAVH_RECOVERY", "1")
    monkeypatch.setenv("NAVH_TOOLSET", "all")
    ts = TaskState(state)
    ts.reset(0, writable=["run/tasks/0/handover.md"])       # what the orchestrator does before the first hop
    sys.modules.pop(MODULE, None)
    B = importlib.import_module(MODULE)
    yield B, stub, ts
    srv.shutdown()
    sys.modules.pop(MODULE, None)


def _tools(B) -> list[str]:
    return sorted(t.name for t in B.mcp._tool_manager.list_tools())


def test_the_tools_registered_are_the_ones_the_state_names(bridge) -> None:
    B, _, _ = bridge
    assert _tools(B) == sorted(["observe", "step", "get_goal", "get_map", "mark", "preview_path", "close_task",
                                "request_recovery", "list_files", "read_file", "write_file"])


def test_frames_steps_and_the_body_land_in_the_state_file(bridge) -> None:
    B, _, ts = bridge
    out = B.observe()
    assert json.loads(out[1])["frame"] == 1
    res = B.step([2, 1])
    assert res["executed"] == 2 and res["steps_remaining"] == 48
    st = ts.read()
    assert st["task"] == 0 and st["steps_used"] == 2 and [f["kind"] for f in st["frames"]] == ["observe"]
    assert Path(st["frames"][0]["path"]).exists() and st["calls"] == {"observe": 1, "step": 1}
    assert st["body"]["explored_m2"] == 2.0 and st["body"]["slam"] == "ok"
    for bad in ([0], [4], [6], [1, 6]):            # STOP, a tilt, SUBTASK_STOP: only the orchestrator closes
        assert "actions must be a non-empty list" in B.step(bad)["error"], bad


def test_a_step_call_says_when_it_did_not_execute_every_action(bridge) -> None:
    B, stub, _ = bridge
    res = B.step([2] * 15)
    assert res["executed"] == 12 and res["requested"] == 15 and res["not_executed"] == 3
    assert "at most 12 actions" in res["note"] and "3 of the 15" in res["note"]
    assert [fn for fn, _ in stub.calls].count("step_discrete") == 12
    res = B.step([1, 1])
    assert res["requested"] == 2 and "not_executed" not in res and "note" not in res
    doc = {t.name: t.description for t in B.mcp._tool_manager.list_tools()}["step"]
    assert "At most 12 actions per call" in doc


def test_preview_path_says_why_no_route_was_drawn(bridge) -> None:
    B, stub, _ = bridge
    stub.replies["preview_path"] = {"error": "no pose estimate yet"}
    assert json.loads(B.preview_path(1.0, 2.0)[-1]) == {
        "reachable": False, "error": "no route could be planned: no pose estimate yet"}
    stub.replies["preview_path"] = {"map": None, "result": {"reachable": False,
                                                             "reason": "no free-space route on this floor's map"}}
    assert json.loads(B.preview_path(1.0, 2.0)[-1]) == {"reachable": False,
                                                         "reason": "no free-space route on this floor's map"}
    stub.replies["preview_path"] = {}
    out = json.loads(B.preview_path(1.0, 2.0)[-1])
    assert out["reachable"] is False and out["reason"], "never an empty object"
    stub.replies["preview_path"] = {"map": PNG, "result": {"reachable": True, "distance_m": 4.2}}
    assert json.loads(B.preview_path(1.0, 2.0)[-1]) == {"reachable": True, "distance_m": 4.2}


def test_mark_says_the_harness_takes_the_places_photographs(bridge) -> None:
    B, stub, ts = bridge
    doc = {t.name: t.description for t in B.mcp._tool_manager.list_tools()}["mark"]
    assert "the harness takes four views here" in doc and "you took" not in doc
    out = json.loads(B.mark("kitchen"))
    assert out["views_banked"] == 4 and [fn for fn, _ in stub.calls] == ["mark", "look_around"]
    assert [v["heading"] for v in ts.read()["marker_photos"]["kitchen"]] == ["front", "right", "back", "left"]


def test_close_task_takes_the_look_around_and_records_the_request_without_a_stop(bridge) -> None:
    B, stub, ts = bridge
    B.observe()
    res = B.close_task("complete", "the sofa is in view")
    assert res["close_requested"] is True and res["claimed"] == "complete"
    st = ts.read()
    assert st["close_request"] == {"status": "complete", "evidence": "the sofa is in view", "steps_used": 0}
    assert st["close_requests"] == 1 and st["stop_frames"] == [2, 3, 4, 5]
    assert [f["kind"] for f in st["frames"]] == ["observe", "stop", "stop", "stop", "stop"]
    assert not st["closed"] and not st["task_over"]
    assert [fn for fn, _ in stub.calls] == ["observe_egocentric", "look_around"], "no STOP was issued"
    assert "already requested" in B.close_task("complete", "again")["note"]
    # the orchestrator answers between hops; once it has issued the STOP every env tool refuses
    ts.update(close_request=None, closed=True, task_over=True)
    assert "already closed" in B.step([1])["error"]
    assert "already closed" in json.loads(B.observe()[0])["error"]


def test_the_file_tools_write_only_what_the_state_says_is_writable(bridge) -> None:
    B, _, ts = bridge
    ws = Path(B.WORKSPACE)
    (ws / "run").mkdir(parents=True, exist_ok=True)
    (ws / "run" / "ledger.md").write_text("task 1 CERTIFIED\n")
    assert json.loads(B.list_files(""))["entries"] == ["house/", "run/"]
    assert json.loads(B.read_file("run/ledger.md")[0])["text"] == "task 1 CERTIFIED\n"
    refused = json.loads(B.write_file("run/tasks/0/lambda.md", "no"))
    assert refused["error"].startswith("you may not write") and refused["writable"] == ["run/tasks/0/handover.md"]
    ok = json.loads(B.write_file("run/tasks/0/handover.md", "Went to the kitchen at (1.5, -2.0); found it."))
    assert ok["written"] == "run/tasks/0/handover.md" and ok["coordinates_removed"] == 1
    assert "(1.5, -2.0)" not in (ws / "run" / "tasks" / "0" / "handover.md").read_text()
    assert ts.read()["written"] == ["run/tasks/0/handover.md"]
    assert "outside the workspace" in json.loads(B.read_file("../secrets")[0])["error"]
    # the orchestrator allows λ for the λ hop; the bridge sees it on the next call
    ts.allow("run/tasks/0/lambda.md")
    assert "written" in json.loads(B.write_file("run/tasks/0/lambda.md", '{"searched": ["kitchen"]}'))


def test_the_lambda_hop_is_file_only(bridge) -> None:
    B, stub, ts = bridge
    ts.update(only_files=True)
    assert "only the file tools" in B.step([1])["error"]
    assert "only the file tools" in json.loads(B.observe()[0])["error"]
    assert stub.calls == []
    assert "written" in json.loads(B.write_file("run/tasks/0/handover.md", "x"))
    ts.update(only_files=False)
    assert B.step([1])["executed"] == 1


def test_the_ladders_proposal_is_recorded_not_acted_on(bridge) -> None:
    B, stub, ts = bridge
    res = B.request_recovery("room", "I cannot tell which room this is", 0.8)
    assert res["recovery_requested"] is True and res["scope"] == "room"
    assert ts.read()["recovery_request"]["scope"] == "room" and stub.calls == []
    assert "already recorded" in B.request_recovery("attempt", "again")["note"]
    assert B.request_recovery("universe", "x")["error"].startswith("scope must be one of")


def test_a_new_task_written_by_the_orchestrator_is_in_force_on_the_next_call(bridge) -> None:
    B, _, ts = bridge
    B.observe()
    B.step([1, 1, 1])
    ts.reset(4, writable=["run/tasks/4/handover.md"])
    res = B.step([1])
    assert res["steps_used"] == 1
    st = ts.read()
    assert st["task"] == 4 and st["frames"] == [] and st["calls"] == {"step": 1}


def test_a_files_only_bridge_registers_the_file_tools_alone(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NAVH_TOOLSET", "files")
    monkeypatch.setenv("NAVH_STATE_FILE", str(tmp_path / "s.json"))
    monkeypatch.setenv("NAVH_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("NAVH_HOUSE_DIR", str(tmp_path / "house"))
    sys.modules.pop(MODULE, None)
    B = importlib.import_module(MODULE)
    try:
        assert _tools(B) == ["list_files", "read_file", "write_file"]
    finally:
        sys.modules.pop(MODULE, None)


def test_the_tasks_step_budget_stops_motion_and_hands_the_close_to_the_harness(bridge) -> None:
    B, _, ts = bridge
    B._load()
    B._state["steps_used"] = 49                       # NAVH_STEP_BUDGET is 50 in this fixture
    B._save()
    res = B.step([1, 1, 1])
    assert res["executed"] == 1 and res["budget_exhausted"] is True and res["steps_remaining"] == 0
    assert ts.read()["budget_exhausted"] is True
    assert "step budget" in B.step([1])["error"] and "step budget" in json.loads(B.observe()[0])["error"]


def test_read_file_hands_back_a_stored_photograph_as_the_picture(bridge) -> None:
    """Every seat can look at a place's photographs: a .png / .jpg in the
    workspace comes back as an image, not as text."""
    from mcp.server.fastmcp import Image

    B, _, _ = bridge
    photos = Path(B.HOUSE_DIR) / "rooms" / "kitchen" / "photos"
    photos.mkdir(parents=True)
    (photos / "front.png").write_bytes(base64.b64decode(PNG))
    (photos / "left.jpg").write_bytes(b"\xff\xd8jpeg")
    got = B.read_file("house/rooms/kitchen/photos/front.png")
    assert isinstance(got[0], Image) and got[0].data == base64.b64decode(PNG)
    assert json.loads(got[1]) == {"path": "house/rooms/kitchen/photos/front.png", "format": "png",
                                   "bytes": len(base64.b64decode(PNG))}
    jpg = B.read_file("house/rooms/kitchen/photos/left.jpg")
    assert isinstance(jpg[0], Image) and json.loads(jpg[1])["format"] == "jpeg"
    assert "outside the workspace" in json.loads(B.read_file("house/../../x.png")[0])["error"]
