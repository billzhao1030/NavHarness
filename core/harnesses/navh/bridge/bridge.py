#!/usr/bin/env python3
"""The bridge — a session's tools, as the MCP server the CLI spawns beside it.

The CLI runs the model ↔ tool loop for one task inside a session, and this
process is the tools: the six env tools, ``close_task``, the ladder's
proposal, and the three file tools over the workspace (toolset ``all``); the
observation and action tools alone — observe, step, get_goal, close_task —
for a context-only session (``act``); the file tools alone for consolidation
(``files``). It forwards env calls
over HTTP, saves frames with a running id, and keeps the STATE FILE
(``state.py``) current — steps, frames, markers, the body's readings, a
close request, a recovery proposal, the files written. The orchestrator
reads that file between hops and edits it between hops; the bridge re-reads
it at the start of every tool call, so an edit is in force on the next one.

What this process never does: judge, STOP, or open a session. ``close_task``
records the claim, takes the look-around at the stopping point and hands
the turn back; the orchestrator runs the gates and issues the STOP. A tool
that decided a task's outcome would be a tool doing the orchestrator's job.

Env vars (all set by the orchestrator when it opens the session):
  NAVH_SERVER_URL    env auto_host base URL
  NAVH_LIVE_DIR      where obs_*.png frames go        NAVH_MARK_DIR   marker photo sets
  NAVH_STATE_FILE    the state file                    NAVH_WORKSPACE  the workspace root
  NAVH_HOUSE_DIR     the house's own directory (workspace/house/ points at it)
  NAVH_STEP_BUDGET   per-task step budget              NAVH_STOP_VIEWS look-around on a close request
  NAVH_BENCH         ivln | goat                       NAVH_TOOLSET    all | act | files
  NAVH_RECOVERY=1    offer request_recovery            NAVH_LEDGER     text | none (none: the file
                                                       tools read only the current task's directory)
  NAVH_ENV_VERB      the nodeset (default env_slam_vlnce)

An arm whose env speaks other verbs imports this module and replaces
``_call`` (see ``exp_workspace/lhnav_es/mcp/navh_bridge.py``); the tools,
the state file and the file tools stay as they are.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP, Image

if __package__ in (None, ""):                     # started as a script: the repo root goes on the path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from core.harnesses.navh.bridge.state import SPATIAL_TOOLS, fresh_state
from core.harnesses.navh.memory.workspace import Workspace

SERVER_URL = os.environ.get("NAVH_SERVER_URL", "http://127.0.0.1:9242")
LIVE_DIR = Path(os.environ.get("NAVH_LIVE_DIR", "live"))
MARK_DIR = Path(os.environ["NAVH_MARK_DIR"]) if os.environ.get("NAVH_MARK_DIR") else LIVE_DIR.parent / "markers"
STATE_FILE = Path(os.environ.get("NAVH_STATE_FILE", "bridge_state.json"))
STEP_BUDGET = int(os.environ.get("NAVH_STEP_BUDGET", "500"))
#: the motions the agent may send: forward 0.25 m, turn left 15°, turn right 15°
MOTIONS = (1, 2, 3)
#: the most actions one ``step`` call executes; the reply says when more were sent
MAX_ACTIONS = 12
STOP_VIEWS = int(os.environ.get("NAVH_STOP_VIEWS", "4"))
RECOVERY = os.environ.get("NAVH_RECOVERY", "0") == "1"
ENV_DEAD_AFTER = int(os.environ.get("NAVH_ENV_DEAD_AFTER", "8"))   # consecutive env failures = dead, not unlucky
BENCH = os.environ.get("NAVH_BENCH", "ivln")
TOOLSET = os.environ.get("NAVH_TOOLSET", "all")                    # all | act | files
LEDGER = os.environ.get("NAVH_LEDGER", "text")                     # text | none
VERB = os.environ.get("NAVH_ENV_VERB", "env_slam_vlnce")
WORKSPACE = os.environ.get("NAVH_WORKSPACE", "")
HOUSE_DIR = os.environ.get("NAVH_HOUSE_DIR", "")

mcp = FastMCP("navharness-env")

_state: dict[str, Any] = fresh_state(-1)
_state["t0"] = time.time()


def _slug(label: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(label))[:40] or "place"


# ── the state file ────────────────────────────────────────────────────

def _save() -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(_state, default=str))
    tmp.replace(STATE_FILE)


def _load() -> None:
    """Pick up what the orchestrator (or an earlier bridge process) left.

    A task spans several hops; a recovery that voids the context starts a
    NEW session, and so a new bridge process, in the middle of a task; and the
    orchestrator edits the file between hops. Reading it at the start of every
    tool call is what makes all three one task."""
    try:
        prior = json.loads(STATE_FILE.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return
    if isinstance(prior, dict):
        prior.pop("t0", None)
        _state.update(prior)


_load()


# ── the env ───────────────────────────────────────────────────────────

_VERB_ALIAS = {"step": "step_discrete", "observe": "observe_egocentric"}
_env_fails = 0            # consecutive failed calls to the env server


def _call(fn: str, inputs: dict | None = None) -> dict:
    """One call to the env server, with a dead-environment detector: an env
    that has lost its SLAM subprocess answers /health and then 500s every
    step, and left alone the agent keeps "working" against a wall."""
    global _env_fails
    fn = _VERB_ALIAS.get(fn, fn)
    try:
        r = requests.post(f"{SERVER_URL}/call/{VERB}__{fn}", json={"inputs": inputs or {}}, timeout=300)
        r.raise_for_status()
        out = r.json()["outputs"]
    except Exception as e:
        _env_fails += 1
        if _env_fails >= ENV_DEAD_AFTER and not _state.get("env_dead"):
            _state["env_dead"] = {"fn": fn, "fails": _env_fails, "error": str(e)[:200],
                                  "steps_used": _state["steps_used"]}
            _save()
        raise
    _env_fails = 0
    return out


def _refusal() -> dict | None:
    """The one answer every env tool owes the model once it must not act."""
    _load()
    if _state.get("env_dead"):
        return {"error": "the environment has stopped responding; this run is being abandoned. "
                         "Stop calling tools and reply DONE."}
    if _state.get("closed") or _state.get("task_over"):
        return {"error": "this task is already closed; stop calling tools and reply DONE"}
    if _state.get("only_files"):
        return {"error": "only the file tools are available in this turn: write the note you were "
                         "asked for with write_file, then reply DONE"}
    if _state.get("budget_exhausted"):
        return {"error": "the step budget of this task is used up; the harness closes the task. "
                         "Stop calling tools and reply DONE."}
    return None


def _count(name: str) -> None:
    _state["calls"][name] = _state["calls"].get(name, 0) + 1


def _note_status(out: dict, *, area: bool = True) -> None:
    """Keep whatever the env volunteered about the body; the ladder's signals
    are read from here, never from the story.

    ``explored_m2`` is the whole map's explored area, which every motion reply
    carries; newly explored area is measured on it without a ``get_map`` call.
    A ``get_map`` reply reports the area of the floor it draws instead, so it
    is noted with ``area=False`` and cannot stand in for the whole map."""
    if not isinstance(out, dict):
        return
    info = out.get("info")
    src = {**out, **(info if isinstance(info, dict) else {})}
    keys = ("explored_m2", "floor", "current_floor", "slam", "reanchors", "floor_mismatch")
    for k in keys if area else keys[1:]:
        if src.get(k) is not None:
            _state["body"][k] = src[k]


def _frame(png_b64: str, kind: str) -> int:
    fid = len(_state["frames"]) + 1
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    p = LIVE_DIR / f"obs_{fid:04d}.png"
    p.write_bytes(base64.b64decode(png_b64))
    _state["frames"].append({"id": fid, "path": str(p), "step": _state["steps_used"], "kind": kind})
    _save()
    return fid


def _views_of(out: dict) -> list[str]:
    """The four pictures a ``look_around`` returns, clockwise from the entry heading."""
    if not isinstance(out, dict):
        return []
    views = [v for v in (out.get("views") or []) if isinstance(v, str) and v]
    if views:
        return views
    return [out[k] for k in ("view_front", "view_right", "view_back", "view_left")
            if isinstance(out.get(k), str) and out[k]]


def _look_around(views: int, kind: str) -> list[int]:
    """A panorama taken by the harness, through the env's own SLAM-silent
    ``look_around``: nothing is charged to the step budget, and no step verb
    is sent that could end a subtask mid-spin."""
    try:
        out = _call("look_around")
    except Exception:
        return []
    if not isinstance(out, dict) or out.get("error"):
        return []
    shots: list[int] = []
    for png in _views_of(out)[:max(1, views)]:
        try:
            shots.append(_frame(png, kind))
        except Exception:
            break
    return shots


#: the look-around's four views, clockwise from the entry heading, and their offsets from it
HEADINGS = ("front", "right", "back", "left")
_HEADING_DEG = (0.0, 90.0, 180.0, -90.0)


def _view_pose(marked: dict, offset_deg: float, floor: Any) -> dict | None:
    """A banked view's SLAM pose estimate: the marker's planar pose, turned to
    the view's heading. None when the env reported no pose for the marker."""
    if not isinstance(marked, dict) or marked.get("x") is None or marked.get("z") is None:
        return None
    pose = {"x": marked["x"], "z": marked["z"], "yaw_deg": None, "floor": floor}
    if marked.get("yaw_deg") is not None:
        pose["yaw_deg"] = round((float(marked["yaw_deg"]) + offset_deg + 180.0) % 360.0 - 180.0, 1)
    return pose


def _bank_views(label: str, views: int = 4, marked: dict | None = None, floor: Any = None) -> int:
    """Photograph the spot a marker was just dropped on, for later recognition
    (a wake-up asks "which of the places I named does this look like"). Kept
    OUT of ``frames``: that list is the judge's evidence.

    A place is a marker and at most four views bucketed by heading, each with
    the navigator's pose estimate: ``marked`` is the env's reply to the mark
    (its planar pose) and ``floor`` the layer it went on."""
    try:
        out = _call("look_around")
    except Exception:
        return 0
    if not isinstance(out, dict) or out.get("error"):
        return 0
    offsets = [float(h) for h in (out.get("headings_deg") or _HEADING_DEG)]
    d = MARK_DIR / _slug(label)
    d.mkdir(parents=True, exist_ok=True)
    got: list[dict] = []
    for i, png in enumerate(_views_of(out)[:max(1, min(views, len(HEADINGS)))]):
        try:
            f = d / f"view_{i}.png"
            f.write_bytes(base64.b64decode(png))
            got.append({"id": f"{label}:{i}", "path": str(f), "step": _state["steps_used"],
                        "heading": HEADINGS[i],
                        "pose": _view_pose(marked or {}, offsets[i] if i < len(offsets) else 0.0, floor)})
        except Exception:
            break
    if got:
        _state["marker_photos"][label] = got             # a re-mark replaces the old set
        _save()
    return len(got)


# ── the env tools ─────────────────────────────────────────────────────

def observe() -> list:
    if (no := _refusal()):
        return [json.dumps(no)]
    _count("observe")
    out = _call("observe")
    _note_status(out)
    fid = _frame(out["rgb"], "observe")
    return [Image(data=base64.b64decode(out["rgb"]), format="png"),
            json.dumps({"frame": fid, "steps_used": _state["steps_used"],
                        "steps_remaining": max(0, STEP_BUDGET - _state["steps_used"])})]


def step(actions: list[int]) -> dict:
    if (no := _refusal()):
        return no
    _count("step")
    asked = [int(a) for a in (actions or [])]
    acts = asked[:MAX_ACTIONS]
    # Only the three motions. The env would also take its closing moves and the
    # camera tilts, and a closing move sent from here would end the task past
    # the handover and the judge's gate: only the orchestrator issues STOP.
    if not acts or any(a not in MOTIONS for a in acts):
        return {"error": "actions must be a non-empty list of 1 (forward), 2 (turn left) and "
                         "3 (turn right); use close_task to finish"}
    executed, collision, last = 0, False, {}
    for a in acts:
        if _state["steps_used"] >= STEP_BUDGET:
            # the task's step budget: no further motion; the orchestrator closes the task
            _state["budget_exhausted"] = True
            break
        out = _call("step", {"action": a})
        if "error" in out:
            _save()
            return out
        executed += 1
        _state["steps_used"] += 1
        _note_status(out)
        last = out.get("info") or {}
        collision = collision or bool(last.get("collision"))
        if out.get("truncated") or (BENCH == "goat" and last.get("subtask_result")):
            _state["task_over"] = True          # the env ended it: the budget ran out
            _state["end_payload"] = out
            _state["end_reason"] = "budget"
            break
    _save()
    res = {"executed": executed, "requested": len(asked), "collision": collision,
           "steps_used": _state["steps_used"],
           "steps_remaining": max(0, STEP_BUDGET - _state["steps_used"]), "slam": last.get("slam")}
    notes: list[str] = []
    if len(asked) > MAX_ACTIONS:
        res["not_executed"] = len(asked) - MAX_ACTIONS
        notes.append(f"A step call executes at most {MAX_ACTIONS} actions: the last "
                     f"{len(asked) - MAX_ACTIONS} of the {len(asked)} you sent were not executed.")
    if _state["task_over"]:
        res["task_over"] = True
        notes.append("Step budget exhausted; the task is closed. Stop calling tools and reply DONE.")
    elif _state.get("budget_exhausted"):
        res["budget_exhausted"] = True
        notes.append("The step budget of this task is used up; the harness closes the task now. "
                     "Stop calling tools and reply DONE.")
    if notes:
        res["note"] = " ".join(notes)
    return res


def get_goal() -> list:
    if (no := _refusal()):
        return [json.dumps(no)]
    _count("get_goal")
    if BENCH == "goat":
        g = _call("get_goal")
        if "error" in g:
            return [json.dumps(g)]
        out: list = [json.dumps({"task": g.get("subtask_index", 0) + 1, "of": g.get("subtask_total"),
                                 "goal": g.get("text"), "modality": g.get("modality")})]
        if g.get("image"):
            out.append(Image(data=base64.b64decode(g["image"]), format="png"))
        return out
    meta = _call("reset", {"trigger": "bridge"})     # idempotent read of the seated task
    return [json.dumps({"task": int(meta.get("tour_slot", 0)) + 1, "of": meta.get("tour_length"),
                        "goal": meta.get("instruction", ""), "modality": "instruction"})]


def _keep_map(png_b64: str) -> None:
    """Keep the map the MODEL was just shown, beside the frames: a few KB, and
    the only way to ask afterwards what its map looked like when it turned."""
    try:
        _state["map_shots"] = n = int(_state.get("map_shots", 0)) + 1
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        (LIVE_DIR / f"map_{n:04d}.png").write_bytes(base64.b64decode(png_b64))
    except Exception:
        pass


def get_map(floor: int = 0, zoom: bool = False) -> list:
    if (no := _refusal()):
        return [json.dumps(no)]
    _count("get_map")
    out = _call("get_map", {"floor": int(floor) if floor else None, "view": "local" if zoom else "global"})
    if out.get("map") is None:
        return [json.dumps({"error": out.get("error", "no map")})]
    _note_status(out, area=False)
    _save()
    status = {k: out.get(k) for k in ("floor", "explored_m2", "markers", "map_window_m", "orientation", "pose", "slam")
              if k in out}
    _keep_map(out["map"])
    res: list = [Image(data=base64.b64decode(out["map"]), format="png")]
    if out.get("map_local"):
        res += ["8 m close-up:", Image(data=base64.b64decode(out["map_local"]), format="png")]
    res.append(json.dumps(status))
    return res


def mark(label: str) -> str:
    if (no := _refusal()):
        return json.dumps(no)
    _count("mark")
    out = _call("mark", {"label": str(label)[:40]})
    out = dict(out.get("result") or out)
    # the layer the marker went on is the banked views' record; the model reads
    # floors as get_map names them
    floor = out.pop("current_floor", None)
    if "marked" in out:
        lab = out["marked"]["label"]
        _state["marks"].append(lab)
        _save()
        n = _bank_views(lab, marked=out["marked"], floor=floor)
        out = dict(out, views_banked=n)
    return json.dumps(out)


def preview_path(x: float, z: float) -> list:
    if (no := _refusal()):
        return [json.dumps(no)]
    _count("preview_path")
    out = _call("preview_path", {"x": float(x), "z": float(z)})
    r = out.get("result") or {k: out.get(k) for k in ("reachable", "distance_m", "snapped", "reason", "error")
                              if k in out}
    if out.get("map") is None:
        return [json.dumps(_no_route(r))]
    return [Image(data=base64.b64decode(out["map"]), format="png"), json.dumps(r)]


def _no_route(r: dict) -> dict:
    """What ``preview_path`` answers when no route was drawn: unreachable and
    why, or the error that kept the route from being planned — never an
    empty object the model has to guess at."""
    if r.get("error"):
        return {"reachable": False, "error": f"no route could be planned: {r['error']}"}
    if r.get("reachable") and r.get("distance_m") is not None:
        return dict(r)                           # a route whose picture did not come back
    return {"reachable": False,
            "reason": str(r.get("reason") or "no route over the known free space to that point")}


# ── the ladder's proposal ─────────────────────────────────────────────

_SCOPES = ("attempt", "place", "room", "floor", "house")


def request_recovery(scope: str, reason: str, confidence: float = 0.5) -> dict:
    """Propose a RECOVERY SCOPE. This records the proposal; the orchestrator's
    controller decides what actually happens — it may honour it, refuse it,
    or go a rung higher than asked."""
    if (no := _refusal()):
        return no
    _count("request_recovery")
    want = str(scope or "attempt").strip().lower()
    if want not in _SCOPES:
        return {"error": f"scope must be one of {', '.join(_SCOPES)}"}
    if _state.get("recovery_request"):
        return {"recovery_requested": True, "note": "already recorded; stop calling tools and reply DONE"}
    try:
        conf = min(1.0, max(0.0, float(confidence)))
    except (TypeError, ValueError):
        conf = 0.5
    _state["recovery_request"] = {"scope": want, "reason": str(reason)[:400],
                                  "confidence": conf, "steps_used": _state["steps_used"]}
    _save()
    return {"recovery_requested": True, "scope": want,
            "note": ("Recorded. The harness decides which scope is actually applied and may choose "
                     "a different one. Stop calling tools now and reply with the single word DONE.")}


# ── the close request ─────────────────────────────────────────────────

def close_task(status: str, evidence: str) -> dict:
    """Ask to END the task. The look-around at the stopping point is taken
    here — independent evidence, before anyone judges — and the request is
    recorded. The gates and the STOP are the orchestrator's, between hops."""
    if (no := _refusal()):
        return no
    _count("close_task")
    if _state.get("close_request"):
        return {"close_requested": True, "note": "already requested; stop calling tools and reply DONE"}
    st = "incomplete" if str(status).lower().startswith("in") else "complete"
    _state["close_requests"] += 1
    _state["stop_frames"] = _look_around(STOP_VIEWS, "stop")
    _state["close_request"] = {"status": st, "evidence": str(evidence)[:400], "steps_used": _state["steps_used"]}
    _save()
    if TOOLSET == "act":
        return {"close_requested": True, "claimed": st, "note": _ACT_CLOSE_NOTE}
    return {"close_requested": True, "claimed": st,
            "note": ("Recorded. The harness checks your files and your claim and answers on the next "
                     "turn. Stop calling tools now and reply with the single word DONE.")}


# ── the file tools ────────────────────────────────────────────────────

_ws: Workspace | None = None


def _workspace() -> Workspace | None:
    global _ws
    if _ws is None and WORKSPACE:
        try:
            _ws = Workspace(Path(WORKSPACE), Path(HOUSE_DIR) if HOUSE_DIR else Path(WORKSPACE) / "house")
        except OSError:
            return None
    return _ws


def _scope() -> str:
    """What the file tools may read: everything (``ledger=text``), or only the
    current task's own directory (``ledger=none``) — its goal, the handover it
    writes and the recovery notes passed between its attempts."""
    return f"run/tasks/{int(_state.get('task', -1))}" if LEDGER == "none" else ""


def list_files(path: str = "") -> str:
    _load()
    _count("list_files")
    ws = _workspace()
    if ws is None:
        return json.dumps({"error": "no workspace"})
    return json.dumps(ws.session_list(str(path or ""), scope=_scope()), ensure_ascii=False)


def read_file(path: str) -> list:
    """A file's text; a stored picture (.png / .jpg) comes back as the image,
    so a seat without file tools of its own can look at a place's photographs."""
    _load()
    _count("read_file")
    ws = _workspace()
    if ws is None:
        return [json.dumps({"error": "no workspace"})]
    out = ws.session_read(str(path or ""), scope=_scope())
    data = out.pop("data", None)
    if data is not None:
        return [Image(data=data, format=out["format"]), json.dumps(out, ensure_ascii=False)]
    return [json.dumps(out, ensure_ascii=False)]


def write_file(path: str, text: str) -> str:
    _load()
    _count("write_file")
    ws = _workspace()
    if ws is None:
        return json.dumps({"error": "no workspace"})
    out = ws.session_write(str(path or ""), str(text or ""), set(_state.get("writable") or ()))
    if "written" in out:
        _state["written"].append(out["written"])
    _save()
    return json.dumps(out, ensure_ascii=False)


# ── registration: what this session is given ─────────────────────────

_ENV_TOOLS = (
    (observe, "Look through the forward camera. Returns an RGB image. Free."),
    (step, "Execute movement actions in order: 1 = forward 0.25 m, 2 = turn left 15°, 3 = turn right 15°. "
           f"Each costs one step. At most {MAX_ACTIONS} actions per call; any more are not executed. "
           "Do not use 0; use close_task to finish."),
    (get_goal, "Re-read the current task's goal (returns the goal photograph if there is one)."),
    (get_map, "Top-down occupancy map built by the robot, persistent across tasks, with your markers on it. "
              "floor=k views another storey; zoom=true adds an 8 m close-up. Free."),
    (mark, "Drop (or move) a named marker at your position. Name it for the PLACE, not for what you came "
           "to find: 'kitchen', 'upstairs landing', 'utility room' -- short and distinct, not 'fridge' or "
           "'the carpet', and not a sentence. Marking photographs the spot: the harness takes four views "
           "here (front, right, back, left) and they become that place's photo set, and on a later day "
           "a room is recognised by comparing what can be seen against those photos; two places whose "
           "names start alike cannot be told apart. Markers outlive tasks. Free."),
    (preview_path, "Draw a route over known free space to map point (x, z) and report its length. "
                   "Nothing moves. Free."),
    (close_task, "Ask to END the current task. status='complete' if you are at the goal, 'incomplete' if "
                 "you searched and it is not here or not reachable. evidence: one sentence by landmark "
                 "names, no coordinates. Write your handover file first: the request is not honoured "
                 "until it exists. This is the only way a task ends."),
)
_RECOVERY_DOC = (
    "Ask the harness for a RECOVERY, naming how much of your current state you believe is "
    "no longer trustworthy. Pick the SMALLEST scope that fits what has gone wrong:\n"
    "  attempt — your reasoning is stuck: same views, same ground, plan no longer matches "
    "what you see. Your position and map are fine.\n"
    "  place   — you have been moved but this still looks like a room you know; what is wrong "
    "is where you think you are standing.\n"
    "  room    — you are in the wrong room, or you cannot say which room this is.\n"
    "  floor   — you are on a different storey than the one you have been planning on.\n"
    "  house   — you have come back after an interruption and need what you knew before.\n"
    "Your map, markers, files and steps survive every one of them. confidence: 0-1, how sure "
    "you are that this is the right scope. This does NOT end the task; only close_task does."
)
_FILE_TOOLS = (
    (list_files, "What is in the workspace (or under one path of it). Free."),
    (read_file, "The text of one file in the workspace; a .png or .jpg comes back as the picture. Free."),
    (write_file, "Write one of the files you own (the instruction names them). Replaces the file. "
                 "No coordinates. Free."),
)
# A context-only session has no handover to write before it closes, so its
# close_task does not ask for one, and its answer does not speak of files.
_ACT_CLOSE_DOC = ("Ask to END the current task. status='complete' if you are at the goal, 'incomplete' if "
                  "you searched and it is not here or not reachable. evidence: one sentence. This is the "
                  "only way a task ends.")
_ACT_CLOSE_NOTE = ("Recorded. The harness answers on the next turn. Stop calling tools now and reply with "
                   "the single word DONE.")

if TOOLSET == "act":
    # observation and action only: no map, no markers, no route preview, no
    # recovery proposal, no files
    for _fn, _doc in _ENV_TOOLS:
        if _fn.__name__ not in SPATIAL_TOOLS:
            mcp.tool(description=_ACT_CLOSE_DOC if _fn is close_task else _doc)(_fn)
elif TOOLSET != "files":
    for _fn, _doc in _ENV_TOOLS:
        mcp.tool(description=_doc)(_fn)
    if RECOVERY:
        mcp.tool(description=_RECOVERY_DOC)(request_recovery)
if TOOLSET != "act":
    for _fn, _doc in _FILE_TOOLS:
        mcp.tool(description=_doc)(_fn)


if __name__ == "__main__":
    _save()
    mcp.run(transport="stdio")
