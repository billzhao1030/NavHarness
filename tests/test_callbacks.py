"""The recorders, on canned input — no server, no tokens.

The budget fuse must raise StopRun; archive_attempt must move every attempt
file aside and leave the plain names free.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.bus import EventBus
from core.callbacks import CostLogger, JsonlLogger, StopRun
from core.callbacks.base import CallbackSet
from runner import archive_attempt

REPO = Path(__file__).resolve().parents[1]


def _ep(index: int = 0, **kw) -> SimpleNamespace:
    return SimpleNamespace(index=index, run_dir=kw.get("run_dir"), raw_dir=kw.get("raw_dir"),
                           server_url="http://stub", bus=None)


def test_budget_fuse_raises_stop_run() -> None:
    cb = CostLogger()
    cb.on_run_start(SimpleNamespace(cfg={"budget_usd": 1.0},
                                    app={"agent": {"model": {"id": "claude-fable-5"}}},   # the seat: priced, not fake
                                    prior_episodes={7: {"agent": {"total_cost_usd": 0.8}}}))
    ep = _ep(0)
    cb.on_episode_start(ep)
    # usd is the provider's tokens x litellm's list price (core/pricing.py), not the
    # harness's own figure: a million input tokens of fable-5 is far past the 0.2 left
    record = {"index": 0, "agent": {"total_cost_usd": 0.3, "num_turns": 2,
                                    "usage": {"input_tokens": 1_000_000, "output_tokens": 0}}}
    with pytest.raises(StopRun):
        cb.on_episode_end(ep, record)
    cost = record["agent"]["cost"]
    assert cb.stopped_by_budget and cost["source"] == "priced" and cost["usd"] > 0.2
    assert cost["settled_usd"] == 0.3   # the harness's figure is kept beside it, not used


def test_archive_attempt_moves_every_file(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    (tmp_path / "live_3").mkdir()
    for name in ("episode_3.jsonl", "raw/episode_3.jsonl", "raw/episode_3.stderr.log",
                 "raw/context_manifest_3.json"):
        (tmp_path / name).write_text("x")
    archive_attempt(tmp_path, 3, 1)
    assert not (tmp_path / "episode_3.jsonl").exists()
    assert (tmp_path / "episode_3.attempt1.jsonl").read_text() == "x"
    assert (tmp_path / "raw" / "episode_3.attempt1.jsonl").exists()
    assert (tmp_path / "raw" / "context_manifest_3.attempt1.json").exists()
    assert (tmp_path / "live_3.attempt1").is_dir()
    archive_attempt(tmp_path, 3, 2)   # nothing to move: a no-op, not an error


def test_jsonl_logger_and_bus_tally(tmp_path: Path) -> None:
    ep = _ep(0, run_dir=tmp_path)
    cbs = CallbackSet([JsonlLogger()])
    bus = EventBus(ep, cbs)
    ep.bus = bus
    cbs.fire("on_episode_start", ep)
    bus.emit("tool_use", {"id": "1", "name": "mcp__env__step", "input": {"actions": [0]}})
    bus.emit("tool_result", {"tool_use_id": "1",
                             "texts": [json.dumps({"steps_taken_total": 1, "end_reason": "stop_called"})]})
    cbs.fire("on_episode_end", ep, {"index": 0})
    lines = [json.loads(ln) for ln in (tmp_path / "episode_0.jsonl").read_text().splitlines()]
    assert [ln["kind"] for ln in lines] == ["tool_use", "tool_result"]
    assert bus.tool_calls == {"step": 1}
    assert bus.last_step_result["end_reason"] == "stop_called"
