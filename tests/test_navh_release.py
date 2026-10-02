"""Harness behaviours, against stubs: the house notes (the journal and the
photo sets the harness files, consolidation the only writer of the notes and
what it reads), the handover's contents, multi-episode GOAT chains for a
deployment day, the IR2R-CE goal numbering and the relocation a placed
transit is, the adapter's memory isolation,
the Claude Code seat's fenced notes reader, and the library defaults."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from test_navh_orchestrator import scripted  # noqa: F401 — scripted is a fixture

from core.harnesses.navh.env.bench_es import GOATEsBench, TourEsBench
from core.harnesses.navh.memory.longterm import NOTE_PATTERNS, HouseNotes
from core.harnesses.navh.memory.workspace import Workspace
from core.harnesses.navharness import NavHarnessAdapter

# ── the house notes ─────────────────────────────────────────────────────


def test_journal_lines_are_task_records_with_the_judges_label(tmp_path: Path) -> None:
    notes = HouseNotes(tmp_path / "house", day="2026-09-19")
    notes.journal(0, "find a fridge", "CERTIFIED", 42, ["kitchen"], "Went to the kitchen at (1.5, -2.0).")
    notes.journal(1, "find a bed", "NOT DONE", 500, [], "")
    text = notes.journal_today.read_text()
    assert text.startswith("# 2026-09-19")
    assert "- task 1 · CERTIFIED · find a fridge · markers: kitchen · 42 steps · account:" in text
    assert "(1.5, -2.0)" not in text, "coordinates are scrubbed from the journal"
    assert "- task 2 · NOT DONE · find a bed · 500 steps" in text


def _views(tmp_path: Path, tag: str, yaw: float = 0.0, step: int = 3) -> list[dict]:
    views = []
    for i, (h, off) in enumerate(zip(("front", "right", "back", "left"), (0, 90, 180, -90))):
        p = tmp_path / f"{tag}_{i}.png"
        p.write_bytes(f"png-{tag}-{h}".encode())
        views.append({"id": f"kitchen:{i}", "path": str(p), "step": step, "heading": h,
                      "pose": {"x": 1.5, "z": -2.0, "yaw_deg": yaw + off, "floor": 0}})
    return views


def test_a_place_keeps_four_views_one_per_heading_with_their_pose(tmp_path: Path) -> None:
    notes = HouseNotes(tmp_path / "house", day="2026-09-19")
    assert notes.attach_photos("Kitchen", _views(tmp_path, "a"), 0) == 4
    assert notes.attach_photos("Kitchen", _views(tmp_path, "a"), 0) == 0, "the same views are not filed twice"
    index = json.loads((notes.dir / "rooms" / "kitchen" / "photos.json").read_text())
    assert [v["heading"] for v in index] == ["front", "right", "back", "left"]
    assert index[1]["pose"] == {"x": 1.5, "z": -2.0, "yaw_deg": 90.0, "floor": 0} and index[0]["label"] == "Kitchen"
    # re-marked on a later day: the place keeps the latest view of each heading, still four
    day2 = HouseNotes(tmp_path / "house", day="2026-09-20")
    assert day2.attach_photos("Kitchen", _views(tmp_path, "b", yaw=30.0, step=7), 4) == 4
    index = json.loads((notes.dir / "rooms" / "kitchen" / "photos.json").read_text())
    assert len(index) == 4 and {v["day"] for v in index} == {"2026-09-20"} and index[0]["task"] == 5
    files = sorted(p.name for p in (notes.dir / "rooms" / "kitchen" / "photos").iterdir())
    assert files == ["back.png", "front.png", "left.png", "right.png"]
    assert (notes.dir / "rooms" / "kitchen" / "photos" / "front.png").read_bytes() == b"png-b-front"


def test_a_later_run_finds_the_views_of_the_places_it_inherits(tmp_path: Path) -> None:
    notes = HouseNotes(tmp_path / "house", day="2026-09-19")
    notes.attach_photos("Kitchen", _views(tmp_path, "a"), 0)
    legacy = notes.dir / "rooms" / "hall"                 # a set filed without its label
    (legacy / "photos").mkdir(parents=True)
    (legacy / "photos" / "front.png").write_bytes(b"png")
    (legacy / "photos.json").write_text(json.dumps([{"file": "front.png", "heading": "front"}]))
    views = HouseNotes(tmp_path / "house", day="2026-09-20").photo_views(["Hall"])
    assert sorted(views) == ["Hall", "Kitchen"]
    assert [v["heading"] for v in views["Kitchen"]] == ["front", "right", "back", "left"]
    assert all(Path(v["path"]).is_absolute() and Path(v["path"]).is_file() for v in views["Kitchen"])
    assert views["Kitchen"][2]["pose"]["yaw_deg"] == 180.0


def test_the_harness_writes_none_of_the_four_house_note_files(tmp_path: Path) -> None:
    """Consolidation is the only writer of the index, the overview, the room
    notes and the skills. The harness files the journal and the photo sets."""
    notes = HouseNotes(tmp_path / "house", day="2026-09-19")
    notes.journal(0, "find a fridge", "CERTIFIED", 10, ["kitchen"], "it was there")
    notes.attach_photos("kitchen", _views(tmp_path, "a"), 0)
    assert not hasattr(notes, "finalize"), "nothing completes the notes after the session"
    written = sorted(p.relative_to(notes.dir).as_posix() for p in notes.dir.rglob("*") if p.is_file())
    assert written == ["journal/2026-09-19.md", "rooms/kitchen/photos.json", "rooms/kitchen/photos/back.png",
                       "rooms/kitchen/photos/front.png", "rooms/kitchen/photos/left.png",
                       "rooms/kitchen/photos/right.png"]
    assert notes.list_notes() == ["journal/2026-09-19.md"]


def test_consolidation_reads_the_ledger_the_handovers_and_the_notes(tmp_path: Path) -> None:
    from core.harnesses.navh import prompts as P
    from core.harnesses.navh.session.consolidate import ConsolidateSession

    text = P.CONSOLIDATE.format(ledger="run/ledger.md", tasks="run/tasks", journal="house/journal/d.md",
                                photos="house/rooms/kitchen/photos.json")
    required = text.split("Read all of these before you write:", 1)[1].split("The photo sets", 1)[0]
    for item in ("run/ledger.md", "run/tasks", "handover.md", "lambda*.md", "house/MEMORY.md",
                 "house/journal/d.md"):
        assert item in required, item
    assert "unclear" not in text, "the ledger and the handovers are required reading, not a fallback"
    assert "house/rooms/kitchen/photos.json" in text
    assert "appends" not in P.CONSOLIDATE_SYSTEM, "the harness adds nothing to the notes after the session"
    reads = ConsolidateSession.reads
    for item in ("run/ledger.md", "run/tasks/*/handover.md", "run/tasks/*/lambda*.md", "house/MEMORY.md",
                 "house/house.md", "house/rooms/*.md", "house/skills/*.md", "house/journal/<day>.md",
                 "house/rooms/*/photos.json"):
        assert item in reads, item


def test_a_handover_names_what_is_still_unsearched_or_uncertain() -> None:
    from core.harnesses.navh import prompts as P

    for text in (P.SYSTEM, P.HANDOVER_REQUIRED):
        flat = " ".join(text.split())
        assert "what is still unsearched or uncertain" in flat and "120 words" in flat


def test_only_the_note_files_are_writable_by_consolidation(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws", tmp_path / "house")
    allowed = set(NOTE_PATTERNS)
    for ok in ("house/MEMORY.md", "house/house.md", "house/rooms/kitchen.md", "house/skills/find-a-bed.md"):
        assert "written" in ws.session_write(ok, "note", allowed), ok
    for bad in ("house/journal/2026-09-19.md", "run/ledger.md", "house/rooms/kitchen/photos/x.md",
                "house/places.json"):
        assert "error" in ws.session_write(bad, "x", allowed), bad


# ── the GOAT deployment day: several episodes of one house as one chain ──


class _ChainEnv:
    """Two GOAT episodes of one house, two sub-goals each."""

    url = "http://stub"

    def __init__(self) -> None:
        self.calls: list = []
        self.ep = 3
        self.sub = 0

    def call(self, fn: str, inputs: dict | None = None, config: dict | None = None) -> dict:
        self.calls.append((fn, dict(inputs or {})))
        if fn == "goal_spec":
            if self.sub >= 2:
                return {"episode_over": True, "n_goals": 2, "goal_index": 2}
            return {"n_goals": 2, "goal_index": self.sub, "scene_id": "data/scene_datasets/hm3d/val/0-abc/abc.glb",
                    "episode_id": str(self.ep), "goal": {"kind": "object", "category": f"thing{self.ep}{self.sub}"}}
        if fn == "place_next":
            self.ep, self.sub = int(inputs["index"]), 0
            return {"episode_id": str(self.ep), "transit": {"lost_ticks": 32,
                                                            "reloc_error_m": 2.4, "slam": "slam_lost"}}
        return {}


def test_a_deployment_day_chains_episodes_with_the_map_kept() -> None:
    env = _ChainEnv()
    bench = GOATEsBench(env, split="val_unseen", step_budget=500)
    g = bench.seat([3, 7], persist_map=True)
    assert (g.index, g.total, bench.house_id) == (0, 2, "abc")
    env.sub = 1                                        # sub-goal 0 closed, episode continues
    end = bench.after_end(SimpleNamespace(chain_finished=False, next_goal=None, wakeup=None), True)
    assert end.next_goal.index == 1 and end.wakeup is None
    env.sub = 2                                        # the episode's last sub-goal closed
    end = bench.after_end(SimpleNamespace(chain_finished=True, next_goal=None, wakeup=None), True)
    assert end.chain_finished is False, "the day goes on with the next episode"
    assert ("place_next", {"index": 7}) in env.calls
    assert end.next_goal.index == 2 and end.next_goal.total == 4, "numbered across the day"
    assert end.wakeup["episode"] == 7 and end.wakeup["map_carried_over"] is True
    assert end.wakeup["slam"] == "slam_lost" and end.wakeup["lost_ticks"] == 32, \
        "the tracker's relocation report is carried to the record"
    env.sub = 2
    end = bench.after_end(SimpleNamespace(chain_finished=True, next_goal=None, wakeup=None), True)
    assert end.chain_finished is True and end.next_goal is None


def test_an_ir2r_tour_numbers_its_tasks_by_slot() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "lhnav_navh_bridge_release", Path(__file__).resolve().parents[1] / "exp_workspace/lhnav_es/mcp/navh_bridge.py")
    assert spec is not None
    source = spec.origin
    text = Path(source).read_text()
    assert '"subtask_index": int(spec.get("tour_slot") or 0) if tour' in text
    bench = TourEsBench(_ChainEnv(), split="val_unseen", step_budget=500)
    g = bench._goal_of({"tour_slot": 4, "tour_size": 9, "goal_index": 0, "n_goals": 1, "task": "walk to the sofa"})
    assert (g.index, g.total) == (4, 9)


class TourEnv:
    """An IR2R-CE tour of three episodes; each transit reports as the env's
    ``advance_tour`` does (``walk_transit`` + ``continue_tour``)."""

    url = "http://stub"

    def __init__(self, transits: list[dict]) -> None:
        self.calls: list = []
        self.slot = 0
        self.transits = list(transits)

    def call(self, fn: str, inputs: dict | None = None, config: dict | None = None) -> dict:
        self.calls.append((fn, dict(inputs or {})))
        if fn == "goal_spec":
            return {"n_goals": 1, "goal_index": 0, "tour_slot": self.slot, "tour_size": 3,
                    "task": f"walk to room {self.slot}", "goal": {"kind": "none"},
                    "episode_id": f"ep{self.slot}", "scene_id": "data/scene_datasets/mp3d/abc/abc.glb"}
        if fn == "advance_tour":
            if self.slot + 1 >= 3:
                return {"tour_over": True, "tour_slot": self.slot, "tour_size": 3}
            self.slot += 1
            return {"tour_over": False, "tour_slot": self.slot, "tour_size": 3, "transit": self.transits.pop(0)}
        if fn == "step" and int((inputs or {}).get("action", -1)) == 0:
            return {"terminated": True, "truncated": False, "info": {"metrics": {"success": 1.0}}}
        if fn == "look_around":
            return {"views": [PNG] * 4}
        return {}


WALKED = {"walked": True, "teleported": False, "transit_feeds": 40, "transit_ticks": 160, "slam": "ok",
          "reloc_error_m": 0.05, "lost_ticks": 0, "spin_ticks": 0, "feeds": 0, "epochs": 1}
PLACED = {"walked": True, "teleported": True, "transit_feeds": 12, "transit_ticks": 48, "slam": "ok",
          "reloc_error_m": 0.4, "pose_jump_m": 1.1, "lost_ticks": 3, "spin_ticks": 120, "feeds": 40,
          "epochs": 1, "level": 0, "explored_m2": 61.0, "markers_kept": 2}
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def test_a_walked_tour_transit_is_no_relocation_and_a_placed_one_is() -> None:
    """The oracle walks the body to the next start, the tracker fed all the
    way: no relocation. When it cannot and puts the body there, the next task
    opens as after a deployment day's relocation, with the transit report."""
    from core.harnesses.navh.env.types import TaskEnd

    env = TourEnv([WALKED, PLACED])
    bench = TourEsBench(env, split="val_unseen", step_budget=500)
    assert bench.seat(0, persist_map=True).index == 0
    end = bench.after_end(TaskEnd(end_reason="stop", score={}, chain_finished=False), True)
    assert end.next_goal.index == 1 and end.chain_finished is False
    assert end.wakeup is None, "a walked transit: the tracker followed the body"
    end = bench.after_end(TaskEnd(end_reason="stop", score={}, chain_finished=False), True)
    assert end.next_goal.index == 2
    assert end.wakeup == {"episode": 2, "map_carried_over": True, "teleported": True, "slam": "ok",
                          "reloc_error_m": 0.4, "pose_jump_m": 1.1, "lost_ticks": 3, "spin_ticks": 120,
                          "epochs": 1, "level": 0, "explored_m2": 61.0}
    end = bench.after_end(TaskEnd(end_reason="stop", score={}, chain_finished=False), True)
    assert end.chain_finished is True and end.next_goal is None and end.wakeup is None


def test_a_placed_transit_opens_the_next_task_with_the_four_views(tmp_path: Path, scripted) -> None:
    """Through the orchestrator: after the walked transit the next task opens
    as any task does; after the placed one, with the four look-around views
    and the instruction to work out which room this is."""
    from test_navh_orchestrator import _agent_closes_properly, _orch, _Wire

    from core.harnesses.navh import prompts as P
    from core.harnesses.navh.session import prompt_text

    openings: dict = {}

    def agent(wire, prompt):
        openings.setdefault(_Wire.orch.k, prompt)
        return _agent_closes_properly(wire, prompt)

    _Wire.handlers["agent"] = agent
    bench = TourEsBench(TourEnv([WALKED, PLACED]), split="val_unseen", step_budget=500,
                        scene_id="data/scene_datasets/mp3d/abc/abc.glb")
    o = _orch(tmp_path, bench, notes=False)
    o.run()
    assert isinstance(openings[1], str) and P.RELOCATED.strip() not in openings[1]
    assert isinstance(openings[2], list) and sum(b.get("type") == "image" for b in openings[2]) == 4
    assert P.RELOCATED.strip() in prompt_text(openings[2])
    rows = o.rec.rows
    assert [bool(r["wakeup"]) for r in rows] == [False, True, False]
    assert [r["after_wakeup"] for r in rows] == [False, False, True]


# ── the adapter: a benchmark episode starts with an empty memory ─────────


def _ctx(tmp_path: Path, index: int):
    return SimpleNamespace(index=index, run_dir=tmp_path / "run", workdir=tmp_path / "run" / "w",
                           max_turns=1500, step_budget=5000, benchmark="val_unseen", instruments=2)


def test_benchmark_episodes_do_not_share_house_notes(tmp_path: Path) -> None:
    a = NavHarnessAdapter(step_budget_per_task=500, max_turns_per_task=200)
    c0, c1 = a._config(_ctx(tmp_path, 0), 5), a._config(_ctx(tmp_path, 1), 5)
    assert c0.memory.memory_root != c1.memory.memory_root
    assert c0.task.step_budget == 500 and c0.task.max_turns_per_task == 200


def test_a_deployment_shares_the_memory_and_keeps_one_map_per_house(tmp_path: Path) -> None:
    a = NavHarnessAdapter(memory_root=str(tmp_path / "mem"), map_state=str(tmp_path / "maps/{house}.pkl"),
                      day="2026-09-21", chain_ids=[3, 7])
    cfg = a._config(_ctx(tmp_path, 3), 5, house="4ok3usBNeis")
    assert cfg.memory.memory_root == str(tmp_path / "mem") and cfg.memory.day == "2026-09-21"
    assert cfg.memory.map_state.endswith("maps/4ok3usBNeis.pkl") and cfg.task.chain_ids == [3, 7]
    json.dumps(cfg.flat())


def test_the_house_is_named_by_its_scene_before_the_seat() -> None:
    bench = GOATEsBench(_ChainEnv(), split="val_unseen", step_budget=500,
                        scene_id="data/scene_datasets/hm3d/val/00829-4ok3usBNeis/4ok3usBNeis.basis.glb")
    assert bench.house_id == "4ok3usBNeis", "the memory directory is the house's, not the chain's"


# ── the Claude Code seat reads its notes with its own file tools, fenced ──


def test_the_notes_fence_lets_a_reader_into_the_notes_and_nowhere_else(tmp_path: Path) -> None:
    import asyncio

    from core.harnesses.navh.session.notes import NotesFence

    house = tmp_path / "mem" / "house"
    (house / "rooms").mkdir(parents=True)
    (house / "rooms" / "kitchen.md").write_text("fridge\n")
    (tmp_path / "secret.md").write_text("x\n")
    (house / "rooms" / "escape.md").symlink_to(tmp_path / "secret.md")
    f = NotesFence(house)
    ok = [("Read", {"file_path": str(house / "rooms" / "kitchen.md")}),
          ("Read", {"file_path": "rooms/kitchen.md"}),
          ("Read", {"file_path": str(house / "rooms" / "kitchen" / "photos" / "front.png")}),
          ("Glob", {"pattern": "**/*.md"}),
          ("Glob", {"pattern": str(house) + "/rooms/*.md"}),
          ("Grep", {"pattern": "fridge", "glob": "*.md"}),
          ("Grep", {"pattern": "fridge", "path": str(house / "rooms")}),
          ("mcp__env__step", {"actions": [1]}), ("mcp__env__read_file", {"path": "run/ledger.md"})]
    for tool, args in ok:
        assert f.allows(tool, args)[0], (tool, args)
    shut = [("Read", {"file_path": str(tmp_path / "secret.md")}),
            ("Read", {"file_path": "../secret.md"}),
            ("Read", {"file_path": str(house / "rooms" / "escape.md")}),
            ("Glob", {"pattern": "../**/*.md"}),
            ("Glob", {"pattern": "/etc/*"}),
            ("Glob", {"pattern": "*.md", "path": str(tmp_path)}),
            ("Grep", {"pattern": "x", "path": "/"}),
            ("Grep", {"pattern": "x", "glob": "../*"}),
            ("Write", {"file_path": str(house / "rooms" / "kitchen.md"), "content": "y"}),
            ("Edit", {"file_path": str(house / "rooms" / "kitchen.md")}),
            ("Bash", {"command": "ls"}), ("WebFetch", {"url": "http://x"}), ("Task", {})]
    for tool, args in shut:
        assert not f.allows(tool, args)[0], (tool, args)
    denied = asyncio.run(f.gate({"tool_name": "Write", "tool_input": {"file_path": "MEMORY.md"}}, "t1", None))
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert asyncio.run(f.gate({"tool_name": "Read", "tool_input": {"file_path": "rooms/kitchen.md"}},
                              "t2", None)) == {}


def test_the_agents_session_on_claude_code_opens_read_glob_grep_on_the_notes(tmp_path: Path, monkeypatch) -> None:
    from core.harnesses.navh import prompts as P
    from core.harnesses.navh.session.agent import AgentSession
    from core.harnesses.navh.session.base import BaseSession
    from core.harnesses.navh.session.notes import READ_TOOLS
    from core.harnesses.navh.session.wire import home

    monkeypatch.setattr(home, "_applied", {"relocated": False})
    house = tmp_path / "mem" / "house"
    house.mkdir(parents=True)
    agent = AgentSession("claude-opus-5", loop=None, cwd=tmp_path / "workspace", system_prompt=P.SYSTEM,
                         tools=["observe", "read_file"], notes=house)
    o = agent.options()
    assert o.tools == list(READ_TOOLS), "`tools` must list them: allowed_tools alone provides nothing"
    assert o.allowed_tools == ["mcp__env__observe", "mcp__env__read_file", *READ_TOOLS]
    assert o.cwd == str(house.resolve()), "a Glob or Grep without a path searches the notes"
    assert o.permission_mode == "bypassPermissions" and o.hooks["PreToolUse"][0].hooks == [agent.notes.gate]
    assert o.system_prompt.startswith(P.SYSTEM) and str(house.resolve()) in o.system_prompt[len(P.SYSTEM):]
    assert P.SYSTEM.count("\n") + P.NOTES_TOOLS.count("\n") == o.system_prompt.count("\n")
    # the other seats read the notes through the bridge; the other sessions get no built-ins
    codex = AgentSession("gpt-6-astra", loop=None, cwd=tmp_path / "workspace", system_prompt=P.SYSTEM,
                         notes=house, transport="codex-session")
    assert codex.notes is None and codex.cwd == str(tmp_path / "workspace") and codex.system_prompt == P.SYSTEM
    other = BaseSession("claude-opus-5", loop=None, cwd=tmp_path, tools=["read_file"]).options()
    assert other.tools == [] and other.hooks is None and other.allowed_tools == ["mcp__env__read_file"]


# ── the defaults are the paper's reference configuration ────────────────


def test_the_library_defaults_are_the_papers_settings(tmp_path: Path) -> None:
    import inspect

    import yaml

    from core.harnesses.navh.config import HarnessConfig

    cfg = HarnessConfig.from_flat(benchmark="es", chain_id=0)
    assert cfg.task.max_turns_per_task == 200 and cfg.task.step_budget == 500
    assert cfg.ladder.recovery is True and cfg.ladder.reset_after_turns == 80
    assert cfg.ladder.max_resets_per_task == 1 and cfg.ladder.recovery_min_turns == 12
    assert cfg.session.chunk_turns == 25
    adapter = NavHarnessAdapter()._config(_ctx(tmp_path, 0), 6)
    assert adapter.task.max_turns_per_task == 200 and adapter.task.step_budget == 500, \
        "not the episode's 5000 steps, nor its turns split over the sub-goals"
    assert adapter.ladder.recovery and adapter.ladder.reset_after_turns == 80
    assert adapter.ladder.max_resets_per_task == 1
    # every switch the experiment's yaml writes out is the adapter's own default
    defaults = {k: p.default for k, p in inspect.signature(NavHarnessAdapter.__init__).parameters.items()
                if p.default is not inspect.Parameter.empty}
    yml = yaml.safe_load((Path(__file__).resolve().parents[1]
                          / "exp_workspace/lhnav_es/configs/harness/navh.yaml").read_text())
    for key, value in yml.items():
        if key in defaults:
            assert defaults[key] == value or (defaults[key] is None and not value), (key, defaults[key], value)


def test_an_inherited_view_is_not_filed_onto_itself(tmp_path: Path) -> None:
    """A place inherited from an earlier day, whose mark this task banked no
    new views, still names that day's files; filing them again changes nothing."""
    notes = HouseNotes(tmp_path / "house", day="2026-09-19")
    notes.attach_photos("kitchen", _views(tmp_path, "a"), 0)
    later = HouseNotes(tmp_path / "house", day="2026-09-20")
    inherited = later.photo_views()["kitchen"]
    assert later.attach_photos("kitchen", inherited, 3) == 0
    index = json.loads((notes.dir / "rooms" / "kitchen" / "photos.json").read_text())
    assert {v["day"] for v in index} == {"2026-09-19"}
