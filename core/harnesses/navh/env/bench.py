"""Bench — the env over HTTP, and the two benchmarks' task semantics.

The env is an ``auto_host`` server exposing ``/call/<fn>`` functions and a
panel (``/env-panel/field/<name>``, ``/env-panel/action/<name>``). Two
nodesets speak the same verbs with different task semantics:

* ``ivln``  IR2R-CE tour. A task is one R2R episode. STOP ends it; the
            harness then calls ``advance_errand`` (oracle transit to the true
            goal, then to the next start; map / SLAM / markers untouched).
* ``goat``  GOAT-Bench episode. A task is one subtask. STOP scores it and
            the env hands the next goal back in the same call. The body does
            not move at the boundary.

The agent never touches this module: its tools are the MCP bridge
(``bridge/bridge.py``), a process of its own. This is the ORCHESTRATOR's
door to the same env — seating a chain, the STOP that closes a task, the
map for the judge, the look-around of a wake-up, the checkpoints.
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import Any

import requests

from .types import EnvUnavailable, Goal, TaskEnd


class EnvClient:
    def __init__(self, url: str, timeout: int = 900) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._fails = 0                          # consecutive failed calls
        requests.get(f"{self.url}/health", timeout=10).raise_for_status()

    DEAD_AFTER = int(os.environ.get("NAVH_ENV_DEAD_AFTER", "8"))   # consecutive failures = dead

    def call(self, fn: str, inputs: dict | None = None, config: dict | None = None) -> dict:
        body: dict[str, Any] = {"inputs": inputs or {}}
        if config:
            body["config"] = config
        # An env that has lost its simulator answers /health and then 500s or
        # refuses every call. Left alone the agent keeps "working" against a
        # wall and the chain records a screen of zeros that reads exactly like a
        # very bad agent. The bridge guards the agent's own tool calls the same
        # way (bridge/bridge.py); this guards the orchestrator's.
        try:
            r = requests.post(f"{self.url}/call/{fn}", json=body, timeout=self.timeout)
            r.raise_for_status()
            out = r.json()["outputs"]
        except Exception as exc:
            self._fails += 1
            if self._fails >= self.DEAD_AFTER:
                raise EnvUnavailable(f"{self._fails} consecutive failures, last on {fn}: "
                                     f"{type(exc).__name__}: {exc}"[:300]) from exc
            raise
        self._fails = 0
        # nodes may nest their payload under info / result / goal; lift those
        # keys to the top level (never overwriting a top-level key) so every
        # node surface reads the same way
        if isinstance(out, dict):
            for sub in ("info", "result", "goal"):
                inner = out.get(sub)
                if isinstance(inner, dict):
                    for k, v in inner.items():
                        out.setdefault(k, v)
        return out

    #: A panel write is a UI courtesy, not part of the episode. It gets seconds,
    #: not the env timeout: a wedged server must not block `seat` on a display
    #: field and take the chain down with it.
    PANEL_TIMEOUT = 15

    def panel(self, name: str, value: Any) -> None:
        with contextlib.suppress(requests.RequestException):
            requests.post(f"{self.url}/env-panel/field/{name}", json={"value": value},
                          timeout=self.PANEL_TIMEOUT).raise_for_status()

    def action(self, name: str) -> None:
        """`play` arms the episode, so this one must be heard — and it is allowed
        to take its time.

        `play` loads the tour and seeks to the episode, which on a long tour
        takes minutes; a short cap here would fail exactly the longest tours
        and look like a flaky environment. It gets the full env timeout."""
        t0 = time.time()
        requests.post(f"{self.url}/env-panel/action/{name}", json={"params": {}},
                      timeout=self.timeout).raise_for_status()
        dt = time.time() - t0
        if dt > 30:
            print(f"[navh] env action {name!r} took {dt:.0f}s", flush=True)


class Bench:
    """Benchmark adapter: how a chain is seated and how tasks begin/end."""

    NODESET = "env_slam_vlnce"

    def __init__(self, env: EnvClient, *, split: str, instruments: int = 2,
                 step_budget: int = 500, rgb_resolution: str = "512") -> None:
        self.env = env
        self.split = split
        self.instruments = instruments
        self.step_budget = step_budget
        self.rgb_resolution = rgb_resolution

    # the nodeset's node names for the harness's short verbs
    VERB_ALIAS = {"step": "step_discrete", "observe": "observe_egocentric"}

    def fn(self, name: str) -> str:
        return f"{self.NODESET}__{self.VERB_ALIAS.get(name, name)}"

    house_id: str = ""          # the scene this chain lives in; set at seat()

    def _note_house(self, meta: dict, default: str) -> None:
        tour = meta.get("tour") or {}
        hid = (tour.get("tour_scene") or meta.get("scene") or meta.get("scene_id")
               or meta.get("scene_name") or default)
        self.house_id = str(hid)

    # to implement per benchmark
    def seat(self, chain_id: int, persist_map: bool, skip: int = 0) -> Goal: ...
    def parse_step(self, out: dict, claimed: str | None) -> tuple[dict, TaskEnd | None]: ...
    def after_end(self, end: TaskEnd, persist_map: bool) -> TaskEnd: ...
    def chain_report(self) -> dict: ...

    def _meta_to_goal(self, meta: dict) -> Goal:
        raise NotImplementedError


class IVLNBench(Bench):
    """IR2R-CE tour: task = episode; boundary = oracle transit."""

    def __init__(self, env: EnvClient, **kw: Any) -> None:
        super().__init__(env, **kw)
        self.tour_id = -1
        self.slot = 0
        self.total = 0

    def seat(self, chain_id: int, persist_map: bool, skip: int = 0) -> Goal:
        e = self.env
        e.panel("split", self.split)
        e.panel("tour", int(chain_id))
        e.panel("instruments", self.instruments)
        # Resume seats at the slot the ledger stopped on. Without this the env
        # replays the whole tour while the ledger says those tasks are done, so
        # every index is written twice and the walking is redone for nothing.
        e.panel("episode_index", int(skip))
        e.action("play")
        meta = e.call(self.fn("reset"), {"trigger": "harness"},
                      {"rgb_resolution": self.rgb_resolution})
        tour = meta.get("tour") or {}
        self._note_house(meta, f"ivln-tour{int(chain_id)}")
        self.tour_id = int(chain_id)
        self.slot = int(meta.get("tour_slot", 0))
        self.total = int(tour.get("tour_length") or meta.get("tour_length") or 1)
        return self._meta_to_goal(meta)

    def _meta_to_goal(self, meta: dict) -> Goal:
        return Goal(index=int(meta.get("tour_slot", self.slot)), total=self.total,
                    text=str(meta.get("instruction", "")).strip(), modality="instruction",
                    meta={"episode_id": meta.get("episode_id"),
                          "map_carried_over": meta.get("map_carried_over")})

    def parse_step(self, out: dict, claimed: str | None) -> tuple[dict, TaskEnd | None]:
        info = out.get("info") or {}
        done = bool(out.get("terminated")) or bool(out.get("truncated"))
        if not done:
            return info, None
        score = dict(out.get("metrics") or info.get("metrics") or {})
        reason = "stop" if out.get("terminated") else "budget"
        return info, TaskEnd(end_reason=reason, score=score, chain_finished=False)

    def after_end(self, end: TaskEnd, persist_map: bool) -> TaskEnd:
        if persist_map:
            try:
                adv = self.env.call(self.fn("advance_errand"), {"trigger": "harness"})
            except EnvUnavailable:
                raise
            except Exception as e:
                # The transit drives hundreds of micro-steps through SLAM, and this
                # is where ORB-SLAM3 can die when its own loop closure throws
                # (`terminate called ... cv::Exception`). One failed
                # verb is not proof the whole env is gone, but nothing downstream can
                # run without a seated episode, and calling it a failed tour buries an
                # environment death in the scores.
                raise EnvUnavailable(f"advance_errand failed: {type(e).__name__}: {e}"[:300]) from e
            end.score = dict(adv.get("scored") or end.score)
            if adv.get("finished"):
                end.chain_finished = True
                return end
            nxt = adv.get("next") or {}
            self.slot = int(nxt.get("tour_slot", self.slot + 1))
            end.next_goal = self._meta_to_goal(nxt)
            return end
        # episodic: a fresh placement at the next slot, map/SLAM/markers reset.
        # The env must honour {"fresh": true}.
        nxt_slot = self.slot + 1
        if nxt_slot >= self.total:
            end.chain_finished = True
            return end
        self.env.panel("episode_index", nxt_slot)
        meta = self.env.call(self.fn("reset"), {"trigger": "harness", "fresh": True},
                             {"rgb_resolution": self.rgb_resolution})
        # The harness is the authority on which slot comes next. If the env
        # reports a different slot the run must not silently loop; flag it.
        self.slot = nxt_slot
        goal = self._meta_to_goal(meta)
        goal.index = nxt_slot
        if int(meta.get("tour_slot", nxt_slot)) != nxt_slot:
            goal.meta["slot_mismatch"] = {"asked": nxt_slot, "env": meta.get("tour_slot")}
        end.next_goal = goal
        return end

    def chain_report(self) -> dict:
        try:
            return self.env.call(self.fn("tour_report"), {"trigger": "harness"})["report"]
        except Exception:
            return {}


class GOATBench(Bench):
    """GOAT-Bench: task = subtask; boundary = in place, auto-advance.

    A chain is one episode (5-10 subtasks) or SEVERAL episodes of the same
    scene concatenated (``chain_ids=[3, 7]``): when one episode's last
    subtask closes, the next episode is seated in the same scene with the
    map / SLAM / markers kept, and the body placed at the new episode's
    start. That placement is the "wake-up" event: the robot must
    re-localise against its own persistent map. The env reports SLAM
    status and re-anchors, recorded as ``TaskEnd.wakeup``.
    """

    def __init__(self, env: EnvClient, **kw: Any) -> None:
        super().__init__(env, **kw)
        self.sub = 0
        self.total = 0                  # subtasks in the CURRENT episode
        self.episode_index = -1
        self.chain_ids: list[int] = []
        self.chain_pos = 0
        self.offset = 0                 # tasks closed in earlier episodes of the chain
        self.chain_total = 0

    def seat(self, chain_id: int | list[int], persist_map: bool, skip: int = 0) -> Goal:
        self.chain_ids = [int(c) for c in (chain_id if isinstance(chain_id, (list, tuple)) else [chain_id])]
        self.chain_pos = 0
        self.offset = 0
        e = self.env
        e.panel("split", self.split)
        e.panel("instruments", self.instruments)
        e.panel("episode_index", self.chain_ids[0])
        e.action("play")
        meta = e.call(self.fn("reset"), {"trigger": "harness"},
                      {"rgb_resolution": self.rgb_resolution})
        self.episode_index = self.chain_ids[0]
        self._note_house(meta, f"goat-ep{self.chain_ids[0]}")
        self.total = int(meta.get("subtask_total", 1))
        self.sub = int(meta.get("subtask_index", 0))
        # A resumed chain: walk the cursor past what is already done. The skip
        # has to cross EPISODE boundaries, not just sub-task ones — episode 0
        # holds only its own handful of sub-tasks, and skipping past its end
        # would let the env wander into whatever episode comes next in the
        # split, ignoring chain_ids entirely.
        left = max(0, int(skip))
        while left:
            room = self.total - self.sub          # sub-tasks left in this episode
            if left < room:
                for _ in range(left):
                    out = self.env.call(self.fn("skip_subtask"), {})
                    self.sub = int(out.get("subtask_index", self.sub + 1))
                left = 0
            elif self.chain_pos + 1 < len(self.chain_ids):
                left -= room                      # this episode is fully done
                self._seat_next_episode(persist_map)
            else:
                for _ in range(min(left, room)):  # last episode: skip within it
                    out = self.env.call(self.fn("skip_subtask"), {})
                    self.sub = int(out.get("subtask_index", self.sub + 1))
                left = 0
        # chain_total is exact only after every episode is seated; until then
        # assume the same length for the rest (refined at each hop)
        self.chain_total = self.total * len(self.chain_ids)
        return self.current_goal()

    def _seat_next_episode(self, persist_map: bool) -> tuple[Goal, dict]:
        """Seat the next episode of the chain in the same scene; return (goal, wakeup)."""
        self.offset += self.total
        self.chain_pos += 1
        eid = self.chain_ids[self.chain_pos]
        self.env.panel("episode_index", eid)
        meta = self.env.call(self.fn("reset"),
                             {"trigger": "harness", "keep_map": bool(persist_map), "fresh": not persist_map},
                             {"rgb_resolution": self.rgb_resolution})
        self.episode_index = eid
        self.total = int(meta.get("subtask_total", 1))
        self.sub = int(meta.get("subtask_index", 0))
        self.chain_total = self.offset + self.total * (len(self.chain_ids) - self.chain_pos)
        wake = {"episode": eid, "slam": meta.get("slam"), "reanchors": meta.get("reanchors"),
                "map_carried_over": meta.get("map_carried_over", bool(persist_map)),
                "explored_m2": meta.get("explored_m2"), "markers_kept": meta.get("markers_kept")}
        # the nodeset's relocalisation report (tracking status / lost_ticks / spin_ticks)
        wake.update({k: v for k, v in (meta.get("wakeup") or {}).items() if k not in wake})
        return self.current_goal(), wake

    def current_goal(self) -> Goal:
        g = self.env.call(self.fn("get_goal"), {})
        if "error" in g:
            return Goal(index=self.offset + self.sub, total=self.chain_total, text="", meta={"error": g["error"]})
        self.sub = int(g.get("subtask_index", self.sub))
        self.total = int(g.get("subtask_total", self.total))
        self.chain_total = self.offset + self.total * (len(self.chain_ids) - self.chain_pos)
        return Goal(index=self.offset + self.sub, total=self.chain_total, text=str(g.get("text", "")),
                    image_b64=g.get("image"), modality=str(g.get("modality", "object")),
                    meta={"category": g.get("category"), "source_hfov": g.get("source_hfov"),
                          "episode": self.episode_index, "subtask": self.sub})

    def parse_step(self, out: dict, claimed: str | None) -> tuple[dict, TaskEnd | None]:
        info = out.get("info") or {}
        res = info.get("subtask_result")
        if not res:
            return info, None
        end = TaskEnd(end_reason=str(res.get("end_reason", "stop")), score=dict(res),
                        chain_finished=bool(out.get("terminated")))
        if not end.chain_finished and info.get("next_goal"):
            self.sub = int(info["next_goal"].get("subtask_index", self.sub + 1))
        return info, end

    def after_end(self, end: TaskEnd, persist_map: bool) -> TaskEnd:
        if end.chain_finished:
            # this episode is done; more episodes in the chain?
            if self.chain_pos + 1 < len(self.chain_ids):
                end.chain_finished = False
                end.next_goal, end.wakeup = self._seat_next_episode(persist_map)
            return end
        if not persist_map:
            # episodic GOAT: same episode, same subtask cursor, fresh map/SLAM.
            # Requires the env to honour {"fresh": true}.
            self.env.call(self.fn("reset"), {"trigger": "harness", "fresh": True,
                                             "keep_subtask": True},
                          {"rgb_resolution": self.rgb_resolution})
        nxt = self.current_goal()
        if self._exhausted(nxt):
            # The episode's subtask list ran out without the env setting
            # `terminated`. Seating that goal asks the agent to find nothing,
            # scores nothing, and — because the chain never reports itself
            # finished — exits `chain_incomplete`, so a resume cannot tell a
            # finished scene from a cut-short one and replays it forever.
            if self.chain_pos + 1 < len(self.chain_ids):
                end.next_goal, end.wakeup = self._seat_next_episode(persist_map)
            else:
                end.chain_finished = True
                end.next_goal = None
            return end
        end.next_goal = nxt
        return end

    def _exhausted(self, goal: Goal) -> bool:
        """No goal left to pursue: nothing to find, or the cursor is past the
        last subtask. Either alone is enough — an empty-text goal is as
        unanswerable as an index out of range."""
        empty = not (goal.text or "").strip() and not goal.image_b64
        past_end = self.total > 0 and self.sub >= self.total
        return empty or past_end

    def chain_report(self) -> dict:
        return self.env.call(self.fn("evaluate"), {"trigger": "harness"})


def map_floor(out: Any) -> int | None:
    """The floor level a get_map answer carries — top level on the GOAT node,
    under ``info`` on the IVLN node."""
    if not isinstance(out, dict):
        return None
    got = out.get("current_floor")
    if got is None and isinstance(out.get("info"), dict):
        got = out["info"].get("current_floor")
    return None if got is None else int(got)


def make_bench(name: str, env: EnvClient, **kw: Any) -> Bench:
    if name == "ivln":
        return IVLNBench(env, **kw)
    if name == "goat":
        return GOATBench(env, **kw)
    raise ValueError(f"unknown benchmark {name!r} (ivln | goat)")
