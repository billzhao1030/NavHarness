"""GOAT-Bench (Ram81/goat-bench ``Goat-v1`` task on HM3D) — 5-10 ordered
sub-goals per episode in three modalities (object / image / description),
closed one by one with SUBTASK_STOP.

Data (under ``EMBODIEDSCORE_DATA_ROOT/goat_bench/hm3d/v1``): ``{split}/{split}.json.gz``
shell + ``content/<scene>.json.gz`` shards, each ``{episodes, goals}``. An
episode is a start pose and ``tasks = [(category, modality, instance_id[, image_index]), ...]``.
The goal binding replicates the upstream loader literally (it mutates the
shard's shared goal lists in place while attaching child categories, so the
same category's list grows as tasks are processed — duplicates are harmless
for a nearest-view-point distance but are kept so the point sets are exactly
upstream's): ``object`` -> every instance of the category (+ children),
``description`` / ``image`` -> the named instance. Descriptions over 55 words
are dropped (0 on val). ``episode_id`` is re-numbered ``str(i)`` per shard.

Variants: ``goat`` (default) runs on the shared STANDARD body; ``goat-upstream``
on goat-bench's ``goat_stretch_hm3d.yaml`` Stretch rig (``presets.bodies.STRETCH``:
0.25 m, turn 30°, tilt 30° unlimited, agent 1.41 / 0.17, RGB 360x640 portrait
hfov 42 at 1.31 m, no depth sensor, sliding off, navmesh recomputed with climb
0.1 / cell 0.05 — the released experiment configs, not the dataclass default).
Both: 5000 steps per episode and none per sub-goal, actions 0-6, metrics =
SequenceNavMetrics(0.25), distance_to_goal re-queried every step.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

from .env import (Benchmark, Episode, GoalSequence, ImageGoal, ObjectGoal, ObjectInstance, SceneRef,
                  SequenceNavMetrics, TextGoal, data_root, scene_root)
from .presets import actions, bodies, depth

SUCCESS_DISTANCE = 0.25
_SCENE_PREFIX = "data/scene_datasets/"
_MAX_DESC_WORDS = 55


def _load_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _instance(g: dict) -> ObjectInstance:
    return ObjectInstance(object_id=str(g["object_id"]), category=str(g["object_category"]),
                          position=tuple(float(x) for x in g["position"]),
                          view_points=tuple(tuple(float(x) for x in vp["agent_state"]["position"]) for vp in g.get("view_points", [])))


def _scene_ref(scene_id: str, dataset_config: str | None, sroot: Path) -> SceneRef:
    rel = scene_id[len(_SCENE_PREFIX):] if scene_id.startswith(_SCENE_PREFIX) else scene_id
    rel = rel.replace("//", "/")
    cfg = None
    if dataset_config and dataset_config != "default":
        c = dataset_config[len("./"):] if dataset_config.startswith("./") else dataset_config
        c = c[len(_SCENE_PREFIX):] if c.startswith(_SCENE_PREFIX) else c
        cfg = str(sroot / c)
    return SceneRef(scene_file=str(sroot / rel), dataset_config=cfg)


def _bind_goals(goals: dict, scene_id: str, tasks: list) -> list[tuple[list[dict], list]]:
    """Upstream GoatDatasetV1.from_json's binding, on the shard's live goal dicts."""
    def same_cat(cat: str) -> list[list[dict]]:
        return [x for x in goals.values() if x[0]["object_category"] == cat]

    filtered = []
    for task in tasks:
        cat, kind, inst = task[0], task[1], task[2]
        if kind == "description":
            cands = same_cat(cat)
            inst_goal = [x for x in cands[0] if x["object_id"] == inst]
            if len(inst_goal[0]["lang_desc"].split(" ")) <= _MAX_DESC_WORDS:
                filtered.append(task)
        else:
            filtered.append(task)
    bound: list[tuple[list[dict], list]] = []
    for task in filtered:
        cat, kind, inst = task[0], task[1], task[2]
        cands = same_cat(cat)
        for child in cands[0][0]["children_object_categories"]:
            key = f"{scene_id.split('/')[-1]}_{child}"
            if key in goals:
                cands[0].extend(goals[key])      # in-place, as upstream
        assert len(cands) == 1, f"more than 1 goal categories for {cat}"
        if kind == "object":
            bound.append((cands[0], task))
        else:
            bound.append(([x for x in cands[0] if x["object_id"] == inst], task))
    return bound


def _sub_goal(goal_dicts: list[dict], task: list):
    kind = task[1]
    if kind == "object":
        return ObjectGoal(category=str(task[0]), instances=tuple(_instance(g) for g in goal_dicts))
    g = goal_dicts[0]
    if kind == "description":
        return TextGoal(instance=_instance(g), description=str(g.get("lang_desc", "") or ""))
    if kind == "image":
        idx = int(task[3]) if len(task) > 3 and task[3] is not None else 0
        ig = g["image_goals"][idx]
        return ImageGoal(instance=_instance(g), camera_position=tuple(float(x) for x in ig["position"]),
                         camera_rotation=tuple(float(x) for x in ig["rotation"]), hfov_deg=float(ig["hfov"]),
                         image_size=(int(ig["image_dimensions"][0]), int(ig["image_dimensions"][1])))
    raise ValueError(f"unknown GOAT modality {kind!r}")


def _episodes_from(raw: dict, sroot: Path, start_index: int) -> list[Episode]:
    goals = raw["goals"]
    out: list[Episode] = []
    for i, e in enumerate(raw["episodes"]):
        scene_id = e["scene_id"]
        bound = _bind_goals(goals, scene_id, e["tasks"])
        seq = GoalSequence(goals=tuple(_sub_goal(gd, task) for gd, task in bound))
        out.append(Episode(
            index=start_index + len(out), episode_id=str(i),
            scene=_scene_ref(scene_id, e.get("scene_dataset_config"), sroot),
            start_position=tuple(float(x) for x in e["start_position"]),
            start_rotation=tuple(float(x) for x in e["start_rotation"]),
            goal=seq,
            info={"scene_id": scene_id, "tasks": e["tasks"], "n_filtered_tasks": len(e["tasks"]) - len(bound)},
        ))
    return out


def load_episodes(split: str, data_root_=None, scene_root_=None) -> list[Episode]:
    base = data_root(data_root_) / "goat_bench" / "hm3d" / "v1" / split
    sroot = scene_root(scene_root_)
    shell = base / f"{split}.json.gz"
    raw = _load_json_gz(shell)
    content = base / "content"
    out: list[Episode] = []
    if content.is_dir():
        for name in sorted(p for p in os.listdir(content) if p.endswith(".json.gz") and not p.startswith("._")):
            out.extend(_episodes_from(_load_json_gz(content / name), sroot, len(out)))
    elif raw.get("episodes"):
        out.extend(_episodes_from(raw, sroot, 0))
    return out


def _decl(upstream: bool) -> Benchmark:
    return Benchmark(
        name="goat-upstream" if upstream else "goat",
        gym_id="EmbodiedScore/GOAT-Upstream-v0" if upstream else "EmbodiedScore/GOAT-v0",
        body=bodies.STRETCH if upstream else bodies.STANDARD, actions=actions.GOAT,
        splits=("val_seen", "val_seen_synonyms", "val_unseen"),
        episodes=lambda split, data_root=None, scene_root=None, **kw: load_episodes(split, data_root, scene_root),
        metrics=lambda env, **o: SequenceNavMetrics(env, success_distance=o.get("success_distance", SUCCESS_DISTANCE)),
        depth=None if upstream else depth.STANDARD, max_episode_steps=5000, dtg_policy="every_step",
        variant="upstream" if upstream else "standard",
        description="GOAT-Bench " + ("on the Stretch rig (navmesh climb 0.1 / cell 0.05)" if upstream else "on the STANDARD body"),
    )


BENCHMARKS = (_decl(False), _decl(True))
