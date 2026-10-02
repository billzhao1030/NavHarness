"""VLN-CE: R2R-CE and RxR-CE (guide role) on the std body.

Data (under ``EMBODIEDSCORE_DATA_ROOT/vlnce``):
    R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz        episodes
    R2R_VLNCE_v1-3_preprocessed/{split}/{split}_gt.json.gz     dense oracle locations (nDTW)
    RxR_VLNCE_v0/{split}/{split}_guide.json.gz  (+ _guide_gt)   RxR, filtered by instruction language
Scenes: ``EMBODIEDSCORE_SCENE_ROOT/<scene_id>`` with scene_id = ``mp3d/<scan>/<scan>.glb``;
the shipped ``.navmesh`` next to it is used as-is (habitat-lab 0.1.7 behaviour).

Variants: ``vlnce-r2r`` / ``vlnce-rxr`` run on the shared STANDARD body
(``presets.bodies.STANDARD``); ``-upstream`` runs each dataset's own evaluator
rig — R2R on VLN-CE ``vlnce_task.yaml`` (RGB 224², 15°, four actions), RxR on
``rxr_vlnce_english_task.yaml`` (the LoCoBot rig with LOOK actions, depth
0.5–5 m). Success distance 3.0 m and VLN_KEYS in both.
Metrics = the seven VLN-CE metrics with success_distance 3.0.
"""

from __future__ import annotations

import gzip
import json
import os
from typing import Iterable

from .env import VLN_KEYS, Benchmark, Episode, NavMetrics, PointGoal, SceneRef, data_root, scene_root
from .presets import actions, bodies, depth

_R2R_DIR = "R2R_VLNCE_v1-3_preprocessed"
_RXR_DIR = "RxR_VLNCE_v0"
SUCCESS_DISTANCE = 3.0



def _files(dataset: str, split: str, root) -> tuple[str, str]:
    base = data_root(root) / "vlnce"
    if dataset == "r2r":
        d = base / _R2R_DIR / split
        return str(d / f"{split}.json.gz"), str(d / f"{split}_gt.json.gz")
    d = base / _RXR_DIR / split
    return str(d / f"{split}_guide.json.gz"), str(d / f"{split}_guide_gt.json.gz")


def _load_json_gz(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_episodes(dataset: str, split: str, data_root_: str | os.PathLike | None = None,
                  scene_root_: str | os.PathLike | None = None,
                  languages: Iterable[str] | None = ("en-US", "en-IN"), gt: bool = True) -> list[Episode]:
    """Episodes in file order (the order episode indices refer to). ``languages``
    filters RxR by ``instruction.language``; ignored for R2R. ``gt`` attaches
    the dense oracle path used by nDTW."""
    ep_file, gt_file = _files(dataset, split, data_root_)
    raw = _load_json_gz(ep_file)["episodes"]
    gt_locs = {str(k): v["locations"] for k, v in _load_json_gz(gt_file).items()} if gt and os.path.isfile(gt_file) else {}
    keep = set(languages) if (dataset == "rxr" and languages) else None
    sroot = scene_root(scene_root_)
    out: list[Episode] = []
    for e in raw:
        ins = e["instruction"]
        lang = ins.get("language")
        if keep is not None and lang not in keep:
            continue
        goal = e["goals"][0]
        eid = str(e["episode_id"])
        info = dict(e.get("info") or {})
        info.update({"trajectory_id": str(e.get("trajectory_id", "")), "language": lang,
                     "instruction_id": str(ins["instruction_id"]) if "instruction_id" in ins else None,
                     "scene_id": e["scene_id"]})
        out.append(Episode(
            index=len(out), episode_id=eid,
            scene=SceneRef(scene_file=str(sroot / e["scene_id"])),
            start_position=tuple(float(x) for x in e["start_position"]),
            start_rotation=tuple(float(x) for x in e["start_rotation"]),
            goal=PointGoal(tuple(float(x) for x in goal["position"]), float(goal.get("radius") or SUCCESS_DISTANCE)),
            instruction=ins["instruction_text"],
            reference_path=tuple(tuple(float(x) for x in p) for p in e.get("reference_path", [])),
            gt_path=tuple(tuple(float(x) for x in p) for p in gt_locs[eid]) if eid in gt_locs else None,
            info=info,
        ))
    return out


# (body, actions, depth) per dataset for the upstream variant
_UPSTREAM = {"r2r": (bodies.VLNCE, actions.NAV, depth.STANDARD),
             "rxr": (bodies.RXR_CE, actions.NAV_LOOK, depth.LOCOBOT)}


def _decl(dataset: str, gym_id: str, splits: tuple[str, ...], upstream: bool) -> Benchmark:
    body, acts, dspec = _UPSTREAM[dataset] if upstream else (bodies.STANDARD, actions.STANDARD, depth.STANDARD)
    return Benchmark(
        name=f"vlnce-{dataset}" + ("-upstream" if upstream else ""),
        gym_id=gym_id.replace("-v0", "-Upstream-v0") if upstream else gym_id,
        body=body, actions=acts, splits=splits,
        episodes=lambda split, data_root=None, scene_root=None, **kw: load_episodes(dataset, split, data_root, scene_root, **kw),
        metrics=lambda env, **o: NavMetrics(env, success_distance=o.get("success_distance", SUCCESS_DISTANCE), keys=VLN_KEYS),
        depth=dspec, max_episode_steps=500, variant="upstream" if upstream else "standard",
        description=f"VLN-CE {dataset.upper()} " + ("on its upstream rig" if upstream else "on the STANDARD body"),
    )


_R2R = ("EmbodiedScore/VLNCE-R2R-v0", ("train", "val_seen", "val_unseen", "test"))
_RXR = ("EmbodiedScore/VLNCE-RxR-v0", ("train", "val_seen", "val_unseen", "test_challenge"))
BENCHMARKS = (
    _decl("r2r", *_R2R, upstream=False), _decl("r2r", *_R2R, upstream=True),
    _decl("rxr", *_RXR, upstream=False), _decl("rxr", *_RXR, upstream=True),
)
