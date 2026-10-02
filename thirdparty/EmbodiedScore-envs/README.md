# EmbodiedScore-envs (NavHarness subset)

The habitat navigation lines of EmbodiedScore-envs (MIT, version 0.2.0) used by NavHarness: **GOAT-Bench**, **IR2R-CE** (IVLN-CE tours)
and **VLN-CE R2R**. They run behind the Gymnasium 1.3 API on one frozen
[habitat-sim 0.3.3](https://github.com/Embodied-Agent-Squad/EmbodiedScore-habitat), with no habitat-lab.

This is a trimmed copy. The Isaac (VLNverse), LIBERO, RoboTwin, RoboCasa, CALVIN and BEHAVIOR engines are gone,
and so are the ObjectNav / OVON, HM-EQA / MT-HM3D and EXPRESS lines. The code that remains is the upstream code,
with the references to the dropped lines removed. The `vlnce` module still declares RxR-CE (`vlnce-rxr`) next to
R2R; NavHarness does not use it.

| line (`make` name) | splits | actions (standard) | metrics |
|---|---|---|---|
| `vlnce-r2r` (`vlnce-rxr`) | train / val_seen / val_unseen / test | 0–5 | VLN-CE: distance_to_goal, success (3.0 m), spl, ndtw, path_length, oracle_success, steps_taken |
| `ivlnce` | train / val_seen / val_unseen | 0–5 | the VLN-CE metrics per episode, plus per-tour t-nDTW in `info["tour"]` |
| `goat` | val_seen / val_seen_synonyms / val_unseen | 0–6 (+ SUBTASK_STOP) | GOAT-Bench partial / composite success and SPL (0.25 m) |

Every line also has a `<line>-upstream` variant that uses the original evaluator's own rig.

## API

```python
import embodiedscore_envs as es
env = es.make("goat", "val_unseen")                # gym.make + TimeLimit + metric wrapper + DepthClip
obs, info = env.reset(options={"episode": 0})      # or {"episode_id": ...}; the goal is in info, never in obs
obs, r, terminated, truncated, info = env.step(es.Act.FORWARD)   # 0 STOP 1 FORWARD 2 LEFT 3 RIGHT 4/5 LOOK 6 SUBTASK_STOP
info["metrics"], info["position"], info["distance_to_goal"]       # ivlnce adds info["tour"]
from embodiedscore_envs.benchmarks import BENCHMARKS, build_env   # build_env(name, split, body=..., actions=...) = the bare HabitatEnv
from embodiedscore_envs.benchmarks.presets import actions, bodies # bodies.STANDARD: 0.25 m / 15°, tilt 30° (±60°), RGB 512² + depth 256²
base = env.unwrapped; base.world                   # SimWorld: pose / place / move / observe / render_at / camera_pose / geodesic / follower
es.benchmark("ivlnce").metrics                     # the line's metric factory (IVLN-CE: TourWrapper over NavMetrics)
env.close()
```

## Data

`EMBODIEDSCORE_DATA_ROOT` holds the episodes and `EMBODIEDSCORE_SCENE_ROOT` holds the scenes. You can also pass
both to each call: `es.make(name, split, data_root=..., scene_root=...)`.

```
$EMBODIEDSCORE_DATA_ROOT/
├── vlnce/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz      R2R-CE episodes (vlnce-r2r, and ivlnce)
│                                       {split}/{split}_gt.json.gz   dense oracle locations (nDTW)
├── vlnce/RxR_VLNCE_v0/{split}/{split}_guide.json.gz (+ _guide_gt)   RxR-CE (vlnce-rxr only)
├── ivlnce/tours.json      {split: {scene: [[episode_id, ...], ...]}}; tour_id = flat index, scene by scene
├── ivlnce/gt_ndtw.json    {split: {tour_id: [{episode_id, position, phase}, ...]}}: the recorded oracle tour
│                          roll-out that t-nDTW is scored against (137 MB, loaded on first use)
└── goat_bench/hm3d/v1/{split}/{split}.json.gz                       GOAT shell
                          {split}/content/<scene>.json.gz            per-scene shards {episodes, goals}

$EMBODIEDSCORE_SCENE_ROOT/
├── mp3d/<scan>/<scan>.glb (+ .navmesh)          VLN-CE / IVLN-CE; the shipped navmesh is used as-is
└── hm3d/...                                     GOAT: the episode's scene_id and scene_dataset_config
                                                 with the leading data/scene_datasets/ stripped
```

ivlnce reuses the R2R-CE episodes and orders them by tour. Within a tour, the agent is walked between episodes by
a geodesic oracle, so it is never teleported (`TourWrapper`, `benchmarks/ivlnce.py`).

## Install

1. Build habitat-sim 0.3.3 from EmbodiedScore-habitat (`./build.sh --env <env> --verify`).
2. In the same environment, run `pip install -e .`. This installs gymnasium, numpy, numpy-quaternion, scipy and
   fastdtw. For nDTW to match habitat-lab 0.1.7 to the last digit, fastdtw has to be its Cython build:
   `pip install cython && pip install --force-reinstall --no-deps --no-cache-dir --no-binary fastdtw --no-build-isolation fastdtw`.
3. Run `pytest tests`. `test_registry.py` needs nothing else. `test_contracts.py` needs habitat-sim, a GPU and both
   data roots, and it is skipped without them.
