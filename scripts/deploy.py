"""Continuous deployment: ten GOAT-Bench tours per house, over simulated days.

Each house's ten val_unseen episodes are walked in order, two a day for five
days. A day is ONE run and one chain: its first episode is placed by the
runner, the second is placed at its prescribed start with the map kept (the
robot is relocated, not walked). Between days the process ends and the next
day starts from what the house kept:

    <out>/memory/<house>/      the house notes and journal (shared by every day)
    <out>/maps/<house>.pkl     the occupancy map and markers (+ <house>.places.json, the place graph)
    <out>/runs/<run name>/     each day's run: task records, sessions, scores

Every run ends with consolidation, unless ``--no-consolidation``: maps and
task records are then kept across days exactly the same, but no house notes
are ever written.

    python scripts/deploy.py --harness navh --model opus-5 --houses <scene id> [<scene id> …]
    python scripts/deploy.py --harness navh --model opus-5 --all --parallel 4
    python scripts/deploy.py --harness navh --model opus-5 --all --no-consolidation --tag nocons

Episode indices are the runner's (GOAT val_unseen in habitat's seed-100
order), computed from the dataset files here; ``--dry-run`` prints the plan.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXPERIMENT = "std_goat_slam_es_lhnav"
SHUFFLE_SEED = 100            # exp_workspace/lhnav_es/mcp/env.py HABITAT_SHUFFLE_SEED (GOAT is a shuffled line)


def goat_episode_scenes(split: str = "val_unseen") -> list[str]:
    """The scene of every runner episode index, in the runner's order."""
    root = Path(os.environ.get("EMBODIEDSCORE_DATA_ROOT") or REPO / "data" / "embodiedscore" / "datasets")
    content = root / "goat_bench" / "hm3d" / "v1" / split / "content"
    if not content.is_dir():
        sys.exit(f"[deploy] no GOAT-Bench data at {content} (set EMBODIEDSCORE_DATA_ROOT)")
    scenes: list[str] = []
    for name in sorted(p for p in os.listdir(content) if p.endswith(".json.gz") and not p.startswith("._")):
        with gzip.open(content / name, "rt", encoding="utf-8") as fh:
            for ep in json.load(fh)["episodes"]:
                scenes.append(Path(str(ep["scene_id"])).name.split(".")[0])
    order = list(range(len(scenes)))
    rng = random.Random()
    rng.seed(SHUFFLE_SEED)
    rng.shuffle(order)
    return [scenes[i] for i in order]


def plan(houses: list[str] | None, per_day: int, days: int) -> dict[str, list[list[int]]]:
    """house -> its days, each a list of runner episode indices."""
    by_house: dict[str, list[int]] = {}
    for idx, scene in enumerate(goat_episode_scenes()):
        by_house.setdefault(scene, []).append(idx)
    if houses:
        missing = [h for h in houses if h not in by_house]
        if missing:
            sys.exit(f"[deploy] not a GOAT val_unseen scene: {', '.join(missing)}")
        by_house = {h: by_house[h] for h in houses}
    out = {}
    for house, eps in sorted(by_house.items()):
        eps = eps[: per_day * days]
        out[house] = [eps[d * per_day:(d + 1) * per_day] for d in range(days) if eps[d * per_day:(d + 1) * per_day]]
    return out


def day_command(a: argparse.Namespace, house: str, d: int, eps: list[int], port: int) -> list[str]:
    root = Path(a.out).resolve()
    date = (dt.date.fromisoformat(a.start_date) + dt.timedelta(days=d)).isoformat()
    name = f"deploy_{a.tag}_{house}_d{d + 1}"
    return [sys.executable, str(REPO / "runner.py"), EXPERIMENT,
            f"harness={a.harness}", f"model={a.model}",
            f"run.episodes={eps[0]}",
            f"harness.chain_ids=[{','.join(map(str, eps))}]",
            f"harness.memory_root={root / 'memory'}",
            f"harness.map_state={root / 'maps' / (house + '.pkl')}",
            f"harness.day={date}",
            f"harness.notes={'false' if a.no_consolidation else 'true'}",
            f"run.name={name}", f"run.dir={root / 'runs' / name}",
            f"env.port={port}", *a.overrides]


def run_house(a: argparse.Namespace, house: str, days: list[list[int]], port: int) -> int:
    """A house's days, one after another (each starts from what the last left)."""
    for d, eps in enumerate(days):
        cmd = day_command(a, house, d, eps, port)
        print(f"[deploy] {house} day {d + 1}: episodes {eps}", flush=True)
        if a.dry_run:
            print("  " + " ".join(cmd))
            continue
        rc = subprocess.call(cmd, cwd=REPO)
        if rc != 0:
            print(f"[deploy] {house} day {d + 1} failed (exit {rc}); the house stops here", flush=True)
            return rc
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harness", default="navh", choices=["navh", "navhcodex", "navhmini"])
    ap.add_argument("--model", required=True)
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--houses", nargs="+", help="scene ids, e.g. 4ok3usBNeis")
    who.add_argument("--all", action="store_true", help="all 36 val_unseen houses")
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--per-day", type=int, default=2, help="tours (episodes) per day")
    ap.add_argument("--start-date", default=dt.date.today().isoformat(), help="the first simulated day (ISO date)")
    ap.add_argument("--no-consolidation", action="store_true",
                    help="keep maps and task records across days, but write no house notes")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--out", default=str(REPO / "outputs" / "deploy"), help="where memory/, maps/ and runs/ live")
    ap.add_argument("--parallel", type=int, default=1, help="houses at once (one env server each)")
    ap.add_argument("--port", type=int, default=9400, help="the first env-server port")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("overrides", nargs="*", help="extra runner overrides, e.g. effort=high")
    a = ap.parse_args()
    if a.no_consolidation and a.tag == "full":
        a.tag = "nocons"
    a.out = str(Path(a.out) / a.tag)
    schedule = plan(None if a.all else a.houses, a.per_day, a.days)
    print(f"[deploy] {len(schedule)} house(s) x {a.days} day(s) x {a.per_day} tour(s); out {a.out}")
    with ThreadPoolExecutor(max_workers=max(1, a.parallel)) as pool:
        futs = [pool.submit(run_house, a, h, days, a.port + i)
                for i, (h, days) in enumerate(schedule.items())]
        failed = sum(1 for f in futs if f.result() != 0)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
