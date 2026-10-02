"""CodeSnapshot — the code a run ran on, under the run dir.

    run_dir/code/             byte copy of the snapshot sources at first launch
                              (core/, runner.py, the agent's arm folder + its configs/)
    run_dir/code_<ts>/        a later launch on CHANGED code gets its own sibling
    run_dir/code_state.json   one entry per launch: timestamp, episodes spec,
                              git anchor (HEAD, branch, dirty files), content
                              hash, and which code*/ dir that launch ran on

A resume whose code is byte-identical to an existing snapshot records the
launch but copies nothing; earlier snapshots are never overwritten. Fires
on_run_start and is FATAL: copy failure aborts the launch (same stance as
the task's `require`); git being unavailable only nulls the anchor —
the physical copy still lands. Also fills summary.provenance with the
content hash and git anchor.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from core.callbacks.base import Callback
from core.paths import REPO_ROOT

_SNAPSHOT_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".mypy_cache")


def _snapshot_sources(arm_dir: Path | None) -> list[Path]:
    ca_root = REPO_ROOT
    sources = [ca_root / "core", ca_root / "runner.py"]  # + the arm folder (its configs/ and mcp/ ride along)
    if arm_dir is not None and arm_dir.is_dir() and arm_dir not in sources:
        sources.append(arm_dir)
    return sources


def _tree_sha256(sources: list[Path]) -> str:
    """Content hash over the snapshot sources (path + bytes, sorted walk)."""
    h = hashlib.sha256()
    for src in sources:
        files = (
            [src]
            if src.is_file()
            else sorted(
                f
                for f in src.rglob("*")
                if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc"
            )
        )
        for f in files:
            h.update(str(f.relative_to(REPO_ROOT)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def _git_anchor() -> dict[str, Any] | None:
    def _run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout

    try:
        return {
            "head": _run("rev-parse", "HEAD").strip(),
            "branch": _run("branch", "--show-current").strip() or "(detached)",
            "dirty": _run("status", "--porcelain", "--", ".").splitlines(),
        }
    except Exception:
        return None


def snapshot_code(
    run_dir: Path, run_name: str, arm_dir: Path | None, episodes_spec: str | None
) -> dict[str, Any]:
    """Snapshot + record the launch; returns the launch entry."""
    sources = _snapshot_sources(arm_dir)
    tree_sha = _tree_sha256(sources)

    state_path = run_dir / "code_state.json"
    state = {"launches": []}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except Exception:
            state = {"launches": [], "note": "prior code_state.json unreadable"}

    existing = next(
        (e["snapshot"] for e in state["launches"] if e.get("code_sha256") == tree_sha), None
    )
    if existing:
        snap_name = existing  # identical code — record the launch, copy nothing
    else:
        snap_name = (
            "code" if not (run_dir / "code").exists() else time.strftime("code_%Y%m%d_%H%M%S")
        )
        snap_dir = run_dir / snap_name
        for src in sources:
            dst = snap_dir / src.relative_to(REPO_ROOT)
            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            else:
                shutil.copytree(src, dst, ignore=_SNAPSHOT_IGNORE)
        try:
            diff = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "diff", "HEAD", "--", "."],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout
            if diff:
                (snap_dir / "git.diff").write_text(diff)
        except Exception:
            pass

    launch = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cell": run_name,
        "episodes": episodes_spec,
        "git": _git_anchor(),
        "code_sha256": tree_sha,
        "snapshot": snap_name,
    }
    state["launches"].append(launch)
    state_path.write_text(json.dumps(state, indent=2))
    print(f"[std] code snapshot -> {snap_name}" + (" (identical, reused)" if existing else ""))
    return launch


class CodeSnapshot(Callback):
    writes = (
        "code/",
        "code_state.json",
        "provenance.code_sha256",
        "provenance.git",
        "provenance.snapshot",
    )
    fatal = True

    def on_run_start(self, run: Any) -> None:
        arm_dir = Path(run.arm.dir) if run.arm is not None else None
        launch = snapshot_code(run.run_dir, run.name, arm_dir, run.episodes_spec)
        run.provenance.update({k: launch[k] for k in ("code_sha256", "git", "snapshot")})
