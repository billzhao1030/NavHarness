"""Where the CLI keeps the session transcripts a run generates.

THE PROBLEM. Every Claude Agent SDK session the harness opens is filed by the
CLI under ``$CLAUDE_CONFIG_DIR/projects/<slug of the session's cwd>/`` as one
jsonl holding the entire conversation, the base64 images included — megabytes
per task, and each run has its own ``cwd``. A benchmark run is therefore one
directory per scene and many gigabytes, landing inside the user's own
``~/.claude`` beside their interactive sessions and their memory files.

Nothing in this repo reads them. What the project scores, views and analyses is
``<out_dir>/episode_N.jsonl`` (thinking, tool calls, tool results, metrics) with
the frames the model actually saw in ``<out_dir>/live_N/``. The only thing
unique to the CLI transcript is the base64 payload of those same frames.

THE FIX. ``CLAUDE_CONFIG_DIR`` relocates the whole config home, transcripts
included: with it set the session lands in the new tree and nothing is written
to ``~/.claude/projects``. Auth is the catch, because
``.credentials.json`` lives in that home too; it is symlinked through and the
link is CHECKED, since a run that silently lost its login would fail every task.
If it cannot be resolved the relocation is abandoned rather than risked.

Settings are deliberately not linked: the sessions already pass ``setting_sources=[]``
so a session never inherits the user's config, and a bare config home makes that
structural instead of a flag.

RETENTION is the other half, and the reason this does more than move the pile.
The CLI's own ``cleanupPeriodDays`` never fires for these sessions — it is read
from the user settings source, which ``setting_sources=[]`` disables — so the
old project directories are pruned here instead.

    NAVH_CLAUDE_CONFIG_DIR    where to put it   (default <repo>/outputs/.claude-sessions)
    NAVH_SESSION_KEEP_DAYS    prune transcripts older than this (default 3; 0 = keep all)
    NAVH_KEEP_HOME_SESSIONS   =1 to switch all of this off and go back to ~/.claude
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

DEFAULT_KEEP_DAYS = 3


def _off(name: str) -> bool:
    return os.environ.get(name, "") not in ("", "0", "false", "False")


def _repo_root() -> Path:
    """The checkout root (the directory holding ``runner.py``). The seat
    passes its own target to ``ensure_relocated``, so this is only the
    fallback."""
    here = Path(__file__).resolve()
    for d in here.parents:
        if (d / "runner.py").is_file():
            return d
    return here.parents[5]


def relocate_claude_sessions(where: Path | str | None = None) -> dict:
    """Point the CLI at its own config home for this process, and prune it.

    Call once, early, in any process that will open an SDK session: the SDK
    passes ``os.environ`` through to the CLI it spawns, so setting the variable
    here covers every session the process goes on to open. Returns what it did,
    so the caller can print one honest line about it.
    """
    if _off("NAVH_KEEP_HOME_SESSIONS"):
        return {"relocated": False, "reason": "NAVH_KEEP_HOME_SESSIONS set"}
    if os.environ.get("CLAUDE_CONFIG_DIR"):
        return {"relocated": False, "reason": "CLAUDE_CONFIG_DIR already set",
                "dir": os.environ["CLAUDE_CONFIG_DIR"]}

    target = Path(os.environ.get("NAVH_CLAUDE_CONFIG_DIR") or where
                  or (_repo_root() / "outputs" / ".claude-sessions"))
    home_creds = Path(os.environ.get("HOME", "~")).expanduser() / ".claude" / ".credentials.json"
    try:
        target = target.resolve()
        (target / "projects").mkdir(parents=True, exist_ok=True)
        link = target / ".credentials.json"
        if home_creds.exists():
            # Re-made every run: a token refresh can replace the link with a
            # regular file, and a stale copy would fail every session of a long run.
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(home_creds)
            if not link.resolve().is_file():
                raise OSError("credential symlink does not resolve to a file")
        elif not link.exists():
            return {"relocated": False, "reason": f"no credentials at {home_creds}"}
    except OSError as e:
        return {"relocated": False, "reason": f"{type(e).__name__}: {e}"}

    os.environ["CLAUDE_CONFIG_DIR"] = str(target)
    return {"relocated": True, "dir": str(target), **prune_sessions(target)}


def prune_sessions(target: Path | str, keep_days: float | None = None) -> dict:
    """Drop transcript directories older than the retention window.

    Age is the directory's own mtime, which the CLI touches on every write, so a
    run in flight — its directories are minutes old — is never a candidate no
    matter how many processes share this home."""
    if keep_days is None:
        keep_days = float(os.environ.get("NAVH_SESSION_KEEP_DAYS", DEFAULT_KEEP_DAYS))
    projects = Path(target) / "projects"
    if keep_days <= 0 or not projects.is_dir():
        return {"pruned": 0, "freed_mb": 0.0}
    cutoff = time.time() - keep_days * 86400
    freed, n = 0, 0
    for d in projects.iterdir():
        try:
            if not d.is_dir() or d.stat().st_mtime >= cutoff:
                continue
            freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            shutil.rmtree(d)
            n += 1
        except OSError:                       # a sibling process got there first
            continue
    return {"pruned": n, "freed_mb": round(freed / 1e6, 1)}


_applied: dict | None = None


def ensure_relocated(where: Path | str | None = None) -> dict:
    """Idempotent ``relocate_claude_sessions``: does the work on the first call
    of a process and returns the same answer to every call after it.

    Call this wherever ``ClaudeAgentOptions`` is built rather than in each entry
    script — a run can be launched from a queue, a notebook or a backend as
    well as from the command line, and the option constructor is the one place
    all of those paths share."""
    global _applied
    if _applied is None:
        _applied = relocate_claude_sessions(where)
    return _applied
