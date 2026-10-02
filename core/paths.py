"""Repository paths shared by the runner and the recorders.

Nothing here composes a config: the four-layer experiment config
(exp_workspace/<arm>/configs/<run name>.yaml — task · env · agent · run) is loaded by Hydra in
runner.py; this module only knows where the tree lives.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_WORKSPACE = REPO_ROOT / "exp_workspace"
OUTPUTS = REPO_ROOT / "outputs"

# harness name root (agent.harness.root) → output root. ``run.dir`` in every
# experiment file is ``outputs/<root dir>/${run.name}`` and must agree with
# this table.
OUTPUT_ROOTS = {
    # NavHarness — one harness, three reasoning cores. They share an output
    # root because they share everything above the wire; the core's name is
    # in the run name.
    "navh": OUTPUTS / "navharness",          # Claude Code holds the session
    "navhcodex": OUTPUTS / "navharness",     # the Codex CLI holds it
    "navhmini": OUTPUTS / "navharness",      # mini-swe-agent is the core
}
