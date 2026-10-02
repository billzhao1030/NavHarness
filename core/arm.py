"""Arm — the agent's interface to the world: one exp_workspace folder's code
(prompts.py + mcp/{env.py, bridge.py}) plus its switches and its surface.

An experiment file (exp_workspace/<arm>/configs/<run name>.yaml) declares it inline::

    agent:
      arm:
        _target_: core.arm.Arm
        dir: exp_workspace/lhnav_es
        word: lhnav                 # the {arm} segment of run.name
        bare: true                  # switches: any extra key is one
        instruments: 2
        toolface: twotool
        surface:
          env_prefix: SLAMB
          allowed_tools: [observe, step, get_goal, get_pose, get_map, …]
          env_map: {SERVER_URL: server_url, INSTRUMENTS: instruments, …}

and runner.py instantiates it. Everything the loop asks about the arm — the
bridge path, the environment rendered into the bridge, the briefing
builder, the first prompt, a switch's value — is a method or attribute
here, so the runner and the adapters ask the arm, never a registry.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Any

from core.paths import REPO_ROOT

DEFAULT_TOOLS = ("observe", "step")


class Arm:
    def __init__(
        self,
        dir: Path | str,
        word: str,
        surface: dict[str, Any],
        prompts: str = "prompts.py",
        **switches: Any,
    ) -> None:
        path = Path(dir)
        self.dir = path if path.is_absolute() else (REPO_ROOT / path).resolve()
        self.word = word
        self.surface = dict(surface)
        self.prompts = prompts
        self._switches = dict(switches)
        self._module: Any = None

    def __getattr__(self, name: str) -> Any:
        # a switch reads as an attribute (arm.wp, arm.toolface …); dataclass-
        # style fields and properties resolve normally and never reach this
        switches = self.__dict__.get("_switches")
        if switches is not None and name in switches:
            return switches[name]
        raise AttributeError(name)

    # ── identity ──

    @property
    def name(self) -> str:
        return self.dir.name

    def __repr__(self) -> str:
        return f"Arm({self.name})"

    # ── switches ──

    def switches(self) -> dict[str, Any]:
        """The arm's switches (bare / toolface / instruments / auto_observe
        …) as declared."""
        return dict(self._switches)

    def switch(self, name: str, default: Any = None) -> Any:
        return self._switches.get(name, default)

    @property
    def bare(self) -> bool:
        return bool(self._switches.get("bare", False))

    # ── the surface ──

    @property
    def bridge_path(self) -> Path:
        """The stdio bridge the session talks to: this arm's ``mcp/bridge.py``
        (the MCP tool set next to its env.py), or the file the surface names
        (``surface.bridge``, relative to the folder)."""
        declared = self.surface.get("bridge")
        return (self.dir / declared).resolve() if declared else self.dir / "mcp" / "bridge.py"

    @property
    def needs_wp_server(self) -> bool:
        return bool(self.surface.get("needs_wp_server"))

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        """The tool names the Claude SDK session is opened with (recorded;
        the bridge's registration is the gate) — observe + step unless the
        surface says otherwise."""
        return tuple(self.surface.get("allowed_tools") or DEFAULT_TOOLS)

    def render_env(self, ctx: Any) -> dict[str, str]:
        """Env for the harnesses that spawn the stdio bridge, from the
        surface's ``env_map``: every key the bridge is started with, in
        order, as <env suffix>: <attribute of the episode context, or of
        this arm>. Rendering: bool -> "1"/"0", None or "" -> the key is left
        OUT (so the bridge keeps its own default), anything else -> str()."""
        prefix = self.surface["env_prefix"]
        env: dict[str, str] = {}
        for suffix, attribute in (self.surface.get("env_map") or {}).items():
            if hasattr(ctx, attribute):
                value = getattr(ctx, attribute)
            else:
                value = self._switches.get(attribute)
            rendered = _render_knob(value)
            if rendered:  # "" and None mean "not set"; "0" is a real value
                env[f"{prefix}_{suffix}"] = rendered
        return env

    # ── the briefing (prompts.py) ──

    @property
    def module(self) -> Any:
        """The folder's prompts.py, loaded once by path (no package import,
        no sys.path)."""
        if self._module is None:
            path = self.dir / self.prompts
            ispec = importlib.util.spec_from_file_location(f"_expp_{self.dir.name}", path)
            mod = importlib.util.module_from_spec(ispec)
            ispec.loader.exec_module(mod)
            self._module = mod
        return self._module

    def first_prompt(self, goal: Any, task_module: Any) -> str:
        """The arm's own FIRST_PROMPT when prompts.py defines one, else the
        shape's."""
        declared = getattr(self.module, "FIRST_PROMPT", None)
        return str(declared) if declared else task_module.first_prompt(goal)

    def briefing(self, goal: Any, step_budget: int, offered: dict[str, Any]) -> str:
        """The arm's briefing builder, called with exactly the kwargs its
        signature names. Builders declare only the knobs their surface needs
        (task / face / tilt …), so each receives exactly what it uses."""
        builder = self.module.build_briefing
        params = inspect.signature(builder).parameters
        kwargs = {name: value for name, value in offered.items() if name in params}
        return builder(goal.text, step_budget, **kwargs)

    # ── for config.yaml / summaries ──

    def describe(self) -> dict[str, Any]:
        return {
            "dir": str(self.dir),
            "word": self.word,
            "prompts": self.prompts,
            "switches": dict(self._switches),
            "surface": dict(self.surface),
        }


def _render_knob(value: Any) -> str | None:
    """One knob value as the bridge will read it. None and "" mean the key is
    left out entirely — a bridge reading its own default is the documented
    behaviour for an unset knob, and writing "None" into the env would break
    that quietly."""
    if value is None:
        return None
    if isinstance(value, bool):  # before int: bool IS an int
        return "1" if value else "0"
    return str(value)
