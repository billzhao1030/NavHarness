"""The two serialisation helpers every recorder shares: ``json_safe`` (SDK
messages / codex events / options objects → JSON, image blobs elided) and
``bridge_tool_schemas`` (the bridge's own tool definitions, introspected
in-process under the session's environment). The env itself is reached
through core.envclient.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

# ── serialisation helpers ──────────────────────────────────────────────────


def json_safe(obj: Any, _depth: int = 0) -> Any:
    """Coerce SDK messages / codex events / options objects into JSON; base64
    image blobs are elided to a marker (frames live in live_*/)."""
    if _depth > 12:
        return "<max-depth>"
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {"_type": type(obj).__name__}
        for f in dataclasses.fields(obj):
            out[f.name] = json_safe(getattr(obj, f.name), _depth + 1)
        return out
    if isinstance(obj, dict):
        return {str(k): json_safe(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x, _depth + 1) for x in obj]
    if isinstance(obj, bytes):
        return f"<bytes {len(obj)}>"
    if isinstance(obj, str):
        # long, space-free string = base64 blob → elide; prose keeps spaces
        if len(obj) > 4000 and " " not in obj[:200]:
            return f"<blob {len(obj)} chars elided>"
        return obj
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    return str(obj)


_TOOL_SCHEMAS_CACHE: dict[tuple, Any] = {}


async def bridge_tool_schemas(bridge_path: Path, env: dict[str, str]) -> Any:
    """The bridge's own tool definitions, introspected in-process from the
    module the sessions actually talk to, under the SAME environment the
    session gives it (EpisodeContext.bridge_env). A knob can gate tool
    registration, so the context's own path and env are used as-is — the
    recorded toolset is the one the agent had.

    Cached per (path, env); never raises - logging must not break a run, and
    a bridge that refuses to import (e.g. one that exits without its
    hardware host) must not take the episode with it."""
    key = (str(bridge_path), tuple(sorted(env.items())))
    if key in _TOOL_SCHEMAS_CACHE:
        return _TOOL_SCHEMAS_CACHE[key]
    saved = {name: os.environ.get(name) for name in env}
    try:
        os.environ.update(env)
        spec = importlib.util.spec_from_file_location("_bridge_introspect", bridge_path)
        mod = importlib.util.module_from_spec(spec)
        # A real session spawns the bridge as a SCRIPT, so its own folder is
        # sys.path[0] and a sibling import resolves. Importing it by path
        # here does not, so mirror the spawn or
        # the whole surface is recorded as an import error.
        sys.path.insert(0, str(Path(bridge_path).parent))
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.path.pop(0)
        tools = await mod.mcp.list_tools()
        _TOOL_SCHEMAS_CACHE[key] = json_safe([
            {"name": getattr(t, "name", None),
             "description": getattr(t, "description", None),
             "input_schema": getattr(t, "inputSchema", None)}
            for t in tools
        ])
    except (Exception, SystemExit) as exc:
        _TOOL_SCHEMAS_CACHE[key] = {"error": f"tool-schema introspection failed: {exc!r}"}
    finally:
        for name, old in saved.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old
    return _TOOL_SCHEMAS_CACHE[key]
