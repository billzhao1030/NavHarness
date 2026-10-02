"""mini-swe-agent's Environment role — a thin session owner over the arm's bridge.

Route parsed tool calls into the toolset (BridgeToolSet: the arm's own MCP
bridge, the same process the SDK and codex seats spawn), and when a call ends
the episode (STOP executed, step budget exhausted) raise ``Submitted`` so the
agent loop exits with the end reason in the trajectory. Episode placement,
reset, and metric collection stay driver-side, exactly like the claude-SDK
path — the agent never sees SR/SPL, reward, pose, or panoramas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minisweagent.exceptions import Submitted


class BridgeEnvironment:
    """mini's Environment role over the arm's own MCP bridge (BridgeToolSet):
    route parsed tool calls into it, raise ``Submitted`` when a result says
    the episode is over. The surface is the arm's, not the harness's."""

    def __init__(self, *, bridge_path: str, env: dict[str, str], server_url: str = "") -> None:
        from bridge_toolset import BridgeToolSet

        self.bridge_path = str(bridge_path)
        self.server_url = server_url
        self.toolset = BridgeToolSet(Path(bridge_path), env)

    def execute(self, action: dict[str, Any], cwd: str = "") -> dict[str, Any]:
        return _execute(self.toolset, action)

    def close(self) -> None:
        self.toolset.close()

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return {"bridge_path": self.bridge_path, "server_url": self.server_url, **kwargs}

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": {"bridge_path": self.bridge_path, "server_url": self.server_url},
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }


def _execute(toolset: Any, action: dict[str, Any]) -> dict[str, Any]:
    """Run one parsed tool call; raise Submitted when the episode ends."""
    result = toolset.execute(action.get("tool", ""), action.get("args") or {})
    output = {"content": result.content, "info": result.info}
    if result.info.get("episode_over"):
        end_reason = result.info.get("end_reason") or "episode_over"
        raise Submitted(
            {
                "role": "exit",
                "content": json.dumps({"end_reason": end_reason, **result.info}),
                "extra": {
                    "exit_status": end_reason,
                    "submission": "",
                    "final_info": result.info,
                },
            }
        )
    return output
