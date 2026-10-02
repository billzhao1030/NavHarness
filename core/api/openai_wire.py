"""openai-chat wire endpoint — a foreign model behind codex's native wire.

An owned ``litellm --config`` subprocess bound to loopback; each Route maps
the alias the harness asks for to the litellm slug that actually serves it,
with the vendor key referenced as ``os.environ/<VAR>`` so key material
never touches the config file on disk. Vendor knowledge (which variable)
is resolved ABOVE this package and arrives inside the Route.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from .service import LOG_DIR, MASTER_KEY, LoopbackService

CODEX_KEY_ENV = "AC_GATEWAY_KEY"  # codex resolves its provider key from env


@dataclass(frozen=True)
class Route:
    alias: str            # model name the harness asks for
    litellm_model: str    # litellm slug the gateway routes to
    key_env: str          # vendor key variable, referenced as os.environ/<VAR>
    api_base: str | None = None


class OpenAIWireGateway(LoopbackService):
    PORT_POOL = tuple(range(4100, 4110))

    def __init__(self, routes: list[Route], tag: str = "gateway") -> None:
        super().__init__(tag)
        self.routes = routes

    def _config_yaml(self) -> str:
        lines = ["model_list:"]
        for r in self.routes:
            lines += [
                f"  - model_name: {r.alias}",
                "    litellm_params:",
                f"      model: {r.litellm_model}",
                f"      api_key: os.environ/{r.key_env}",
            ]
            if r.api_base:
                lines.append(f"      api_base: {r.api_base}")
        lines += [
            "litellm_settings:",
            "  drop_params: true",
            "general_settings:",
            f"  master_key: {MASTER_KEY}",
        ]
        return "\n".join(lines) + "\n"

    def command(self) -> list[str]:
        cfg = LOG_DIR / f"{self.tag}_{self.port}.yaml"
        cfg.write_text(self._config_yaml())
        os.environ[CODEX_KEY_ENV] = MASTER_KEY
        # litellm has no __main__ — the proxy entry is a console script. The
        # gateway may run from its OWN env (ac-litellm-gw, a recent litellm)
        # so the main env's litellm pin is left alone. Override with
        # $AC_GATEWAY_LITELLM; fall back to this env.
        gw_env_bin = Path.home() / "miniforge3" / "envs" / "ac-litellm-gw" / "bin" / "litellm"
        litellm_bin = os.environ.get("AC_GATEWAY_LITELLM") or (
            str(gw_env_bin) if gw_env_bin.exists()
            else str(Path(sys.executable).parent / "litellm"))
        return [litellm_bin, "--config", str(cfg),
                "--host", "127.0.0.1", "--port", str(self.port)]

    # ── harness-facing projections ──

    def anthropic_env(self) -> dict[str, str]:
        """Env for the SDK harness: Claude Code speaks anthropic-messages to us."""
        return {"ANTHROPIC_BASE_URL": self.base_url,
                "ANTHROPIC_AUTH_TOKEN": MASTER_KEY}

    def codex_args(self, provider_id: str = "acgw") -> list[str]:
        """-c overrides for codex exec: a custom openai-responses provider."""
        return codex_provider_args(self.base_url, provider_id, "NavHarness gateway")

    def describe(self) -> dict:
        """Recorded into harness_inherent — the run must say it was gatewayed."""
        return {"backend": "litellm-gateway", "auth": "litellm gateway (vendor API billing)",
                "port": self.port, "routes": {r.alias: r.litellm_model for r in self.routes}}


def codex_provider_args(base_url: str, provider_id: str, name: str) -> list[str]:
    """-c overrides that make a loopback endpoint codex's model provider (its
    key read from ``CODEX_KEY_ENV``). codex 0.153+ refuses ``wire_api =
    "chat"`` — a custom provider serves ``/v1/responses`` (the litellm proxy
    does; verified for the fake endpoint, see fake_wire)."""
    return [
        "-c", f'model_providers.{provider_id}.name = "{name}"',
        "-c", f'model_providers.{provider_id}.base_url = "{base_url}/v1"',
        "-c", f'model_providers.{provider_id}.env_key = "{CODEX_KEY_ENV}"',
        "-c", f'model_providers.{provider_id}.wire_api = "responses"',
        "-c", f'model_provider = "{provider_id}"',
    ]
