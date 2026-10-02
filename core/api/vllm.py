"""vLLM serving — a local ``vllm serve`` behind the standard lifecycle.

The vLLM endpoint is plain openai-chat: sampling (seed / temperature / top_p)
is honored PER REQUEST, so none of the ollama machinery (Modelfile baking,
/api/show verification) exists here — the model's own params are the pin.

Two ways a hosted_vllm model reaches an endpoint:

- an ALREADY-RUNNING server (an SSH-tunneled cluster endpoint, or one the
  operator started): ``probe()`` checks ``VLLM_URL`` / the model's api_base
  and never launches anything;
- a LOCAL launch through ``VllmServer`` — an owned subprocess on the pool
  ports (not :8000, commonly held by other local services), binary from
  ``$AC_VLLM_BIN`` or the ac-vllm conda env. Loading a big model takes
  minutes, hence the long start timeout.
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.request
from pathlib import Path

from .service import LoopbackService


def vllm_bin() -> str | None:
    env_bin = Path.home() / "miniforge3" / "envs" / "ac-vllm" / "bin" / "vllm"
    return os.environ.get("AC_VLLM_BIN") or (
        str(env_bin) if env_bin.exists() else shutil.which("vllm"))


def probe(api_base: str | None) -> dict:
    """Verify an already-running endpoint and describe it — never launches."""
    if not api_base:
        raise RuntimeError(
            "no vllm endpoint configured — set VLLM_URL (or the cell's "
            "api_base) to a running server / SSH tunnel, or launch one via "
            "core.api.vllm.VllmServer")
    try:
        with urllib.request.urlopen(f"{api_base}/v1/models", timeout=5) as resp:
            served = [m.get("id") for m in json.loads(resp.read()).get("data", [])]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"vllm endpoint not reachable at {api_base} ({exc}) — start the "
            "server or bring the tunnel up before the run") from exc
    return {
        "backend": "vllm",
        "url": api_base,
        "served_models": served,
        "sampling": "per-request (openai-chat params honored; pin seed/"
                    "temperature in the cell)",
    }


class VllmServer(LoopbackService):
    PORT_POOL = tuple(range(8801, 8806))
    HEALTH_PATH = "/health"
    START_TIMEOUT = 600.0  # weight load + graph capture on a 27B is minutes
    POLL_INTERVAL = 2.0

    def __init__(self, model: str, tag: str = "vllm",
                 max_model_len: int | None = None,
                 gpu_memory_utilization: float | None = None,
                 extra_args: tuple[str, ...] = ()) -> None:
        super().__init__(tag)
        self.model = model
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.extra_args = extra_args

    def spawn_env(self) -> dict[str, str] | None:
        """Prepend the vllm env's own lib/ to LD_LIBRARY_PATH — without it the
        loader mixes the conda env's libicui18n with the SYSTEM libstdc++ and
        the server dies on CXXABI_1.3.15 before it can log anything useful."""
        binary = vllm_bin()
        if binary is None:
            return None
        lib = Path(binary).resolve().parents[1] / "lib"
        if not lib.is_dir():
            return None
        merged = dict(os.environ)
        merged["LD_LIBRARY_PATH"] = ":".join(
            p for p in (str(lib), merged.get("LD_LIBRARY_PATH")) if p)
        return merged

    def command(self) -> list[str]:
        binary = vllm_bin()
        if binary is None:
            raise RuntimeError(
                "vllm binary not found — set $AC_VLLM_BIN or create the "
                "ac-vllm conda env")
        argv = [binary, "serve", self.model,
                "--host", "127.0.0.1", "--port", str(self.port)]
        if self.max_model_len is not None:
            argv += ["--max-model-len", str(self.max_model_len)]
        if self.gpu_memory_utilization is not None:
            argv += ["--gpu-memory-utilization", str(self.gpu_memory_utilization)]
        return argv + list(self.extra_args)

    def describe(self) -> dict:
        return {"backend": "vllm", "port": self.port, "model": self.model,
                "max_model_len": self.max_model_len,
                "sampling": "per-request (openai-chat params honored)"}
