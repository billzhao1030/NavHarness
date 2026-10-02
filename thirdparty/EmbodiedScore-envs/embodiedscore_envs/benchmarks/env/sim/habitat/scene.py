"""SceneRef — where a scene lives on disk (pure data, layer L0)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SceneRef:
    scene_file: str                    # .glb (MP3D) or .basis.glb (HM3D), absolute
    dataset_config: str | None = None  # scene_dataset_config.json (HM3D annotated), or None for a bare glb
    navmesh_file: str | None = None    # defaults to scene_file with a .navmesh suffix

    @property
    def navmesh(self) -> str:
        if self.navmesh_file:
            return self.navmesh_file
        p = Path(self.scene_file)
        name = p.name
        for suf in (".basis.glb", ".glb"):
            if name.endswith(suf):
                return str(p.with_name(name[: -len(suf)] + (".basis.navmesh" if suf == ".basis.glb" else ".navmesh")))
        return str(p.with_suffix(".navmesh"))
