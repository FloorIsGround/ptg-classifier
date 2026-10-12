from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactPaths:
    index_path: Path
    metas_path: Path


def artifact_paths(artifacts_dir: Path, run_prefix: str) -> ArtifactPaths:
    return ArtifactPaths(
        index_path=artifacts_dir / f"{run_prefix}index.faiss",
        metas_path=artifacts_dir / f"{run_prefix}metas.parquet",
    )
