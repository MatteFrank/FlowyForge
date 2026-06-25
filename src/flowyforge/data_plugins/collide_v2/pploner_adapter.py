"""Adapter for pploner-style COLLIDE-2V pipeline paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from flowyforge.data_plugins.collide_v2.source_resolver import ResolvedDatasetSource


@dataclass(frozen=True, slots=True)
class PplonerPipelinePaths:
    dataset_dir: Path
    tmp_data_dir: Path
    processed_data_dir: Path
    file_event_counts_path: Path
    split_manifest_path: Path | None = None


def prepare_pploner_paths(
    source: ResolvedDatasetSource,
    create_dirs: bool = True,
) -> PplonerPipelinePaths:
    """Prepare the paths expected by the pploner-style data pipeline."""

    manifests_dir = source.processed_data_dir / "manifests"
    if create_dirs:
        source.tmp_data_dir.mkdir(parents=True, exist_ok=True)
        source.processed_data_dir.mkdir(parents=True, exist_ok=True)
        manifests_dir.mkdir(parents=True, exist_ok=True)

    return PplonerPipelinePaths(
        dataset_dir=source.dataset_dir,
        tmp_data_dir=source.tmp_data_dir,
        processed_data_dir=source.processed_data_dir,
        file_event_counts_path=manifests_dir / "file_event_counts.json",
        split_manifest_path=manifests_dir / "split_manifest.json",
    )


__all__ = ["PplonerPipelinePaths", "prepare_pploner_paths"]
