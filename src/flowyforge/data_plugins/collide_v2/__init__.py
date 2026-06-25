"""COLLIDE-2V fast-track data plugin placeholders."""

from flowyforge.data_plugins.collide_v2.eos_paths import (
    CollideV2Paths,
    list_parquet_files,
    resolve_base_path,
)
from flowyforge.data_plugins.collide_v2.hf_collide1m import materialize_hf_collide1m_subset
from flowyforge.data_plugins.collide_v2.manifest import (
    create_split_manifest,
    scan_parquet_event_counts,
    write_file_event_counts,
    write_split_manifest,
)
from flowyforge.data_plugins.collide_v2.pploner_adapter import (
    PplonerPipelinePaths,
    prepare_pploner_paths,
)
from flowyforge.data_plugins.collide_v2.schema_inspector import (
    inspect_dataset_schema,
    inspect_parquet_schema,
    write_schema_report,
)
from flowyforge.data_plugins.collide_v2.source_resolver import (
    ResolvedDatasetSource,
    resolve_dataset_source,
)

__all__ = [
    "CollideV2Paths",
    "PplonerPipelinePaths",
    "ResolvedDatasetSource",
    "create_split_manifest",
    "inspect_dataset_schema",
    "inspect_parquet_schema",
    "list_parquet_files",
    "materialize_hf_collide1m_subset",
    "prepare_pploner_paths",
    "resolve_base_path",
    "resolve_dataset_source",
    "scan_parquet_event_counts",
    "write_file_event_counts",
    "write_schema_report",
    "write_split_manifest",
]
