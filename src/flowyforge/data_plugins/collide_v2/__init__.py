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
from flowyforge.data_plugins.collide_v2.pipeline_checks import (
    collect_pipeline_artifacts,
    file_exists,
)
from flowyforge.data_plugins.collide_v2.preprocessing import (
    PreprocessingConfig,
    PreprocessingResult,
    apply_standardization,
    compute_standardization_stats,
    preprocess_vectorized_dataset,
    preprocessing_config_from_dict,
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
from flowyforge.data_plugins.collide_v2.vectorization import (
    VectorizationConfig,
    VectorizationResult,
    build_feature_map,
    encode_labels,
    infer_feature_columns,
    load_feature_map,
    save_feature_map,
    vectorization_config_from_dict,
    vectorize_parquet_files,
)

__all__ = [
    "CollideV2Paths",
    "PplonerPipelinePaths",
    "PreprocessingConfig",
    "PreprocessingResult",
    "ResolvedDatasetSource",
    "VectorizationConfig",
    "VectorizationResult",
    "apply_standardization",
    "build_feature_map",
    "collect_pipeline_artifacts",
    "compute_standardization_stats",
    "create_split_manifest",
    "encode_labels",
    "file_exists",
    "infer_feature_columns",
    "inspect_dataset_schema",
    "inspect_parquet_schema",
    "list_parquet_files",
    "load_feature_map",
    "materialize_hf_collide1m_subset",
    "preprocess_vectorized_dataset",
    "preprocessing_config_from_dict",
    "prepare_pploner_paths",
    "resolve_base_path",
    "resolve_dataset_source",
    "save_feature_map",
    "scan_parquet_event_counts",
    "vectorization_config_from_dict",
    "vectorize_parquet_files",
    "write_file_event_counts",
    "write_schema_report",
    "write_split_manifest",
]
