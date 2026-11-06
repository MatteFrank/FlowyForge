"""Structured COLLIDE objects, streaming and raw Hugging Face preparation."""

from flowyforge.data_plugins.collide_v2.object_dataset import (
    CollideCollectionSpec, collate_event_batches, collection_feature_dims,
    default_collection_specs, expand_parquet_paths, iter_parquet_events,
    load_process_events, required_columns, row_to_event_batch,
)
from flowyforge.data_plugins.collide_v2.streaming import (
    StreamingDataset, build_manifest, inspect_dataset,
)

__all__ = [
    "CollideCollectionSpec", "collate_event_batches", "collection_feature_dims",
    "default_collection_specs", "expand_parquet_paths", "iter_parquet_events",
    "load_process_events", "required_columns", "row_to_event_batch",
    "StreamingDataset", "build_manifest", "inspect_dataset",
]
