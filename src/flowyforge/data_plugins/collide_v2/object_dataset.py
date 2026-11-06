"""Small, explicit object reader for original COLLIDE nested Parquet columns.

Columns are exactly ``{view}_{collection}_{field}``; no aliases or scalar
summary fallbacks are inferred. One Parquet row must describe one event.
"""

from __future__ import annotations

import glob
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from flowyforge.core.typing import EventBatch, ObjectGroup


@dataclass(frozen=True, slots=True)
class CollideCollectionSpec:
    name: str
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = ()
    max_objects: int | None = None
    sort_by_pt: bool = True

    def __post_init__(self) -> None:
        supported = {"PT", "MET", "Mass", "Eta", "Phi", "PID", "Charge"}
        if not self.name or not self.fields or len(set(self.fields)) != len(self.fields):
            raise ValueError("Collection names and unique fields must be nonempty.")
        if set(self.fields) - supported:
            raise ValueError(f"Unsupported fields for {self.name}: {self.fields}")
        if not set(self.optional_fields) < set(self.fields):
            raise ValueError("Optional fields must be a proper subset of fields.")
        if self.max_objects is not None and (
            type(self.max_objects) is not int or self.max_objects < 1
        ):
            raise ValueError("max_objects must be a positive integer or None.")
        if self.sort_by_pt and "PT" not in self.fields:
            raise ValueError("Sorting by PT requires a PT field.")

    @property
    def feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for field in self.fields:
            if field == "Phi":
                names.extend(("sin_phi", "cos_phi"))
            elif field in {"PT", "MET", "Mass"}:
                names.append(f"log1p_{field}")
            elif field == "Eta":
                names.append("scaled_eta")
            elif field == "PID":
                names.append("signed_log1p_PID")
            else:
                names.append(field)
        return tuple(names)


def default_collection_specs(
    collection_limits: Mapping[str, int] | None = None,
) -> tuple[CollideCollectionSpec, ...]:
    """Canonical requested collections; optional fields never use aliases.

    Limits belong in YAML. Without limits, the reusable reader keeps all objects.
    A limit of zero explicitly disables a collection.
    """
    specs = (
        CollideCollectionSpec("PUPPIPart", ("PT", "Eta", "Phi", "PID", "Charge"), ("PID", "Charge")),
        CollideCollectionSpec("JetPuppiAK4", ("PT", "Eta", "Phi", "Mass"), ("Mass",)),
        CollideCollectionSpec("Electron", ("PT", "Eta", "Phi", "Charge"), ("Charge",)),
        CollideCollectionSpec("MuonTight", ("PT", "Eta", "Phi", "Charge"), ("Charge",)),
        CollideCollectionSpec("PhotonTight", ("PT", "Eta", "Phi")),
        CollideCollectionSpec("PUPPIMET", ("MET", "Phi"), sort_by_pt=False),
    )
    limits = dict(collection_limits or {})
    if set(limits) - {spec.name for spec in specs}:
        raise ValueError(f"Unknown collection limits: {sorted(set(limits) - {s.name for s in specs})}")
    if any(type(value) is not int or value < 0 for value in limits.values()):
        raise ValueError("Collection limits must be nonnegative integers.")
    return _specs(tuple(
        replace(spec, max_objects=limits.get(spec.name))
        for spec in specs if limits.get(spec.name) != 0
    ))


def _specs(specs: Sequence[CollideCollectionSpec] | None) -> tuple[CollideCollectionSpec, ...]:
    result = default_collection_specs() if specs is None else tuple(specs)
    if not result or len({spec.name for spec in result}) != len(result):
        raise ValueError("Provide at least one collection with unique names.")
    return result


def _view(view: str) -> str:
    if view not in {"FullReco", "L1T"}:
        raise ValueError("view must be FullReco or L1T.")
    return view


def collection_feature_dims(
    specs: Sequence[CollideCollectionSpec] | None = None,
) -> dict[str, int]:
    return {spec.name: len(spec.feature_names) for spec in _specs(specs)}


def required_columns(
    view: str = "FullReco",
    specs: Sequence[CollideCollectionSpec] | None = None,
    *,
    include_optional: bool = False,
) -> list[str]:
    return [
        f"{_view(view)}_{spec.name}_{field}"
        for spec in _specs(specs) for field in spec.fields
        if include_optional or field not in spec.optional_fields
    ]


def _object_values(value: Any, column: str) -> np.ndarray:
    """Accept a vector, or singleton list wrappers around one object vector.

    Multiple nested vectors are ambiguous (events/subobjects) and are rejected,
    never flattened into a fabricated event.
    """
    if value is None:
        return np.empty(0, dtype=np.float64)
    while isinstance(value, (list, tuple, np.ndarray)) and len(value) == 1:
        if not isinstance(value[0], (list, tuple, np.ndarray)):
            break
        value = value[0]
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{column}: expected an object list, not a scalar/summary.")
    try:
        values = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column}: expected one numeric object vector.") from exc
    if values.ndim != 1:
        raise ValueError(f"{column}: expected one object vector; ambiguous nested shape {values.shape}.")
    return values


def row_to_event_batch(
    row: Mapping[str, Any],
    view: str = "FullReco",
    specs: Sequence[CollideCollectionSpec] | None = None,
    *,
    target: int | None = None,
    eta_clip: float = 5.0,
    value_clip: float = 1.0e6,
) -> EventBatch:
    """Convert a row to float32 [1, N, F] groups with True = real object.

    NaN/inf/null numeric entries become zero before transforms. Missing optional
    columns become zero features and are recorded. Missing required columns or
    inconsistent object counts raise, including on the L1T path.
    """
    specs = _specs(specs)
    _view(view)
    if not all(math.isfinite(v) and v > 0 for v in (eta_clip, value_clip)):
        raise ValueError("eta_clip and value_clip must be finite and positive.")
    missing = sorted(set(required_columns(view, specs)) - row.keys())
    if missing:
        raise ValueError(f"Missing required nested COLLIDE columns: {missing}. Scalar-summary data is unsupported.")
    groups: dict[str, ObjectGroup] = {}
    for spec in specs:
        columns = {field: f"{view}_{spec.name}_{field}" for field in spec.fields}
        arrays = {
            field: _object_values(row[column], column)
            for field, column in columns.items() if column in row and row[column] is not None
        }
        # A null required list is an empty collection, not a missing field.
        for field in spec.fields:
            if field not in spec.optional_fields and field not in arrays:
                arrays[field] = np.empty(0, dtype=np.float64)
        anchor = next(field for field in spec.fields if field not in spec.optional_fields)
        count = len(arrays[anchor])
        if any(len(array) != count for array in arrays.values()):
            raise ValueError(f"{view}_{spec.name}: inconsistent field lengths { {k: len(v) for k, v in arrays.items()} }.")
        missing_optional = [field for field in spec.optional_fields if field not in arrays]
        nonfinite = sum(int((~np.isfinite(array)).sum()) for array in arrays.values())
        clean = {
            field: np.nan_to_num(arrays.get(field, np.zeros(count)), nan=0.0, posinf=0.0, neginf=0.0)
            for field in spec.fields
        }
        indices = np.arange(count)
        if spec.sort_by_pt:
            indices = np.argsort(-np.maximum(clean["PT"], 0), kind="stable")
        indices = indices[:spec.max_objects]
        features: list[np.ndarray] = []
        for field in spec.fields:
            values = clean[field][indices]
            if field in {"PT", "MET", "Mass"}:
                features.append(np.log1p(np.clip(values, 0, value_clip)))
            elif field == "Eta":
                features.append(np.clip(values, -eta_clip, eta_clip) / eta_clip)
            elif field == "Phi":
                angles = np.remainder(values, 2 * np.pi)
                features.extend((np.sin(angles), np.cos(angles)))
            elif field == "PID":
                features.append(np.sign(values) * np.log1p(np.minimum(np.abs(values), value_clip)))
            else:
                features.append(np.clip(values, -3, 3))
        tensor = torch.from_numpy(np.stack(features, axis=-1).astype(np.float32)).unsqueeze(0)
        groups[spec.name] = ObjectGroup(
            name=spec.name,
            features=tensor,
            mask=torch.ones((1, len(indices)), dtype=torch.bool),
            metadata={
                "view": view, "feature_names": list(spec.feature_names),
                "source_columns": list(columns.values()), "max_objects": spec.max_objects,
                "original_count": count, "retained_count": len(indices),
                "truncated_count": count - len(indices), "nonfinite_values": nonfinite,
                "missing_optional_fields": missing_optional,
            },
        )
    return EventBatch(
        objects=groups,
        targets=None if target is None else torch.tensor([target], dtype=torch.long),
        metadata={"view": view, "batch_size": 1},
    )


def collate_event_batches(events: Sequence[EventBatch]) -> EventBatch:
    """Pad single-event groups to batch-local maxima; absent groups are masked."""
    if not events:
        raise ValueError("Cannot collate an empty event sequence.")
    views = {event.metadata.get("view") for event in events} - {None}
    if len(views) > 1:
        raise ValueError("Cannot collate FullReco and L1T into one batch.")
    names = dict.fromkeys(name for event in events for name in event.objects)
    objects: dict[str, ObjectGroup] = {}
    for name in names:
        present = [event.objects[name] for event in events if name in event.objects]
        first = present[0]
        for group in present:
            if group.features.ndim != 3 or group.features.shape[0] != 1:
                raise ValueError("Collation expects single-event [1, N, F] groups.")
            if group.features.shape[-1] != first.features.shape[-1]:
                raise ValueError(f"Feature dimension mismatch for {name}.")
            if group.metadata.get("feature_names") != first.metadata.get("feature_names"):
                raise ValueError(f"Feature ordering mismatch for {name}.")
            if group.mask is None or group.mask.dtype != torch.bool or group.mask.shape != group.features.shape[:2]:
                raise ValueError(f"{name} requires a boolean [1, N] mask.")
        length = max(group.features.shape[1] for group in present)
        features = first.features.new_zeros((len(events), length, first.features.shape[-1]))
        mask = torch.zeros((len(events), length), dtype=torch.bool, device=features.device)
        for index, event in enumerate(events):
            if name in event.objects:
                group = event.objects[name]
                size = group.features.shape[1]
                features[index, :size] = group.features[0].masked_fill(~group.mask[0, :, None], 0)
                mask[index, :size] = group.mask[0]
        objects[name] = ObjectGroup(name, features, mask, {
            "feature_names": first.metadata.get("feature_names"),
            "events": [event.objects[name].metadata if name in event.objects else None for event in events],
        })
    labeled = [event.targets is not None for event in events]
    if any(labeled) and not all(labeled):
        raise ValueError("Cannot mix labeled and unlabeled events.")
    if all(labeled) and any(event.targets.shape != (1,) for event in events):
        raise ValueError("Single-event targets must have shape [1].")
    return EventBatch(
        objects=objects,
        targets=torch.cat([event.targets for event in events]) if all(labeled) else None,
        metadata={"batch_size": len(events), "view": next(iter(views), None),
                  "events": [event.metadata for event in events]},
    )


def expand_parquet_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    """Resolve local files/directories/globs deterministically, with deduplication."""
    entries = [paths] if isinstance(paths, (str, Path)) else paths
    selected: set[Path] = set()
    for entry in entries:
        pattern = str(Path(entry).expanduser())
        matches: list[Path] = []
        for match in sorted(glob.glob(pattern, recursive=True)):
            path = Path(match)
            matches.extend(path.rglob("*.parquet") if path.is_dir() else [path])
        files = [path.resolve() for path in matches if path.is_file() and path.suffix == ".parquet"]
        if not files:
            raise FileNotFoundError(f"No local Parquet files match: {entry}")
        selected.update(files)
    if not selected:
        raise ValueError("Provide at least one Parquet path.")
    return sorted(selected)


def iter_parquet_events(
    paths: str | Path | Sequence[str | Path],
    view: str = "FullReco",
    specs: Sequence[CollideCollectionSpec] | None = None,
    *,
    max_events: int | None = None,
    read_batch_size: int = 256,
    target: int | None = None,
    eta_clip: float = 5.0,
    value_clip: float = 1.0e6,
) -> Iterator[EventBatch]:
    """Read projected Arrow batches, preserving one source row per event."""
    specs = _specs(specs)
    columns = required_columns(view, specs, include_optional=True)
    required = set(required_columns(view, specs))
    if max_events is not None and (type(max_events) is not int or max_events < 0):
        raise ValueError("max_events must be a nonnegative integer or None.")
    if type(read_batch_size) is not int or read_batch_size < 1:
        raise ValueError("read_batch_size must be a positive integer.")
    seen = 0
    for path in expand_parquet_paths(paths):
        if max_events is not None and seen >= max_events:
            return
        with pq.ParquetFile(path) as parquet:
            available = set(parquet.schema_arrow.names)
            if missing := sorted(required - available):
                raise ValueError(f"{path}: missing required nested COLLIDE columns {missing}; scalar summaries are unsupported.")
            selected = [column for column in columns if column in available]
            batch_size = min(read_batch_size, max_events - seen) if max_events is not None else read_batch_size
            row_index = 0
            for record_batch in parquet.iter_batches(batch_size=batch_size, columns=selected):
                for row in record_batch.to_pylist():
                    try:
                        event = row_to_event_batch(row, view, specs, target=target,
                                                   eta_clip=eta_clip, value_clip=value_clip)
                    except ValueError as exc:
                        raise ValueError(f"{path}, row {row_index}: {exc}") from exc
                    event.metadata.update({"source_file": str(path), "source_row": row_index})
                    yield event
                    seen += 1
                    row_index += 1
                    if max_events is not None and seen >= max_events:
                        return


def load_process_events(
    processes: Mapping[str, str | Path | Sequence[str | Path]],
    view: str = "FullReco",
    specs: Sequence[CollideCollectionSpec] | None = None,
    *,
    max_events_per_class: int | None = None,
    eta_clip: float = 5.0,
    value_clip: float = 1.0e6,
) -> tuple[list[EventBatch], dict[str, int]]:
    """Load a bounded smoke sample; labels derive only from process names."""
    if not processes or any(not name.strip() for name in processes):
        raise ValueError("Provide nonempty process names.")
    label_map = {name: index for index, name in enumerate(sorted(processes))}
    events: list[EventBatch] = []
    used_files: set[Path] = set()
    for name, label in label_map.items():
        paths = expand_parquet_paths(processes[name])
        if overlap := used_files.intersection(paths):
            raise ValueError(f"Parquet files assigned to multiple processes: {sorted(map(str, overlap))}")
        used_files.update(paths)
        sample = list(iter_parquet_events(paths, view, specs, max_events=max_events_per_class,
                                         target=label, eta_clip=eta_clip, value_clip=value_clip))
        if not sample:
            raise ValueError(f"No events loaded for process {name}.")
        for event in sample:
            event.metadata["process_name"] = name
        events.extend(sample)
    return events, label_map


__all__ = [
    "CollideCollectionSpec", "default_collection_specs", "collection_feature_dims",
    "required_columns", "row_to_event_batch", "collate_event_batches",
    "expand_parquet_paths", "iter_parquet_events", "load_process_events",
]
