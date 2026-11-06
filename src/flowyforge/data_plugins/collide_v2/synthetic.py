"""Deterministic nested fixtures for offline Perceiver feasibility runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from flowyforge.data_plugins.collide_v2.object_dataset import default_collection_specs


def make_tiny_parquet_sample(
    output_dir: str | Path, *, events_per_class: int = 40,
    files_per_class: int = 2, seed: int = 42,
) -> list[Path]:
    """Write both views and all six collections; never overwrite existing data.

    Classes differ in a toy momentum scale. These fixtures exercise the data
    contract, not detector simulation or physics performance.
    """
    if type(events_per_class) is not int or events_per_class < 3:
        raise ValueError("events_per_class must be an integer >= 3.")
    if type(files_per_class) is not int or not 1 <= files_per_class <= events_per_class:
        raise ValueError("files_per_class must be between 1 and events_per_class.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    output = Path(output_dir).expanduser()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite existing sample: {output}. Choose an empty output directory.")
    rng = np.random.default_rng(seed)
    specs = default_collection_specs()
    fields = []
    for view in ("FullReco", "L1T"):
        for spec in specs:
            for field in spec.fields:
                metadata = {b"unit": b"GeV"} if field in {"PT", "MET", "Mass"} else (
                    {b"unit": b"rad"} if field == "Phi" else None)
                fields.append(pa.field(f"{view}_{spec.name}_{field}", pa.list_(pa.float32()), metadata=metadata))
    schema = pa.schema(fields)
    written = []
    for label, name in enumerate(("background", "signal")):
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        for part, indices in enumerate(np.array_split(np.arange(events_per_class), files_per_class)):
            rows = []
            for _ in indices:
                row = {}
                for view in ("FullReco", "L1T"):
                    for spec in specs:
                        count = 1 if spec.name == "PUPPIMET" else int(rng.integers(0, 13 if spec.name == "PUPPIPart" else 4))
                        for field in spec.fields:
                            if field in {"PT", "MET", "Mass"}:
                                values = rng.exponential(10 + label * 15, count)
                            elif field == "Eta":
                                values = rng.uniform(-4, 4, count)
                            elif field == "Phi":
                                values = rng.uniform(-np.pi, np.pi, count)
                            elif field == "PID":
                                values = rng.choice([-211, -13, -11, 11, 13, 22, 211], count)
                            else:
                                values = rng.choice([-1, 0, 1], count)
                            row[f"{view}_{spec.name}_{field}"] = values.tolist()
                rows.append(row)
            path = directory / f"sample_{part:03d}.parquet"
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, row_group_size=16)
            written.append(path)
    return written


__all__ = ["make_tiny_parquet_sample"]
