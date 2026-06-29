#!/usr/bin/env python
"""Run a safe HF COLLIDE-1M tiny-sample smoke pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from flowyforge.core.config import load_config
from flowyforge.data_plugins.collide_v2.hf_collide1m import materialize_hf_collide1m_subset
from flowyforge.data_plugins.collide_v2.manifest import (
    create_split_manifest,
    scan_parquet_event_counts,
    write_file_event_counts,
    write_split_manifest,
)
from flowyforge.data_plugins.collide_v2.pipeline_checks import collect_pipeline_artifacts
from flowyforge.data_plugins.collide_v2.pploner_adapter import prepare_pploner_paths
from flowyforge.data_plugins.collide_v2.preprocessing import (
    preprocess_vectorized_dataset,
    preprocessing_config_from_dict,
)
from flowyforge.data_plugins.collide_v2.schema_inspector import inspect_dataset_schema
from flowyforge.data_plugins.collide_v2.source_resolver import ResolvedDatasetSource, resolve_dataset_source
from flowyforge.data_plugins.collide_v2.vectorization import (
    vectorization_config_from_dict,
    vectorize_parquet_files,
)
from flowyforge.training.classification_trainer import train_tiny_mlp_classifier
from flowyforge.training.config import training_config_from_dict
from flowyforge.training.evaluation import evaluate_trained_classifier, evaluation_config_from_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a safe HF tiny-sample smoke pipeline.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def run_hf_smoke_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    data = config.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("Config key 'data' must be a mapping.")

    source = resolve_dataset_source(config)
    if source.backend != "hf":
        raise ValueError(f"HF smoke pipeline requires backend='hf', got {source.backend!r}.")

    source = _materialize_if_needed(config, source)
    if not source.parquet_files:
        raise FileNotFoundError(f"No parquet files found under HF dataset directory: {source.dataset_dir}")

    pipeline_paths = prepare_pploner_paths(source)
    _write_manifests(source, pipeline_paths, data)
    schema_report = inspect_dataset_schema(source.parquet_files, max_files=data.get("max_files"))

    vector_config = vectorization_config_from_dict(config)
    vectorize_parquet_files(
        parquet_files=source.parquet_files,
        output_dir=pipeline_paths.processed_data_dir / vector_config.output_subdir,
        config=vector_config,
    )
    preprocess_vectorized_dataset(
        processed_data_dir=pipeline_paths.processed_data_dir,
        config=preprocessing_config_from_dict(config),
    )

    artifacts = collect_pipeline_artifacts(pipeline_paths.processed_data_dir)
    status = "x_only_complete"
    next_action = (
        f"HF sample did not contain configured label_column={vector_config.label_column!r}; "
        "set data.label_column to an available supervised target column if training is desired."
    )
    if artifacts["preprocessed_y"]:
        training_result = train_tiny_mlp_classifier(
            processed_data_dir=pipeline_paths.processed_data_dir,
            config=training_config_from_dict(config),
        )
        artifacts = collect_pipeline_artifacts(pipeline_paths.processed_data_dir)
        try:
            evaluate_trained_classifier(
                processed_data_dir=pipeline_paths.processed_data_dir,
                config=evaluation_config_from_dict(config),
            )
            artifacts = collect_pipeline_artifacts(pipeline_paths.processed_data_dir)
        except FileNotFoundError:
            pass
        status = "supervised_complete"
        next_action = f"Inspect metrics and checkpoint under {training_result.output_dir}."

    report = _build_report(
        source=source,
        processed_data_dir=pipeline_paths.processed_data_dir,
        parquet_count=len(source.parquet_files),
        artifacts=artifacts,
        status=status,
        next_action=next_action,
        schema_columns=schema_report.get("union_columns", []),
    )
    _write_hf_smoke_reports(report, pipeline_paths.processed_data_dir / "reports")
    return report


def main() -> int:
    args = parse_args()
    try:
        report = run_hf_smoke_pipeline(load_config(args.config))
    except Exception as exc:  # noqa: BLE001 - CLI should fail without traceback.
        print(f"HF smoke pipeline failed: {exc}", file=sys.stderr)
        return 1

    print(f"Backend: {report['backend']}")
    print(f"Status: {report['status']}")
    print(f"Parquet files: {report['number_of_parquet_files']}")
    print(f"X.npy exists: {report['artifacts']['vectorized_x']}")
    print(f"X_preprocessed.npy exists: {report['artifacts']['preprocessed_x']}")
    print(f"y.npy exists: {report['artifacts']['preprocessed_y']}")
    print(f"Report JSON: {report['report_json_path']}")
    print(f"Report Markdown: {report['report_md_path']}")
    return 0


def _materialize_if_needed(config: dict[str, Any], source: ResolvedDatasetSource) -> ResolvedDatasetSource:
    data = config.get("data", {})
    if source.parquet_files or not data.get("materialize_local_parquet", False):
        return source
    if source.hf_dataset_name is None or source.hf_split is None:
        raise ValueError("HF backend requires paths.hf_dataset_name and paths.hf_split.")
    materialize_hf_collide1m_subset(
        dataset_name=source.hf_dataset_name,
        split=source.hf_split,
        output_dir=source.dataset_dir,
        max_rows=int(data.get("max_rows") or 20),
        max_files=int(data.get("max_files") or 1),
        data_dir=source.hf_data_dir,
        data_files=source.hf_data_files,
        columns=data.get("hf_summary_columns"),
        materialization_mode=str(data.get("hf_materialization_mode", "summary")),
    )
    return resolve_dataset_source(config)


def _write_manifests(source: ResolvedDatasetSource, pipeline_paths: Any, data: dict[str, Any]) -> None:
    counts = scan_parquet_event_counts(source.dataset_dir)
    write_file_event_counts(counts, pipeline_paths.file_event_counts_path)
    split_manifest = create_split_manifest(
        counts=counts,
        split_counts=data.get("train_val_test_split_per_class"),
        classnames=data.get("to_classify") or data.get("classnames"),
        folder_map=data.get("process_to_folder"),
    )
    if pipeline_paths.split_manifest_path is not None:
        write_split_manifest(split_manifest, pipeline_paths.split_manifest_path)


def _build_report(
    source: ResolvedDatasetSource,
    processed_data_dir: Path,
    parquet_count: int,
    artifacts: dict[str, bool],
    status: str,
    next_action: str,
    schema_columns: list[str],
) -> dict[str, Any]:
    return {
        "backend": source.backend,
        "hf_dataset_name": source.hf_dataset_name,
        "hf_split": source.hf_split,
        "hf_data_dir": source.hf_data_dir,
        "dataset_dir": str(source.dataset_dir),
        "processed_data_dir": str(processed_data_dir),
        "number_of_parquet_files": parquet_count,
        "x_npy_exists": artifacts["vectorized_x"],
        "x_preprocessed_npy_exists": artifacts["preprocessed_x"],
        "y_npy_exists": artifacts["preprocessed_y"],
        "artifacts": artifacts,
        "status": status,
        "next_recommended_action": next_action,
        "available_columns": schema_columns,
    }


def _write_hf_smoke_reports(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "hf_smoke_report.json"
    md_path = output_dir / "hf_smoke_report.md"
    report["report_json_path"] = str(json_path)
    report["report_md_path"] = str(md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_format_markdown_report(report), encoding="utf-8")


def _format_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# HF COLLIDE-1M Smoke Report",
        "",
        f"- Backend: {report['backend']}",
        f"- Dataset: {report.get('hf_dataset_name')}",
        f"- Split: {report.get('hf_split')}",
        f"- HF data_dir: {report.get('hf_data_dir') or '<all>'}",
        f"- Status: {report['status']}",
        f"- Parquet files: {report['number_of_parquet_files']}",
        f"- X.npy exists: {report['x_npy_exists']}",
        f"- X_preprocessed.npy exists: {report['x_preprocessed_npy_exists']}",
        f"- y.npy exists: {report['y_npy_exists']}",
        f"- Dataset dir: `{report['dataset_dir']}`",
        f"- Processed dir: `{report['processed_data_dir']}`",
        "",
        "## Next Recommended Action",
        "",
        report["next_recommended_action"],
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
