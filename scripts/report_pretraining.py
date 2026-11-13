#!/usr/bin/env python
"""Report masked pretraining history and deterministic validation reconstruction."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Directory containing metrics/history/config and optional checkpoint.pt")
    parser.add_argument("--output-dir", help="Default: RUN_DIR/pretraining_report")
    parser.add_argument("--overwrite", action="store_true", help="Replace files only inside an existing diagnostic report")
    parser.add_argument("--history-only", action="store_true", help="No model or source data access")
    for name in ("max-events", "batch-size", "seed", "bins", "min-bin-count"):
        parser.add_argument("--" + name, type=int)
    args = parser.parse_args()
    try:
        from flowyforge.analysis.pretraining_report import generate_pretraining_report
        options = {key: value for key, value in vars(args).items()
                   if key in {"max_events", "batch_size", "seed", "bins", "min_bin_count"} and value is not None}
        if args.history_only:
            options["inference"] = False
        result = generate_pretraining_report(args.run_dir, output_dir=args.output_dir,
                                              options=options, overwrite=args.overwrite)
        print(f"Report: {result['html']}; inference={result['inference']['status']}; test excluded.")
        if result["inference"].get("reason"):
            print(result["inference"]["reason"])
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError, FloatingPointError) as exc:
        print(f"Pretraining report failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
