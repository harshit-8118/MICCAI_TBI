"""Upload an internal fold-monitor summary to a separate W&B monitoring run."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--project", default="AIMS-TBI-standard-nnUNet")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--name", required=True, help="For example: standard_nnunet_fold0_monitor_epoch050")
    parser.add_argument("--mode", choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not any(key.startswith("internal_validation_") for key in rows[0]):
        raise ValueError("Expected an internal_validation_* summary from the fold monitor evaluator.")
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("wandb is required to upload internal-monitor metrics.") from error
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name,
        mode=args.mode,
        job_type="internal-fold-monitor",
        tags=["standard-nnunet", "fold-0", "internal-validation", "monitoring-only"],
        config={
            "analysis_type": "full_volume_internal_fold_monitor",
            "data_status": "internal_development_validation_not_paper_benchmark",
            "summary_csv": str(args.summary.resolve()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    for row in rows:
        metrics: dict[str, float] = {}
        for key, value in row.items():
            if key.startswith("internal_validation_") or key in {"threshold", "min_component_voxels", "n_cases"}:
                try:
                    metrics[f"internal_monitor/{key}"] = float(value)
                except (TypeError, ValueError):
                    continue
        run.log(metrics)
    run.summary["data_status"] = "internal_development_validation_not_paper_benchmark"
    run.finish()
    manifest_path = args.summary.parent / "wandb_internal_monitor_upload.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump({"summary": str(args.summary.resolve()), "name": args.name, "project": args.project}, handle, indent=2)
        handle.write("\n")
    print(f"[DONE] Uploaded {len(rows)} internal-monitor setting(s) to W&B.")


if __name__ == "__main__":
    main()
