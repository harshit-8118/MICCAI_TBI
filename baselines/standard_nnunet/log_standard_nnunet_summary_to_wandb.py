"""Upload fixed, full-volume standard nnU-Net summaries to a W&B analysis run.

This script accepts only CSV metrics already produced by
evaluate_standard_nnunet_external.py. It never trains, selects a checkpoint,
or changes predictions. The source release must remain labelled development
data, not an independent test set.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="Fixed-setting summary.csv from the evaluator.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No metric rows in {path}")
    for row in rows:
        if any(key.startswith("test_") for key in row):
            raise ValueError(
                "Refusing a summary with test_* metric names. Re-run evaluate_standard_nnunet_external.py "
                "so this development release is labelled released_validation_*."
            )
        if not any(key.startswith("released_validation_dice_") for key in row):
            raise ValueError(f"{path} is not a released-validation summary produced by the paper-safe evaluator.")
    return rows


def numeric_values(row: dict[str, str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, raw_value in row.items():
        if raw_value in (None, ""):
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if math.isfinite(value):
            values[key] = value
    return values


def setting_tag(row: dict[str, str]) -> str:
    threshold = float(row["threshold"])
    min_component = int(float(row["min_component_voxels"]))
    return f"tau_{threshold:.2f}_mincc_{min_component}"


def main() -> None:
    args = parse_args()
    if not args.summary.is_file():
        raise FileNotFoundError(args.summary)
    rows = read_rows(args.summary)
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("Install wandb in the evaluation environment before uploading summaries.") from error

    output_dir = args.output_dir or args.summary.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name,
        mode=args.mode,
        dir=str(output_dir),
        config={
            "analysis_type": "fixed_post_training_released_validation_summary",
            "data_status": "development_only_not_independent_test",
            "summary_csv": str(args.summary.resolve()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        tags=["standard-nnunet", "paper-baseline", "released-validation", "development-only"],
    )
    uploaded: list[str] = []
    for row in rows:
        tag = setting_tag(row)
        metrics = {
            f"released_validation/{tag}/{key.removeprefix('released_validation_')}": value
            for key, value in numeric_values(row).items()
            if key.startswith("released_validation_") or key.startswith("n_")
        }
        metrics["released_validation/threshold"] = float(row["threshold"])
        metrics["released_validation/min_component_voxels"] = int(float(row["min_component_voxels"]))
        run.log(metrics)
        uploaded.append(tag)
    run.summary["data_status"] = "development_only_not_independent_test"
    run.summary["uploaded_settings"] = uploaded
    run.finish()
    manifest = {
        "summary": str(args.summary.resolve()),
        "project": args.project,
        "entity": args.entity,
        "name": args.name,
        "mode": args.mode,
        "data_status": "development_only_not_independent_test",
        "uploaded_settings": uploaded,
    }
    manifest_path = output_dir / "wandb_summary_upload_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[DONE] Uploaded {len(uploaded)} fixed released-validation setting(s) to W&B.")
    print(f"[DONE] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
