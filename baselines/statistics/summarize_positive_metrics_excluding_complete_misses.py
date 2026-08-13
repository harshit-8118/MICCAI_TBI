#!/usr/bin/env python3
"""Summarize positive-case metrics with and without complete positive misses.

This is an analysis-only descriptive error decomposition for the two locked
Phase-2 operating points (tau=0.20/minCC=40 and tau=0.50/minCC=0). For each
model report it writes the usual ground-truth-positive Dice+/HD95+/ASSD+ mean,
then the same means after excluding *complete* misses only:
``gt_voxels > 0 and missed_positive == 1``.

The conditional values describe overlap/surface quality among positives for
which a non-empty prediction remained after post-processing. They must be
reported together with the complete-miss count/rate and must not replace the
ordinary lesion-positive metrics or be used as a causal/significance test.
No model, prediction, threshold, minCC value, or source case-metric CSV is
modified by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


DEFAULT_REPORTS = (
    ("Model A no-TTA", "single_ddp_fft_finetuned_kpycyjb66_no_tta_mA_103"),
    ("Model A TTA", "single_ddp_fft_finetuned_kpycyjb66_tta_mA_103"),
    ("Model B no-TTA", "best_val_f0_e5ohnz5w_no_tta_mB_103"),
    ("Model B TTA", "best_val_f0_e5ohnz5w_tta_mB_103"),
    ("A+B no-TTA/no-TTA", "ensemble_2_models_ddp_kpcyjb66_no_tta_e5ohnz5w_no_tta"),
    ("A+B no-TTA/TTA (submitted)", "ensemble_2_models_ddp_kpcyjb66_no_tta_hybrid_e5ohnz5w_tta"),
    ("A+B TTA/no-TTA", "ensemble_2_models_ddp_kpcyjb66_tta_hybrid_e5ohnz5w_no_tta"),
    ("A+B TTA/TTA", "ensemble_2_models_ddp_kpcyjb66_tta_e5ohnz5w_tta"),
    (
        "Standard nnU-Net (final, mirror TTA)",
        "standard_nnunet_model_b_matched/checkpoint_final_standard_mirror_tta_fixed_metrics",
    ),
    ("Random residual nnU-Net no-TTA", "random_residual_nnunet_no_tta_103"),
    ("Random residual nnU-Net TTA", "random_residual_nnunet_tta_103"),
)

METRICS = (
    ("dice", "dice"),
    ("hd95", "hd95_mm"),
    ("assd", "assd_mm"),
)


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    min_component_voxels: int

    @property
    def display(self) -> str:
        return f"tau={self.threshold:g}, minCC={self.min_component_voxels}"

    @property
    def filename(self) -> str:
        token = f"{self.threshold:g}".replace("-", "m").replace(".", "p")
        return f"case_metrics_thr{token}_mincc{self.min_component_voxels}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("checkpoints/Validation2025_100"),
        help="Parent directory containing per-case model report folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/Validation2025_100/statistics/positive_metrics_excluding_complete_misses"),
        help="New directory for the summary CSV, case-inclusion manifest, and provenance JSON.",
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        default=("0.2:40", "0.5:0"),
        metavar="TAU:MINCC",
        help="Locked operating points. Defaults to 0.2:40 and 0.5:0.",
    )
    parser.add_argument(
        "--add-report",
        action="append",
        default=[],
        metavar="LABEL=RELATIVE_REPORT_DIRECTORY",
        help=(
            "Append a completed report, for example 'Random residual nnU-Net=random_residual_nnunet/...'. "
            "Do not add a report until it has both 103-case fixed-setting CSVs."
        ),
    )
    parser.add_argument("--expected-postprocess", default="none")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of this script's existing output files in --output-dir.",
    )
    return parser.parse_args()


def _parse_point(text: str) -> OperatingPoint:
    match = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*:\s*([0-9]+)\s*", text)
    if match is None:
        raise ValueError(f"Invalid setting {text!r}; use TAU:MINCC, e.g. 0.2:40.")
    threshold = float(match.group(1))
    min_component_voxels = int(match.group(2))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be in [0, 1], got {threshold}.")
    return OperatingPoint(threshold, min_component_voxels)


def _parse_additional_reports(specifications: Iterable[str]) -> list[tuple[str, str]]:
    reports: list[tuple[str, str]] = []
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(
                f"Invalid --add-report value {specification!r}; use LABEL=RELATIVE_REPORT_DIRECTORY."
            )
        label, directory = (part.strip() for part in specification.split("=", 1))
        if not label or not directory:
            raise ValueError(f"Invalid --add-report value {specification!r}.")
        reports.append((label, directory))
    return reports


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(path: Path, point: OperatingPoint, expected_postprocess: str) -> list[dict[str, str]]:
    required = {
        "case_id",
        "gt_voxels",
        "gt_category",
        "threshold",
        "min_component_voxels",
        "postprocess",
        "dice",
        "hd95",
        "assd",
        "missed_positive",
        "empty_false_positive",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header.")
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path} lacks required columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no case rows.")
    identifiers = [row["case_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{path} contains duplicate case IDs.")
    for row in rows:
        case_id = row["case_id"]
        try:
            threshold = float(row["threshold"])
            min_component_voxels = int(row["min_component_voxels"])
            gt_voxels = int(row["gt_voxels"])
            missed_positive = int(row["missed_positive"])
            int(row["empty_false_positive"])
            metric_values = [float(row[key]) for key, _ in METRICS]
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}: invalid row for {case_id!r}: {error}") from error
        if gt_voxels < 0 or missed_positive not in (0, 1):
            raise ValueError(f"{path}: invalid gt_voxels or missed_positive for {case_id!r}.")
        if not math.isclose(threshold, point.threshold, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{path}: {case_id} has tau={threshold}, expected {point.threshold}.")
        if min_component_voxels != point.min_component_voxels:
            raise ValueError(f"{path}: {case_id} has minCC={min_component_voxels}, expected {point.min_component_voxels}.")
        if row["postprocess"].lower() != expected_postprocess.lower():
            raise ValueError(
                f"{path}: {case_id} has postprocess={row['postprocess']!r}, expected {expected_postprocess!r}."
            )
        if gt_voxels > 0 and not all(math.isfinite(value) for value in metric_values):
            raise ValueError(f"{path}: positive case {case_id} has non-finite metric(s).")
        if gt_voxels == 0 and missed_positive:
            raise ValueError(f"{path}: empty case {case_id} is incorrectly marked as missed_positive.")
    return rows


def _mean(rows: list[dict[str, str]], metric_key: str) -> float:
    if not rows:
        raise ValueError(f"Cannot calculate {metric_key} mean from zero rows.")
    return math.fsum(float(row[metric_key]) for row in rows) / len(rows)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    report_root = args.report_root.resolve()
    if not report_root.is_dir():
        raise FileNotFoundError(report_root)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Pass --overwrite to replace outputs.")
    output_dir.mkdir(parents=True, exist_ok=True)

    points = tuple(_parse_point(text) for text in args.settings)
    if len(set(points)) != len(points):
        raise ValueError("List each operating point only once.")
    reports = list(DEFAULT_REPORTS) + _parse_additional_reports(args.add_report)
    labels = [label for label, _ in reports]
    if len(labels) != len(set(labels)):
        raise ValueError("Report labels must be unique.")

    summary_rows: list[dict[str, object]] = []
    inclusion_rows: list[dict[str, object]] = []
    input_hashes: dict[str, str] = {}
    common_metadata: dict[str, tuple[str, str]] | None = None

    for point in points:
        for label, relative_directory in reports:
            report_directory = report_root / relative_directory
            metric_path = report_directory / point.filename
            if not metric_path.is_file():
                raise FileNotFoundError(
                    f"Missing locked-setting case metrics for {label}: {metric_path}. "
                    "Do not substitute an archived sweep or another threshold/minCC setting."
                )
            rows = _read_rows(metric_path, point, args.expected_postprocess)
            if len(rows) != 103:
                raise ValueError(f"{metric_path}: expected exactly 103 Phase-2 cases, found {len(rows)}.")
            metadata = {row["case_id"]: (row["gt_voxels"], row["gt_category"]) for row in rows}
            if common_metadata is None:
                common_metadata = metadata
            elif metadata != common_metadata:
                raise ValueError(f"{metric_path}: case IDs or ground-truth metadata differ from the controlled report set.")

            positive_rows = [row for row in rows if int(row["gt_voxels"]) > 0]
            empty_rows = [row for row in rows if int(row["gt_voxels"]) == 0]
            missed_rows = [row for row in positive_rows if int(row["missed_positive"]) == 1]
            retained_rows = [row for row in positive_rows if int(row["missed_positive"]) == 0]
            if len(positive_rows) != 55 or len(empty_rows) != 48:
                raise ValueError(
                    f"{metric_path}: expected 55 positive/48 empty cases, got {len(positive_rows)}/{len(empty_rows)}."
                )
            if not retained_rows:
                raise ValueError(f"{metric_path}: all positive cases are complete misses; conditional metrics are undefined.")

            summary: dict[str, object] = {
                "model": label,
                "report_directory": str(report_directory),
                "threshold": point.threshold,
                "min_component_voxels": point.min_component_voxels,
                "operating_point": point.display,
                "n_cases": len(rows),
                "n_positive": len(positive_rows),
                "n_empty": len(empty_rows),
                "n_missed_positive": len(missed_rows),
                "miss_rate_positive": len(missed_rows) / len(positive_rows),
                "n_positive_excluding_complete_misses": len(retained_rows),
                "fraction_positive_retained_after_excluding_complete_misses": len(retained_rows) / len(positive_rows),
                "n_empty_false_positive": sum(int(row["empty_false_positive"]) for row in rows),
            }
            for metric_key, output_name in METRICS:
                all_mean = _mean(positive_rows, metric_key)
                retained_mean = _mean(retained_rows, metric_key)
                summary[f"positive_all_{output_name}"] = all_mean
                summary[f"positive_excluding_complete_misses_{output_name}"] = retained_mean
                summary[f"excluding_complete_misses_minus_all_{output_name}"] = retained_mean - all_mean
            summary_rows.append(summary)

            for row in positive_rows:
                inclusion_rows.append(
                    {
                        "model": label,
                        "threshold": point.threshold,
                        "min_component_voxels": point.min_component_voxels,
                        "operating_point": point.display,
                        "case_id": row["case_id"],
                        "gt_voxels": int(row["gt_voxels"]),
                        "gt_category": row["gt_category"],
                        "missed_positive": int(row["missed_positive"]),
                        "included_in_positive_excluding_complete_misses_metrics": int(row["missed_positive"]) == 0,
                    }
                )
            input_hashes[str(metric_path)] = _sha256(metric_path)

    _write_csv(output_dir / "positive_metrics_excluding_complete_misses_summary.csv", summary_rows)
    _write_csv(output_dir / "positive_excluding_complete_misses_case_manifest.csv", inclusion_rows)
    run_config = {
        "analysis": "descriptive positive-case metrics with and without complete positive misses",
        "report_root": str(report_root),
        "output_dir": str(output_dir),
        "settings": [
            {"threshold": point.threshold, "min_component_voxels": point.min_component_voxels}
            for point in points
        ],
        "models": [{"label": label, "relative_report_directory": directory} for label, directory in reports],
        "complete_miss_definition": "ground-truth-positive case with missed_positive == 1 (non-empty prediction absent after fixed post-processing)",
        "primary_metrics": "positive_all_* columns include all 55 ground-truth-positive cases, including complete misses and their configured diagonal surface penalty",
        "secondary_conditional_metrics": "positive_excluding_complete_misses_* columns include only ground-truth-positive cases with missed_positive == 0",
        "interpretation": "Secondary descriptive error decomposition only; do not replace primary positive metrics or use this outcome-conditioned subset for causal/significance claims.",
        "expected_postprocess": args.expected_postprocess,
        "input_sha256": input_hashes,
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)
        handle.write("\n")

    print(
        f"[DONE] Wrote {len(summary_rows)} model-setting rows to "
        f"{output_dir / 'positive_metrics_excluding_complete_misses_summary.csv'}"
    )
    print(
        f"[DONE] Wrote {len(inclusion_rows)} positive-case inclusion rows to "
        f"{output_dir / 'positive_excluding_complete_misses_case_manifest.csv'}"
    )
    print("[INFO] This is a descriptive conditional analysis; no inference, training, or setting selection was performed.")


if __name__ == "__main__":
    main()
