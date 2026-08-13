#!/usr/bin/env python3
"""Audit final fixed-setting CI and paired-test output packages.

This is a read-only guard for the workshop revision. It verifies that the
four expected analysis packages cover the intended configurations/contrasts at
the two locked operating points, and that all analyses were based on the same
complete 103-case set (55 positive, 48 empty). It deliberately does not load
models, run inference, calculate new statistics, or select settings.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from bootstrap_fixed_setting_cis import (
    CONFIGURATION_PRESETS,
    METRICS,
    OperatingPoint,
    _read_case_rows,
    _sha256,
)
from paired_tta_ablation_tests import ANALYSIS_SPECS


LOCKED_POINTS = (OperatingPoint(0.2, 40), OperatingPoint(0.5, 0))
EXPECTED_CASES = 103
EXPECTED_POSITIVE = 55
EXPECTED_EMPTY = 48


@dataclass(frozen=True)
class Package:
    directory_name: str
    result_filename: str
    kind: str
    preset: str
    analysis_id: str | None


PACKAGES = (
    Package(
        "tta_ablation_bootstrap_cis",
        "bootstrap_ci_summary.csv",
        "ci",
        "tta_ablation",
        None,
    ),
    Package(
        "tta_ablation_paired_tests",
        "paired_effects_summary.csv",
        "paired",
        "tta_ablation",
        "tta_ablation",
    ),
    Package(
        "standard_nnunet_benchmark_bootstrap_cis",
        "bootstrap_ci_summary.csv",
        "ci",
        "standard_nnunet_benchmark",
        None,
    ),
    Package(
        "standard_nnunet_benchmark_paired_tests",
        "paired_effects_summary.csv",
        "paired",
        "standard_nnunet_benchmark",
        "standard_nnunet_benchmark",
    ),
)

RANDOM_RESIDUAL_PACKAGES = (
    Package(
        "random_residual_benchmark_bootstrap_cis",
        "bootstrap_ci_summary.csv",
        "ci",
        "random_residual_benchmark",
        None,
    ),
    Package(
        "random_residual_benchmark_paired_tests",
        "paired_effects_summary.csv",
        "paired",
        "random_residual_benchmark",
        "random_residual_benchmark",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("checkpoints/Validation2025_100"),
        help="Parent directory containing the original per-case report folders.",
    )
    parser.add_argument(
        "--statistics-root",
        type=Path,
        default=Path("checkpoints/Validation2025_100/statistics"),
        help="Parent directory containing the four final analysis packages.",
    )
    parser.add_argument(
        "--include-random-residual",
        action="store_true",
        help=(
            "Also require the completed random-residual-versus-Model-B CI and paired-test packages. "
            "Leave unset until the scratch model has finished Phase-2 inference."
        ),
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header.")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no rows.")
    return rows


def _points_from_rows(rows: list[dict[str, str]]) -> set[tuple[float, int]]:
    return {(float(row["threshold"]), int(row["min_component_voxels"])) for row in rows}


def _expected_point_set() -> set[tuple[float, int]]:
    return {(point.threshold, point.min_component_voxels) for point in LOCKED_POINTS}


def _validate_manifest(path: Path) -> None:
    rows = _read_csv(path)
    by_point: dict[tuple[float, int], list[dict[str, str]]] = {}
    for row in rows:
        text = row.get("operating_point", "")
        point = next((item for item in LOCKED_POINTS if item.display == text), None)
        if point is None:
            raise ValueError(f"{path}: unexpected operating_point {text!r}.")
        if int(row["gt_voxels"]) <= 0 or row.get("analysis_subset") != "ground_truth_positive":
            raise ValueError(f"{path}: manifest includes a non-positive or wrongly labelled case: {row!r}")
        by_point.setdefault((point.threshold, point.min_component_voxels), []).append(row)
    if set(by_point) != _expected_point_set():
        raise ValueError(f"{path}: manifest operating points are {set(by_point)}, expected {_expected_point_set()}.")
    reference: dict[str, tuple[str, str]] | None = None
    for point, point_rows in by_point.items():
        if len(point_rows) != EXPECTED_POSITIVE:
            raise ValueError(f"{path}: {point} contains {len(point_rows)} positive cases, expected {EXPECTED_POSITIVE}.")
        metadata = {row["case_id"]: (row["gt_voxels"], row["gt_category"]) for row in point_rows}
        if len(metadata) != len(point_rows):
            raise ValueError(f"{path}: {point} contains duplicated positive case IDs.")
        if reference is None:
            reference = metadata
        elif metadata != reference:
            raise ValueError(f"{path}: positive-case manifest differs between locked operating points.")


def _validate_input_reports(run_config: dict[str, object], report_root: Path) -> None:
    configurations = run_config.get("configurations")
    if not isinstance(configurations, list) or not configurations:
        raise ValueError("run_config.json lacks configurations.")
    expected_postprocess = str(run_config.get("expected_postprocess", "none"))
    reference_metadata: dict[str, tuple[str, str]] | None = None
    for point in LOCKED_POINTS:
        for configuration in configurations:
            if not isinstance(configuration, dict):
                raise ValueError("run_config.json has malformed configuration metadata.")
            directory_text = configuration.get("report_directory")
            if not directory_text:
                raise ValueError("run_config.json configuration lacks report_directory.")
            directory = Path(str(directory_text))
            if not directory.is_dir():
                # The reports are normally recorded as absolute paths. This
                # fallback supports moving the complete report root intact.
                directory = report_root / directory.name
            metric_path = directory / point.filename
            rows = _read_case_rows(metric_path, point, expected_postprocess)
            if len(rows) != EXPECTED_CASES:
                raise ValueError(f"{metric_path}: {len(rows)} cases, expected {EXPECTED_CASES}.")
            metadata = {row["case_id"]: (row["gt_voxels"], row["gt_category"]) for row in rows}
            if len(metadata) != len(rows):
                raise ValueError(f"{metric_path}: duplicated case IDs.")
            positive = sum(int(gt_voxels) > 0 for gt_voxels, _ in metadata.values())
            empty = sum(int(gt_voxels) == 0 for gt_voxels, _ in metadata.values())
            if positive != EXPECTED_POSITIVE or empty != EXPECTED_EMPTY:
                raise ValueError(
                    f"{metric_path}: positive={positive}, empty={empty}; expected "
                    f"{EXPECTED_POSITIVE}/{EXPECTED_EMPTY}."
                )
            if reference_metadata is None:
                reference_metadata = metadata
            elif metadata != reference_metadata:
                raise ValueError(f"{metric_path}: case or ground-truth metadata differs from the controlled set.")

    input_hashes = run_config.get("input_sha256", {})
    if not isinstance(input_hashes, dict) or not input_hashes:
        raise ValueError("run_config.json lacks input_sha256 provenance.")
    for recorded_path, expected_hash in input_hashes.items():
        path = Path(str(recorded_path))
        if not path.is_file():
            path = report_root / path.parent.name / path.name
        if not path.is_file():
            raise FileNotFoundError(f"Cannot recheck input hash: {recorded_path}")
        observed_hash = _sha256(path)
        if observed_hash != expected_hash:
            raise ValueError(f"Input changed after analysis: {path}")


def _validate_ci_rows(rows: list[dict[str, str]], package: Package) -> None:
    expected_labels = {spec[0] for spec in CONFIGURATION_PRESETS[package.preset]}
    expected = {
        (label, point.threshold, point.min_component_voxels, metric_key)
        for label in expected_labels
        for point in LOCKED_POINTS
        for metric_key, _, _, _ in METRICS
    }
    observed = {
        (row["configuration"], float(row["threshold"]), int(row["min_component_voxels"]), row["metric_key"])
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise ValueError(f"{package.directory_name}: CI rows do not cover exactly the expected configuration/point/metric grid.")
    for row in rows:
        mean = float(row["mean"])
        lower = float(row["ci_95_lower"])
        upper = float(row["ci_95_upper"])
        if not all(math.isfinite(value) for value in (mean, lower, upper)) or not lower <= mean <= upper:
            raise ValueError(f"{package.directory_name}: invalid CI row {row!r}")
        if int(row["n_positive"]) != EXPECTED_POSITIVE or int(row["n_empty_excluded"]) != EXPECTED_EMPTY:
            raise ValueError(f"{package.directory_name}: wrong case count in CI row {row!r}")


def _validate_paired_rows(rows: list[dict[str, str]], package: Package) -> None:
    if package.analysis_id is None:
        raise RuntimeError("Internal package configuration error.")
    expected_contrasts = {contrast[0] for contrast in ANALYSIS_SPECS[package.analysis_id]["contrasts"]}
    expected = {
        (contrast, point.threshold, point.min_component_voxels, metric_key)
        for contrast in expected_contrasts
        for point in LOCKED_POINTS
        for metric_key, _, _, _ in METRICS
    }
    observed = {
        (row["contrast_id"], float(row["threshold"]), int(row["min_component_voxels"]), row["metric_key"])
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise ValueError(f"{package.directory_name}: paired rows do not cover exactly the expected contrast/point/metric grid.")
    for row in rows:
        mean = float(row["mean_difference"])
        lower = float(row["ci_95_lower"])
        upper = float(row["ci_95_upper"])
        p_value = float(row["p_value_two_sided_label_swap"])
        p_holm = float(row["p_value_holm"])
        if not all(math.isfinite(value) for value in (mean, lower, upper, p_value, p_holm)):
            raise ValueError(f"{package.directory_name}: non-finite paired result {row!r}")
        if not lower <= mean <= upper or not 0.0 <= p_value <= 1.0 or not 0.0 <= p_holm <= 1.0:
            raise ValueError(f"{package.directory_name}: invalid paired result {row!r}")
        if int(row["n_positive"]) != EXPECTED_POSITIVE or int(row["n_empty_excluded"]) != EXPECTED_EMPTY:
            raise ValueError(f"{package.directory_name}: wrong case count in paired row {row!r}")


def main() -> None:
    args = parse_args()
    report_root = args.report_root.resolve()
    statistics_root = args.statistics_root.resolve()
    if not report_root.is_dir():
        raise FileNotFoundError(report_root)
    if not statistics_root.is_dir():
        raise FileNotFoundError(statistics_root)

    packages = PACKAGES + (RANDOM_RESIDUAL_PACKAGES if args.include_random_residual else ())
    for package in packages:
        directory = statistics_root / package.directory_name
        config_path = directory / "run_config.json"
        result_path = directory / package.result_filename
        manifest_path = directory / "positive_case_manifest.csv"
        if not directory.is_dir() or not config_path.is_file() or not result_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Incomplete analysis package: {directory}")
        run_config = json.loads(config_path.read_text(encoding="utf-8"))
        # ``analysis_id`` was added after the first TTA paired-test package
        # had already been generated.  A missing identifier is therefore
        # accepted as legacy provenance; the exact configuration labels,
        # contrast IDs, endpoint grid, case manifests, and input hashes below
        # remain mandatory. A present-but-wrong identifier is never accepted.
        observed_analysis_id = run_config.get("analysis_id")
        if package.analysis_id is not None and observed_analysis_id not in (None, package.analysis_id):
            raise ValueError(
                f"{config_path}: analysis_id={observed_analysis_id!r}, expected {package.analysis_id!r}."
            )
        if package.analysis_id is not None and observed_analysis_id is None:
            print(f"[INFO] {package.directory_name}: legacy run_config without analysis_id; validating full result grid.")
        _validate_input_reports(run_config, report_root)
        _validate_manifest(manifest_path)
        rows = _read_csv(result_path)
        if package.kind == "ci":
            _validate_ci_rows(rows, package)
        else:
            _validate_paired_rows(rows, package)
        print(f"[PASS] {package.directory_name}: {len(rows)} result rows; 103 cases / 55 positive / 48 empty.")
    print("[PASS] All final CI and paired-test packages are complete and internally consistent.")


if __name__ == "__main__":
    main()
