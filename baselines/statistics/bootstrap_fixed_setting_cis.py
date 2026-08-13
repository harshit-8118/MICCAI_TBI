#!/usr/bin/env python3
"""Bootstrap confidence intervals for fixed-setting, case-level reports.

This script is deliberately an analysis-only tool. It never loads a model,
performs inference, searches operating points, or changes any prediction.
It reads existing ``case_metrics_*.csv`` files and reports percentile 95%
bootstrap confidence intervals for the ground-truth-positive subgroup:

* Dice+ (higher is better)
* HD95+ in millimetres (lower is better)
* ASSD+ in millimetres (lower is better)

The presets encode the controlled A+B ensemble conditions, the four A/B
single-model conditions, their complete TTA ablation, or the conventional
standard-nnU-Net-versus-Model-B benchmark under the two locked operating
points used in the revision: tau=0.20/minCC=40 and tau=0.50/minCC=0.
Input case sets, reference labels, post-processing, and checkpoint provenance
must agree exactly; the script stops rather than silently analysing a
mismatched comparison.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np


ENSEMBLE_CONFIGURATION_SPECS = (
    (
        "A+B no-TTA/no-TTA",
        "ensemble_2_models_ddp_kpcyjb66_no_tta_e5ohnz5w_no_tta",
        "none",
        "none",
        ("model_b", "model_a"),
        ("none", "none"),
    ),
    (
        "A+B no-TTA/TTA (submitted)",
        "ensemble_2_models_ddp_kpcyjb66_no_tta_hybrid_e5ohnz5w_tta",
        "none",
        "flips3",
        ("model_b", "model_a"),
        ("flips3", "none"),
    ),
    (
        "A+B TTA/no-TTA",
        "ensemble_2_models_ddp_kpcyjb66_tta_hybrid_e5ohnz5w_no_tta",
        "flips3",
        "none",
        ("model_b", "model_a"),
        ("none", "flips3"),
    ),
    (
        "A+B TTA/TTA",
        "ensemble_2_models_ddp_kpcyjb66_tta_e5ohnz5w_tta",
        "flips3",
        "flips3",
        ("model_b", "model_a"),
        ("flips3", "flips3"),
    ),
)

SINGLE_MODEL_CONFIGURATION_SPECS = (
    (
        "Model A no-TTA",
        "single_ddp_fft_finetuned_kpycyjb66_no_tta_mA_103",
        "none",
        None,
        ("model_a",),
        ("none",),
    ),
    (
        "Model A TTA",
        "single_ddp_fft_finetuned_kpycyjb66_tta_mA_103",
        "flips3",
        None,
        ("model_a",),
        ("flips3",),
    ),
    (
        "Model B no-TTA",
        "best_val_f0_e5ohnz5w_no_tta_mB_103",
        None,
        "none",
        ("model_b",),
        ("none",),
    ),
    (
        "Model B TTA",
        "best_val_f0_e5ohnz5w_tta_mB_103",
        None,
        "flips3",
        ("model_b",),
        ("flips3",),
    ),
)

# This is intentionally a conventional end-to-end benchmark, rather than a
# causal pretraining comparison: PlainConvUNet/random initialization/standard
# eight-way mirror TTA differ from the residual MultiTalentV2 Model-B setup.
# Model B with its recorded flip TTA is the closest available historical
# inference comparator for the standard nnU-Net's native mirror-TTA output.
STANDARD_NNUNET_BENCHMARK_CONFIGURATION_SPECS = (
    (
        "Standard nnU-Net (final, mirror TTA)",
        "standard_nnunet_model_b_matched/checkpoint_final_standard_mirror_tta_fixed_metrics",
        None,
        None,
        (),
        (),
    ),
    (
        "Model B TTA",
        "best_val_f0_e5ohnz5w_tta_mB_103",
        None,
        "flips3",
        ("model_b",),
        ("flips3",),
    ),
)

# This paired control preserves Model B's residual architecture and training
# protocol. The two TTA modes are reported separately so each scratch/pretrained
# contrast keeps inference augmentation fixed. The no-TTA contrast is the
# primary initialization comparison; the TTA contrast is a fixed sensitivity
# analysis, not a setting selected on Phase-2 results.
RANDOM_RESIDUAL_BENCHMARK_CONFIGURATION_SPECS = (
    (
        "Random residual nnU-Net no-TTA",
        "random_residual_nnunet_no_tta_103",
        None,
        "none",
        ("random_residual",),
        ("none",),
    ),
    (
        "Random residual nnU-Net TTA",
        "random_residual_nnunet_tta_103",
        None,
        "flips3",
        ("random_residual",),
        ("flips3",),
    ),
    (
        "Model B no-TTA",
        "best_val_f0_e5ohnz5w_no_tta_mB_103",
        None,
        "none",
        ("model_b",),
        ("none",),
    ),
    (
        "Model B TTA",
        "best_val_f0_e5ohnz5w_tta_mB_103",
        None,
        "flips3",
        ("model_b",),
        ("flips3",),
    ),
)

CONFIGURATION_PRESETS = {
    "ensembles": ENSEMBLE_CONFIGURATION_SPECS,
    "single_models": SINGLE_MODEL_CONFIGURATION_SPECS,
    "tta_ablation": SINGLE_MODEL_CONFIGURATION_SPECS + ENSEMBLE_CONFIGURATION_SPECS,
    "standard_nnunet_benchmark": STANDARD_NNUNET_BENCHMARK_CONFIGURATION_SPECS,
    "random_residual_benchmark": RANDOM_RESIDUAL_BENCHMARK_CONFIGURATION_SPECS,
}

METRICS = (
    ("dice", "Dice+", "unitless", "higher_is_better"),
    ("hd95", "HD95+", "mm", "lower_is_better"),
    ("assd", "ASSD+", "mm", "lower_is_better"),
)


@dataclass(frozen=True)
class Configuration:
    """A labelled report directory and its intended A/B TTA assignment."""

    label: str
    directory: Path
    model_a_tta: str | None
    model_b_tta: str | None
    checkpoint_roles: tuple[str, ...]
    expected_checkpoint_ttas: tuple[str, ...]


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    min_component_voxels: int

    @property
    def display(self) -> str:
        return f"tau={self.threshold:g}, minCC={self.min_component_voxels}"

    @property
    def filename(self) -> str:
        threshold_token = f"{self.threshold:g}".replace("-", "m").replace(".", "p")
        return f"case_metrics_thr{threshold_token}_mincc{self.min_component_voxels}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("checkpoints/Validation2025_100"),
        help="Parent directory containing the case-metric report folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/Validation2025_100/statistics/ensemble_fixed_setting_bootstrap_cis"),
        help="New directory for analysis CSV/JSON outputs.",
    )
    parser.add_argument(
        "--preset",
        choices=tuple(CONFIGURATION_PRESETS),
        default="ensembles",
        help=(
            "Report group to analyse: ensembles (default), single_models, or "
            "tta_ablation (the four single-model plus four ensemble conditions), "
            "standard_nnunet_benchmark (a conventional, non-causal baseline comparison), "
            "or random_residual_benchmark (the matched initialization control)."
        ),
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        default=("0.2:40", "0.5:0"),
        metavar="TAU:MINCC",
        help="Locked operating points. Defaults to 0.2:40 and 0.5:0.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=10_000,
        help="Number of case-resampling bootstrap replicates (default: 10000).",
    )
    parser.add_argument("--seed", type=int, default=42, help="NumPy random seed (default: 42).")
    parser.add_argument(
        "--expected-postprocess",
        default="none",
        help="Required post-processing label in every input case row (default: none).",
    )
    parser.add_argument(
        "--checkpoint-path-alias",
        action="append",
        default=[],
        metavar="RECORDED_PATH=ACTUAL_PATH",
        help=(
            "Map a stale checkpoint path recorded in a historical run_config.json to its actual local copy. "
            "The raw path remains in provenance; the resolved file is SHA-256 checked before use. Repeat if needed."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing files in --output-dir.",
    )
    return parser.parse_args()


def _canonical_tta(value: str) -> str:
    normalized = str(value).lower()
    if normalized in {"flips", "flips3"}:
        return "flips3"
    return normalized


def _output_tta(value: str | None) -> str:
    return _canonical_tta(value) if value is not None else ""


def _parse_operating_point(text: str) -> OperatingPoint:
    match = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*:\s*([0-9]+)\s*", text)
    if match is None:
        raise ValueError(f"Invalid operating point '{text}'. Use TAU:MINCC, for example 0.2:40.")
    threshold = float(match.group(1))
    min_component_voxels = int(match.group(2))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be in [0, 1], got {threshold}.")
    return OperatingPoint(threshold=threshold, min_component_voxels=min_component_voxels)


def _parse_checkpoint_path_aliases(specifications: Iterable[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(
                f"Invalid checkpoint path alias {specification!r}. Use RECORDED_PATH=ACTUAL_PATH."
            )
        recorded_path, actual_path = (part.strip() for part in specification.split("=", 1))
        if not recorded_path or not actual_path:
            raise ValueError(
                f"Invalid checkpoint path alias {specification!r}. Both paths must be non-empty."
            )
        previous = aliases.setdefault(recorded_path, actual_path)
        if previous != actual_path:
            raise ValueError(
                f"Conflicting aliases for recorded checkpoint path {recorded_path!r}: {previous!r} vs {actual_path!r}."
            )
    return aliases


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_case_rows(path: Path, point: OperatingPoint, expected_postprocess: str) -> list[dict[str, str]]:
    required_columns = {
        "case_id",
        "gt_voxels",
        "gt_category",
        "threshold",
        "min_component_voxels",
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
        has_postprocess_column = "postprocess" in reader.fieldnames
        # The native standard-nnU-Net evaluator predates the common per-case
        # schema and omits this column. Its fixed reports have no additional
        # post-processing, so accept that legacy schema only for ``none``.
        if not has_postprocess_column and expected_postprocess.lower() != "none":
            raise ValueError(
                f"{path} omits postprocess; it can only be analysed as postprocess='none', "
                f"not {expected_postprocess!r}."
            )
        missing_columns = sorted(required_columns - set(reader.fieldnames))
        if missing_columns:
            raise ValueError(f"{path} lacks required columns: {missing_columns}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no case rows.")
    case_ids = [row["case_id"] for row in rows]
    if len(case_ids) != len(set(case_ids)):
        duplicates = sorted(case_id for case_id in set(case_ids) if case_ids.count(case_id) > 1)
        raise ValueError(f"{path} contains duplicate case IDs: {duplicates[:10]}")
    for row in rows:
        try:
            threshold = float(row["threshold"])
            min_component_voxels = int(row["min_component_voxels"])
            int(row["gt_voxels"])
            for metric_name, _, _, _ in METRICS:
                if not np.isfinite(float(row[metric_name])):
                    raise ValueError(f"non-finite {metric_name}")
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}: invalid row for case {row.get('case_id')!r}: {error}") from error
        if not np.isclose(threshold, point.threshold, rtol=0.0, atol=1e-12):
            raise ValueError(f"{path}: case {row['case_id']} has threshold {threshold}, expected {point.threshold}.")
        if min_component_voxels != point.min_component_voxels:
            raise ValueError(
                f"{path}: case {row['case_id']} has minCC {min_component_voxels}, "
                f"expected {point.min_component_voxels}."
            )
        observed_postprocess = row.get("postprocess", "none")
        if str(observed_postprocess).lower() != expected_postprocess.lower():
            raise ValueError(
                f"{path}: case {row['case_id']} has postprocess={observed_postprocess!r}, "
                f"expected {expected_postprocess!r}."
            )
    return rows


def _validate_tta_provenance(configuration: Configuration) -> dict[str, object]:
    config_path = configuration.directory / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run provenance: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoints = payload.get("checkpoints", [])
    checkpoint_ttas = payload.get("checkpoint_ttas")
    if checkpoint_ttas is None:
        checkpoint_ttas = [payload.get("tta", "none")] * len(checkpoints)
    if len(checkpoints) != len(configuration.checkpoint_roles):
        raise ValueError(
            f"{config_path}: expected {len(configuration.checkpoint_roles)} checkpoint(s) for "
            f"{configuration.label!r}, got {checkpoints!r}."
        )
    if len(checkpoint_ttas) != len(configuration.expected_checkpoint_ttas):
        raise ValueError(
            f"{config_path}: expected {len(configuration.expected_checkpoint_ttas)} checkpoint TTA values, "
            f"got {checkpoint_ttas!r}."
        )
    observed_ttas = tuple(_canonical_tta(value) for value in checkpoint_ttas)
    expected_ttas = tuple(_canonical_tta(value) for value in configuration.expected_checkpoint_ttas)
    if observed_ttas != expected_ttas:
        role_text = ", ".join(configuration.checkpoint_roles)
        raise ValueError(
            f"{config_path}: TTA assignment is {checkpoint_ttas!r} in [{role_text}] checkpoint order, "
            f"but {configuration.label!r} requires {list(expected_ttas)!r}."
        )
    return {
        "run_config": str(config_path.resolve()),
        "run_config_sha256": _sha256(config_path),
        "checkpoints": checkpoints,
        "checkpoint_roles": list(configuration.checkpoint_roles),
        "checkpoints_by_role": dict(zip(configuration.checkpoint_roles, checkpoints)),
        "checkpoint_ttas_raw": checkpoint_ttas,
        "checkpoint_ttas_canonical": list(observed_ttas),
    }


def _validate_checkpoint_identity(
    configurations: tuple[Configuration, ...],
    provenance: dict[str, dict[str, object]],
    path_aliases: dict[str, str] | None = None,
) -> dict[str, dict[str, object]]:
    """Require any differently named checkpoint copies to be byte-identical.

    A Model-A or Model-B checkpoint may legitimately have been copied or
    renamed between historical report runs. In that case path equality is too
    strict, while filename equality is unsafe. This audit accepts aliases only
    after SHA-256 equality of the actual checkpoint files.
    """
    by_role: dict[str, dict[str, object]] = {}
    digest_cache: dict[str, str] = {}
    path_aliases = path_aliases or {}

    def resolved_path(recorded_path: str) -> str:
        return path_aliases.get(recorded_path, recorded_path)

    def digest(checkpoint_path_text: str) -> str:
        if checkpoint_path_text not in digest_cache:
            checkpoint_path = Path(checkpoint_path_text)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    "Cannot verify that differently named checkpoint copies are identical because "
                    f"the checkpoint is not accessible: {checkpoint_path}. Run this analysis from "
                    "the Linux workspace that contains the archived checkpoints."
                )
            digest_cache[checkpoint_path_text] = _sha256(checkpoint_path)
        return digest_cache[checkpoint_path_text]

    for configuration in configurations:
        checkpoint_map = provenance[configuration.label]["checkpoints_by_role"]
        for role, checkpoint in checkpoint_map.items():
            checkpoint = str(checkpoint)
            if role not in by_role:
                by_role[role] = {
                    "canonical_recorded_checkpoint_path": checkpoint,
                    "canonical_resolved_checkpoint_path": resolved_path(checkpoint),
                    "accepted_recorded_checkpoint_paths": [checkpoint],
                    "accepted_resolved_checkpoint_paths": [resolved_path(checkpoint)],
                    "sha256": None,
                }
                continue
            record = by_role[role]
            canonical_recorded_path = str(record["canonical_recorded_checkpoint_path"])
            canonical_resolved_path = str(record["canonical_resolved_checkpoint_path"])
            candidate_resolved_path = resolved_path(checkpoint)
            if checkpoint == canonical_recorded_path:
                if checkpoint not in record["accepted_recorded_checkpoint_paths"]:
                    record["accepted_recorded_checkpoint_paths"].append(checkpoint)
                if candidate_resolved_path not in record["accepted_resolved_checkpoint_paths"]:
                    record["accepted_resolved_checkpoint_paths"].append(candidate_resolved_path)
                continue
            canonical_hash = digest(canonical_resolved_path)
            checkpoint_hash = digest(candidate_resolved_path)
            if checkpoint_hash != canonical_hash:
                raise ValueError(
                    f"{configuration.label}: {role} checkpoint differs in content from the controlled report set. "
                    f"Reference {canonical_recorded_path} -> {canonical_resolved_path} SHA-256={canonical_hash}; "
                    f"candidate {checkpoint} -> {candidate_resolved_path} SHA-256={checkpoint_hash}."
                )
            record["sha256"] = canonical_hash
            if checkpoint not in record["accepted_recorded_checkpoint_paths"]:
                record["accepted_recorded_checkpoint_paths"].append(checkpoint)
            if candidate_resolved_path not in record["accepted_resolved_checkpoint_paths"]:
                record["accepted_resolved_checkpoint_paths"].append(candidate_resolved_path)
    return by_role


def _reference_case_metadata(rows: Iterable[dict[str, str]]) -> dict[str, tuple[str, str]]:
    return {row["case_id"]: (row["gt_voxels"], row["gt_category"]) for row in rows}


def _percentile_bootstrap_ci(values: np.ndarray, indices: np.ndarray) -> tuple[float, float, float]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Bootstrap values must be a non-empty one-dimensional array.")
    if indices.ndim != 2 or indices.shape[1] != values.size:
        raise ValueError("Bootstrap index matrix does not match the number of observations.")
    bootstrap_means = values[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, (0.025, 0.975), method="linear")
    return float(values.mean()), float(lower), float(upper)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty result table: {path}")
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
    if args.bootstrap_replicates < 1_000:
        raise ValueError("Use at least 1,000 bootstrap replicates; 10,000 is the paper default.")

    report_root = args.report_root.resolve()
    if not report_root.is_dir():
        raise FileNotFoundError(f"Report root not found: {report_root}")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Pass --overwrite to replace its outputs.")
    output_dir.mkdir(parents=True, exist_ok=True)

    points = tuple(_parse_operating_point(text) for text in args.settings)
    if len(set(points)) != len(points):
        raise ValueError("Each operating point must be listed once.")
    configurations = tuple(
        Configuration(
            label=label,
            directory=report_root / folder,
            model_a_tta=model_a_tta,
            model_b_tta=model_b_tta,
            checkpoint_roles=checkpoint_roles,
            expected_checkpoint_ttas=expected_checkpoint_ttas,
        )
        for label, folder, model_a_tta, model_b_tta, checkpoint_roles, expected_checkpoint_ttas
        in CONFIGURATION_PRESETS[args.preset]
    )
    for configuration in configurations:
        if not configuration.directory.is_dir():
            raise FileNotFoundError(f"Missing report directory for {configuration.label}: {configuration.directory}")

    checkpoint_path_aliases = _parse_checkpoint_path_aliases(args.checkpoint_path_alias)
    provenance = {configuration.label: _validate_tta_provenance(configuration) for configuration in configurations}
    checkpoint_identity = _validate_checkpoint_identity(configurations, provenance, checkpoint_path_aliases)
    result_rows: list[dict[str, object]] = []
    input_hashes: dict[str, str] = {}
    case_manifest_rows: list[dict[str, object]] = []

    for point_index, point in enumerate(points):
        rows_by_label: dict[str, list[dict[str, str]]] = {}
        for configuration in configurations:
            metric_path = configuration.directory / point.filename
            if not metric_path.is_file():
                raise FileNotFoundError(
                    f"Missing fixed-setting case metrics for {configuration.label}: {metric_path}. "
                    "Do not substitute a different operating point."
                )
            rows_by_label[configuration.label] = _read_case_rows(
                metric_path,
                point,
                args.expected_postprocess,
            )
            input_hashes[str(metric_path.resolve())] = _sha256(metric_path)

        reference_label = configurations[0].label
        reference_rows = rows_by_label[reference_label]
        reference_metadata = _reference_case_metadata(reference_rows)
        for configuration in configurations[1:]:
            current_metadata = _reference_case_metadata(rows_by_label[configuration.label])
            if current_metadata != reference_metadata:
                reference_ids = set(reference_metadata)
                current_ids = set(current_metadata)
                missing = sorted(reference_ids - current_ids)
                extra = sorted(current_ids - reference_ids)
                changed = sorted(
                    case_id
                    for case_id in reference_ids & current_ids
                    if reference_metadata[case_id] != current_metadata[case_id]
                )
                raise ValueError(
                    f"{point.display}: case/ground-truth mismatch for {configuration.label}. "
                    f"missing={missing[:10]}, extra={extra[:10]}, changed_ground_truth={changed[:10]}"
                )

        positive_case_ids = sorted(
            case_id for case_id, (gt_voxels, _) in reference_metadata.items() if int(gt_voxels) > 0
        )
        empty_case_ids = sorted(
            case_id for case_id, (gt_voxels, _) in reference_metadata.items() if int(gt_voxels) == 0
        )
        if not positive_case_ids:
            raise ValueError(f"{point.display}: no ground-truth-positive cases available for Dice+/HD95+/ASSD+.")
        if len(positive_case_ids) + len(empty_case_ids) != len(reference_metadata):
            raise ValueError(f"{point.display}: found a negative ground-truth voxel count.")

        # One deterministic set of case-resampling indices is shared across all
        # configurations and all three endpoints at this operating point.
        # This does not perform paired inference; it merely keeps every reported
        # single-model CI reproducible under identical resample weights.
        setting_rng = np.random.default_rng(np.random.SeedSequence((args.seed, point_index)))
        bootstrap_indices = setting_rng.integers(
            0,
            len(positive_case_ids),
            size=(args.bootstrap_replicates, len(positive_case_ids)),
        )

        for case_id in positive_case_ids:
            gt_voxels, gt_category = reference_metadata[case_id]
            case_manifest_rows.append(
                {
                    "operating_point": point.display,
                    "case_id": case_id,
                    "gt_voxels": int(gt_voxels),
                    "gt_category": gt_category,
                    "analysis_subset": "ground_truth_positive",
                }
            )

        for configuration in configurations:
            rows_by_case = {row["case_id"]: row for row in rows_by_label[configuration.label]}
            for metric_key, metric_label, unit, direction in METRICS:
                values = np.asarray(
                    [float(rows_by_case[case_id][metric_key]) for case_id in positive_case_ids], dtype=np.float64
                )
                mean, ci_lower, ci_upper = _percentile_bootstrap_ci(values, bootstrap_indices)
                result_rows.append(
                    {
                        "configuration": configuration.label,
                        "model_a_tta": _output_tta(configuration.model_a_tta),
                        "model_b_tta": _output_tta(configuration.model_b_tta),
                        "threshold": point.threshold,
                        "min_component_voxels": point.min_component_voxels,
                        "operating_point": point.display,
                        "metric": metric_label,
                        "metric_key": metric_key,
                        "unit": unit,
                        "direction": direction,
                        "mean": mean,
                        "ci_95_lower": ci_lower,
                        "ci_95_upper": ci_upper,
                        "n_positive": len(positive_case_ids),
                        "n_empty_excluded": len(empty_case_ids),
                        "bootstrap_replicates": args.bootstrap_replicates,
                        "bootstrap_seed": args.seed,
                        "ci_method": "percentile bootstrap (case resampling with replacement)",
                        "analysis_subset": "ground_truth_positive (gt_voxels > 0)",
                    }
                )

    _write_csv(output_dir / "bootstrap_ci_summary.csv", result_rows)
    _write_csv(output_dir / "positive_case_manifest.csv", case_manifest_rows)
    run_config = {
        "analysis": "single-configuration percentile bootstrap confidence intervals; no paired hypothesis tests",
        "preset": args.preset,
        "report_root": str(report_root),
        "output_dir": str(output_dir),
        "operating_points": [
            {"threshold": point.threshold, "min_component_voxels": point.min_component_voxels}
            for point in points
        ],
        "configurations": [
            {
                "label": configuration.label,
                "report_directory": str(configuration.directory.resolve()),
                "model_a_tta": _output_tta(configuration.model_a_tta),
                "model_b_tta": _output_tta(configuration.model_b_tta),
                "provenance": provenance[configuration.label],
            }
            for configuration in configurations
        ],
        "checkpoint_identity_by_role": checkpoint_identity,
        "checkpoint_path_aliases": checkpoint_path_aliases,
        "metrics": [
            {"metric_key": key, "metric": label, "unit": unit, "direction": direction}
            for key, label, unit, direction in METRICS
        ],
        "case_subset": "ground_truth_positive (gt_voxels > 0)",
        "expected_postprocess": args.expected_postprocess,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.seed,
        "confidence_level": 0.95,
        "interval_method": "percentile bootstrap (2.5th and 97.5th percentiles)",
        "resampling_unit": "case ID; sampled with replacement within each fixed operating point",
        "selection_note": "This script evaluates only the supplied locked operating points and does not select models, thresholds, or minCC values.",
        "input_sha256": input_hashes,
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)
        handle.write("\n")

    print(f"[DONE] Wrote {len(result_rows)} CI rows to {output_dir / 'bootstrap_ci_summary.csv'}")
    print(f"[DONE] Wrote positive-case manifest to {output_dir / 'positive_case_manifest.csv'}")
    print("[INFO] No inference, training, threshold search, or paired tests were performed.")


if __name__ == "__main__":
    main()
