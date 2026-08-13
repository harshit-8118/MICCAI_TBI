#!/usr/bin/env python3
"""Paired bootstrap and permutation analyses for locked case-level comparisons.

Its default analysis reads the eight fixed-setting A/B reports:

* Model A, with and without TTA
* Model B, with and without TTA
* the four A+B TTA-placement combinations

For each pre-specified contrast and each ground-truth-positive endpoint
(Dice+, HD95+, ASSD+), it writes:

* mean paired difference (left configuration minus right configuration),
* percentile 95% paired-bootstrap confidence interval,
* two-sided paired label-swap permutation p-value, and
* Holm-adjusted p-value within the metric/operating-point contrast family.

The default analysis is the controlled TTA ablation. ``--analysis
standard_nnunet_benchmark`` instead tests the conventional standard nnU-Net
mirror-TTA baseline against Model B with recorded TTA. That latter comparison
is deliberately labelled non-causal because both architecture and
initialization differ.

The script evaluates only locked operating points. It never loads models,
runs inference, selects a model, or searches thresholds/minCC values.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from bootstrap_fixed_setting_cis import (
    CONFIGURATION_PRESETS,
    METRICS,
    Configuration,
    _parse_operating_point,
    _parse_checkpoint_path_aliases,
    _read_case_rows,
    _sha256,
    _validate_checkpoint_identity,
    _validate_tta_provenance,
    _write_csv,
)


TTA_ABLATION_CONTRASTS = (
    (
        "single_a_tta_effect",
        "Model A TTA - Model A no-TTA",
        "Model A TTA",
        "Model A no-TTA",
        "TTA effect for Model A alone",
    ),
    (
        "single_b_tta_effect",
        "Model B TTA - Model B no-TTA",
        "Model B TTA",
        "Model B no-TTA",
        "TTA effect for Model B alone",
    ),
    (
        "ensemble_effect_vs_model_a",
        "A+B no-TTA/no-TTA - Model A no-TTA",
        "A+B no-TTA/no-TTA",
        "Model A no-TTA",
        "Ensemble effect relative to Model A without TTA",
    ),
    (
        "ensemble_effect_vs_model_b",
        "A+B no-TTA/no-TTA - Model B no-TTA",
        "A+B no-TTA/no-TTA",
        "Model B no-TTA",
        "Ensemble effect relative to Model B without TTA",
    ),
    (
        "ensemble_b_tta_given_a_no_tta",
        "A+B no-TTA/TTA - A+B no-TTA/no-TTA",
        "A+B no-TTA/TTA (submitted)",
        "A+B no-TTA/no-TTA",
        "Model B TTA effect within ensemble when Model A has no TTA",
    ),
    (
        "ensemble_a_tta_given_b_no_tta",
        "A+B TTA/no-TTA - A+B no-TTA/no-TTA",
        "A+B TTA/no-TTA",
        "A+B no-TTA/no-TTA",
        "Model A TTA effect within ensemble when Model B has no TTA",
    ),
    (
        "ensemble_a_tta_given_b_tta",
        "A+B TTA/TTA - A+B no-TTA/TTA",
        "A+B TTA/TTA",
        "A+B no-TTA/TTA (submitted)",
        "Model A TTA effect within ensemble when Model B has TTA",
    ),
    (
        "ensemble_b_tta_given_a_tta",
        "A+B TTA/TTA - A+B TTA/no-TTA",
        "A+B TTA/TTA",
        "A+B TTA/no-TTA",
        "Model B TTA effect within ensemble when Model A has TTA",
    ),
    (
        "ensemble_tta_placement",
        "A+B no-TTA/TTA - A+B TTA/no-TTA",
        "A+B no-TTA/TTA (submitted)",
        "A+B TTA/no-TTA",
        "Direct asymmetric TTA-placement comparison",
    ),
)


STANDARD_NNUNET_BENCHMARK_CONTRASTS = (
    (
        "standard_nnunet_vs_model_b_tta",
        "Standard nnU-Net (final, mirror TTA) - Model B TTA",
        "Standard nnU-Net (final, mirror TTA)",
        "Model B TTA",
        (
            "Conventional end-to-end benchmark; architecture, initialization, and native "
            "TTA scheme differ, so this does not isolate pretraining"
        ),
    ),
)


RANDOM_RESIDUAL_BENCHMARK_CONTRASTS = (
    (
        "model_b_vs_random_residual_no_tta",
        "Model B no-TTA - Random residual nnU-Net no-TTA",
        "Model B no-TTA",
        "Random residual nnU-Net no-TTA",
        (
            "Primary initialization comparison: identical residual architecture, training protocol, "
            "and no-TTA inference; only MultiTalentV2 initialization differs"
        ),
    ),
    (
        "model_b_vs_random_residual_tta",
        "Model B TTA - Random residual nnU-Net TTA",
        "Model B TTA",
        "Random residual nnU-Net TTA",
        (
            "Fixed TTA sensitivity analysis for the initialization comparison; residual architecture "
            "and TTA inference are held matched"
        ),
    ),
)


ANALYSIS_SPECS = {
    "tta_ablation": {
        "preset": "tta_ablation",
        "contrasts": TTA_ABLATION_CONTRASTS,
        "description": "paired fixed-setting TTA/ensemble effects",
        "multiplicity_family": "pre-specified TTA/ensemble contrasts",
    },
    "standard_nnunet_benchmark": {
        "preset": "standard_nnunet_benchmark",
        "contrasts": STANDARD_NNUNET_BENCHMARK_CONTRASTS,
        "description": "paired fixed-setting conventional standard-nnU-Net versus Model-B benchmark",
        "multiplicity_family": "pre-specified conventional baseline contrast",
    },
    "random_residual_benchmark": {
        "preset": "random_residual_benchmark",
        "contrasts": RANDOM_RESIDUAL_BENCHMARK_CONTRASTS,
        "description": "paired fixed-setting matched random-residual versus MultiTalentV2 Model-B initialization comparison",
        "multiplicity_family": "pre-specified matched initialization contrasts",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("checkpoints/Validation2025_100"),
        help="Parent directory containing the required case-metric report folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/Validation2025_100/statistics/tta_ablation_paired_tests"),
        help="New directory for analysis CSV/JSON outputs.",
    )
    parser.add_argument(
        "--analysis",
        choices=tuple(ANALYSIS_SPECS),
        default="tta_ablation",
        help=(
            "Analysis family: tta_ablation (default), standard_nnunet_benchmark, or "
            "random_residual_benchmark. The standard nnU-Net analysis is conventional/non-causal; "
            "the random-residual analysis is the matched initialization control."
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
        help="Paired bootstrap replicates (default: 10000).",
    )
    parser.add_argument(
        "--permutation-replicates",
        type=int,
        default=10_000,
        help="Two-sided paired label-swap permutations (default: 10000).",
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
            "The resolved file is SHA-256 checked before use. Repeat if needed."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing files in --output-dir.",
    )
    return parser.parse_args()


def _reference_case_metadata(rows: Iterable[dict[str, str]]) -> dict[str, tuple[str, str]]:
    return {row["case_id"]: (row["gt_voxels"], row["gt_category"]) for row in rows}


def _paired_bootstrap_ci(differences: np.ndarray, indices: np.ndarray) -> tuple[float, float, float]:
    if differences.ndim != 1 or differences.size == 0:
        raise ValueError("Paired differences must be a non-empty one-dimensional array.")
    if indices.ndim != 2 or indices.shape[1] != differences.size:
        raise ValueError("Paired bootstrap index matrix does not match the case count.")
    bootstrap_means = differences[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, (0.025, 0.975), method="linear")
    return float(differences.mean()), float(lower), float(upper)


def _paired_label_swap_p_value(differences: np.ndarray, rng: np.random.Generator, replicates: int) -> float:
    """Two-sided randomization p-value under within-case model-label exchangeability."""
    observed = abs(float(differences.mean()))
    if np.allclose(differences, 0.0, rtol=0.0, atol=0.0):
        return 1.0
    signs = rng.choice(np.array((-1.0, 1.0), dtype=np.float64), size=(replicates, differences.size))
    null_means = (signs * differences).mean(axis=1)
    # Add one to numerator and denominator for a valid finite-resampling p-value.
    return float((1 + np.count_nonzero(np.abs(null_means) >= observed)) / (replicates + 1))


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down family-wise adjustment, preserving original row order."""
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running_maximum = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running_maximum = max(running_maximum, value)
        adjusted[index] = running_maximum
    return adjusted


def _make_configurations(report_root: Path, preset: str) -> tuple[Configuration, ...]:
    return tuple(
        Configuration(
            label=label,
            directory=report_root / folder,
            model_a_tta=model_a_tta,
            model_b_tta=model_b_tta,
            checkpoint_roles=checkpoint_roles,
            expected_checkpoint_ttas=expected_checkpoint_ttas,
        )
        for label, folder, model_a_tta, model_b_tta, checkpoint_roles, expected_checkpoint_ttas
        in CONFIGURATION_PRESETS[preset]
    )


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1_000 or args.permutation_replicates < 1_000:
        raise ValueError("Use at least 1,000 bootstrap and permutation replicates; 10,000 is the paper default.")

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
    analysis_spec = ANALYSIS_SPECS[args.analysis]
    contrasts = analysis_spec["contrasts"]
    configurations = _make_configurations(report_root, str(analysis_spec["preset"]))
    configuration_by_label = {configuration.label: configuration for configuration in configurations}
    if len(configuration_by_label) != len(configurations):
        raise RuntimeError("Internal configuration labels are not unique.")
    for configuration in configurations:
        if not configuration.directory.is_dir():
            raise FileNotFoundError(f"Missing report directory for {configuration.label}: {configuration.directory}")
    for _, _, left, right, _ in contrasts:
        if left not in configuration_by_label or right not in configuration_by_label:
            raise RuntimeError(f"Internal contrast refers to an unknown configuration: {left!r} vs {right!r}.")

    checkpoint_path_aliases = _parse_checkpoint_path_aliases(args.checkpoint_path_alias)
    provenance = {configuration.label: _validate_tta_provenance(configuration) for configuration in configurations}
    checkpoint_identity = _validate_checkpoint_identity(configurations, provenance, checkpoint_path_aliases)
    input_hashes: dict[str, str] = {}
    result_rows: list[dict[str, object]] = []
    positive_case_manifest: list[dict[str, object]] = []

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
        reference_metadata = _reference_case_metadata(rows_by_label[reference_label])
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
        empty_count = sum(int(gt_voxels) == 0 for gt_voxels, _ in reference_metadata.values())
        if not positive_case_ids:
            raise ValueError(f"{point.display}: no ground-truth-positive cases available for paired metrics.")
        if len(positive_case_ids) + empty_count != len(reference_metadata):
            raise ValueError(f"{point.display}: found a negative ground-truth voxel count.")

        for case_id in positive_case_ids:
            gt_voxels, gt_category = reference_metadata[case_id]
            positive_case_manifest.append(
                {
                    "operating_point": point.display,
                    "case_id": case_id,
                    "gt_voxels": int(gt_voxels),
                    "gt_category": gt_category,
                    "analysis_subset": "ground_truth_positive",
                }
            )

        rows_by_case = {
            label: {row["case_id"]: row for row in rows}
            for label, rows in rows_by_label.items()
        }
        setting_seed = np.random.SeedSequence((args.seed, point_index))
        bootstrap_rng, permutation_rng = (np.random.default_rng(child) for child in setting_seed.spawn(2))
        bootstrap_indices = bootstrap_rng.integers(
            0,
            len(positive_case_ids),
            size=(args.bootstrap_replicates, len(positive_case_ids)),
        )

        for metric_index, (metric_key, metric_label, unit, direction) in enumerate(METRICS):
            metric_rows: list[dict[str, object]] = []
            for contrast_index, (contrast_id, contrast_label, left, right, scientific_question) in enumerate(contrasts):
                left_values = np.asarray(
                    [float(rows_by_case[left][case_id][metric_key]) for case_id in positive_case_ids],
                    dtype=np.float64,
                )
                right_values = np.asarray(
                    [float(rows_by_case[right][case_id][metric_key]) for case_id in positive_case_ids],
                    dtype=np.float64,
                )
                differences = left_values - right_values
                mean_difference, ci_lower, ci_upper = _paired_bootstrap_ci(differences, bootstrap_indices)
                comparison_rng = np.random.default_rng(
                    np.random.SeedSequence((args.seed, point_index, metric_index, contrast_index, 1))
                )
                p_value = _paired_label_swap_p_value(
                    differences,
                    comparison_rng,
                    args.permutation_replicates,
                )
                metric_rows.append(
                    {
                        "contrast_id": contrast_id,
                        "contrast": contrast_label,
                        "scientific_question": scientific_question,
                        "left_configuration": left,
                        "right_configuration": right,
                        "effect_definition": "mean(left per-case metric - right per-case metric)",
                        "threshold": point.threshold,
                        "min_component_voxels": point.min_component_voxels,
                        "operating_point": point.display,
                        "metric": metric_label,
                        "metric_key": metric_key,
                        "unit": unit,
                        "direction": direction,
                        "mean_left": float(left_values.mean()),
                        "mean_right": float(right_values.mean()),
                        "mean_difference": mean_difference,
                        "ci_95_lower": ci_lower,
                        "ci_95_upper": ci_upper,
                        "p_value_two_sided_label_swap": p_value,
                        "n_positive": len(positive_case_ids),
                        "n_empty_excluded": empty_count,
                        "bootstrap_replicates": args.bootstrap_replicates,
                        "permutation_replicates": args.permutation_replicates,
                        "seed": args.seed,
                        "ci_method": "paired percentile bootstrap (case IDs resampled with replacement)",
                        "test_method": "two-sided paired label-swap permutation test",
                        "analysis_subset": "ground_truth_positive (gt_voxels > 0)",
                    }
                )
            adjusted = _holm_adjust([float(row["p_value_two_sided_label_swap"]) for row in metric_rows])
            for row, adjusted_p_value in zip(metric_rows, adjusted):
                row["p_value_holm"] = adjusted_p_value
                row["holm_family"] = (
                    f"{point.display}; {metric_label}; {len(metric_rows)} "
                    f"{analysis_spec['multiplicity_family']}"
                )
            result_rows.extend(metric_rows)

    _write_csv(output_dir / "paired_effects_summary.csv", result_rows)
    _write_csv(output_dir / "positive_case_manifest.csv", positive_case_manifest)
    run_config = {
        "analysis": f"{analysis_spec['description']} with percentile bootstrap CIs and two-sided paired label-swap tests",
        "analysis_id": args.analysis,
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
                "model_a_tta": configuration.model_a_tta or "",
                "model_b_tta": configuration.model_b_tta or "",
                "provenance": provenance[configuration.label],
            }
            for configuration in configurations
        ],
        "checkpoint_identity_by_role": checkpoint_identity,
        "checkpoint_path_aliases": checkpoint_path_aliases,
        "contrasts": [
            {
                "contrast_id": contrast_id,
                "contrast": contrast_label,
                "left_configuration": left,
                "right_configuration": right,
                "scientific_question": scientific_question,
            }
            for contrast_id, contrast_label, left, right, scientific_question in contrasts
        ],
        "metrics": [
            {"metric_key": key, "metric": label, "unit": unit, "direction": direction}
            for key, label, unit, direction in METRICS
        ],
        "case_subset": "ground_truth_positive (gt_voxels > 0)",
        "expected_postprocess": args.expected_postprocess,
        "bootstrap_replicates": args.bootstrap_replicates,
        "permutation_replicates": args.permutation_replicates,
        "seed": args.seed,
        "confidence_level": 0.95,
        "interval_method": "percentile paired bootstrap (2.5th and 97.5th percentiles)",
        "test_method": "two-sided within-case label-swap permutation test",
        "multiplicity": (
            f"Holm adjustment separately for the {len(contrasts)} "
            f"{analysis_spec['multiplicity_family']} within each metric and operating point"
        ),
        "selection_note": "This script evaluates only the supplied locked operating points and does not select models, thresholds, or minCC values.",
        "input_sha256": input_hashes,
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)
        handle.write("\n")

    print(f"[DONE] Wrote {len(result_rows)} paired-effect rows to {output_dir / 'paired_effects_summary.csv'}")
    print(f"[DONE] Wrote positive-case manifest to {output_dir / 'positive_case_manifest.csv'}")
    print("[INFO] No inference, training, threshold search, or checkpoint modification was performed.")


if __name__ == "__main__":
    main()
