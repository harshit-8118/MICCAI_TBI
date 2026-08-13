"""Evaluate nnU-Net probability exports at fixed operating points.

By default, this uses the same canonical 1-mm target grid and local empty-mask
surface convention as evaluate_external_validation.py. For internal nnU-Net
monitoring, --reference-label-dir instead enforces a direct comparison on the
prepared native T1/prediction grid. The script is intentionally limited to
fixed thresholds/minimum-component sizes: do not use it to search the released
Phase-2 validation set for a new operating point.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output

from evaluate_external_validation import (
    _discover_validation_cases,
    _metrics_from_rows,
    _surface_metrics,
    _write_dict_rows,
)
from multitalent_tbi.case_filters import build_case_infos, lesion_category
from multitalent_tbi.data import load_case
from multitalent_tbi.engine import dice_score
from multitalent_tbi.postprocessing import postprocess_prediction, setting_tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory containing T1/lesion pairs to evaluate.")
    parser.add_argument("--prediction-dir", type=Path, required=True, help="nnU-Net output directory with .nii.gz and .npz files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-label-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing binary reference labels named <case_id>.nii.gz on the "
            "same native grid as the exported nnU-Net predictions. Use the prepared Dataset501 "
            "labelsTr directory for internal fold monitoring."
        ),
    )
    parser.add_argument(
        "--case-ids-json",
        type=Path,
        default=None,
        help="Optional JSON list (or {case_ids: [...]}) restricting evaluation to named cases.",
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.20, 0.50])
    parser.add_argument("--min-components", nargs="+", type=int, default=[40, 0])
    parser.add_argument("--target-spacing", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument("--probability-key", default="probabilities")
    parser.add_argument(
        "--probability-axis-order",
        choices=("auto", "nibabel_xyz", "simpleitk_zyx"),
        default="auto",
        help=(
            "Axis order of saved nnU-Net probabilities. 'auto' chooses the orientation that "
            "reproduces the corresponding exported segmentation and refuses an ambiguous export."
        ),
    )
    parser.add_argument(
        "--prediction-source",
        choices=("probability", "exported_segmentation"),
        default="probability",
        help=(
            "Use probability maps for fixed operating-point evaluation (default), or the exported "
            "nnU-Net segmentation masks for a tau=0.5 internal sanity monitor."
        ),
    )
    parser.add_argument(
        "--metric-prefix",
        default="released_validation",
        help="CSV metric prefix. Keep the default for the released Phase-2 benchmark.",
    )
    parser.add_argument(
        "--split-name",
        default="released_validation",
        help="Case-info split label recorded in outputs.",
    )
    parser.add_argument(
        "--interpretation",
        default="Local released-validation metrics; not an independent test.",
        help="Exact interpretation written to run_config.json.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _binary_dice(first: np.ndarray, second: np.ndarray) -> float:
    first_mask = np.asarray(first, dtype=bool)
    second_mask = np.asarray(second, dtype=bool)
    denominator = int(first_mask.sum()) + int(second_mask.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(first_mask, second_mask).sum() / denominator)


def _orient_probability_to_exported_segmentation(
    probabilities: np.ndarray,
    segmentation: nib.Nifti1Image,
    requested_order: str,
    case_id: str,
) -> tuple[np.ndarray, str]:
    """Convert saved nnU-Net probabilities to the exported NIfTI voxel order.

    nnU-Net's default SimpleITK reader uses ZYX arrays, whereas nibabel exposes
    NIfTI voxel data in XYZ order. The exact array convention can vary with the
    nnU-Net reader/writer combination, so ``auto`` exhaustively checks the
    spatial axis permutations and flips that preserve the exported shape. The
    exported argmax segmentation is the local, label-free invariant used to
    select and verify the conversion. This is stricter and more appropriate
    than requiring foreground probability >= 0.5 to match the exported mask:
    nnU-Net exports argmax labels, while a user-selected operating threshold
    need not be 0.5 after probability resampling.
    """
    expected_shape = tuple(int(value) for value in segmentation.shape[:3])
    exported_segmentation = np.asarray(segmentation.dataobj)
    if not np.all(np.isfinite(exported_segmentation)):
        raise ValueError(f"{case_id}: exported segmentation contains non-finite values.")
    exported_segmentation = np.rint(exported_segmentation).astype(np.int16, copy=False)
    raw_probabilities = np.asarray(probabilities, dtype=np.float32)
    if raw_probabilities.ndim != 4 or raw_probabilities.shape[0] < 2:
        raise ValueError(
            f"{case_id}: expected CxXxYxZ probabilities with two classes, got {raw_probabilities.shape}."
        )
    raw_shape = tuple(int(value) for value in raw_probabilities.shape[1:])
    candidates: list[tuple[str, np.ndarray]] = []

    # Preserve explicit, documented choices for callers that need them. The
    # robust full search is deliberately used only by the default auto mode.
    if requested_order == "nibabel_xyz":
        if raw_shape == expected_shape:
            candidates.append(("nibabel_xyz", raw_probabilities))
    elif requested_order == "simpleitk_zyx":
        reversed_probabilities = raw_probabilities.transpose(0, 3, 2, 1)
        if tuple(reversed_probabilities.shape[1:]) == expected_shape:
            candidates.append(("simpleitk_zyx", reversed_probabilities))
    else:
        for permutation in itertools.permutations(range(3)):
            reordered = raw_probabilities.transpose((0,) + tuple(axis + 1 for axis in permutation))
            if tuple(reordered.shape[1:]) != expected_shape:
                continue
            for flip_flags in itertools.product((False, True), repeat=3):
                slices = (slice(None),) + tuple(slice(None, None, -1) if flag else slice(None) for flag in flip_flags)
                candidate = reordered[slices]
                name = "auto_perm_{}{}{}_flip{}".format(
                    permutation[0], permutation[1], permutation[2], "".join("1" if flag else "0" for flag in flip_flags)
                )
                candidates.append((name, candidate))
    if not candidates:
        raise ValueError(
            f"{case_id}: probability shape {raw_shape} cannot be mapped to exported NIfTI shape "
            f"{expected_shape} using requested order '{requested_order}'."
        )

    scored: list[tuple[float, float, str, np.ndarray]] = []
    for name, candidate in candidates:
        argmax_dice = _binary_dice(np.argmax(candidate, axis=0) > 0, exported_segmentation > 0)
        tau_half_dice = _binary_dice(candidate[1] >= 0.5, exported_segmentation > 0)
        scored.append((argmax_dice, tau_half_dice, name, candidate))
    agreement, tau_half_agreement, selected_name, selected_probabilities = max(scored, key=lambda item: item[0])
    if agreement < 0.999:
        best_half_agreement, best_half_name = max((score, name) for _, score, name, _ in scored)
        top_argmax = sorted(scored, reverse=True, key=lambda item: item[0])[:3]
        argmax_scores = ", ".join(f"{name}={score:.6f}" for score, _, name, _ in top_argmax)
        raise ValueError(
            f"{case_id}: saved probability map does not reproduce its exported segmentation by argmax "
            f"(best: {argmax_scores}; best foreground>=0.5: {best_half_name}={best_half_agreement:.6f}). "
            "Refusing to evaluate potentially mixed or malformed prediction artifacts."
        )
    if tau_half_agreement < 0.999:
        print(
            f"[PROBABILITY AUDIT] {case_id}: {selected_name} reproduces the exported segmentation by argmax "
            f"(Dice={agreement:.6f}), while foreground>=0.5 agreement is {tau_half_agreement:.6f}. "
            "Proceeding because the exported nnU-Net segmentation is argmax-based."
        )
    return np.asarray(selected_probabilities[1], dtype=np.float32), selected_name


def _load_native_probability(
    prediction_dir: Path,
    case_id: str,
    key: str,
    requested_order: str,
) -> tuple[np.ndarray, nib.Nifti1Image, str]:
    npz_path = prediction_dir / f"{case_id}.npz"
    segmentation_path = prediction_dir / f"{case_id}.nii.gz"
    if not npz_path.exists() or not segmentation_path.exists():
        raise FileNotFoundError(
            f"Missing nnU-Net outputs for {case_id}. Expected {npz_path.name} and {segmentation_path.name}. "
            "Run nnUNetv2_predict with --save_probabilities."
        )
    with np.load(npz_path, allow_pickle=False) as data:
        if key not in data:
            raise KeyError(f"{npz_path} lacks '{key}'. Available keys: {list(data.keys())}")
        probabilities = np.asarray(data[key], dtype=np.float32)
    segmentation = nib.load(str(segmentation_path))
    if probabilities.ndim != 4 or probabilities.shape[0] < 2:
        raise ValueError(f"{npz_path}: expected CxXxYxZ probability array with at least two classes, got {probabilities.shape}.")
    if not np.isfinite(probabilities).all() or probabilities.min() < -1e-4 or probabilities.max() > 1.0001:
        raise ValueError(f"{npz_path}: invalid foreground probabilities.")
    probability, resolved_order = _orient_probability_to_exported_segmentation(
        np.clip(probabilities, 0.0, 1.0),
        segmentation,
        requested_order,
        case_id,
    )
    return probability, segmentation, resolved_order


def _load_exported_segmentation(prediction_dir: Path, case_id: str) -> nib.Nifti1Image:
    segmentation_path = prediction_dir / f"{case_id}.nii.gz"
    if not segmentation_path.is_file():
        raise FileNotFoundError(f"Missing exported nnU-Net segmentation for {case_id}: {segmentation_path}")
    segmentation = nib.load(str(segmentation_path))
    if len(segmentation.shape) != 3:
        raise ValueError(f"{segmentation_path}: expected a 3-D exported segmentation, got {segmentation.shape}.")
    return segmentation


def _load_resampled_probability(
    prediction_dir: Path,
    case_id: str,
    key: str,
    spacing: tuple[float, float, float],
    requested_order: str,
) -> tuple[np.ndarray, str]:
    probability, segmentation, resolved_order = _load_native_probability(
        prediction_dir,
        case_id,
        key,
        requested_order,
    )
    probability_image = nib.Nifti1Image(probability, segmentation.affine)
    probability_image = nib.as_closest_canonical(probability_image)
    resampled = resample_to_output(probability_image, voxel_sizes=spacing, order=1)
    resampled_probability = np.asarray(resampled.dataobj, dtype=np.float32)
    if not np.isfinite(resampled_probability).all() or resampled_probability.min() < -1e-4 or resampled_probability.max() > 1.0001:
        raise ValueError(f"{case_id}: invalid foreground probabilities after resampling.")
    return np.clip(resampled_probability, 0.0, 1.0), resolved_order


def _load_native_reference_label(
    reference_label_dir: Path,
    case_id: str,
    prediction_segmentation: nib.Nifti1Image,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    label_path = reference_label_dir / f"{case_id}.nii.gz"
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing native reference label for {case_id}: {label_path}")
    label_image = nib.load(str(label_path))
    label = np.asarray(label_image.dataobj, dtype=np.float32)
    if label.ndim != 3:
        raise ValueError(f"{label_path}: expected a 3-D reference label, got shape {label.shape}.")
    if tuple(label.shape) != tuple(prediction_segmentation.shape[:3]):
        raise ValueError(
            f"{case_id}: reference-label shape {label.shape} does not match nnU-Net prediction grid "
            f"{prediction_segmentation.shape[:3]}. Prepare labels on the T1/prediction grid first."
        )
    if not np.allclose(label_image.affine, prediction_segmentation.affine, rtol=1e-5, atol=1e-4):
        raise ValueError(
            f"{case_id}: reference-label affine does not match the nnU-Net prediction affine. "
            "Refusing a voxelwise comparison on different grids."
        )
    if not np.isfinite(label).all():
        raise ValueError(f"{label_path}: reference label contains non-finite values.")
    spacing = tuple(float(value) for value in nib.affines.voxel_sizes(prediction_segmentation.affine))
    return (label > 0.5).astype(np.uint8), label_image.affine, spacing


def _requested_case_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    values = payload.get("case_ids") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{path} must contain a JSON string list or {{'case_ids': [...]}}.")
    return set(values)


def main() -> None:
    args = parse_args()
    if len(args.thresholds) != len(args.min_components):
        raise ValueError("Pass exactly one --min-components value per --thresholds value.")
    if args.prediction_source == "exported_segmentation":
        if args.reference_label_dir is None:
            raise ValueError("--prediction-source exported_segmentation requires --reference-label-dir.")
        if any(not np.isclose(threshold, 0.5) for threshold in args.thresholds):
            raise ValueError(
                "Exported segmentation masks are valid only for the fixed tau=0.5 monitor. "
                "Use --prediction-source probability for other thresholds."
            )
    dataset_dir = args.dataset_dir.resolve()
    prediction_dir = args.prediction_dir.resolve()
    if not prediction_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")
    reference_label_dir = args.reference_label_dir.resolve() if args.reference_label_dir is not None else None
    if reference_label_dir is not None and not reference_label_dir.is_dir():
        raise FileNotFoundError(f"Reference-label directory not found: {reference_label_dir}")
    spacing = tuple(float(value) for value in args.target_spacing)
    records, dataset_manifest = _discover_validation_cases(dataset_dir)
    if args.case_ids_json is not None:
        wanted_ids = _requested_case_ids(args.case_ids_json.resolve())
        records = [record for record in records if record.case_id in wanted_ids]
        dataset_manifest = [row for row in dataset_manifest if str(row.get("case_id")) in wanted_ids]
        found_ids = {record.case_id for record in records}
        missing_ids = sorted(wanted_ids - found_ids)
        if missing_ids:
            raise ValueError(f"Requested case IDs are absent from {dataset_dir}: {missing_ids[:20]}")
        if not records:
            raise ValueError("No cases remain after --case-ids-json filtering.")
    infos = {info.record.case_id: info for info in build_case_infos(records, split=args.split_name)}
    rows_by_setting: dict[str, list[dict[str, object]]] = {}
    settings = [
        {"threshold": float(threshold), "min_component_voxels": int(min_component), "postprocess": "none"}
        for threshold, min_component in zip(args.thresholds, args.min_components)
    ]
    for setting in settings:
        rows_by_setting[setting_tag(setting)] = []

    prediction_hashes: dict[str, str] = {}
    for record in records:
        if args.prediction_source == "exported_segmentation":
            prediction_segmentation = _load_exported_segmentation(prediction_dir, record.case_id)
            target, target_affine, case_spacing = _load_native_reference_label(
                reference_label_dir,  # checked above
                record.case_id,
                prediction_segmentation,
            )
            probability = (np.asarray(prediction_segmentation.dataobj) > 0).astype(np.float32)
            evaluation_grid = "native_prediction_grid_with_prepared_reference_label"
            prediction_path = prediction_dir / f"{record.case_id}.nii.gz"
            resolved_probability_axis_order = "exported_segmentation"
        elif reference_label_dir is None:
            _, target, _, target_affine = load_case(
                case=record,
                target_spacing=spacing,
                include_dmri=False,
                dmri_reduce="mean",
                dmri_b0_threshold=50.0,
                normalize_foreground_only=True,
            )
            probability, resolved_probability_axis_order = _load_resampled_probability(
                prediction_dir,
                record.case_id,
                args.probability_key,
                spacing,
                args.probability_axis_order,
            )
            if probability.shape != target.shape:
                raise ValueError(
                    f"{record.case_id}: resampled nnU-Net probability shape {probability.shape} does not match "
                    f"the common 1-mm target shape {target.shape}."
                )
            case_spacing = spacing
            evaluation_grid = "canonical_1mm"
            prediction_path = prediction_dir / f"{record.case_id}.npz"
        else:
            probability, prediction_segmentation, resolved_probability_axis_order = _load_native_probability(
                prediction_dir,
                record.case_id,
                args.probability_key,
                args.probability_axis_order,
            )
            target, target_affine, case_spacing = _load_native_reference_label(
                reference_label_dir,
                record.case_id,
                prediction_segmentation,
            )
            evaluation_grid = "native_prediction_grid_with_prepared_reference_label"
            prediction_path = prediction_dir / f"{record.case_id}.npz"
        prediction_hashes[record.case_id] = _sha256(prediction_path)
        info = infos[record.case_id]
        for setting in settings:
            prediction = postprocess_prediction(
                lesion_probability=probability,
                threshold=float(setting["threshold"]),
                min_component_voxels=int(setting["min_component_voxels"]),
                postprocess="none",
            )
            pred_voxels = int(np.count_nonzero(prediction))
            hd95, assd = _surface_metrics(prediction, target, case_spacing)
            rows_by_setting[setting_tag(setting)].append(
                {
                    "case_id": record.case_id,
                    "gt_voxels": int(info.gt_voxels),
                    "gt_category": info.category,
                    "threshold": float(setting["threshold"]),
                    "min_component_voxels": int(setting["min_component_voxels"]),
                    "postprocess": str(setting["postprocess"]),
                    "pred_voxels": pred_voxels,
                    "pred_category": lesion_category(pred_voxels),
                    "dice": dice_score(prediction, target),
                    "hd95": hd95,
                    "assd": assd,
                    "missed_positive": int(info.gt_voxels > 0 and pred_voxels == 0),
                    "empty_false_positive": int(info.gt_voxels == 0 and pred_voxels > 0),
                    "target_affine": np.asarray(target_affine).round(8).tolist(),
                    "evaluation_grid": evaluation_grid,
                    "surface_spacing_mm": list(case_spacing),
                    "prediction_source": args.prediction_source,
                    "probability_axis_order": resolved_probability_axis_order,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []
    for setting in settings:
        tag = setting_tag(setting)
        rows = rows_by_setting[tag]
        _write_dict_rows(args.output_dir / f"case_metrics_{tag}.csv", rows)
        summary.append({**setting, **_metrics_from_rows(rows, prefix=args.metric_prefix)})
    _write_dict_rows(args.output_dir / "summary.csv", summary)
    _write_dict_rows(args.output_dir / "dataset_manifest.csv", dataset_manifest)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "analysis": "fixed operating-point evaluation of standard nnU-Net probability exports",
                "dataset_dir": str(dataset_dir),
                "prediction_dir": str(prediction_dir),
                "target_spacing": spacing,
                "reference_label_dir": str(reference_label_dir) if reference_label_dir is not None else None,
                "evaluation_grid": (
                    "native_prediction_grid_with_prepared_reference_label"
                    if reference_label_dir is not None
                    else "canonical_1mm"
                ),
                "thresholds": args.thresholds,
                "min_components": args.min_components,
                "probability_key": args.probability_key,
                "probability_axis_order_requested": args.probability_axis_order,
                "prediction_source": args.prediction_source,
                "case_ids_json": str(args.case_ids_json.resolve()) if args.case_ids_json else None,
                "metric_prefix": args.metric_prefix,
                "split_name": args.split_name,
                "n_cases": len(records),
                "prediction_sha256": prediction_hashes,
                "surface_empty_policy": "both empty: 0; one empty: image diagonal penalty",
                "interpretation": args.interpretation,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    print(f"[DONE] Wrote {len(summary)} fixed-setting summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
