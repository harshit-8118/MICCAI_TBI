from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
from tqdm.auto import tqdm

from multitalent_tbi.case_filters import build_case_infos, lesion_category
from multitalent_tbi.config import load_config, resolve_path
from multitalent_tbi.data import CaseRecord, TBIDataset, load_nifti_robust
from multitalent_tbi.engine import build_model, configure_torch_for_speed, dice_score, set_seed
from multitalent_tbi.infer import predict_logits
from multitalent_tbi.losses import dice_ce_loss
from multitalent_tbi.postprocessing import (
    conditional_m2_component_acceptance,
    postprocess_prediction,
    postprocess_setting_values,
    setting_tag,
)
from train_hierarchical_segmenter import _apply_branch_config, _get


CATEGORIES = ["empty", "very_tiny", "tiny", "small", "large"]
POSITIVE_CATEGORIES = ["very_tiny", "tiny", "small", "large"]
METRIC_GROUPS = ["all", "positive", "gt50", "micro", *CATEGORIES]
TTA_CHOICES = ["none", "flips", "flips3", "flips2", "sagittal_coronal"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints on the released Validation2025_100 hold-out dataset."
    )
    parser.add_argument("--config", default="config.yml")
    parser.add_argument(
        "--branch",
        default="micro128",
        help="Branch under hierarchical_clean.branches, or 'none' to use the base config unchanged.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Validation2025_100 folder. Defaults to external_validation.dataset_dir in config.yml.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="One or more .pt checkpoints. Passing multiple checkpoints averages probabilities as an ensemble.",
    )
    parser.add_argument(
        "--ensemble-strategy",
        choices=["probability_average", "strategy2_conditional_m2"],
        default="probability_average",
        help=(
            "probability_average keeps the old behavior. strategy2_conditional_m2 expects exactly two checkpoints: "
            "checkpoint 1 is safe M1, checkpoint 2 is aggressive M2; M2 components are accepted only when supported by M1."
        ),
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--min-components", nargs="+", type=int, default=None)
    parser.add_argument(
        "--postprocess",
        choices=["none", "dilate", "core_halo"],
        default="none",
        help="Optional mask expansion: plain dilation or seed/core growth into a lower-threshold halo.",
    )
    parser.add_argument(
        "--halo-threshold",
        "--halo-thresholds",
        dest="halo_thresholds",
        nargs="+",
        type=float,
        default=[0.10],
        help="Lower probability threshold(s) used by --postprocess core_halo.",
    )
    parser.add_argument(
        "--core-threshold",
        "--core-thresholds",
        dest="core_thresholds",
        nargs="+",
        type=float,
        default=None,
        help=(
            "High-confidence seed threshold(s) used by --postprocess core_halo. "
            "Defaults to --thresholds for backward compatibility."
        ),
    )
    parser.add_argument(
        "--dilation-radius",
        type=int,
        default=0,
        help="3D binary dilation iterations after thresholding/core_halo. Try 1 first.",
    )
    parser.add_argument(
        "--dilate-min-component-voxels",
        type=int,
        default=1,
        help="Only dilate predicted components with at least this many voxels.",
    )
    parser.add_argument(
        "--core-min-component-voxels",
        type=int,
        nargs="+",
        default=[1],
        help="For core_halo, grow a core component only if its confident core has at least this many voxels.",
    )
    parser.add_argument(
        "--max-growth-ratio",
        "--max-growth-ratios",
        dest="max_growth_ratios",
        nargs="+",
        type=float,
        default=[0.0],
        help="For core_halo, reject halo expansion if grown/core voxel ratio exceeds this value. 0 disables.",
    )
    parser.add_argument(
        "--max-grown-component-voxels",
        type=int,
        default=0,
        help="For core_halo, reject halo expansion if the grown component exceeds this voxel count. 0 disables.",
    )
    parser.add_argument(
        "--m1-thresholds",
        nargs="+",
        type=float,
        default=None,
        help="Strategy 2 only: thresholds for safe M1. Defaults to --thresholds.",
    )
    parser.add_argument(
        "--m1-min-components",
        nargs="+",
        type=int,
        default=None,
        help="Strategy 2 only: min-component voxels for safe M1. Defaults to --min-components.",
    )
    parser.add_argument(
        "--m2-thresholds",
        nargs="+",
        type=float,
        default=None,
        help="Strategy 2 only: thresholds for aggressive M2. Defaults to --thresholds.",
    )
    parser.add_argument(
        "--m2-min-components",
        nargs="+",
        type=int,
        default=None,
        help="Strategy 2 only: min-component voxels for aggressive M2. Defaults to --min-components.",
    )
    parser.add_argument(
        "--m1-support-thresholds",
        nargs="+",
        type=float,
        default=[0.15],
        help="Strategy 2 only: low M1 probability threshold used to decide whether an M2 component is supported.",
    )
    parser.add_argument(
        "--support-radii",
        nargs="+",
        type=int,
        default=[0],
        help="Strategy 2 only: optional voxel dilation radius for the low-threshold M1 support mask.",
    )
    parser.add_argument(
        "--support-min-overlap-voxels",
        type=int,
        default=1,
        help="Strategy 2 only: minimum voxels where an M2 component must overlap the M1 support mask.",
    )
    parser.add_argument(
        "--support-min-overlap-ratio",
        type=float,
        default=0.0,
        help="Strategy 2 only: minimum overlap ratio between each M2 component and the M1 support mask.",
    )
    parser.add_argument(
        "--final-min-components",
        nargs="+",
        type=int,
        default=[0],
        help="Strategy 2 only: final min-component filter after M1 + accepted M2 components are merged.",
    )
    parser.add_argument(
        "--tta",
        choices=TTA_CHOICES,
        default="none",
        help=(
            "TTA mode for every checkpoint unless --checkpoint-ttas is given. "
            "'flips'/'flips3' uses all three spatial flips. "
            "'flips2'/'sagittal_coronal' uses only spatial axes 0 and 1."
        ),
    )
    parser.add_argument(
        "--checkpoint-ttas",
        nargs="+",
        choices=TTA_CHOICES,
        default=None,
        help=(
            "Optional per-checkpoint TTA modes. Must have the same length and order as --checkpoints. "
            "If omitted, --tta is used for every checkpoint."
        ),
    )
    parser.add_argument(
        "--load-mode",
        choices=["preload", "sequential"],
        default="preload",
        help="preload keeps all checkpoints on GPU; sequential loads/unloads per case.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--save-lesions",
        action="store_true",
        help="Save predicted binary lesion masks as .nii.gz files for ITK-SNAP inspection.",
    )
    parser.add_argument(
        "--skip-surface",
        action="store_true",
        help="Skip HD95/ASSD. Useful for quick Dice-only sweeps before final report runs.",
    )
    parser.add_argument(
        "--no-preflight-file-check",
        action="store_true",
        help="Skip early NIfTI load/decompression checks. By default, corrupt image/mask files fail before inference.",
    )
    return parser.parse_args()


def _torch_load_checkpoint(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _strip_nii_suffix(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return name[:-7]
    if lower.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def _discover_validation_cases(dataset_dir: Path) -> tuple[list[CaseRecord], list[dict[str, object]]]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"External validation dataset not found: {dataset_dir}")

    images: dict[str, Path] = {}
    labels: dict[str, Path] = {}
    for path in sorted(dataset_dir.glob("scan_*.nii*")):
        if not path.is_file():
            continue
        stem = _strip_nii_suffix(path.name)
        lower_stem = stem.lower()
        if lower_stem.endswith("_lesion"):
            labels[stem[:-7]] = path
        elif lower_stem.endswith("_t1"):
            images[stem[:-3]] = path
        elif lower_stem.endswith("_dmri"):
            continue
        else:
            images[stem] = path

    all_ids = sorted(set(images) | set(labels))
    records: list[CaseRecord] = []
    manifest: list[dict[str, object]] = []
    for case_id in all_ids:
        image_path = images.get(case_id)
        label_path = labels.get(case_id)
        status = "matched" if image_path is not None and label_path is not None else "missing_label"
        if image_path is None:
            status = "missing_image"
        manifest.append(
            {
                "case_id": case_id,
                "status": status,
                "image_path": str(image_path or ""),
                "lesion_mask_path": str(label_path or ""),
            }
        )
        if image_path is not None and label_path is not None:
            records.append(CaseRecord(case_id=case_id, t1_path=image_path, lesion_path=label_path))

    if not records:
        preview = [path.name for path in sorted(dataset_dir.iterdir())[:20]]
        raise FileNotFoundError(
            f"No matched scan_XXXX(.nii.gz) + scan_XXXX_lesion(.nii.gz) or "
            f"scan_XXXX_T1(.nii.gz) + scan_XXXX_Lesion(.nii.gz) pairs found in {dataset_dir}. "
            f"Preview: {preview}"
        )
    return records, manifest


def _load_nifti_array_for_check(path: Path) -> tuple[tuple[int, ...], np.dtype]:
    image = load_nifti_robust(path)
    data = np.asarray(image.dataobj)
    return tuple(int(dim) for dim in data.shape), data.dtype


def _preflight_check_records(records: list[CaseRecord]) -> None:
    """Force-load every matched image/mask so corrupt gzip/NIfTI files fail early."""
    failures: list[str] = []
    for record in tqdm(records, desc="Preflight NIfTI check", leave=False):
        try:
            image_shape, image_dtype = _load_nifti_array_for_check(record.t1_path)
            lesion_shape, lesion_dtype = _load_nifti_array_for_check(record.lesion_path)
            if len(image_shape) < 3:
                raise ValueError(f"T1 image is not 3D/4D: shape={image_shape}, dtype={image_dtype}")
            if len(lesion_shape) != 3:
                raise ValueError(f"Lesion mask is not 3D: shape={lesion_shape}, dtype={lesion_dtype}")
            if image_shape[:3] != lesion_shape:
                raise ValueError(f"T1/mask shape mismatch: image={image_shape[:3]}, lesion={lesion_shape}")
        except Exception as error:  # noqa: BLE001 - report all load/decompression failures together.
            failures.append(f"{record.case_id}: {error}")

    if failures:
        preview = "\n".join(f"  - {failure}" for failure in failures[:20])
        extra = "" if len(failures) <= 20 else f"\n  ... and {len(failures) - 20} more"
        raise RuntimeError(
            f"Preflight NIfTI sanity check failed for {len(failures)} case(s). "
            f"Fix/remove these files before running inference:\n{preview}{extra}"
        )


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _rows_for_group(rows: list[dict[str, object]], group: str) -> list[dict[str, object]]:
    if group == "all":
        return rows
    if group == "positive":
        return [row for row in rows if int(row["gt_voxels"]) > 0]
    if group == "gt50":
        return [row for row in rows if int(row["gt_voxels"]) >= 50]
    if group == "micro":
        return [row for row in rows if row["gt_category"] in {"very_tiny", "tiny"}]
    return [row for row in rows if row["gt_category"] == group]


def _metrics_from_rows(rows: list[dict[str, object]], prefix: str = "test") -> dict[str, float]:
    metrics: dict[str, float] = {}
    for group in METRIC_GROUPS:
        group_rows = _rows_for_group(rows, group)
        metrics[f"{prefix}_dice_{group}"] = _mean(float(row["dice"]) for row in group_rows)
        metrics[f"{prefix}_hd95_{group}"] = _mean(float(row["hd95"]) for row in group_rows)
        metrics[f"{prefix}_assd_{group}"] = _mean(float(row["assd"]) for row in group_rows)

    metrics["n_cases"] = float(len(rows))
    metrics["n_positive"] = float(len(_rows_for_group(rows, "positive")))
    metrics["n_gt50"] = float(len(_rows_for_group(rows, "gt50")))
    for category in CATEGORIES:
        metrics[f"n_{category}"] = float(len(_rows_for_group(rows, category)))
    metrics["n_missed_positive"] = float(sum(int(row["missed_positive"]) for row in rows))
    metrics["n_empty_false_positive"] = float(sum(int(row["empty_false_positive"]) for row in rows))

    empty = metrics[f"{prefix}_dice_empty"]
    positive = metrics[f"{prefix}_dice_positive"]
    metrics[f"{prefix}_dice_balanced"] = _mean([empty, positive])
    return metrics


def _tta_specs(mode: str) -> list[int | None]:
    normalized = str(mode).lower()
    if normalized == "none":
        return [None]
    if normalized in {"flips", "flips3"}:
        return [None, 0, 1, 2]
    if normalized in {"flips2", "sagittal_coronal"}:
        return [None, 0, 1]
    raise ValueError(f"Unknown TTA mode: {mode}")


@torch.inference_mode()
def _predict_model_probs(
    model: torch.nn.Module,
    image: np.ndarray,
    config,
    device: torch.device,
    use_amp: bool,
    tta: str,
) -> np.ndarray:
    probs_sum = None
    specs = _tta_specs(tta)
    for axis in specs:
        aug_image = image if axis is None else np.flip(image, axis=axis + 1).copy()
        logits = predict_logits(
            model=model,
            image=aug_image,
            roi_size=tuple(config.inference.roi_size),
            overlap=float(config.inference.overlap),
            batch_size=int(config.inference.sw_batch_size),
            device=device,
            use_amp=use_amp and device.type == "cuda",
        )
        probabilities = torch.softmax(torch.from_numpy(logits).float(), dim=0).numpy()
        if axis is not None:
            probabilities = np.flip(probabilities, axis=axis + 1).copy()
        probs_sum = probabilities if probs_sum is None else probs_sum + probabilities
    return probs_sum / float(len(specs))


def _build_loaded_model(config, base_dir: Path, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = build_model(config, base_dir).to(device)
    payload = _torch_load_checkpoint(checkpoint_path, map_location=device)
    state = payload.get("model_state", payload)
    model.load_state_dict(state)
    model.eval()
    return model


def _unload_model(model: torch.nn.Module) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _predict_probs_preload(
    models: list[torch.nn.Module],
    image: np.ndarray,
    config,
    device: torch.device,
    use_amp: bool,
    checkpoint_ttas: list[str],
) -> np.ndarray:
    probs_sum = None
    for model, tta in zip(models, checkpoint_ttas):
        probabilities = _predict_model_probs(model, image, config, device, use_amp, tta)
        probs_sum = probabilities if probs_sum is None else probs_sum + probabilities
    return probs_sum / float(len(models))


def _predict_probs_by_checkpoint_preload(
    models: list[torch.nn.Module],
    image: np.ndarray,
    config,
    device: torch.device,
    use_amp: bool,
    checkpoint_ttas: list[str],
) -> list[np.ndarray]:
    return [
        _predict_model_probs(model, image, config, device, use_amp, tta)
        for model, tta in zip(models, checkpoint_ttas)
    ]


def _predict_probs_sequential(
    checkpoint_paths: list[Path],
    image: np.ndarray,
    config,
    base_dir: Path,
    device: torch.device,
    use_amp: bool,
    checkpoint_ttas: list[str],
) -> np.ndarray:
    probs_sum = None
    for checkpoint_path, tta in zip(checkpoint_paths, checkpoint_ttas):
        model = _build_loaded_model(config, base_dir, checkpoint_path, device)
        probabilities = _predict_model_probs(model, image, config, device, use_amp, tta)
        probs_sum = probabilities if probs_sum is None else probs_sum + probabilities
        _unload_model(model)
    return probs_sum / float(len(checkpoint_paths))


def _predict_probs_by_checkpoint_sequential(
    checkpoint_paths: list[Path],
    image: np.ndarray,
    config,
    base_dir: Path,
    device: torch.device,
    use_amp: bool,
    checkpoint_ttas: list[str],
) -> list[np.ndarray]:
    probabilities_by_checkpoint: list[np.ndarray] = []
    for checkpoint_path, tta in zip(checkpoint_paths, checkpoint_ttas):
        model = _build_loaded_model(config, base_dir, checkpoint_path, device)
        probabilities_by_checkpoint.append(_predict_model_probs(model, image, config, device, use_amp, tta))
        _unload_model(model)
    return probabilities_by_checkpoint


def _surface_mask(mask: np.ndarray) -> np.ndarray:
    structure = generate_binary_structure(rank=3, connectivity=1)
    mask_bool = mask.astype(bool)
    eroded = binary_erosion(mask_bool, structure=structure, border_value=0)
    surface = mask_bool & ~eroded
    return surface if surface.any() else mask_bool


def _diagonal_penalty(shape: tuple[int, ...], spacing: tuple[float, float, float]) -> float:
    extents = [(max(int(size) - 1, 1) * float(space)) for size, space in zip(shape, spacing)]
    return float(math.sqrt(sum(extent * extent for extent in extents)))


def _surface_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[float, float]:
    pred = prediction.astype(bool)
    tgt = target.astype(bool)
    pred_any = bool(pred.any())
    tgt_any = bool(tgt.any())
    if not pred_any and not tgt_any:
        return 0.0, 0.0
    if pred_any != tgt_any:
        penalty = _diagonal_penalty(tuple(target.shape), spacing)
        return penalty, penalty

    pred_surface = _surface_mask(pred)
    tgt_surface = _surface_mask(tgt)
    dist_to_tgt = distance_transform_edt(~tgt_surface, sampling=spacing)
    dist_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate([dist_to_tgt[pred_surface], dist_to_pred[tgt_surface]]).astype(np.float64)
    if distances.size == 0:
        return 0.0, 0.0
    hd95 = float(np.percentile(distances, 95))
    assd = float(np.mean(distances))
    return hd95, assd


def _write_dict_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_case_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "case_id",
        "gt_voxels",
        "gt_category",
        "ensemble_strategy",
        "postprocess",
        "core_threshold",
        "halo_threshold",
        "dilation_radius",
        "dilate_min_component_voxels",
        "core_min_component_voxels",
        "max_growth_ratio",
        "max_grown_component_voxels",
        "threshold",
        "min_component_voxels",
        "m1_threshold",
        "m1_min_component_voxels",
        "m2_threshold",
        "m2_min_component_voxels",
        "m1_support_threshold",
        "support_radius",
        "support_min_overlap_voxels",
        "support_min_overlap_ratio",
        "final_min_component_voxels",
        "m1_pred_voxels",
        "m2_pred_voxels",
        "accepted_m2_components",
        "rejected_m2_components",
        "accepted_m2_voxels",
        "rejected_m2_voxels",
        "pred_voxels",
        "pred_category",
        "dice",
        "hd95",
        "assd",
        "missed_positive",
        "empty_false_positive",
    ]
    _write_dict_rows(path, rows, fieldnames=fieldnames)


def _save_lesion_prediction(
    output_dir: Path,
    record: CaseRecord,
    prediction: np.ndarray,
    image_affine: np.ndarray,
    original_image: nib.Nifti1Image,
    setting: dict[str, object],
) -> Path:
    nifti_prediction = nib.Nifti1Image(prediction.astype(np.uint8), affine=np.asarray(image_affine))
    restored = resample_from_to(nifti_prediction, original_image, order=0)
    restored_data = (np.asarray(restored.dataobj) > 0.5).astype(np.uint8)
    restored = nib.Nifti1Image(restored_data, restored.affine, restored.header)
    restored.set_data_dtype(np.uint8)

    save_dir = output_dir / "predicted_lesions" / setting_tag(setting)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"{record.case_id}_pred_lesion.nii.gz"
    nib.save(restored, str(output_path))
    return output_path


def _rank_values(values: list[float], higher_is_better: bool) -> list[int]:
    valid = [(index, value) for index, value in enumerate(values) if math.isfinite(value)]
    valid.sort(key=lambda item: item[1], reverse=higher_is_better)
    ranks = [len(values) + 1 for _ in values]
    last_value = None
    last_rank = 0
    for position, (index, value) in enumerate(valid, start=1):
        if last_value is not None and math.isclose(value, last_value, rel_tol=1e-12, abs_tol=1e-12):
            rank = last_rank
        else:
            rank = position
            last_value = value
            last_rank = rank
        ranks[index] = rank
    return ranks


def _add_ranking_proxy(summary_rows: list[dict[str, object]]) -> None:
    dice_ranks = _rank_values([float(row["test_dice_all"]) for row in summary_rows], higher_is_better=True)
    hd95_ranks = _rank_values([float(row["test_hd95_all"]) for row in summary_rows], higher_is_better=False)
    assd_ranks = _rank_values([float(row["test_assd_all"]) for row in summary_rows], higher_is_better=False)
    for row, dice_rank, hd95_rank, assd_rank in zip(summary_rows, dice_ranks, hd95_ranks, assd_ranks):
        row["ranking_proxy_dice_rank"] = int(dice_rank)
        row["ranking_proxy_hd95_rank"] = int(hd95_rank)
        row["ranking_proxy_assd_rank"] = int(assd_rank)
        row["ranking_proxy_sum"] = int(dice_rank + hd95_rank + assd_rank)


def _to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, SimpleNamespace):
        return {key: _to_jsonable(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _log_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def _format_float(value: object, digits: int = 4) -> str:
    number = float(value)
    return "nan" if not math.isfinite(number) else f"{number:.{digits}f}"


def _build_sweep_settings(
    args: argparse.Namespace,
    thresholds: list[float],
    min_components: list[int],
    core_thresholds: list[float],
    core_min_components: list[int],
    halo_thresholds: list[float],
    max_growth_ratios: list[float],
) -> list[dict[str, object]]:
    if args.ensemble_strategy == "strategy2_conditional_m2":
        m1_thresholds = [float(value) for value in (args.m1_thresholds or thresholds)]
        m1_min_components = [int(value) for value in (args.m1_min_components or min_components)]
        m2_thresholds = [float(value) for value in (args.m2_thresholds or thresholds)]
        m2_min_components = [int(value) for value in (args.m2_min_components or min_components)]
        support_thresholds = [float(value) for value in args.m1_support_thresholds]
        support_radii = [int(value) for value in args.support_radii]
        final_min_components = [int(value) for value in args.final_min_components]
        settings: list[dict[str, object]] = []
        for (
            m1_threshold,
            m1_min_component_voxels,
            m2_threshold,
            m2_min_component_voxels,
            support_threshold,
            support_radius,
            final_min_component_voxels,
        ) in itertools.product(
            m1_thresholds,
            m1_min_components,
            m2_thresholds,
            m2_min_components,
            support_thresholds,
            support_radii,
            final_min_components,
        ):
            settings.append(
                {
                    "ensemble_strategy": args.ensemble_strategy,
                    "postprocess": "strategy2_conditional_m2",
                    "threshold": float(m1_threshold),
                    "min_component_voxels": int(m1_min_component_voxels),
                    "m1_threshold": float(m1_threshold),
                    "m1_min_component_voxels": int(m1_min_component_voxels),
                    "m2_threshold": float(m2_threshold),
                    "m2_min_component_voxels": int(m2_min_component_voxels),
                    "m1_support_threshold": float(support_threshold),
                    "support_radius": int(support_radius),
                    "support_min_overlap_voxels": int(args.support_min_overlap_voxels),
                    "support_min_overlap_ratio": float(args.support_min_overlap_ratio),
                    "final_min_component_voxels": int(final_min_component_voxels),
                    "core_threshold": float("nan"),
                    "halo_threshold": float("nan"),
                    "dilation_radius": 0,
                    "dilate_min_component_voxels": 0,
                    "core_min_component_voxels": 0,
                    "max_growth_ratio": float("nan"),
                    "max_grown_component_voxels": 0,
                }
            )
        return settings

    settings = []
    for threshold, min_component_voxels, core_threshold, core_min_component_voxels, halo_threshold, max_growth_ratio in itertools.product(
        thresholds,
        min_components,
        core_thresholds,
        core_min_components,
        halo_thresholds,
        max_growth_ratios,
    ):
        settings.append(
            {
                "ensemble_strategy": args.ensemble_strategy,
                "postprocess": args.postprocess,
                "threshold": float(threshold),
                "min_component_voxels": int(min_component_voxels),
                "core_threshold": float(core_threshold),
                "halo_threshold": float(halo_threshold),
                "dilation_radius": int(args.dilation_radius),
                "dilate_min_component_voxels": int(args.dilate_min_component_voxels),
                "core_min_component_voxels": int(core_min_component_voxels),
                "max_growth_ratio": float(max_growth_ratio),
                "max_grown_component_voxels": int(args.max_grown_component_voxels),
                "m1_threshold": float("nan"),
                "m1_min_component_voxels": -1,
                "m2_threshold": float("nan"),
                "m2_min_component_voxels": -1,
                "m1_support_threshold": float("nan"),
                "support_radius": -1,
                "support_min_overlap_voxels": -1,
                "support_min_overlap_ratio": float("nan"),
                "final_min_component_voxels": -1,
            }
        )
    return settings


def _sort_summary_key(row: dict[str, object]) -> tuple[object, ...]:
    if str(row.get("ensemble_strategy", "probability_average")) == "strategy2_conditional_m2":
        return (
            float(row["m1_threshold"]),
            int(row["m1_min_component_voxels"]),
            float(row["m2_threshold"]),
            int(row["m2_min_component_voxels"]),
            float(row["m1_support_threshold"]),
            int(row["support_radius"]),
            int(row["final_min_component_voxels"]),
        )
    return (
        float(row["threshold"]),
        int(row["min_component_voxels"]),
        float(row.get("core_threshold", float("nan"))) if math.isfinite(float(row.get("core_threshold", float("nan")))) else -1.0,
        float(row.get("halo_threshold", float("nan"))) if math.isfinite(float(row.get("halo_threshold", float("nan")))) else -1.0,
        int(row.get("core_min_component_voxels", 0)),
        float(row.get("max_growth_ratio", float("nan"))) if math.isfinite(float(row.get("max_growth_ratio", float("nan")))) else -1.0,
    )


def _format_setting_for_log(row: dict[str, object]) -> str:
    if str(row.get("ensemble_strategy", "probability_average")) == "strategy2_conditional_m2":
        return (
            f"m1_thr={float(row['m1_threshold']):.3f} "
            f"m1_mincc={int(row['m1_min_component_voxels'])} "
            f"m2_thr={float(row['m2_threshold']):.3f} "
            f"m2_mincc={int(row['m2_min_component_voxels'])} "
            f"support={float(row['m1_support_threshold']):.3f} "
            f"radius={int(row['support_radius'])} "
            f"final_mincc={int(row['final_min_component_voxels'])}"
        )
    parts = [f"thr={float(row['threshold']):.3f}", f"mincc={int(row['min_component_voxels'])}"]
    if str(row.get("postprocess", "none")) == "core_halo":
        parts.extend(
            [
                f"core={float(row['core_threshold']):.3f}",
                f"halo={float(row['halo_threshold']):.3f}",
                f"core_mincc={int(row['core_min_component_voxels'])}",
                f"grow={float(row['max_growth_ratio']):.2f}",
            ]
        )
    return " ".join(parts)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    base_dir = Path(args.config).expanduser().resolve().parent

    branch_cfg = None
    if str(args.branch).lower() != "none":
        hierarchical_cfg = _get(config, "hierarchical_clean", None)
        if hierarchical_cfg is None:
            raise ValueError("config.yml needs a hierarchical_clean section, or pass --branch none.")
        branch_cfg = getattr(hierarchical_cfg.branches, args.branch)
        dummy_args = SimpleNamespace(epochs=None, batch_size=None, num_workers=args.num_workers, lr=None)
        _apply_branch_config(config, branch_cfg, dummy_args)

    external_cfg = _get(config, "external_validation", SimpleNamespace())
    dataset_dir_value = args.dataset_dir or _get(external_cfg, "dataset_dir", None)
    if dataset_dir_value is None:
        raise ValueError("Pass --dataset-dir or set external_validation.dataset_dir in config.yml.")
    dataset_dir = resolve_path(base_dir, dataset_dir_value)
    records, manifest = _discover_validation_cases(dataset_dir)
    if not args.no_preflight_file_check:
        print(f"[INFO] Preflight   : checking {len(records)} matched image/mask pairs")
        _preflight_check_records(records)
    case_infos = build_case_infos(records, split="external_validation")

    thresholds = args.thresholds
    if thresholds is None and branch_cfg is not None:
        thresholds = [float(_get(branch_cfg.validation, "threshold", 0.5))]
    if thresholds is None:
        thresholds = [float(_get(config.training, "metric_threshold", 0.5))]

    min_components = args.min_components
    if min_components is None and branch_cfg is not None:
        min_components = [int(_get(branch_cfg.validation, "min_component_voxels", 0))]
    if min_components is None:
        min_components = [0]

    checkpoint_paths = [resolve_path(base_dir, path) for path in args.checkpoints]
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if args.checkpoint_ttas is not None:
        if len(args.checkpoint_ttas) != len(checkpoint_paths):
            raise ValueError(
                f"--checkpoint-ttas must have exactly one value per checkpoint: "
                f"got {len(args.checkpoint_ttas)} TTA modes for {len(checkpoint_paths)} checkpoints."
            )
        checkpoint_ttas = [str(value) for value in args.checkpoint_ttas]
    else:
        checkpoint_ttas = [str(args.tta) for _ in checkpoint_paths]
    if args.ensemble_strategy == "strategy2_conditional_m2" and len(checkpoint_paths) != 2:
        raise ValueError("--ensemble-strategy strategy2_conditional_m2 requires exactly two checkpoints: M1 first, M2 second.")
    halo_thresholds, max_growth_ratios = postprocess_setting_values(
        args.postprocess,
        [float(value) for value in args.halo_thresholds],
        [float(value) for value in args.max_growth_ratios],
    )
    core_thresholds = (
        [float(value) for value in args.core_thresholds]
        if args.postprocess == "core_halo" and args.core_thresholds is not None
        else (list(thresholds) if args.postprocess == "core_halo" else [float("nan")])
    )
    core_min_components = (
        [int(value) for value in args.core_min_component_voxels]
        if args.postprocess == "core_halo"
        else [int(args.core_min_component_voxels[0])]
    )
    sweep_settings = _build_sweep_settings(
        args,
        thresholds,
        min_components,
        core_thresholds,
        core_min_components,
        halo_thresholds,
        max_growth_ratios,
    )

    output_root = resolve_path(base_dir, _get(external_cfg, "output_root", "checkpoints/Validation2025_100"))
    output_dir = (
        resolve_path(base_dir, args.output_dir)
        if args.output_dir
        else output_root / (str(args.branch).lower() if str(args.branch).lower() != "none" else "base_config")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = None
    if bool(config.data.cache_preprocessed):
        cache_dir = resolve_path(base_dir, _get(external_cfg, "cache_dir", config.paths.cache_dir))

    set_seed(int(config.training.seed))
    configure_torch_for_speed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config.training.use_amp) and not args.no_amp and device.type == "cuda"
    spacing = tuple(float(value) for value in config.data.target_spacing)

    dataset = TBIDataset(
        records,
        patch_size=tuple(config.data.patch_size),
        target_spacing=spacing,
        include_dmri=bool(config.data.include_dmri),
        dmri_reduce=str(config.data.dmri_reduce),
        dmri_b0_threshold=float(config.data.dmri_b0_threshold),
        normalize_foreground_only=bool(config.data.normalize_foreground_only),
        oversample_foreground_prob=0.0,
        training=False,
        cache_dir=cache_dir,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )

    manifest_fields = ["case_id", "status", "image_path", "lesion_mask_path"]
    _write_dict_rows(output_dir / "dataset_manifest.csv", manifest, fieldnames=manifest_fields)
    run_config = {
        "config": str(Path(args.config).resolve()),
        "branch": args.branch,
        "dataset_dir": dataset_dir,
        "output_dir": output_dir,
        "checkpoints": checkpoint_paths,
        "ensemble_strategy": args.ensemble_strategy,
        "thresholds": thresholds,
        "min_components": min_components,
        "m1_thresholds": args.m1_thresholds,
        "m1_min_components": args.m1_min_components,
        "m2_thresholds": args.m2_thresholds,
        "m2_min_components": args.m2_min_components,
        "m1_support_thresholds": args.m1_support_thresholds,
        "support_radii": args.support_radii,
        "support_min_overlap_voxels": int(args.support_min_overlap_voxels),
        "support_min_overlap_ratio": float(args.support_min_overlap_ratio),
        "final_min_components": args.final_min_components,
        "postprocess": args.postprocess,
        "core_thresholds": core_thresholds,
        "halo_thresholds": halo_thresholds,
        "dilation_radius": int(args.dilation_radius),
        "dilate_min_component_voxels": int(args.dilate_min_component_voxels),
        "core_min_component_voxels": core_min_components,
        "max_growth_ratios": max_growth_ratios,
        "max_grown_component_voxels": int(args.max_grown_component_voxels),
        "tta": args.tta,
        "checkpoint_ttas": checkpoint_ttas,
        "checkpoint_tta_pairs": [
            {"checkpoint": str(path), "tta": tta}
            for path, tta in zip(checkpoint_paths, checkpoint_ttas)
        ],
        "load_mode": args.load_mode,
        "save_lesions": bool(args.save_lesions),
        "skip_surface": bool(args.skip_surface),
        "target_spacing": spacing,
        "roi_size": list(config.inference.roi_size),
        "overlap": float(config.inference.overlap),
        "sw_batch_size": int(config.inference.sw_batch_size),
        "n_sweep_settings": len(sweep_settings),
        "n_manifest_rows": len(manifest),
        "n_matched_cases": len(records),
        "n_missing_labels": sum(1 for row in manifest if row["status"] == "missing_label"),
        "n_missing_images": sum(1 for row in manifest if row["status"] == "missing_image"),
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(run_config), handle, indent=2)

    print(f"[INFO] Dataset     : {dataset_dir}")
    print(f"[INFO] Matched     : {len(records)} cases")
    print(f"[INFO] Missing lab : {run_config['n_missing_labels']} | Missing image: {run_config['n_missing_images']}")
    print(f"[INFO] Branch      : {args.branch}")
    print(f"[INFO] Checkpoints : {len(checkpoint_paths)}")
    print(f"[INFO] Strategy    : {args.ensemble_strategy}")
    print(f"[INFO] Device      : {device} | AMP={use_amp} | TTA={args.tta} | checkpoint_ttas={checkpoint_ttas} | load_mode={args.load_mode}")
    print(f"[INFO] Thresholds  : {thresholds}")
    print(f"[INFO] Min comp    : {min_components}")
    print(f"[INFO] Sweep combos: {len(sweep_settings)}")
    if args.ensemble_strategy == "strategy2_conditional_m2":
        print(
            "[INFO] Strategy 2  : "
            f"M1 thr={args.m1_thresholds or thresholds} mincc={args.m1_min_components or min_components} | "
            f"M2 thr={args.m2_thresholds or thresholds} mincc={args.m2_min_components or min_components} | "
            f"support={args.m1_support_thresholds} radii={args.support_radii} "
            f"overlap_vox={args.support_min_overlap_voxels} overlap_ratio={args.support_min_overlap_ratio} "
            f"final_mincc={args.final_min_components}"
        )
    print(
        "[INFO] Postprocess : "
        f"{args.postprocess} | halos={halo_thresholds} | "
        f"dilate_radius={args.dilation_radius} | dilate_mincc={args.dilate_min_component_voxels} | "
        f"core_thresholds={core_thresholds} | core_mincc={core_min_components} | max_growth={max_growth_ratios} | "
        f"max_grown={args.max_grown_component_voxels}"
    )
    print(f"[INFO] Save lesions: {args.save_lesions}")
    print(f"[INFO] Surface     : {'off' if args.skip_surface else 'HD95 + ASSD'}")
    print(f"[INFO] Output dir  : {output_dir}")

    loaded_models = None
    if args.load_mode == "preload":
        loaded_models = [_build_loaded_model(config, base_dir, checkpoint_path, device) for checkpoint_path in checkpoint_paths]

    class_weights = torch.tensor(config.training.class_weights, dtype=torch.float32, device=device)
    setting_by_tag = {setting_tag(setting): setting for setting in sweep_settings}
    if len(setting_by_tag) != len(sweep_settings):
        raise ValueError("Duplicate sweep setting tags were generated. Please check threshold/mincc/support sweeps.")
    rows_by_setting: dict[str, list[dict[str, object]]] = {tag: [] for tag in setting_by_tag}
    losses: list[float] = []
    saved_lesion_paths: list[dict[str, object]] = []

    for index, batch in enumerate(tqdm(loader, desc="External validation")):
        image = batch["image"].numpy()[0]
        mask = batch["mask"].numpy()[0]
        image_affine = batch["image_affine"].numpy()[0]
        info = case_infos[index]
        original_image = load_nifti_robust(info.record.t1_path) if args.save_lesions else None
        probabilities_by_checkpoint: list[np.ndarray] | None = None
        if args.ensemble_strategy == "strategy2_conditional_m2":
            if loaded_models is not None:
                probabilities_by_checkpoint = _predict_probs_by_checkpoint_preload(
                    loaded_models,
                    image,
                    config,
                    device,
                    use_amp,
                    checkpoint_ttas,
                )
            else:
                probabilities_by_checkpoint = _predict_probs_by_checkpoint_sequential(
                    checkpoint_paths,
                    image,
                    config,
                    base_dir,
                    device,
                    use_amp,
                    checkpoint_ttas,
                )
            probabilities = np.mean(np.stack(probabilities_by_checkpoint, axis=0), axis=0)
        elif loaded_models is not None:
            probabilities = _predict_probs_preload(loaded_models, image, config, device, use_amp, checkpoint_ttas)
        else:
            probabilities = _predict_probs_sequential(checkpoint_paths, image, config, base_dir, device, use_amp, checkpoint_ttas)

        logits_for_loss = torch.log(torch.from_numpy(np.clip(probabilities, 1e-7, 1.0)).float()).unsqueeze(0).to(device)
        mask_tensor = torch.from_numpy(mask.copy()).unsqueeze(0).to(device=device, dtype=torch.long)
        losses.append(float(dice_ce_loss(logits_for_loss, mask_tensor, class_weights=class_weights).detach().cpu()))

        lesion_probability = probabilities[1]
        for tag, setting in setting_by_tag.items():
            strategy_stats: dict[str, object] = {
                "m1_pred_voxels": -1,
                "m2_pred_voxels": -1,
                "accepted_m2_components": -1,
                "rejected_m2_components": -1,
                "accepted_m2_voxels": -1,
                "rejected_m2_voxels": -1,
            }
            if args.ensemble_strategy == "strategy2_conditional_m2":
                if probabilities_by_checkpoint is None:
                    raise RuntimeError("Internal error: Strategy 2 requires per-checkpoint probabilities.")
                prediction, stats = conditional_m2_component_acceptance(
                    m1_probability=probabilities_by_checkpoint[0][1],
                    m2_probability=probabilities_by_checkpoint[1][1],
                    m1_threshold=float(setting["m1_threshold"]),
                    m1_min_component_voxels=int(setting["m1_min_component_voxels"]),
                    m2_threshold=float(setting["m2_threshold"]),
                    m2_min_component_voxels=int(setting["m2_min_component_voxels"]),
                    m1_support_threshold=float(setting["m1_support_threshold"]),
                    support_radius=int(setting["support_radius"]),
                    support_min_overlap_voxels=int(setting["support_min_overlap_voxels"]),
                    support_min_overlap_ratio=float(setting["support_min_overlap_ratio"]),
                    final_min_component_voxels=int(setting["final_min_component_voxels"]),
                )
                strategy_stats.update(vars(stats))
            else:
                prediction = postprocess_prediction(
                    lesion_probability=lesion_probability,
                    threshold=float(setting["threshold"]),
                    min_component_voxels=int(setting["min_component_voxels"]),
                    postprocess=str(setting["postprocess"]),
                    core_threshold=float(setting["core_threshold"]),
                    halo_threshold=float(setting["halo_threshold"]),
                    dilation_radius=int(setting["dilation_radius"]),
                    dilate_min_component_voxels=int(setting["dilate_min_component_voxels"]),
                    core_min_component_voxels=int(setting["core_min_component_voxels"]),
                    max_growth_ratio=float(setting["max_growth_ratio"]),
                    max_grown_component_voxels=int(setting["max_grown_component_voxels"]),
                )
            pred_voxels = int(np.count_nonzero(prediction))
            if args.skip_surface:
                hd95 = float("nan")
                assd = float("nan")
            else:
                hd95, assd = _surface_metrics(prediction, mask, spacing)
            if args.save_lesions and original_image is not None:
                saved_path = _save_lesion_prediction(
                    output_dir=output_dir,
                    record=info.record,
                    prediction=prediction,
                    image_affine=image_affine,
                    original_image=original_image,
                    setting=setting,
                )
                saved_lesion_paths.append(
                    {
                        "case_id": info.record.case_id,
                        **setting,
                        "path": str(saved_path),
                    }
                )
            rows_by_setting[tag].append(
                {
                    "case_id": info.record.case_id,
                    "gt_voxels": int(info.gt_voxels),
                    "gt_category": info.category,
                    **setting,
                    **strategy_stats,
                    "pred_voxels": pred_voxels,
                    "pred_category": lesion_category(pred_voxels),
                    "dice": dice_score(prediction, mask),
                    "hd95": hd95,
                    "assd": assd,
                    "missed_positive": int(info.gt_voxels > 0 and pred_voxels == 0),
                    "empty_false_positive": int(info.gt_voxels == 0 and pred_voxels > 0),
                }
            )

    if loaded_models is not None:
        for model in loaded_models:
            _unload_model(model)

    summary_rows: list[dict[str, object]] = []
    for tag, rows in rows_by_setting.items():
        setting = setting_by_tag[tag]
        metrics = _metrics_from_rows(rows, prefix="test")
        row = {
            **setting,
            "test_loss": _mean(losses),
            **metrics,
        }
        summary_rows.append(row)
        detail_name = "case_metrics_" + setting_tag(setting) + ".csv"
        _write_case_rows(output_dir / detail_name, rows)

    summary_rows.sort(key=_sort_summary_key)
    _add_ranking_proxy(summary_rows)
    _write_dict_rows(output_dir / "summary.csv", summary_rows)
    if saved_lesion_paths:
        _write_dict_rows(output_dir / "saved_lesions.csv", saved_lesion_paths)

    best_by_proxy = sorted(summary_rows, key=lambda row: (int(row["ranking_proxy_sum"]), -float(row["test_dice_all"])))
    _write_dict_rows(output_dir / "best_by_ranking_proxy.csv", best_by_proxy)

    log_lines = [
        "External Validation2025_100 evaluation",
        f"Dataset: {dataset_dir}",
        f"Output: {output_dir}",
        f"Matched cases: {len(records)}",
        f"Missing labels: {run_config['n_missing_labels']}",
        f"Missing images: {run_config['n_missing_images']}",
        f"Branch: {args.branch}",
        f"Checkpoints: {len(checkpoint_paths)}",
        f"Ensemble strategy: {args.ensemble_strategy}",
        f"Postprocess: {args.postprocess}",
        f"Core thresholds: {core_thresholds}",
        f"Halo thresholds: {halo_thresholds}",
        f"Dilation radius: {args.dilation_radius}",
        f"Dilate min component voxels: {args.dilate_min_component_voxels}",
        f"Core min component voxels: {core_min_components}",
        f"Max growth ratios: {max_growth_ratios}",
        f"Max grown component voxels: {args.max_grown_component_voxels}",
        f"TTA: {args.tta}",
        f"Checkpoint TTAs: {checkpoint_ttas}",
        f"Load mode: {args.load_mode}",
        f"Saved lesion masks: {len(saved_lesion_paths)}",
        f"Surface metrics: {'skipped' if args.skip_surface else 'HD95 and ASSD'}",
        "",
        "Top settings by local ranking proxy (Dice rank + HD95 rank + ASSD rank within this sweep):",
    ]
    for row in best_by_proxy[:10]:
        log_lines.append(
            "{setting} rank_sum={rank:3d} "
            "dice={dice} hd95={hd95} assd={assd} pos={pos} gt50={gt50} "
            "missed={missed} empty_fp={empty_fp}".format(
                setting=_format_setting_for_log(row),
                rank=int(row["ranking_proxy_sum"]),
                dice=_format_float(row["test_dice_all"]),
                hd95=_format_float(row["test_hd95_all"]),
                assd=_format_float(row["test_assd_all"]),
                pos=_format_float(row["test_dice_positive"]),
                gt50=_format_float(row["test_dice_gt50"]),
                missed=int(float(row["n_missed_positive"])),
                empty_fp=int(float(row["n_empty_false_positive"])),
            )
        )
    _log_lines(output_dir / "evaluation_log.txt", log_lines)

    print("\n=== External Validation Summary ===")
    for row in best_by_proxy[:10]:
        print(
            f"{_format_setting_for_log(row)} "
            f"rank_sum={int(row['ranking_proxy_sum']):3d} "
            f"dice={_format_float(row['test_dice_all'])} "
            f"hd95={_format_float(row['test_hd95_all'])} "
            f"assd={_format_float(row['test_assd_all'])} "
            f"pos={_format_float(row['test_dice_positive'])} "
            f"missed={int(float(row['n_missed_positive']))} "
            f"empty_fp={int(float(row['n_empty_false_positive']))}"
        )
    print(f"[DONE] Wrote {output_dir / 'summary.csv'}")
    print(f"[DONE] Wrote {output_dir / 'best_by_ranking_proxy.csv'}")
    print(f"[DONE] Wrote {output_dir / 'evaluation_log.txt'}")


if __name__ == "__main__":
    main()


"""
python evaluate_external_validation.py \
  --config config.yml \
  --branch none \
  --ensemble-strategy probability_average \
  --checkpoints \
    /data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning//data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning/archive/ls/best_val_f0_e5ohnz5w.pt \
    /data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning/archive/Approach_2_Ensemble_Callibrater/checkpoints/trained_models/best_ddp_fft_finetuned_kpcyjb66_data_leaked_0.54_rank1_leaderboard.pt \
  --checkpoint-ttas none none \
  --m1-thresholds 0.5 \
  --m1-min-components 10 \
  --m2-thresholds 0.5 \
  --m2-min-components 10 \
  --m1-support-thresholds 0.10 \
  --support-radii 1 \
  --support-min-overlap-voxels 5 \
  --support-min-overlap-ratio 0.0 \
  --final-min-components 10 \
  --output-dir checkpoints/Validation2025_100/ensemble_2_models_e5ohnz5w_ddp_kpcyjb66_no_tta__save_preview \
  --save-lesions
"""

"""
python evaluate_external_validation.py \
  --config config.yml \
  --branch none \
  --ensemble-strategy probability_average \
  --checkpoints \
    archive/Approach_2_Ensemble_Callibrater/checkpoints/trained_models/best_val_f0_e5ohnz5w.pt \
  --checkpoint-ttas none \
  --thresholds 0.2 \
  --min-components 0 \
  --postprocess none \
  --output-dir checkpoints/Validation2025_100/best_val_f0_e5ohnz5w_tta_mB_previews_0p2_0 \
  --save-lesions
"""

'''
python evaluate_external_validation.py \
  --config config.yml \
  --branch none \
  --ensemble-strategy probability_average \
  --checkpoints \
    archive/Approach_2_Ensemble_Callibrater/checkpoints/trained_models/best_val_f0_e5ohnz5w.pt \
    archive/Approach_2_Ensemble_Callibrater/checkpoints/trained_models/best_ddp_fft_finetuned_kpcyjb66_data_leaked_0.54_rank1_leaderboard.pt \
  --checkpoint-ttas none flips3 \
  --thresholds 0.1 0.2 0.3 0.4 0.5 \
  --min-components 3 5 10 20 40 \
  --postprocess none \
  --output-dir checkpoints/Validation2025_100/ensemble_2_models_e5ohnz5w_no_tta_ddp_kpcyjb66_tta_hybrid_previews \
  --save-lesions
''' 

'''
python evaluate_external_validation.py \
  --config config.yml \
  --branch none \
  --ensemble-strategy probability_average \
  --checkpoints \
    /data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning//data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning/archive/ls/best_val_f0_e5ohnz5w.pt \
    /data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning/archive/Approach_2_Ensemble_Callibrater/checkpoints/trained_models/best_ddp_fft_finetuned_kpcyjb66_data_leaked_0.54_rank1_leaderboard.pt \
  --checkpoint-ttas flips none \
  --thresholds 0.20 \
  --min-components 40 \
  --postprocess core_halo \
  --core-thresholds 0.50 0.60 \
  --halo-thresholds 0.05 0.10 \
  --core-min-component-voxels 10 20 \
  --max-growth-ratios 2 3 \
  --dilation-radius 0 \
  --output-dir checkpoints/Validation2025_100/hybrid_thr0p2_mincc40_core_halo_seed_sweep
'''