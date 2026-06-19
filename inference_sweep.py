from __future__ import annotations

import argparse
import csv
import gc
import itertools
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to
from scipy.ndimage import label
from tqdm.auto import tqdm

from multitalent_tbi.case_filters import build_case_infos, lesion_category
from multitalent_tbi.config import load_config, resolve_path
from multitalent_tbi.data import CaseRecord, discover_cases, load_case_cached, load_nifti_robust
from multitalent_tbi.engine import build_model
from multitalent_tbi.infer import predict_logits
from multitalent_tbi.splits import load_splits, split_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run AIMS-TBI inference/validation sweeps with the same sliding-window "
            "predict_logits path used by validate.py/validate2.py."
        )
    )
    parser.add_argument("--config", default="config.yml", help="Path to config.yml.")
    parser.add_argument("--fold", type=int, default=None, help="Fold index. Defaults to config training.fold.")
    parser.add_argument(
        "--split",
        choices=["val", "train", "all"],
        default="val",
        help="Cases to evaluate from the configured split.",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Evaluate only cases with GT lesion voxels > 0.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="One or more checkpoints. Probabilities are averaged across checkpoints.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.5],
        help="Lesion probability thresholds to sweep, e.g. 0.35 0.4 0.45 0.5.",
    )
    parser.add_argument(
        "--min-components",
        nargs="+",
        type=int,
        default=[0],
        help="Remove connected components smaller than these voxel counts.",
    )
    parser.add_argument(
        "--tta",
        choices=["none", "flips"],
        default="none",
        help="none = original only; flips = original + D/H/W single-axis flips.",
    )
    parser.add_argument(
        "--keep-models-loaded",
        action="store_true",
        help="Deprecated alias for --load-mode preload.",
    )
    parser.add_argument(
        "--load-mode",
        choices=["auto", "preload", "sequential"],
        default="auto",
        help=(
            "auto preloads on large GPUs, preload keeps all checkpoints on GPU, "
            "sequential loads/unloads each checkpoint per case for low-memory GPUs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to checkpoints/tbi_multitalentv2/inference_sweeps/fold_X.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save NIfTI predictions for one selected threshold/min-component combo.",
    )
    parser.add_argument("--save-threshold", type=float, default=0.5)
    parser.add_argument("--save-min-component", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP at inference.")
    return parser.parse_args()


def _load_state(model: torch.nn.Module, checkpoint_path: Path) -> None:
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[WARN] {checkpoint_path.name}: missing keys={len(missing)}")
    if unexpected:
        print(f"[WARN] {checkpoint_path.name}: unexpected keys={len(unexpected)}")


def _build_inference_model(config, base_dir: Path, device: torch.device) -> torch.nn.Module:
    original_load_pretrained = bool(getattr(config.model, "load_pretrained", False))
    config.model.load_pretrained = False
    try:
        model = build_model(config, base_dir).to(device)
    finally:
        config.model.load_pretrained = original_load_pretrained
    return model


def _build_loaded_models(config, base_dir: Path, checkpoint_paths: list[Path], device: torch.device) -> list[torch.nn.Module]:
    models = []
    for checkpoint_path in checkpoint_paths:
        print(f"[INFO] Loading model: {checkpoint_path}")
        model = _build_inference_model(config, base_dir, device)
        _load_state(model, checkpoint_path)
        model.eval()
        models.append(model)
    return models


def _unload_model(model: torch.nn.Module) -> None:
    model.cpu()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _tta_specs(mode: str) -> list[tuple[str, int | None]]:
    if mode == "none":
        return [("orig", None)]
    return [("orig", None), ("flip_d", 0), ("flip_h", 1), ("flip_w", 2)]


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
    for _, axis in specs:
        if axis is None:
            aug_image = image
        else:
            aug_image = np.flip(image, axis=axis + 1).copy()

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


def _predict_ensemble_probs_sequential(
    config,
    base_dir: Path,
    checkpoint_paths: list[Path],
    image: np.ndarray,
    device: torch.device,
    use_amp: bool,
    tta: str,
) -> np.ndarray:
    probs_sum = None
    for checkpoint_path in checkpoint_paths:
        model = _build_inference_model(config, base_dir, device)
        _load_state(model, checkpoint_path)
        model.eval()
        probabilities = _predict_model_probs(model, image, config, device, use_amp, tta)
        probs_sum = probabilities if probs_sum is None else probs_sum + probabilities
        _unload_model(model)
    return probs_sum / float(len(checkpoint_paths))


def _predict_ensemble_probs_loaded(
    models: list[torch.nn.Module],
    image: np.ndarray,
    config,
    device: torch.device,
    use_amp: bool,
    tta: str,
) -> np.ndarray:
    probs_sum = None
    for model in models:
        probabilities = _predict_model_probs(model, image, config, device, use_amp, tta)
        probs_sum = probabilities if probs_sum is None else probs_sum + probabilities
    return probs_sum / float(len(models))


def _gpu_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.get_device_properties(device).total_memory) / 1e9


def _resolve_load_mode(args: argparse.Namespace, device: torch.device, checkpoint_count: int) -> str:
    if args.keep_models_loaded:
        return "preload"
    if args.load_mode != "auto":
        return args.load_mode
    if device.type == "cuda" and _gpu_memory_gb(device) >= 35.0:
        return "preload"
    if checkpoint_count == 1:
        return "preload"
    return "sequential"


def _filter_components(mask: np.ndarray, min_component_voxels: int) -> np.ndarray:
    if min_component_voxels <= 1 or not mask.any():
        return mask.astype(np.uint8)
    components, n_components = label(mask.astype(bool))
    filtered = np.zeros_like(mask, dtype=np.uint8)
    for component_index in range(1, n_components + 1):
        component = components == component_index
        if int(component.sum()) >= min_component_voxels:
            filtered[component] = 1
    return filtered


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = prediction.astype(bool)
    tgt = target.astype(bool)
    tp = int(np.logical_and(pred, tgt).sum())
    fp = int(np.logical_and(pred, ~tgt).sum())
    fn = int(np.logical_and(~pred, tgt).sum())
    denominator = 2 * tp + fp + fn
    return float((2 * tp) / denominator) if denominator > 0 else 1.0


def _binary_counts(prediction: np.ndarray, target: np.ndarray) -> tuple[int, int, int]:
    pred = prediction.astype(bool)
    tgt = target.astype(bool)
    tp = int(np.logical_and(pred, tgt).sum())
    fp = int(np.logical_and(pred, ~tgt).sum())
    fn = int(np.logical_and(~pred, tgt).sum())
    return tp, fp, fn


def _case_metrics(
    case_id: str,
    prediction: np.ndarray,
    target: np.ndarray,
    threshold: float,
    min_component_voxels: int,
    gt_voxels: int,
) -> dict[str, object]:
    pred_voxels = int(np.count_nonzero(prediction > 0))
    tp, fp, fn = _binary_counts(prediction, target)
    precision = float(tp / (tp + fp)) if tp + fp > 0 else 0.0
    recall = float(tp / (tp + fn)) if tp + fn > 0 else 0.0
    return {
        "threshold": threshold,
        "min_component_voxels": min_component_voxels,
        "case_id": case_id,
        "dice": _dice(prediction, target),
        "precision": precision,
        "recall": recall,
        "gt_voxels": gt_voxels,
        "gt_category": lesion_category(gt_voxels),
        "pred_voxels": pred_voxels,
        "pred_category": lesion_category(pred_voxels),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "missed_positive": int(gt_voxels > 0 and pred_voxels == 0),
        "empty_false_positive": int(gt_voxels == 0 and pred_voxels > 0),
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[float, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["threshold"]), int(row["min_component_voxels"]))].append(row)

    summaries: list[dict[str, object]] = []
    categories = ["empty", "very_tiny", "tiny", "small", "large"]
    for (threshold, min_component_voxels), combo_rows in sorted(grouped.items()):
        positive_rows = [row for row in combo_rows if int(row["gt_voxels"]) > 0]
        gt50_rows = [row for row in combo_rows if int(row["gt_voxels"]) >= 50]
        summary = {
            "threshold": threshold,
            "min_component_voxels": min_component_voxels,
            "n_cases": len(combo_rows),
            "mean_dice_all": _mean([float(row["dice"]) for row in combo_rows]),
            "mean_dice_positive": _mean([float(row["dice"]) for row in positive_rows]),
            "mean_dice_gt50": _mean([float(row["dice"]) for row in gt50_rows]),
            "n_positive": len(positive_rows),
            "n_gt50": len(gt50_rows),
            "n_missed_positive": sum(int(row["missed_positive"]) for row in combo_rows),
            "n_empty_false_positive": sum(int(row["empty_false_positive"]) for row in combo_rows),
        }
        for category in categories:
            category_rows = [row for row in combo_rows if row["gt_category"] == category]
            summary[f"mean_dice_{category}"] = _mean([float(row["dice"]) for row in category_rows])
            summary[f"n_{category}"] = len(category_rows)
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_prediction(
    output_dir: Path,
    record: CaseRecord,
    prediction: np.ndarray,
    image_affine: np.ndarray,
    original_image: nib.Nifti1Image,
) -> None:
    nifti_prediction = nib.Nifti1Image(prediction.astype(np.uint8), affine=np.asarray(image_affine))
    restored = resample_from_to(nifti_prediction, original_image, order=0)
    output_path = output_dir / "predictions" / f"scan_{record.case_id}_Lesion.nii.gz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(restored, str(output_path))


def _select_records(config, base_dir: Path, fold: int, split_name: str) -> list[CaseRecord]:
    records = discover_cases(resolve_path(base_dir, config.paths.dataset_dir))
    if split_name == "all":
        return records
    splits = load_splits(resolve_path(base_dir, config.paths.splits_file))
    train_records, val_records = split_records(records, splits[fold])
    return val_records if split_name == "val" else train_records


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    base_dir = Path(args.config).expanduser().resolve().parent
    fold = int(args.fold if args.fold is not None else config.training.fold)
    output_dir = resolve_path(
        base_dir,
        args.output_dir
        or (resolve_path(base_dir, config.paths.work_dir) / "inference_sweeps" / f"fold_{fold}"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = [resolve_path(base_dir, checkpoint) for checkpoint in args.checkpoints]
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    records = _select_records(config, base_dir, fold, args.split)
    infos = build_case_infos(records, split=args.split)
    if args.positive_only:
        infos = [info for info in infos if info.gt_voxels > 0]
        records = [info.record for info in infos]
    else:
        records = [info.record for info in infos]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config.inference.use_amp) and not args.no_amp
    load_mode = _resolve_load_mode(args, device, len(checkpoint_paths))
    print(f"[INFO] Cases: {len(records)} | split={args.split} | positive_only={args.positive_only}")
    print(f"[INFO] Device: {device} | AMP={use_amp} | TTA={args.tta}")
    if device.type == "cuda":
        print(f"[INFO] GPU memory: {_gpu_memory_gb(device):.1f} GB")
    print(f"[INFO] Load mode: {load_mode}")
    print(f"[INFO] Thresholds: {args.thresholds}")
    print(f"[INFO] Min components: {args.min_components}")
    print("[INFO] Using multitalent_tbi.infer.predict_logits sliding-window inference.")

    loaded_models = None
    if load_mode == "preload":
        loaded_models = _build_loaded_models(config, base_dir, checkpoint_paths, device)

    all_rows: list[dict[str, object]] = []
    for info in tqdm(infos, desc="Inference sweep"):
        record = info.record
        original_image = load_nifti_robust(record.t1_path)
        image, mask, image_affine, _ = load_case_cached(
            case=record,
            target_spacing=tuple(config.data.target_spacing),
            include_dmri=bool(config.data.include_dmri),
            dmri_reduce=str(config.data.dmri_reduce),
            dmri_b0_threshold=float(config.data.dmri_b0_threshold),
            normalize_foreground_only=bool(config.data.normalize_foreground_only),
            cache_dir=resolve_path(base_dir, config.paths.cache_dir) if config.data.cache_preprocessed else None,
        )

        if loaded_models is None:
            mean_probs = _predict_ensemble_probs_sequential(
                config=config,
                base_dir=base_dir,
                checkpoint_paths=checkpoint_paths,
                image=image,
                device=device,
                use_amp=use_amp,
                tta=args.tta,
            )
        else:
            mean_probs = _predict_ensemble_probs_loaded(
                models=loaded_models,
                image=image,
                config=config,
                device=device,
                use_amp=use_amp,
                tta=args.tta,
            )

        lesion_probability = mean_probs[1]
        save_prediction_written = False
        for threshold, min_component_voxels in itertools.product(args.thresholds, args.min_components):
            prediction = (lesion_probability >= threshold).astype(np.uint8)
            prediction = _filter_components(prediction, min_component_voxels)
            all_rows.append(
                _case_metrics(
                    case_id=record.case_id,
                    prediction=prediction,
                    target=mask,
                    threshold=threshold,
                    min_component_voxels=min_component_voxels,
                    gt_voxels=info.gt_voxels,
                )
            )
            if (
                args.save_predictions
                and not save_prediction_written
                and abs(threshold - args.save_threshold) < 1e-8
                and min_component_voxels == args.save_min_component
            ):
                _save_prediction(output_dir, record, prediction, image_affine, original_image)
                save_prediction_written = True

    if loaded_models is not None:
        for model in loaded_models:
            _unload_model(model)

    summary_rows = _summarize(all_rows)
    _write_csv(output_dir / "per_case_metrics.csv", all_rows)
    _write_csv(output_dir / "summary_metrics.csv", summary_rows)

    print("\n=== Summary ===")
    for row in summary_rows:
        print(
            f"thr={row['threshold']:.3f} mincc={row['min_component_voxels']:>3} "
            f"gt50={row['mean_dice_gt50']:.4f} "
            f"pos={row['mean_dice_positive']:.4f} "
            f"tiny={row['mean_dice_tiny']:.4f} "
            f"small={row['mean_dice_small']:.4f} "
            f"large={row['mean_dice_large']:.4f} "
            f"missed={row['n_missed_positive']} "
            f"empty_fp={row['n_empty_false_positive']}"
        )
    print(f"\n[INFO] Per-case metrics: {output_dir / 'per_case_metrics.csv'}")
    print(f"[INFO] Summary metrics : {output_dir / 'summary_metrics.csv'}")


if __name__ == "__main__":
    main()
