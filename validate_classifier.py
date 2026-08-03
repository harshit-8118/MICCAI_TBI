from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from multitalent_tbi.config import load_config, resolve_path
from multitalent_tbi.data import discover_cases, load_case_cached
from multitalent_tbi.engine import build_model, configure_torch_for_speed
from multitalent_tbi.infer import _pad_volume, _sliding_positions
from evaluate_external_validation import (
    TTA_CHOICES,
    _predict_probs_preload,
    _unload_model,
)
from multitalent_tbi.postprocessing import conditional_m2_component_acceptance, filter_components as _filter_components
from train_classifier import (
    EncoderGAPClassifier,
    _classification_metrics,
    _confusion_counts,
    encoder_feature,
    _load_segmentation_checkpoint,
    _torch_load,
)


def _build_loaded_segmentation_model(config, base_dir: Path, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    original_load_pretrained = bool(config.model.load_pretrained)
    config.model.load_pretrained = False
    try:
        model = build_model(config, base_dir).to(device)
    finally:
        config.model.load_pretrained = original_load_pretrained
    payload = _torch_load(checkpoint_path, map_location=device)
    state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify MRI-level lesion presence from a trained classifier or from "
            "single/ensemble segmentation probability maps."
        )
    )
    parser.add_argument("--mode", choices=["classifier", "segmentation"], default="segmentation")
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--classifier-checkpoint", default=None)
    parser.add_argument("--seg-checkpoint", default=None, help="Override init checkpoint stored in classifier checkpoint.")
    parser.add_argument(
        "--checkpoints",
        "--seg-checkpoints",
        dest="seg_checkpoints",
        nargs="+",
        default=None,
        help=(
            "Segmentation checkpoint(s) for mode=segmentation. One checkpoint runs a "
            "single model; multiple checkpoints are averaged as probability maps."
        ),
    )
    parser.add_argument("--thresholds", "--seg-thresholds", dest="seg_thresholds", nargs="+", type=float, default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--min-components", "--seg-min-components", dest="seg_min_components", nargs="+", type=int, default=[0, 3, 5, 10, 20, 40, 80, 120])
    parser.add_argument("--ensemble-strategy", "--seg-ensemble-strategy", dest="seg_ensemble_strategy", choices=["probability_average", "strategy2_conditional_m2"], default="probability_average")
    parser.add_argument("--m1-thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--m1-min-components", nargs="+", type=int, default=None)
    parser.add_argument("--m2-thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--m2-min-components", nargs="+", type=int, default=None)
    parser.add_argument("--m1-support-thresholds", nargs="+", type=float, default=[0.15])
    parser.add_argument("--support-radii", nargs="+", type=int, default=[0])
    parser.add_argument("--support-min-overlap-voxels", type=int, default=1)
    parser.add_argument("--support-min-overlap-ratio", type=float, default=0.0)
    parser.add_argument("--final-min-components", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--seg-tta",
        "--tta",
        dest="seg_tta",
        choices=TTA_CHOICES,
        default="none",
        help="TTA applied to every segmentation checkpoint unless --checkpoint-ttas is supplied.",
    )
    parser.add_argument(
        "--seg-checkpoint-ttas",
        "--checkpoint-ttas",
        dest="seg_checkpoint_ttas",
        nargs="+",
        choices=TTA_CHOICES,
        default=None,
        help="Per-checkpoint TTA modes in the same order as --checkpoints.",
    )
    parser.add_argument(
        "--seg-load-mode",
        "--load-mode",
        dest="seg_load_mode",
        choices=["preload", "sequential"],
        default="preload",
    )
    parser.add_argument("--dataset-dir", default="Validation2025_100/Validation2025_100")
    parser.add_argument("--output-dir", default="checkpoints/tbi_classifier/depth_vector/validation2025_100")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--aggregation", choices=["max", "topk_mean", "mean"], default="max")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--sw-batch-size",
        type=int,
        default=None,
        help=(
            "Optional segmentation sliding-window batch-size override. When omitted, "
            "config.yml inference.sw_batch_size is preserved."
        ),
    )
    parser.add_argument(
        "--selection-metric",
        choices=["balanced_accuracy", "accuracy", "f1", "sensitivity", "specificity"],
        default="balanced_accuracy",
        help="Metric used to select the best threshold/minCC row from a sweep.",
    )
    parser.add_argument(
        "--require-zero-fn",
        action="store_true",
        help="Select the best setting only among rows with zero missed positive scans, if any exist.",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _aggregate(values: list[float], mode: str, top_k: int) -> float:
    if not values:
        return 0.0
    array = np.asarray(values, dtype=np.float32)
    if mode == "mean":
        return float(array.mean())
    if mode == "topk_mean":
        k = max(1, min(int(top_k), len(array)))
        return float(np.sort(array)[-k:].mean())
    return float(array.max())


@torch.no_grad()
def predict_case_probability(
    model: EncoderGAPClassifier,
    image: np.ndarray,
    patch_size: tuple[int, int, int],
    overlap: float,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    aggregation: str,
    top_k: int,
) -> tuple[float, int]:
    padded, _ = _pad_volume(image, patch_size)
    _, depth, height, width = padded.shape
    stride = [max(1, int(size * (1.0 - float(overlap)))) for size in patch_size]
    z_positions = _sliding_positions(depth, patch_size[0], stride[0])
    y_positions = _sliding_positions(height, patch_size[1], stride[1])
    x_positions = _sliding_positions(width, patch_size[2], stride[2])

    probs: list[float] = []
    patches: list[np.ndarray] = []
    for z in z_positions:
        for y in y_positions:
            for x in x_positions:
                patches.append(
                    padded[
                        :,
                        z : z + patch_size[0],
                        y : y + patch_size[1],
                        x : x + patch_size[2],
                    ]
                )
                if len(patches) >= int(batch_size):
                    probs.extend(_predict_patch_batch(model, patches, device, use_amp))
                    patches = []
    if patches:
        probs.extend(_predict_patch_batch(model, patches, device, use_amp))
    return _aggregate(probs, aggregation, top_k), len(probs)


@torch.no_grad()
def _predict_patch_batch(
    model: EncoderGAPClassifier,
    patches: list[np.ndarray],
    device: torch.device,
    use_amp: bool,
) -> list[float]:
    tensor = torch.from_numpy(np.stack(patches)).to(device=device, dtype=torch.float32).contiguous(memory_format=torch.channels_last_3d)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        logits = model(tensor)
    return [float(value) for value in torch.softmax(logits.float(), dim=1)[:, 1].detach().cpu().tolist()]


def _write_outputs(output_dir: Path, rows: list[dict[str, object]], metrics: dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_case_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "gt_positive", "probability", "prediction", "n_patches", "gt_voxels"],
        )
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (output_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "pred_negative", "pred_positive"])
        writer.writerow(["true_negative", int(metrics["tn"]), int(metrics["fp"])])
        writer.writerow(["true_positive", int(metrics["fn"]), int(metrics["tp"])])


def _classification_metrics_from_predictions(
    y_true: list[int],
    y_prob: list[float],
    y_pred: list[int],
    threshold: float,
) -> dict[str, float]:
    counts = _confusion_counts(y_true, y_pred)
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    negative_predictive_value = tn / max(tn + fn, 1)
    f1 = 2.0 * precision * sensitivity / max(precision + sensitivity, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    balanced_accuracy = 0.5 * (sensitivity + specificity)
    pos_probs = [prob for label, prob in zip(y_true, y_prob) if label == 1]
    neg_probs = [prob for label, prob in zip(y_true, y_prob) if label == 0]
    return {
        **{key: float(value) for key, value in counts.items()},
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "tpr": float(sensitivity),
        "tnr": float(specificity),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "negative_predictive_value": float(negative_predictive_value),
        "f1": float(f1),
        "n_cases": float(len(y_true)),
        "n_positive": float(sum(y_true)),
        "n_empty": float(len(y_true) - sum(y_true)),
        "mean_prob_positive_cases": float(np.mean(pos_probs)) if pos_probs else float("nan"),
        "mean_prob_negative_cases": float(np.mean(neg_probs)) if neg_probs else float("nan"),
    }


def _select_best_segmentation_setting(
    summary_rows: list[dict[str, object]],
    selection_metric: str,
    require_zero_fn: bool,
) -> tuple[dict[str, object], bool]:
    candidates = summary_rows
    zero_fn_applied = False
    if require_zero_fn:
        zero_fn_rows = [row for row in summary_rows if int(row["fn"]) == 0]
        if zero_fn_rows:
            candidates = zero_fn_rows
            zero_fn_applied = True

    def ranking_key(row: dict[str, object]) -> tuple[float, ...]:
        return (
            float(row[selection_metric]),
            float(row["balanced_accuracy"]),
            float(row["sensitivity"]),
            float(row["specificity"]),
            float(row["f1"]),
            float(row["accuracy"]),
            -float(row["fp"]),
            -float(row["fn"]),
        )

    return max(candidates, key=ranking_key), zero_fn_applied


def _filtered_case_probability(
    lesion_probability: np.ndarray,
    threshold: float,
    min_component_voxels: int,
) -> tuple[float, int, int]:
    raw_mask = lesion_probability >= float(threshold)
    filtered = _filter_components(raw_mask.astype(np.uint8), int(min_component_voxels))
    pred_voxels = int(np.count_nonzero(filtered))
    raw_voxels = int(np.count_nonzero(raw_mask))
    if pred_voxels == 0:
        return 0.0, raw_voxels, pred_voxels
    return float(np.max(lesion_probability[filtered > 0])), raw_voxels, pred_voxels


def _predict_probs_by_checkpoint_preload_local(
    models: list[torch.nn.Module],
    image: np.ndarray,
    config,
    device: torch.device,
    use_amp: bool,
    checkpoint_ttas: list[str],
) -> list[np.ndarray]:
    return [
        _predict_probs_preload([model], image, config, device, use_amp, [tta])
        for model, tta in zip(models, checkpoint_ttas)
    ]


def _predict_probs_by_checkpoint_sequential_local(
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
        model = _build_loaded_segmentation_model(config, base_dir, checkpoint_path, device)
        probabilities_by_checkpoint.append(_predict_probs_preload([model], image, config, device, use_amp, [tta]))
        _unload_model(model)
    return probabilities_by_checkpoint


def _predict_probs_sequential_local(
    checkpoint_paths: list[Path],
    image: np.ndarray,
    config,
    base_dir: Path,
    device: torch.device,
    use_amp: bool,
    checkpoint_ttas: list[str],
) -> np.ndarray:
    probabilities_by_checkpoint = _predict_probs_by_checkpoint_sequential_local(
        checkpoint_paths, image, config, base_dir, device, use_amp, checkpoint_ttas
    )
    probs_sum = None
    for probabilities in probabilities_by_checkpoint:
        probs_sum = probabilities if probs_sum is None else probs_sum + probabilities
    return probs_sum / float(len(probabilities_by_checkpoint))


def _probability_from_mask(mask: np.ndarray, *probability_maps: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    values = [float(np.max(probability_map[mask > 0])) for probability_map in probability_maps]
    return max(values) if values else 1.0


def _write_dict_rows(path: Path, rows: list[dict[str, object]], preferred_fields: list[str]) -> None:
    extras = sorted({key for row in rows for key in row.keys()} - set(preferred_fields))
    fieldnames = preferred_fields + extras
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_segmentation_outputs(
    output_dir: Path,
    summary_rows: list[dict[str, object]],
    per_case_rows: list[dict[str, object]],
    best_row: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_fields = [
        "selected",
        "model_mode",
        "n_checkpoints",
        "checkpoint_names",
        "checkpoint_ttas",
        "setting_index",
        "ensemble_strategy",
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
        "tn",
        "fp",
        "fn",
        "tp",
        "accuracy",
        "balanced_accuracy",
        "tpr",
        "tnr",
        "sensitivity",
        "specificity",
        "precision",
        "negative_predictive_value",
        "f1",
        "n_cases",
        "n_positive",
        "n_empty",
        "mean_prob_positive_cases",
        "mean_prob_negative_cases",
    ]
    per_case_fields = [
        "model_mode",
        "checkpoint_names",
        "checkpoint_ttas",
        "setting_index",
        "case_id",
        "gt_positive",
        "gt_voxels",
        "ensemble_strategy",
        "threshold",
        "min_component_voxels",
        "m1_threshold",
        "m1_min_component_voxels",
        "m2_threshold",
        "m2_min_component_voxels",
        "m1_support_threshold",
        "support_radius",
        "final_min_component_voxels",
        "probability",
        "prediction",
        "raw_voxels",
        "pred_voxels",
    ]
    _write_dict_rows(output_dir / "segmentation_calibrator_summary.csv", summary_rows, summary_fields)
    _write_dict_rows(output_dir / "segmentation_calibrator_per_case.csv", per_case_rows, per_case_fields)
    with (output_dir / "segmentation_calibrator_best.json").open("w", encoding="utf-8") as handle:
        json.dump(best_row, handle, indent=2)
    with (output_dir / "best_segmentation_classifier_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(best_row, handle, indent=2)
    with (output_dir / "best_segmentation_classifier_confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "pred_negative", "pred_positive"])
        writer.writerow(["true_negative", int(best_row["tn"]), int(best_row["fp"])])
        writer.writerow(["true_positive", int(best_row["fn"]), int(best_row["tp"])])

    best_setting_index = int(best_row["setting_index"])
    best_case_rows = [
        row for row in per_case_rows
        if int(row["setting_index"]) == best_setting_index
    ]
    _write_dict_rows(output_dir / "best_segmentation_classifier_per_case.csv", best_case_rows, per_case_fields)


def _build_segmentation_detection_settings(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.seg_ensemble_strategy == "strategy2_conditional_m2":
        m1_thresholds = [float(value) for value in (args.m1_thresholds or args.seg_thresholds)]
        m1_min_components = [int(value) for value in (args.m1_min_components or args.seg_min_components)]
        m2_thresholds = [float(value) for value in (args.m2_thresholds or args.seg_thresholds)]
        m2_min_components = [int(value) for value in (args.m2_min_components or args.seg_min_components)]
        settings = []
        for setting_index, (
            m1_threshold,
            m1_mincc,
            m2_threshold,
            m2_mincc,
            support_threshold,
            support_radius,
            final_mincc,
        ) in enumerate(
            itertools.product(
                m1_thresholds,
                m1_min_components,
                m2_thresholds,
                m2_min_components,
                [float(value) for value in args.m1_support_thresholds],
                [int(value) for value in args.support_radii],
                [int(value) for value in args.final_min_components],
            )
        ):
            settings.append(
                {
                    "setting_index": setting_index,
                    "ensemble_strategy": args.seg_ensemble_strategy,
                    "threshold": 0.5,
                    "min_component_voxels": int(final_mincc),
                    "m1_threshold": float(m1_threshold),
                    "m1_min_component_voxels": int(m1_mincc),
                    "m2_threshold": float(m2_threshold),
                    "m2_min_component_voxels": int(m2_mincc),
                    "m1_support_threshold": float(support_threshold),
                    "support_radius": int(support_radius),
                    "support_min_overlap_voxels": int(args.support_min_overlap_voxels),
                    "support_min_overlap_ratio": float(args.support_min_overlap_ratio),
                    "final_min_component_voxels": int(final_mincc),
                }
            )
        return settings

    return [
        {
            "setting_index": setting_index,
            "ensemble_strategy": args.seg_ensemble_strategy,
            "threshold": float(threshold),
            "min_component_voxels": int(mincc),
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
        for setting_index, (threshold, mincc) in enumerate(
            itertools.product([float(value) for value in args.seg_thresholds], [int(value) for value in args.seg_min_components])
        )
    ]


def _format_segmentation_detection_setting(row: dict[str, object]) -> str:
    if str(row.get("ensemble_strategy")) == "strategy2_conditional_m2":
        return (
            f"m1_thr={float(row['m1_threshold']):.3f} "
            f"m1_mincc={int(row['m1_min_component_voxels'])} "
            f"m2_thr={float(row['m2_threshold']):.3f} "
            f"m2_mincc={int(row['m2_min_component_voxels'])} "
            f"support={float(row['m1_support_threshold']):.3f} "
            f"radius={int(row['support_radius'])} "
            f"final_mincc={int(row['final_min_component_voxels'])}"
        )
    return f"thr={float(row['threshold']):.3f} mincc={int(row['min_component_voxels'])}"


def run_segmentation_calibrator(args: argparse.Namespace) -> None:
    if not args.seg_checkpoints:
        raise ValueError("For --mode segmentation, pass --seg-checkpoints.")

    config = load_config(args.config)
    base_dir = Path(args.config).expanduser().resolve().parent
    output_dir = Path(args.output_dir).expanduser().resolve()
    configure_torch_for_speed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(not args.no_amp and device.type == "cuda")
    if args.sw_batch_size is not None:
        if int(args.sw_batch_size) < 1:
            raise ValueError("--sw-batch-size must be at least 1.")
        config.inference.sw_batch_size = int(args.sw_batch_size)

    if not args.seg_thresholds or any(not 0.0 <= float(value) <= 1.0 for value in args.seg_thresholds):
        raise ValueError("Every --thresholds value must be in [0, 1].")
    if not args.seg_min_components or any(int(value) < 0 for value in args.seg_min_components):
        raise ValueError("Every --min-components value must be non-negative.")

    checkpoint_paths = [resolve_path(base_dir, path) for path in args.seg_checkpoints]
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if args.seg_ensemble_strategy == "strategy2_conditional_m2" and len(checkpoint_paths) != 2:
        raise ValueError("--ensemble-strategy strategy2_conditional_m2 requires exactly two checkpoints: M1 first, M2 second.")
    if args.seg_checkpoint_ttas is not None:
        if len(args.seg_checkpoint_ttas) != len(checkpoint_paths):
            raise ValueError(
                f"--seg-checkpoint-ttas must match --seg-checkpoints length: "
                f"{len(args.seg_checkpoint_ttas)} modes for {len(checkpoint_paths)} checkpoints."
            )
        checkpoint_ttas = [str(value) for value in args.seg_checkpoint_ttas]
    else:
        checkpoint_ttas = [str(args.seg_tta) for _ in checkpoint_paths]

    model_mode = "single" if len(checkpoint_paths) == 1 else "ensemble"
    checkpoint_names = [path.name for path in checkpoint_paths]
    run_metadata: dict[str, object] = {
        "model_mode": model_mode,
        "n_checkpoints": len(checkpoint_paths),
        "checkpoint_names": " | ".join(checkpoint_names),
        "checkpoint_ttas": " | ".join(checkpoint_ttas),
    }

    records = discover_cases(resolve_path(base_dir, args.dataset_dir))
    cache_dir = resolve_path(base_dir, args.cache_dir) if args.cache_dir else (
        resolve_path(base_dir, config.paths.cache_dir) if config.data.cache_preprocessed else None
    )
    loaded_models = None
    if args.seg_load_mode == "preload":
        loaded_models = [_build_loaded_segmentation_model(config, base_dir, checkpoint_path, device) for checkpoint_path in checkpoint_paths]

    settings = _build_segmentation_detection_settings(args)
    for setting in settings:
        setting.update(run_metadata)
    y_true_by_setting: dict[int, list[int]] = {int(setting["setting_index"]): [] for setting in settings}
    y_prob_by_setting: dict[int, list[float]] = {int(setting["setting_index"]): [] for setting in settings}
    y_pred_by_setting: dict[int, list[int]] = {int(setting["setting_index"]): [] for setting in settings}
    per_case_rows: list[dict[str, object]] = []

    print(f"[INFO] Mode        : segmentation-output calibrator")
    print(f"[INFO] Dataset     : {resolve_path(base_dir, args.dataset_dir)}")
    print(f"[INFO] Cases       : {len(records)}")
    print(f"[INFO] Model mode  : {model_mode}")
    print(f"[INFO] Checkpoints : {checkpoint_names}")
    print(f"[INFO] Strategy    : {args.seg_ensemble_strategy}")
    print(f"[INFO] Ensemble    : probability-map average or M1-supported M2 components before classification")
    print(f"[INFO] TTA modes   : {checkpoint_ttas}")
    if args.seg_ensemble_strategy == "strategy2_conditional_m2":
        effective_m1_thresholds = args.m1_thresholds or args.seg_thresholds
        effective_m1_min_components = args.m1_min_components or args.seg_min_components
        effective_m2_thresholds = args.m2_thresholds or args.seg_thresholds
        effective_m2_min_components = args.m2_min_components or args.seg_min_components
        print(
            f"[INFO] Hybrid M1/M2: m1_tta={checkpoint_ttas[0]} m2_tta={checkpoint_ttas[1]} "
            f"support={args.m1_support_thresholds} radii={args.support_radii}"
        )
        print(f"[INFO] M1 thresholds/mincc : {effective_m1_thresholds} / {effective_m1_min_components}")
        print(f"[INFO] M2 thresholds/mincc : {effective_m2_thresholds} / {effective_m2_min_components}")
        print(f"[INFO] Final mincc         : {args.final_min_components}")
        print(f"[INFO] Sweep settings      : {len(settings)}")
    else:
        print(f"[INFO] Thresholds  : {args.seg_thresholds}")
        print(f"[INFO] Min comp    : {args.seg_min_components}")
        print(f"[INFO] Sweep settings: {len(settings)}")
    print(f"[INFO] Device      : {device} | AMP={use_amp} | load_mode={args.seg_load_mode}")
    print(f"[INFO] SW batch    : {config.inference.sw_batch_size}")
    print(
        f"[INFO] Selection   : metric={args.selection_metric} "
        f"require_zero_fn={bool(args.require_zero_fn)}"
    )

    try:
        for record in tqdm(records, desc="Segmentation-output calibrator"):
            image, mask, _, _ = load_case_cached(
                case=record,
                target_spacing=config.data.target_spacing,
                include_dmri=config.data.include_dmri,
                dmri_reduce=config.data.dmri_reduce,
                dmri_b0_threshold=config.data.dmri_b0_threshold,
                normalize_foreground_only=config.data.normalize_foreground_only,
                cache_dir=cache_dir,
            )
            probabilities_by_checkpoint = None
            lesion_probability = None
            if args.seg_ensemble_strategy == "strategy2_conditional_m2":
                if loaded_models is not None:
                    probabilities_by_checkpoint = _predict_probs_by_checkpoint_preload_local(
                        loaded_models, image, config, device, use_amp, checkpoint_ttas
                    )
                else:
                    probabilities_by_checkpoint = _predict_probs_by_checkpoint_sequential_local(
                        checkpoint_paths, image, config, base_dir, device, use_amp, checkpoint_ttas
                    )
            else:
                if loaded_models is not None:
                    probabilities = _predict_probs_preload(loaded_models, image, config, device, use_amp, checkpoint_ttas)
                else:
                    probabilities = _predict_probs_sequential_local(checkpoint_paths, image, config, base_dir, device, use_amp, checkpoint_ttas)
                lesion_probability = probabilities[1]
            gt_positive = int(np.any(mask > 0))
            gt_voxels = int(np.count_nonzero(mask > 0))
            for setting in settings:
                setting_index = int(setting["setting_index"])
                row_extra: dict[str, object] = {}
                if args.seg_ensemble_strategy == "strategy2_conditional_m2":
                    if probabilities_by_checkpoint is None:
                        raise RuntimeError("Internal error: missing per-checkpoint probabilities for strategy2_conditional_m2.")
                    m1_probability = probabilities_by_checkpoint[0][1]
                    m2_probability = probabilities_by_checkpoint[1][1]
                    prediction_mask, stats = conditional_m2_component_acceptance(
                        m1_probability=m1_probability,
                        m2_probability=m2_probability,
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
                    pred_voxels = int(np.count_nonzero(prediction_mask > 0))
                    raw_voxels = int(stats.m1_pred_voxels + stats.m2_pred_voxels)
                    probability = _probability_from_mask(prediction_mask, m1_probability, m2_probability)
                    prediction = int(pred_voxels > 0)
                    row_extra = {
                        "m1_pred_voxels": int(stats.m1_pred_voxels),
                        "m2_pred_voxels": int(stats.m2_pred_voxels),
                        "accepted_m2_components": int(stats.accepted_m2_components),
                        "rejected_m2_components": int(stats.rejected_m2_components),
                        "accepted_m2_voxels": int(stats.accepted_m2_voxels),
                        "rejected_m2_voxels": int(stats.rejected_m2_voxels),
                    }
                else:
                    if lesion_probability is None:
                        raise RuntimeError("Internal error: missing averaged lesion probability map.")
                    probability, raw_voxels, pred_voxels = _filtered_case_probability(
                        lesion_probability=lesion_probability,
                        threshold=float(setting["threshold"]),
                        min_component_voxels=int(setting["min_component_voxels"]),
                    )
                    prediction = int(pred_voxels > 0)
                y_true_by_setting[setting_index].append(gt_positive)
                y_prob_by_setting[setting_index].append(probability)
                y_pred_by_setting[setting_index].append(prediction)
                per_case_rows.append(
                    {
                        **setting,
                        "case_id": record.case_id,
                        "gt_positive": gt_positive,
                        "gt_voxels": gt_voxels,
                        "probability": probability,
                        "prediction": prediction,
                        "raw_voxels": raw_voxels,
                        "pred_voxels": pred_voxels,
                        **row_extra,
                    }
                )
    finally:
        if loaded_models is not None:
            for model in loaded_models:
                _unload_model(model)

    summary_rows: list[dict[str, object]] = []
    for setting in settings:
        setting_index = int(setting["setting_index"])
        metrics = _classification_metrics_from_predictions(
            y_true_by_setting[setting_index],
            y_prob_by_setting[setting_index],
            y_pred_by_setting[setting_index],
            threshold=float(setting["threshold"]),
        )
        summary_rows.append(
            {
                **setting,
                **metrics,
            }
        )
    best_row, zero_fn_applied = _select_best_segmentation_setting(
        summary_rows=summary_rows,
        selection_metric=str(args.selection_metric),
        require_zero_fn=bool(args.require_zero_fn),
    )
    best_setting_index = int(best_row["setting_index"])
    for row in summary_rows:
        row["selected"] = int(int(row["setting_index"]) == best_setting_index)
    best_row["selection_metric"] = str(args.selection_metric)
    best_row["require_zero_fn"] = bool(args.require_zero_fn)
    best_row["zero_fn_filter_applied"] = bool(zero_fn_applied)
    _write_segmentation_outputs(output_dir, summary_rows, per_case_rows, best_row)
    run_manifest = {
        "dataset_dir": str(resolve_path(base_dir, args.dataset_dir)),
        "output_dir": str(output_dir),
        "model_mode": model_mode,
        "ensemble_strategy": str(args.seg_ensemble_strategy),
        "checkpoints": [str(path) for path in checkpoint_paths],
        "checkpoint_ttas": checkpoint_ttas,
        "thresholds": [float(value) for value in args.seg_thresholds],
        "min_components": [int(value) for value in args.seg_min_components],
        "selection_metric": str(args.selection_metric),
        "require_zero_fn": bool(args.require_zero_fn),
        "zero_fn_filter_applied": bool(zero_fn_applied),
        "device": str(device),
        "amp": bool(use_amp),
        "load_mode": str(args.seg_load_mode),
        "sw_batch_size": int(config.inference.sw_batch_size),
    }
    with (output_dir / "segmentation_calibrator_run.json").open("w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, indent=2)
    print(
        "Best setting: "
        f"{_format_segmentation_detection_setting(best_row)} "
        f"cm=[[TN={int(best_row['tn'])}, FP={int(best_row['fp'])}], "
        f"[FN={int(best_row['fn'])}, TP={int(best_row['tp'])}]] "
        f"TPR={float(best_row['tpr']):.4f} TNR={float(best_row['tnr']):.4f} "
        f"F1={float(best_row['f1']):.4f} balanced={float(best_row['balanced_accuracy']):.4f}"
    )
    print(
        "Saved best confusion matrix: "
        f"{output_dir / 'best_segmentation_classifier_confusion_matrix.csv'}"
    )
    print(f"Wrote segmentation-output calibrator outputs to {output_dir}")


def run_classifier_validation(args: argparse.Namespace) -> None:
    if args.classifier_checkpoint is None:
        raise ValueError("For --mode classifier, pass --classifier-checkpoint.")
    config = load_config(args.config)
    base_dir = Path(args.config).expanduser().resolve().parent
    output_dir = Path(args.output_dir).expanduser().resolve()
    configure_torch_for_speed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(not args.no_amp and device.type == "cuda")

    checkpoint_path = Path(args.classifier_checkpoint).expanduser().resolve()
    checkpoint = _torch_load(checkpoint_path, map_location="cpu")
    patch_size = tuple(int(value) for value in checkpoint.get("patch_size", config.data.patch_size))
    threshold = float(args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.5))
    architecture_name = str(checkpoint.get("architecture_name", checkpoint.get("architecture", "Depth Vector")))
    seg_checkpoint = args.seg_checkpoint or checkpoint.get("init_checkpoint") or checkpoint.get("seg_checkpoint")
    if not seg_checkpoint:
        raise ValueError("Init checkpoint is missing. Pass --seg-checkpoint or store init_checkpoint in the classifier checkpoint.")

    seg_model = build_model(config, base_dir).to(device).eval()
    _load_segmentation_checkpoint(seg_model, resolve_path(base_dir, seg_checkpoint))
    with torch.no_grad():
        dummy = torch.zeros((1, int(config.model.in_channels), *patch_size), device=device)
        inferred_feature_depth = int(encoder_feature(seg_model, dummy).shape[2])
    model = EncoderGAPClassifier(
        seg_model=seg_model,
        feature_channels=int(checkpoint["feature_channels"]),
        feature_depth=int(checkpoint.get("feature_depth", inferred_feature_depth)),
        dropout=0.0,
        head_hidden_dim=int(checkpoint.get("args", {}).get("head_hidden_dim", 0)),
        freeze_encoder=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    print(f"[INFO] Architecture: {architecture_name}")

    records = discover_cases(resolve_path(base_dir, args.dataset_dir))
    cache_dir = resolve_path(base_dir, args.cache_dir) if args.cache_dir else (
        resolve_path(base_dir, config.paths.cache_dir) if config.data.cache_preprocessed else None
    )
    rows: list[dict[str, object]] = []
    y_true: list[int] = []
    y_prob: list[float] = []
    for record in tqdm(records, desc="Validation2025 classifier"):
        image, mask, _, _ = load_case_cached(
            case=record,
            target_spacing=config.data.target_spacing,
            include_dmri=config.data.include_dmri,
            dmri_reduce=config.data.dmri_reduce,
            dmri_b0_threshold=config.data.dmri_b0_threshold,
            normalize_foreground_only=config.data.normalize_foreground_only,
            cache_dir=cache_dir,
        )
        probability, n_patches = predict_case_probability(
            model=model,
            image=image,
            patch_size=patch_size,
            overlap=float(args.overlap),
            batch_size=int(args.batch_size),
            device=device,
            use_amp=use_amp,
            aggregation=str(args.aggregation),
            top_k=int(args.top_k),
        )
        gt_positive = int(np.any(mask > 0))
        prediction = int(probability >= threshold)
        y_true.append(gt_positive)
        y_prob.append(probability)
        rows.append(
            {
                "case_id": record.case_id,
                "gt_positive": gt_positive,
                "probability": probability,
                "prediction": prediction,
                "n_patches": n_patches,
                "gt_voxels": int(np.count_nonzero(mask > 0)),
            }
        )

    metrics = _classification_metrics(y_true, y_prob, threshold)
    metrics.update(
        {
            "n_cases": float(len(records)),
            "aggregation": str(args.aggregation),
            "top_k": float(args.top_k),
            "overlap": float(args.overlap),
        }
    )
    _write_outputs(output_dir, rows, metrics)
    print(
        "Confusion matrix [[TN, FP], [FN, TP]] = "
        f"[[{int(metrics['tn'])}, {int(metrics['fp'])}], [{int(metrics['fn'])}, {int(metrics['tp'])}]]"
    )
    print(
        f"TPR/sensitivity={metrics['tpr']:.4f} "
        f"TNR/specificity={metrics['tnr']:.4f} "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f} "
        f"accuracy={metrics['accuracy']:.4f}"
    )
    print(f"Wrote classifier validation outputs to {output_dir}")


def main() -> None:
    args = parse_args()
    if args.mode == "segmentation":
        run_segmentation_calibrator(args)
    else:
        run_classifier_validation(args)


if __name__ == "__main__":
    main()


'''
python validate_classifier.py \
  --mode segmentation \
  --config config.yml \
  --checkpoints checkpoints/tbi_multitalentv2/fold_0/best_val_f0_e5ohnz5w.pt \
  --checkpoint-ttas flips \
  --thresholds 0.50 \
  --min-components 3 \
  --dataset-dir Validation2025_100/Validation2025_100 \
  --output-dir checkpoints/segmentation_classifier/model_B_tta
'''


'''
python validate_classifier.py \
  --mode segmentation \
  --config config.yml \
  --ensemble-strategy probability_average \
  --checkpoints \
    checkpoints/tbi_multitalentv2/fold_0/best_val_f0_e5ohnz5w.pt \
    archive/Approach_2_Ensemble_Callibrater/checkpoints/trained_models/best_ddp_fft_finetuned_kpcyjb66_data_leaked_0.54_rank1_leaderboard.pt \
  --checkpoint-ttas flips flips \
  --thresholds 0.5 \
  --min-components 3 \
  --dataset-dir Validation2025_100/Validation2025_100 \
  --load-mode preload \
  --sw-batch-size 2 \
  --output-dir checkpoints/segmentation_classifier/hybrid_A_tta_B_tta_thr050_mincc3
'''