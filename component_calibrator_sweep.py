from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from inference_sweep import (
    _build_loaded_models,
    _gpu_memory_gb,
    _predict_ensemble_probs_loaded,
    _predict_ensemble_probs_sequential,
    _save_prediction,
    _unload_model,
)
from multitalent_tbi.case_filters import build_case_infos
from multitalent_tbi.component_calibrator import (
    apply_component_decisions,
    case_metric_row,
    extract_component_features,
    load_calibrator,
    predict_component_probabilities,
    summarize_case_metrics,
    write_csv,
)
from multitalent_tbi.config import load_config, resolve_path
from multitalent_tbi.data import discover_cases, load_case_cached, load_nifti_robust
from multitalent_tbi.splits import load_splits, split_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase-H: sweep component-calibrator thresholds on a fold split."
    )
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--positive-only", action="store_true")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--calibrator", required=True)
    parser.add_argument("--candidate-threshold", type=float, default=None)
    parser.add_argument("--calibrator-thresholds", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.30, 0.50])
    parser.add_argument("--min-components", nargs="+", type=int, default=[0])
    parser.add_argument("--tta", choices=["none", "flips"], default="flips")
    parser.add_argument("--load-mode", choices=["auto", "preload", "sequential"], default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--save-calibrator-threshold", type=float, default=0.10)
    parser.add_argument("--save-min-component", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _resolve_load_mode(args: argparse.Namespace, device: torch.device, checkpoint_count: int) -> str:
    if args.load_mode != "auto":
        return args.load_mode
    if device.type == "cuda" and _gpu_memory_gb(device) >= 35.0:
        return "preload"
    if checkpoint_count == 1:
        return "preload"
    return "sequential"


def _select_records(config, base_dir: Path, fold: int, split_name: str):
    records = discover_cases(resolve_path(base_dir, config.paths.dataset_dir))
    if split_name == "all":
        return records
    splits = load_splits(resolve_path(base_dir, config.paths.splits_file))
    train_records, val_records = split_records(records, splits[fold])
    return val_records if split_name == "val" else train_records


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    base_dir = Path.cwd()
    fold = int(args.fold if args.fold is not None else config.training.fold)
    output_dir = (
        resolve_path(base_dir, args.output_dir)
        if args.output_dir
        else resolve_path(base_dir, config.paths.work_dir) / "component_calibrator_sweeps" / f"fold_{fold}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    calibrator_payload = load_calibrator(resolve_path(base_dir, args.calibrator))
    candidate_threshold = (
        float(args.candidate_threshold)
        if args.candidate_threshold is not None
        else float(calibrator_payload["candidate_threshold"])
    )
    min_candidate_voxels = int(calibrator_payload["min_component_voxels"])
    min_true_overlap_voxels = int(calibrator_payload["min_true_overlap_voxels"])

    checkpoint_paths = [resolve_path(base_dir, checkpoint) for checkpoint in args.checkpoints]
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

    records = _select_records(config, base_dir, fold, args.split)
    infos = build_case_infos(records, split=args.split)
    if args.positive_only:
        infos = [info for info in infos if info.gt_voxels > 0]
        records = [info.record for info in infos]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp
    load_mode = _resolve_load_mode(args, device, len(checkpoint_paths))
    print(f"[INFO] Cases: {len(infos)} | split={args.split} | positive_only={args.positive_only}")
    print(f"[INFO] Device: {device} | AMP={use_amp} | TTA={args.tta} | load_mode={load_mode}")
    print(f"[INFO] Candidate threshold={candidate_threshold} | candidate min voxels={min_candidate_voxels}")
    print(f"[INFO] Calibrator thresholds={args.calibrator_thresholds}")
    print(f"[INFO] Min components={args.min_components}")

    loaded_models = None
    if load_mode == "preload":
        loaded_models = _build_loaded_models(config, base_dir, checkpoint_paths, device)

    per_case_rows: list[dict[str, object]] = []
    component_rows_out: list[dict[str, object]] = []
    try:
        for info in tqdm(infos, desc="Calibrator sweep"):
            record = info.record
            original_image = load_nifti_robust(record.t1_path)
            image, mask, image_affine, _ = load_case_cached(
                case=record,
                target_spacing=tuple(config.data.target_spacing),
                include_dmri=bool(config.data.include_dmri),
                dmri_reduce=str(config.data.dmri_reduce),
                dmri_b0_threshold=float(config.data.dmri_b0_threshold),
                normalize_foreground_only=bool(config.data.normalize_foreground_only),
                cache_dir=resolve_path(base_dir, config.paths.cache_dir)
                if config.data.cache_preprocessed
                else None,
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
            candidate_mask = lesion_probability >= candidate_threshold
            component_rows = extract_component_features(
                case_id=record.case_id,
                lesion_probability=lesion_probability,
                image=image,
                target=mask,
                candidate_threshold=candidate_threshold,
                min_component_voxels=min_candidate_voxels,
                min_true_overlap_voxels=min_true_overlap_voxels,
            )
            component_probabilities = predict_component_probabilities(calibrator_payload, component_rows)
            for row, component_probability in zip(component_rows, component_probabilities):
                row_out = dict(row)
                row_out["calibrator_probability"] = float(component_probability)
                component_rows_out.append(row_out)

            save_prediction_written = False
            for calibrator_threshold, min_component_voxels in itertools.product(
                args.calibrator_thresholds, args.min_components
            ):
                prediction, kept_components, rejected_components = apply_component_decisions(
                    candidate_mask=candidate_mask,
                    component_rows=component_rows,
                    component_probabilities=component_probabilities,
                    calibrator_threshold=float(calibrator_threshold),
                    min_component_voxels=int(min_component_voxels),
                )
                per_case_rows.append(
                    case_metric_row(
                        case_id=record.case_id,
                        prediction=prediction,
                        target=mask,
                        gt_voxels=info.gt_voxels,
                        candidate_threshold=candidate_threshold,
                        calibrator_threshold=float(calibrator_threshold),
                        min_component_voxels=int(min_component_voxels),
                        kept_components=kept_components,
                        rejected_components=rejected_components,
                    )
                )
                if (
                    args.save_predictions
                    and not save_prediction_written
                    and abs(float(calibrator_threshold) - float(args.save_calibrator_threshold)) < 1e-8
                    and int(min_component_voxels) == int(args.save_min_component)
                ):
                    _save_prediction(output_dir, record, prediction, image_affine, original_image)
                    save_prediction_written = True
    finally:
        if loaded_models is not None:
            for model in loaded_models:
                _unload_model(model)

    summary_rows = summarize_case_metrics(per_case_rows)
    write_csv(output_dir / "per_case_metrics.csv", per_case_rows)
    write_csv(output_dir / "summary_metrics.csv", summary_rows)
    write_csv(output_dir / "component_predictions.csv", component_rows_out)

    print("\n=== Calibrator Summary ===")
    for row in summary_rows:
        print(
            f"cal={row['calibrator_threshold']:.3f} mincc={row['min_component_voxels']:>3} "
            f"all={row['mean_dice_all']:.4f} "
            f"pos={row['mean_dice_positive']:.4f} "
            f"gt50={row['mean_dice_gt50']:.4f} "
            f"tiny={row['mean_dice_tiny']:.4f} "
            f"small={row['mean_dice_small']:.4f} "
            f"large={row['mean_dice_large']:.4f} "
            f"missed={row['n_missed_positive']} "
            f"empty_fp={row['n_empty_false_positive']}"
        )
    print(f"\n[INFO] Per-case metrics: {output_dir / 'per_case_metrics.csv'}")
    print(f"[INFO] Summary metrics : {output_dir / 'summary_metrics.csv'}")
    print(f"[INFO] Components      : {output_dir / 'component_predictions.csv'}")


if __name__ == "__main__":
    main()
