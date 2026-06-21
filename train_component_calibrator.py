from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from inference_sweep import (
    _build_loaded_models,
    _gpu_memory_gb,
    _predict_ensemble_probs_loaded,
    _predict_ensemble_probs_sequential,
    _unload_model,
)
from multitalent_tbi.case_filters import build_case_infos
from multitalent_tbi.component_calibrator import (
    extract_component_features,
    save_calibrator,
    train_random_forest_calibrator,
    write_csv,
)
from multitalent_tbi.config import load_config, resolve_path
from multitalent_tbi.data import discover_cases, load_case_cached
from multitalent_tbi.splits import load_splits, split_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase-H: train a component-level lesion/false-positive calibrator."
    )
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--split", choices=["train", "val", "all"], default="train")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--candidate-threshold", type=float, default=0.10)
    parser.add_argument("--min-candidate-voxels", type=int, default=1)
    parser.add_argument("--min-true-overlap-voxels", type=int, default=1)
    parser.add_argument("--tta", choices=["none", "flips"], default="flips")
    parser.add_argument("--load-mode", choices=["auto", "preload", "sequential"], default="auto")
    parser.add_argument("--positive-weight", type=float, default=8.0)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
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
    return train_records if split_name == "train" else val_records


def main() -> None:
    args = parse_args()
    base_dir = Path.cwd()
    config = load_config(args.config)
    fold = int(args.fold if args.fold is not None else config.training.fold)
    output_dir = (
        resolve_path(base_dir, args.output_dir)
        if args.output_dir
        else resolve_path(base_dir, config.paths.work_dir) / "component_calibrator" / f"fold_{fold}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = [resolve_path(base_dir, checkpoint) for checkpoint in args.checkpoints]
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

    records = _select_records(config, base_dir, fold, args.split)
    if args.max_cases is not None:
        records = records[: int(args.max_cases)]
    infos = build_case_infos(records, split=args.split)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp
    load_mode = _resolve_load_mode(args, device, len(checkpoint_paths))
    print(f"[INFO] Cases: {len(records)} | split={args.split} | fold={fold}")
    print(f"[INFO] Device: {device} | AMP={use_amp} | TTA={args.tta} | load_mode={load_mode}")
    print(f"[INFO] Candidate threshold={args.candidate_threshold} min_candidate_voxels={args.min_candidate_voxels}")

    loaded_models = None
    if load_mode == "preload":
        loaded_models = _build_loaded_models(config, base_dir, checkpoint_paths, device)

    all_rows: list[dict[str, object]] = []
    try:
        for info in tqdm(infos, desc="Mining component features"):
            record = info.record
            image, mask, _, _ = load_case_cached(
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

            rows = extract_component_features(
                case_id=record.case_id,
                lesion_probability=mean_probs[1],
                image=image,
                target=mask,
                candidate_threshold=float(args.candidate_threshold),
                min_component_voxels=int(args.min_candidate_voxels),
                min_true_overlap_voxels=int(args.min_true_overlap_voxels),
            )
            all_rows.extend(rows)
    finally:
        if loaded_models is not None:
            for model in loaded_models:
                _unload_model(model)

    feature_csv = output_dir / "component_features_train.csv"
    write_csv(feature_csv, all_rows)
    positives = sum(int(row["label"]) for row in all_rows)
    negatives = len(all_rows) - positives
    print(f"[INFO] Components: {len(all_rows)} | positive={positives} | negative={negatives}")
    print(f"[INFO] Feature CSV: {feature_csv}")

    model = train_random_forest_calibrator(
        all_rows,
        positive_weight=float(args.positive_weight),
        n_estimators=int(args.n_estimators),
        seed=int(config.training.seed),
    )
    calibrator_path = output_dir / "component_calibrator.joblib"
    metadata = {
        "fold": fold,
        "split": args.split,
        "cases": len(records),
        "components": len(all_rows),
        "positive_components": positives,
        "negative_components": negatives,
        "checkpoints": [str(path) for path in checkpoint_paths],
        "tta": args.tta,
        "load_mode": load_mode,
        "positive_weight": float(args.positive_weight),
        "n_estimators": int(args.n_estimators),
    }
    save_calibrator(
        calibrator_path,
        model,
        candidate_threshold=float(args.candidate_threshold),
        min_component_voxels=int(args.min_candidate_voxels),
        min_true_overlap_voxels=int(args.min_true_overlap_voxels),
        metadata=metadata,
    )
    (output_dir / "training_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[INFO] Calibrator: {calibrator_path}")


if __name__ == "__main__":
    main()
