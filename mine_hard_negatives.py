from __future__ import annotations

import argparse
import csv
import json
import gc
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import label
from tqdm.auto import tqdm

from inference_sweep import (
    _build_loaded_models,
    _predict_ensemble_probs_loaded,
    _predict_ensemble_probs_sequential,
    _unload_model,
)
from multitalent_tbi.case_filters import build_case_infos, empty_infos
from multitalent_tbi.config import load_config, resolve_path
from multitalent_tbi.data import discover_cases, load_case_cached
from multitalent_tbi.engine import configure_torch_for_speed, set_seed
from multitalent_tbi.splits import load_splits, split_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mine false-positive patches on empty training MRIs for Segmenter A "
            "hard-negative fine-tuning."
        )
    )
    parser.add_argument("--config", default="config.yml", help="Path to config.yml.")
    parser.add_argument("--fold", type=int, default=None, help="Fold index. Defaults to config training.fold.")
    parser.add_argument(
        "--split",
        choices=["train", "val", "all"],
        default="train",
        help="Split to mine. Use train for hard-negative fine-tuning.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="One or more checkpoints. Probabilities are averaged across checkpoints.",
    )
    parser.add_argument("--threshold", type=float, default=0.10, help="Lesion probability threshold for FP mining.")
    parser.add_argument("--min-component-voxels", type=int, default=3, help="Ignore components smaller than this.")
    parser.add_argument("--max-components-per-case", type=int, default=8, help="Keep top-K FP components per empty case.")
    parser.add_argument("--max-cases", type=int, default=0, help="Optional debug limit for empty cases.")
    parser.add_argument(
        "--load-mode",
        choices=["auto", "preload", "sequential"],
        default="auto",
        help="preload for A6000, sequential for low-memory GPUs.",
    )
    parser.add_argument("--output-csv", default=None, help="Output CSV path.")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP.")
    return parser.parse_args()


def _gpu_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.get_device_properties(device).total_memory) / 1e9


def _resolve_load_mode(mode: str, device: torch.device, checkpoint_count: int) -> str:
    if mode != "auto":
        return mode
    if checkpoint_count == 1:
        return "preload"
    if device.type == "cuda" and _gpu_memory_gb(device) >= 35.0:
        return "preload"
    return "sequential"


def _select_records(split: str, records, train_records, val_records):
    if split == "train":
        return train_records
    if split == "val":
        return val_records
    return records


def _filter_mask(mask: np.ndarray, min_component_voxels: int) -> np.ndarray:
    if min_component_voxels <= 1 or not mask.any():
        return mask.astype(np.uint8)
    components, n_components = label(mask.astype(bool))
    filtered = np.zeros_like(mask, dtype=np.uint8)
    for component_index in range(1, n_components + 1):
        component = components == component_index
        if int(component.sum()) >= min_component_voxels:
            filtered[component] = 1
    return filtered


def _component_rows(
    case_id: str,
    lesion_probs: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    min_component_voxels: int,
    max_components_per_case: int,
    fold: int,
    split: str,
    checkpoint_names: list[str],
) -> list[dict[str, object]]:
    components, n_components = label(mask.astype(bool))
    rows: list[dict[str, object]] = []
    for component_index in range(1, n_components + 1):
        component = components == component_index
        component_voxels = int(component.sum())
        if component_voxels < min_component_voxels:
            continue

        coords = np.argwhere(component)
        component_probs = lesion_probs[component]
        peak_offset = int(np.argmax(component_probs))
        peak_coord = coords[peak_offset]
        centroid = coords.mean(axis=0)

        rows.append(
            {
                "fold": fold,
                "split": split,
                "case_id": case_id,
                "component_index": component_index,
                "component_voxels": component_voxels,
                "center_i": int(peak_coord[0]),
                "center_j": int(peak_coord[1]),
                "center_k": int(peak_coord[2]),
                "centroid_i": float(centroid[0]),
                "centroid_j": float(centroid[1]),
                "centroid_k": float(centroid[2]),
                "max_prob": float(component_probs.max()),
                "mean_prob": float(component_probs.mean()),
                "threshold": threshold,
                "min_component_voxels": min_component_voxels,
                "image_i": int(mask.shape[0]),
                "image_j": int(mask.shape[1]),
                "image_k": int(mask.shape[2]),
                "checkpoints": "|".join(checkpoint_names),
            }
        )

    rows.sort(key=lambda row: (float(row["max_prob"]), int(row["component_voxels"])), reverse=True)
    if max_components_per_case > 0:
        rows = rows[:max_components_per_case]
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fold",
        "split",
        "case_id",
        "component_index",
        "component_voxels",
        "center_i",
        "center_j",
        "center_k",
        "centroid_i",
        "centroid_j",
        "centroid_k",
        "max_prob",
        "mean_prob",
        "threshold",
        "min_component_voxels",
        "image_i",
        "image_j",
        "image_k",
        "checkpoints",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    base_dir = Path(args.config).expanduser().resolve().parent
    fold = int(args.fold if args.fold is not None else config.training.fold)
    seed = int(config.training.seed) + fold
    set_seed(seed)
    configure_torch_for_speed()

    dataset_dir = resolve_path(base_dir, config.paths.dataset_dir)
    splits_path = resolve_path(base_dir, config.paths.splits_file)
    work_dir = resolve_path(base_dir, config.paths.work_dir)
    output_csv = (
        resolve_path(base_dir, args.output_csv)
        if args.output_csv
        else work_dir / "hard_negatives" / f"fold_{fold}" / f"{args.split}_empty_fp_components.csv"
    )

    records = discover_cases(dataset_dir)
    splits = load_splits(splits_path)
    train_records, val_records = split_records(records, splits[fold])
    selected_records = _select_records(args.split, records, train_records, val_records)
    empty_cases = empty_infos(build_case_infos(selected_records, split=args.split))
    if args.max_cases > 0:
        empty_cases = empty_cases[: args.max_cases]

    checkpoint_paths = [resolve_path(base_dir, path) for path in args.checkpoints]
    missing = [str(path) for path in checkpoint_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Checkpoint(s) not found: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda" and bool(config.inference.use_amp)
    load_mode = _resolve_load_mode(args.load_mode, device, len(checkpoint_paths))
    checkpoint_names = [path.name for path in checkpoint_paths]

    print(f"[INFO] Mining split={args.split} fold={fold} empty_cases={len(empty_cases)}")
    print(f"[INFO] Device={device} load_mode={load_mode} threshold={args.threshold}")
    print(f"[INFO] Checkpoints={checkpoint_names}")

    models = []
    if load_mode == "preload":
        models = _build_loaded_models(config, base_dir, checkpoint_paths, device)

    cache_dir = resolve_path(base_dir, config.paths.cache_dir) if config.data.cache_preprocessed else None
    rows: list[dict[str, object]] = []
    cases_with_fp = 0
    try:
        for info in tqdm(empty_cases, desc="Mining empty MRIs"):
            image, _, _, _ = load_case_cached(
                case=info.record,
                target_spacing=config.data.target_spacing,
                include_dmri=config.data.include_dmri,
                dmri_reduce=config.data.dmri_reduce,
                dmri_b0_threshold=config.data.dmri_b0_threshold,
                normalize_foreground_only=config.data.normalize_foreground_only,
                cache_dir=cache_dir,
            )

            if load_mode == "preload":
                probabilities = _predict_ensemble_probs_loaded(models, image, config, device, use_amp, tta="none")
            else:
                probabilities = _predict_ensemble_probs_sequential(
                    config, base_dir, checkpoint_paths, image, device, use_amp, tta="none"
                )

            lesion_probs = probabilities[1]
            fp_mask = (lesion_probs >= float(args.threshold)).astype(np.uint8)
            fp_mask = _filter_mask(fp_mask, int(args.min_component_voxels))
            case_rows = _component_rows(
                case_id=info.record.case_id,
                lesion_probs=lesion_probs,
                mask=fp_mask,
                threshold=float(args.threshold),
                min_component_voxels=int(args.min_component_voxels),
                max_components_per_case=int(args.max_components_per_case),
                fold=fold,
                split=args.split,
                checkpoint_names=checkpoint_names,
            )
            if case_rows:
                cases_with_fp += 1
                rows.extend(case_rows)
    finally:
        for model in models:
            _unload_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_rows(output_csv, rows)
    summary = {
        "fold": fold,
        "split": args.split,
        "empty_cases": len(empty_cases),
        "cases_with_fp": cases_with_fp,
        "components_saved": len(rows),
        "threshold": float(args.threshold),
        "min_component_voxels": int(args.min_component_voxels),
        "max_components_per_case": int(args.max_components_per_case),
        "load_mode": load_mode,
        "checkpoints": checkpoint_names,
        "output_csv": str(output_csv),
    }
    summary_path = output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote {len(rows)} hard-negative centers: {output_csv}")
    print(f"[INFO] Summary: {summary_path}")


if __name__ == "__main__":
    main()
