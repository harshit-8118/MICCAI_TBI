from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from turtle import mode
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure, label
from tqdm.auto import tqdm

from multitalent_tbi import config
from multitalent_tbi.case_filters import build_case_infos, lesion_category
from multitalent_tbi.config import load_config, resolve_path
from multitalent_tbi.data import CaseRecord, TBIDataset
from multitalent_tbi.engine import build_model, configure_torch_for_speed, dice_score, set_seed
from multitalent_tbi.infer import predict_logits
from multitalent_tbi.losses import dice_ce_loss
from train_hierarchical_segmenter import _apply_branch_config, _get


CATEGORIES = ["empty", "very_tiny", "tiny", "small", "large"]
POSITIVE_CATEGORIES = ["very_tiny", "tiny", "small", "large"]
METRIC_GROUPS = ["all", "positive", "gt50", "micro", *CATEGORIES]


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
    parser.add_argument("--thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--min-components", nargs="+", type=int, default=None)
    parser.add_argument("--tta", choices=["none", "flips"], default="none")
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
        "--skip-surface",
        action="store_true",
        help="Skip HD95/ASSD. Useful for quick Dice-only sweeps before final report runs.",
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
            f"No matched scan_XXXX.nii.gz + scan_XXXX_lesion.nii.gz pairs found in {dataset_dir}. "
            f"Preview: {preview}"
        )
    return records, manifest


def _filter_components(mask: np.ndarray, min_component_voxels: int) -> np.ndarray:
    if min_component_voxels <= 1 or not mask.any():
        return mask.astype(np.uint8)
    components, n_components = label(mask > 0)
    filtered = np.zeros_like(mask, dtype=np.uint8)
    for component_index in range(1, n_components + 1):
        component = components == component_index
        if int(component.sum()) >= int(min_component_voxels):
            filtered[component] = 1
    return filtered


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
    if mode == "none":
        return [None]
    return [None, 0, 1, 2]


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
    tta: str,
) -> np.ndarray:
    probs_sum = None
    for model in models:
        probabilities = _predict_model_probs(model, image, config, device, use_amp, tta)
        probs_sum = probabilities if probs_sum is None else probs_sum + probabilities
    return probs_sum / float(len(models))


def _predict_probs_sequential(
    checkpoint_paths: list[Path],
    image: np.ndarray,
    config,
    base_dir: Path,
    device: torch.device,
    use_amp: bool,
    tta: str,
) -> np.ndarray:
    probs_sum = None
    for checkpoint_path in checkpoint_paths:
        model = _build_loaded_model(config, base_dir, checkpoint_path, device)
        probabilities = _predict_model_probs(model, image, config, device, use_amp, tta)
        probs_sum = probabilities if probs_sum is None else probs_sum + probabilities
        _unload_model(model)
    return probs_sum / float(len(checkpoint_paths))


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
        "threshold",
        "min_component_voxels",
        "pred_voxels",
        "pred_category",
        "dice",
        "hd95",
        "assd",
        "missed_positive",
        "empty_false_positive",
    ]
    _write_dict_rows(path, rows, fieldnames=fieldnames)


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

    output_root = resolve_path(base_dir, _get(external_cfg, "output_root", "checkpoints/validation2025_100"))
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
        "thresholds": thresholds,
        "min_components": min_components,
        "tta": args.tta,
        "load_mode": args.load_mode,
        "skip_surface": bool(args.skip_surface),
        "target_spacing": spacing,
        "roi_size": list(config.inference.roi_size),
        "overlap": float(config.inference.overlap),
        "sw_batch_size": int(config.inference.sw_batch_size),
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
    print(f"[INFO] Device      : {device} | AMP={use_amp} | TTA={args.tta} | load_mode={args.load_mode}")
    print(f"[INFO] Thresholds  : {thresholds}")
    print(f"[INFO] Min comp    : {min_components}")
    print(f"[INFO] Surface     : {'off' if args.skip_surface else 'HD95 + ASSD'}")
    print(f"[INFO] Output dir  : {output_dir}")

    loaded_models = None
    if args.load_mode == "preload":
        loaded_models = [_build_loaded_model(config, base_dir, checkpoint_path, device) for checkpoint_path in checkpoint_paths]

    class_weights = torch.tensor(config.training.class_weights, dtype=torch.float32, device=device)
    rows_by_setting: dict[tuple[float, int], list[dict[str, object]]] = {
        (float(threshold), int(min_component_voxels)): []
        for threshold, min_component_voxels in itertools.product(thresholds, min_components)
    }
    losses: list[float] = []

    for index, batch in enumerate(tqdm(loader, desc="External validation")):
        image = batch["image"].numpy()[0]
        mask = batch["mask"].numpy()[0]
        info = case_infos[index]
        if loaded_models is not None:
            probabilities = _predict_probs_preload(loaded_models, image, config, device, use_amp, args.tta)
        else:
            probabilities = _predict_probs_sequential(checkpoint_paths, image, config, base_dir, device, use_amp, args.tta)

        logits_for_loss = torch.log(torch.from_numpy(np.clip(probabilities, 1e-7, 1.0)).float()).unsqueeze(0).to(device)
        mask_tensor = torch.from_numpy(mask.copy()).unsqueeze(0).to(device=device, dtype=torch.long)
        losses.append(float(dice_ce_loss(logits_for_loss, mask_tensor, class_weights=class_weights).detach().cpu()))

        lesion_probability = probabilities[1]
        for threshold, min_component_voxels in rows_by_setting:
            prediction = (lesion_probability >= float(threshold)).astype(np.uint8)
            prediction = _filter_components(prediction, int(min_component_voxels))
            pred_voxels = int(np.count_nonzero(prediction))
            if args.skip_surface:
                hd95 = float("nan")
                assd = float("nan")
            else:
                hd95, assd = _surface_metrics(prediction, mask, spacing)
            rows_by_setting[(threshold, min_component_voxels)].append(
                {
                    "case_id": info.record.case_id,
                    "gt_voxels": int(info.gt_voxels),
                    "gt_category": info.category,
                    "threshold": float(threshold),
                    "min_component_voxels": int(min_component_voxels),
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
    for (threshold, min_component_voxels), rows in rows_by_setting.items():
        metrics = _metrics_from_rows(rows, prefix="test")
        row = {
            "threshold": threshold,
            "min_component_voxels": min_component_voxels,
            "test_loss": _mean(losses),
            **metrics,
        }
        summary_rows.append(row)
        threshold_tag = f"{threshold:g}".replace(".", "p")
        detail_name = f"case_metrics_thr{threshold_tag}_mincc{min_component_voxels}.csv"
        _write_case_rows(output_dir / detail_name, rows)

    summary_rows.sort(key=lambda row: (float(row["threshold"]), int(row["min_component_voxels"])))
    _add_ranking_proxy(summary_rows)
    _write_dict_rows(output_dir / "summary.csv", summary_rows)

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
        f"TTA: {args.tta}",
        f"Load mode: {args.load_mode}",
        f"Surface metrics: {'skipped' if args.skip_surface else 'HD95 and ASSD'}",
        "",
        "Top settings by local ranking proxy (Dice rank + HD95 rank + ASSD rank within this sweep):",
    ]
    for row in best_by_proxy[:10]:
        log_lines.append(
            "thr={thr:.3f} mincc={mincc:3d} rank_sum={rank:3d} "
            "dice={dice} hd95={hd95} assd={assd} pos={pos} gt50={gt50} "
            "missed={missed} empty_fp={empty_fp}".format(
                thr=float(row["threshold"]),
                mincc=int(row["min_component_voxels"]),
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
            f"thr={float(row['threshold']):.3f} mincc={int(row['min_component_voxels']):3d} "
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

# python evaluate_external_validation.py --config config.yml --branch none --checkpoints checkpoints/tbi_multitalentv2/fold_0/best_ep_280.pt --thresholds 0.10 0.20 0.30 0.40 0.50 --min-components 0 3 5 10 20 40 --tta none --output-dir checkpoints/validation2025_100/single_ep_280_no_tta

"/data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning/archive/Approach_2_Ensemble_Callibrater/checkpoints/trained_models/best_714109.pt"

# python evaluate_external_validation.py --config config.yml --branch none --checkpoints checkpoints/tbi_multitalentv2/fold_0/best_ep_280.pt --thresholds 0.10 0.20 0.30 0.40 0.50 --min-components 0 3 5 10 20 40 --tta flips --output-dir checkpoints/validation2025_100/single_ep_280_tta

# python evaluate_external_validation.py --config config.yml --branch none --checkpoints checkpoints/tbi_multitalentv2/fold_0/best_ep_201.pt checkpoints/tbi_multitalentv2/fold_0/best_ep_222.pt checkpoints/tbi_multitalentv2/fold_0/best_ep_280.pt --thresholds 0.10 0.20 0.30 0.40 0.50 --min-components 0 3 5 10 20 40 --tta flips --load-mode sequential --output-dir checkpoints/validation2025_100/ensemble_ep201_222_280_tta

# python visualize_test_evaluations.py --root checkpoints/validation2025_100 --select-metric comparison_ranking_proxy_sum --select-mode min