from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.ndimage import label
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from multitalent_tbi.case_filters import (
    CaseInfo,
    build_case_infos,
    empty_infos,
    lesion_category,
    positive_infos,
    write_case_manifest,
)
from multitalent_tbi.config import load_config, resolve_path
from multitalent_tbi.data import TBIDataset, discover_cases
from multitalent_tbi.engine import (
    _apply_stage_freezing,
    _get_staged_tuning,
    _normalize_patterns,
    _stage_for_epoch,
    apply_overrides,
    build_model,
    build_stage_optimizer,
    configure_torch_for_speed,
    dice_score,
    set_seed,
)
from multitalent_tbi.infer import predict_logits
from multitalent_tbi.losses import dice_ce_loss
from multitalent_tbi.lr_scheduler import StagedPolyLRScheduler
from multitalent_tbi.splits import load_splits, split_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Segmenter A: lesion-positive-focused fine-tuning for AIMS-TBI."
    )
    parser.add_argument("--config", default="config.yml", help="Path to config.yml.")
    parser.add_argument("--fold", type=int, default=None, help="Fold index. Defaults to config training.fold.")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--full-lr", type=float, default=None, help="Override full fine-tuning LR.")
    parser.add_argument("--partial-lr", type=float, default=None, help="Override partial fine-tuning LR.")
    parser.add_argument("--head-lr", type=float, default=None, help="Override head LR.")
    parser.add_argument("--init-checkpoint", default=None, help="Checkpoint to initialize from.")
    parser.add_argument("--output-subdir", default=None, help="Subfolder under config paths.work_dir.")
    parser.add_argument("--probability-threshold", type=float, default=None, help="Lesion probability threshold.")
    parser.add_argument("--min-component-voxels", type=int, default=None, help="Remove predicted components smaller than this.")
    parser.add_argument("--include-empty-fraction", type=float, default=None, help="Optional empty case fraction for hard-negative phase.")
    parser.add_argument("--hard-negative-manifest", default=None, help="CSV from mine_hard_negatives.py with FP patch centers.")
    parser.add_argument("--hard-negative-center-prob", type=float, default=None, help="Probability of sampling a mined FP center for empty cases.")
    parser.add_argument("--hard-negative-max-centers-per-case", type=int, default=None, help="Limit mined FP centers loaded per empty case.")
    parser.add_argument("--wandb-name", default=None, help="Override W&B run name.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B for this run.")
    return parser.parse_args()


def _namespace_get(namespace, name: str, default):
    return getattr(namespace, name, default) if namespace is not None else default


def _segmenter_config(config) -> SimpleNamespace:
    return getattr(config, "segmenter_a", SimpleNamespace())


def _build_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if args.epochs is not None:
        overrides.setdefault("training", {})["max_epochs"] = args.epochs
    if args.batch_size is not None:
        overrides.setdefault("training", {})["batch_size"] = args.batch_size
    if args.num_workers is not None:
        overrides.setdefault("training", {})["num_workers"] = args.num_workers
    if args.head_lr is not None:
        overrides.setdefault("training", {}).setdefault("staged_tuning", {})["head_lr"] = args.head_lr
    if args.partial_lr is not None:
        overrides.setdefault("training", {}).setdefault("staged_tuning", {})["partial_lr"] = args.partial_lr
    if args.full_lr is not None:
        overrides.setdefault("training", {}).setdefault("staged_tuning", {})["full_lr"] = args.full_lr
    if args.init_checkpoint is not None:
        overrides.setdefault("segmenter_a", {})["init_checkpoint"] = args.init_checkpoint
    if args.output_subdir is not None:
        overrides.setdefault("segmenter_a", {})["output_subdir"] = args.output_subdir
    if args.probability_threshold is not None:
        overrides.setdefault("segmenter_a", {})["probability_threshold"] = args.probability_threshold
    if args.min_component_voxels is not None:
        overrides.setdefault("segmenter_a", {})["min_component_voxels"] = args.min_component_voxels
    if args.include_empty_fraction is not None:
        overrides.setdefault("segmenter_a", {}).setdefault("hard_negative", {})["empty_fraction"] = args.include_empty_fraction
        overrides.setdefault("segmenter_a", {}).setdefault("hard_negative", {})["enabled"] = args.include_empty_fraction > 0
    if args.hard_negative_manifest is not None:
        overrides.setdefault("segmenter_a", {}).setdefault("hard_negative", {})["manifest_path"] = args.hard_negative_manifest
        overrides.setdefault("segmenter_a", {}).setdefault("hard_negative", {})["enabled"] = True
    if args.hard_negative_center_prob is not None:
        overrides.setdefault("segmenter_a", {}).setdefault("hard_negative", {})["center_sampling_prob"] = args.hard_negative_center_prob
    if args.hard_negative_max_centers_per_case is not None:
        overrides.setdefault("segmenter_a", {}).setdefault("hard_negative", {})["max_centers_per_case"] = args.hard_negative_max_centers_per_case
    if args.no_wandb:
        overrides.setdefault("wandb", {})["enabled"] = False
    if args.wandb_name is not None:
        overrides.setdefault("wandb", {})["name"] = args.wandb_name
    return overrides


def _select_empty_subset(empty: list[CaseInfo], positive_count: int, empty_fraction: float, seed: int) -> list[CaseInfo]:
    if empty_fraction <= 0 or not empty:
        return []
    empty_fraction = min(max(empty_fraction, 0.0), 0.95)
    target_empty = int(round((positive_count * empty_fraction) / max(1e-8, 1.0 - empty_fraction)))
    target_empty = min(len(empty), max(0, target_empty))
    rng = random.Random(seed)
    selected = empty[:]
    rng.shuffle(selected)
    return selected[:target_empty]


def _size_weights(segmenter_cfg) -> dict[str, float]:
    default = {"very_tiny": 8.0, "tiny": 6.0, "small": 3.0, "large": 1.5, "empty": 1.0}
    configured = _namespace_get(segmenter_cfg, "size_weights", None)
    if configured is None:
        return default
    if isinstance(configured, dict):
        return {**default, **{key: float(value) for key, value in configured.items()}}
    return {**default, **{key: float(value) for key, value in vars(configured).items()}}


def _build_sampler(infos: list[CaseInfo], segmenter_cfg, seed: int) -> WeightedRandomSampler:
    weights_by_size = _size_weights(segmenter_cfg)
    weights = [weights_by_size.get(info.category, 1.0) for info in infos]
    epoch_length_multiplier = float(_namespace_get(segmenter_cfg, "epoch_length_multiplier", 1.0))
    num_samples = max(1, int(round(len(infos) * epoch_length_multiplier)))
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def _load_hard_negative_centers(
    manifest_path: Path,
    max_centers_per_case: int = 0,
) -> dict[str, list[tuple[int, int, int]]]:
    centers: dict[str, list[tuple[int, int, int]]] = {}
    if not manifest_path.exists():
        print(f"[WARN] Hard-negative manifest not found: {manifest_path}")
        return centers

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                case_id = str(row["case_id"])
                center = (
                    int(float(row["center_i"])),
                    int(float(row["center_j"])),
                    int(float(row["center_k"])),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid hard-negative row in {manifest_path}: {row}") from error
            bucket = centers.setdefault(case_id, [])
            if max_centers_per_case <= 0 or len(bucket) < max_centers_per_case:
                bucket.append(center)
    return centers


def _filter_components(mask: np.ndarray, min_component_voxels: int) -> np.ndarray:
    if min_component_voxels <= 1 or not mask.any():
        return mask.astype(np.uint8)
    components, n_components = label(mask.astype(bool))
    if n_components == 0:
        return mask.astype(np.uint8)
    filtered = np.zeros_like(mask, dtype=np.uint8)
    for component_index in range(1, n_components + 1):
        component = components == component_index
        if int(component.sum()) >= min_component_voxels:
            filtered[component] = 1
    return filtered


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


@torch.no_grad()
def evaluate_segmenter_a(
    model: torch.nn.Module,
    loader: DataLoader,
    val_infos: list[CaseInfo],
    config,
    segmenter_cfg,
    device: torch.device,
    use_amp: bool,
    class_weights: torch.Tensor,
) -> tuple[float, dict[str, float], list[dict[str, object]]]:
    model.eval()
    probability_threshold = float(_namespace_get(segmenter_cfg, "probability_threshold", 0.5))
    min_component_voxels = int(_namespace_get(segmenter_cfg, "min_component_voxels", 0))
    selection_min_voxels = int(_namespace_get(segmenter_cfg, "selection_min_lesion_voxels", 50))

    losses: list[float] = []
    rows: list[dict[str, object]] = []
    category_scores: dict[str, list[float]] = {
        "very_tiny": [],
        "tiny": [],
        "small": [],
        "large": [],
        "gt_over_selection_min": [],
        "all_positive": [],
    }

    for index, batch in enumerate(tqdm(loader, desc="SegmenterA validation", leave=False)):
        image = batch["image"].numpy()[0]
        mask = batch["mask"].numpy()[0]
        info = val_infos[index]

        logits = predict_logits(
            model=model,
            image=image,
            roi_size=tuple(config.inference.roi_size),
            overlap=float(config.inference.overlap),
            batch_size=int(config.inference.sw_batch_size),
            device=device,
            use_amp=use_amp,
        )

        logits_tensor = torch.from_numpy(logits).unsqueeze(0).to(device=device, dtype=torch.float32)
        mask_tensor = torch.from_numpy(mask.copy()).unsqueeze(0).to(device=device, dtype=torch.long)
        loss = dice_ce_loss(logits_tensor, mask_tensor, class_weights=class_weights)
        losses.append(float(loss.detach().cpu()))

        probabilities = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
        prediction = (probabilities[1] >= probability_threshold).astype(np.uint8)
        prediction = _filter_components(prediction, min_component_voxels)

        score = dice_score(prediction, mask)
        pred_voxels = int(np.count_nonzero(prediction > 0))
        pred_category = lesion_category(pred_voxels)
        category_scores["all_positive"].append(score)
        if info.gt_voxels >= selection_min_voxels:
            category_scores["gt_over_selection_min"].append(score)
        if info.category in category_scores:
            category_scores[info.category].append(score)

        rows.append(
            {
                "case_id": info.record.case_id,
                "loss": float(loss.detach().cpu()),
                "dice": score,
                "gt_voxels": info.gt_voxels,
                "gt_category": info.category,
                "pred_voxels": pred_voxels,
                "pred_category": pred_category,
                "missed": int(pred_voxels == 0),
            }
        )

    metrics = {
        "val_loss": _mean(losses),
        "dice_all_positive": _mean(category_scores["all_positive"]),
        "dice_gt_over_selection_min": _mean(category_scores["gt_over_selection_min"]),
        "dice_very_tiny": _mean(category_scores["very_tiny"]),
        "dice_tiny": _mean(category_scores["tiny"]),
        "dice_small": _mean(category_scores["small"]),
        "dice_large": _mean(category_scores["large"]),
        "n_val_positive": float(len(category_scores["all_positive"])),
        "n_val_gt_over_selection_min": float(len(category_scores["gt_over_selection_min"])),
        "n_missed_positive": float(sum(int(row["missed"]) for row in rows)),
    }
    return metrics["val_loss"], metrics, rows


def _write_history_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "stage",
                "train_loss",
                "train_dice",
                "val_loss",
                "dice_all_positive",
                "dice_gt_over_selection_min",
                "dice_very_tiny",
                "dice_tiny",
                "dice_small",
                "dice_large",
                "n_missed_positive",
                "selection_metric",
                "selection_value",
                "lr_head",
                "lr_partial",
                "lr_full",
            ]
        )


def _append_history(path: Path, row: list[object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(row)


def _write_val_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "loss",
                "dice",
                "gt_voxels",
                "gt_category",
                "pred_voxels",
                "pred_category",
                "missed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_metric_specs(selection_metric: str) -> dict[str, tuple[str, str]]:
    return {
        "best": (selection_metric, "max"),
        "best_all_positive": ("dice_all_positive", "max"),
        "best_gt50": ("dice_gt_over_selection_min", "max"),
        "best_very_tiny": ("dice_very_tiny", "max"),
        "best_tiny": ("dice_tiny", "max"),
        "best_small": ("dice_small", "max"),
        "best_large": ("dice_large", "max"),
        "best_low_missed": ("n_missed_positive", "min"),
    }


def _is_better_metric(value: float, best_value: float, mode: str) -> bool:
    if not math.isfinite(value):
        return False
    if mode == "min":
        return value < best_value
    return value > best_value


def _init_wandb(config, output_dir: Path, fold: int, train_infos: list[CaseInfo], val_infos: list[CaseInfo]):
    wandb_config = getattr(config, "wandb", None)
    if wandb_config is None or not bool(getattr(wandb_config, "enabled", False)):
        return None
    try:
        import wandb
    except ImportError:
        print("Warning: wandb enabled but package is unavailable; continuing without W&B.")
        return None
    if not hasattr(wandb, "init"):
        print(
            "Warning: imported wandb module has no init() function; "
            "continuing without W&B. Check for a local wandb.py or broken install."
        )
        return None
    run_name = getattr(wandb_config, "name", None) or f"segmenter_a_fold_{fold}"
    run = wandb.init(
        project=getattr(wandb_config, "project", "AIMS-TBI-MultiTalentV2"),
        entity=getattr(wandb_config, "entity", None),
        name=run_name,
        dir=str(output_dir / "wandb"),
        mode=str(getattr(wandb_config, "mode", "offline")),
        tags=list(getattr(wandb_config, "tags", []) or []) + ["segmenter-a", "lesion-positive"],
        config=json.loads(json.dumps(config, default=lambda value: getattr(value, "__dict__", str(value)))),
        reinit=True,
    )
    run.summary["segmenter_a/train_cases"] = len(train_infos)
    run.summary["segmenter_a/val_cases"] = len(val_infos)
    return run


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, _build_overrides(args) or None)
    segmenter_cfg = _segmenter_config(config)

    base_dir = Path(args.config).expanduser().resolve().parent
    fold = int(args.fold if args.fold is not None else config.training.fold)
    seed = int(config.training.seed) + fold
    set_seed(seed)
    configure_torch_for_speed()

    dataset_dir = resolve_path(base_dir, config.paths.dataset_dir)
    splits_path = resolve_path(base_dir, config.paths.splits_file)
    work_dir = resolve_path(base_dir, config.paths.work_dir)
    output_subdir = str(_namespace_get(segmenter_cfg, "output_subdir", "segmenter_a"))
    output_dir = work_dir / output_subdir / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = discover_cases(dataset_dir)
    splits = load_splits(splits_path)
    train_records, val_records = split_records(records, splits[fold])

    train_all_infos = build_case_infos(train_records, split="train")
    val_all_infos = build_case_infos(val_records, split="val")
    train_positive = positive_infos(train_all_infos, int(_namespace_get(segmenter_cfg, "train_min_lesion_voxels", 1)))
    val_positive = positive_infos(val_all_infos, int(_namespace_get(segmenter_cfg, "val_min_lesion_voxels", 1)))

    hard_negative_cfg = _namespace_get(segmenter_cfg, "hard_negative", SimpleNamespace())
    include_empty = bool(_namespace_get(hard_negative_cfg, "enabled", False))
    empty_fraction = float(_namespace_get(hard_negative_cfg, "empty_fraction", 0.0)) if include_empty else 0.0

    hard_negative_centers: dict[str, list[tuple[int, int, int]]] = {}
    hard_negative_manifest = str(_namespace_get(hard_negative_cfg, "manifest_path", "") or "")
    if include_empty and hard_negative_manifest:
        hard_negative_centers = _load_hard_negative_centers(
            resolve_path(base_dir, hard_negative_manifest),
            max_centers_per_case=int(_namespace_get(hard_negative_cfg, "max_centers_per_case", 0)),
        )

    all_empty_infos = empty_infos(train_all_infos)
    hard_negative_empty_infos = [
        info for info in all_empty_infos if info.record.case_id in hard_negative_centers
    ]
    empty_pool = hard_negative_empty_infos if hard_negative_empty_infos else all_empty_infos
    selected_empty = _select_empty_subset(empty_pool, len(train_positive), empty_fraction, seed)
    selected_empty_ids = {info.record.case_id for info in selected_empty}
    hard_negative_centers = {
        case_id: centers
        for case_id, centers in hard_negative_centers.items()
        if case_id in selected_empty_ids
    }
    hard_negative_center_prob = (
        min(max(float(_namespace_get(hard_negative_cfg, "center_sampling_prob", 1.0)), 0.0), 1.0)
        if hard_negative_centers
        else 0.0
    )
    train_infos = train_positive + selected_empty

    if not train_positive:
        raise RuntimeError("No lesion-positive training cases found. Check dataset_dir, split file, and masks.")
    if not val_positive:
        raise RuntimeError("No lesion-positive validation cases found. Check split file and masks.")

    manifests_dir = output_dir / "manifests"
    write_case_manifest(manifests_dir / "train_cases.csv", train_infos)
    write_case_manifest(manifests_dir / "train_lesion_positive_cases.csv", train_positive)
    write_case_manifest(manifests_dir / "train_empty_selected_cases.csv", selected_empty)
    write_case_manifest(manifests_dir / "val_lesion_positive_cases.csv", val_positive)

    train_dataset = TBIDataset(
        records=[info.record for info in train_infos],
        patch_size=config.data.patch_size,
        target_spacing=config.data.target_spacing,
        include_dmri=config.data.include_dmri,
        dmri_reduce=config.data.dmri_reduce,
        dmri_b0_threshold=config.data.dmri_b0_threshold,
        normalize_foreground_only=config.data.normalize_foreground_only,
        oversample_foreground_prob=config.data.oversample_foreground_prob,
        training=True,
        cache_dir=resolve_path(base_dir, config.paths.cache_dir) if config.data.cache_preprocessed else None,
        forced_patch_centers=hard_negative_centers,
        forced_center_prob=hard_negative_center_prob,
    )
    val_dataset = TBIDataset(
        records=[info.record for info in val_positive],
        patch_size=config.data.patch_size,
        target_spacing=config.data.target_spacing,
        include_dmri=config.data.include_dmri,
        dmri_reduce=config.data.dmri_reduce,
        dmri_b0_threshold=config.data.dmri_b0_threshold,
        normalize_foreground_only=config.data.normalize_foreground_only,
        oversample_foreground_prob=0.0,
        training=False,
        cache_dir=resolve_path(base_dir, config.paths.cache_dir) if config.data.cache_preprocessed else None,
    )

    loader_kwargs = {"pin_memory": True}
    if int(config.training.num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(getattr(config.training, "prefetch_factor", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.training.batch_size),
        sampler=_build_sampler(train_infos, segmenter_cfg, seed),
        shuffle=False,
        num_workers=int(config.training.num_workers),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, base_dir).to(device)
    init_checkpoint = _namespace_get(
        segmenter_cfg,
        "init_checkpoint",
        "checkpoints/trained_models/best_tr_f1_kpcyjb66.pt",
    )
    init_checkpoint_path = resolve_path(base_dir, init_checkpoint)
    if init_checkpoint_path.exists():
        payload = torch.load(str(init_checkpoint_path), map_location="cpu", weights_only=False)
        state_dict = payload.get("model_state", payload)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Initialized from {init_checkpoint_path}")
        print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    else:
        raise FileNotFoundError(f"Segmenter A init checkpoint not found: {init_checkpoint_path}")

    staged = _get_staged_tuning(config)
    head_patterns = _normalize_patterns(
        getattr(staged, "head_patterns", None) if staged is not None else None,
        ["seg", "final", "classifier", "output"],
    )
    partial_patterns = _normalize_patterns(
        getattr(staged, "partial_patterns", None) if staged is not None else None,
        ["decoder", "up", "localization", "stages.4", "stages.5", "stages.6"],
    )
    optimizer = build_stage_optimizer(model, config)
    scheduler = StagedPolyLRScheduler(optimizer, config)
    amp_enabled = bool(config.training.use_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    class_weights = torch.tensor(config.training.class_weights, dtype=torch.float32, device=device)
    validation_interval = int(getattr(config.training, "validation_interval_epochs", 1))
    selection_metric = str(_namespace_get(segmenter_cfg, "selection_metric", "dice_gt_over_selection_min"))
    checkpoint_specs = _checkpoint_metric_specs(selection_metric)
    best_values = {
        name: math.inf if mode == "min" else -math.inf
        for name, (_, mode) in checkpoint_specs.items()
    }

    history_path = output_dir / "history.csv"
    _write_history_header(history_path)
    with (output_dir / "run_segmenter_a.txt").open("w", encoding="utf-8") as run_log:
        run_log.write("Segmenter A lesion-positive-focused training\n")
        run_log.write(f"Fold: {fold}\n")
        run_log.write(f"Init checkpoint: {init_checkpoint_path}\n")
        run_log.write(f"Train positive: {len(train_positive)}\n")
        run_log.write(f"Train selected empty: {len(selected_empty)}\n")
        run_log.write(f"Hard-negative manifest: {hard_negative_manifest or 'none'}\n")
        run_log.write(f"Hard-negative cases with centers: {len(hard_negative_centers)}\n")
        run_log.write(f"Hard-negative centers loaded: {sum(len(v) for v in hard_negative_centers.values())}\n")
        run_log.write(f"Hard-negative center sampling prob: {hard_negative_center_prob}\n")
        run_log.write(f"Val positive: {len(val_positive)}\n")
        run_log.write(f"Selection metric: {selection_metric}\n")
        run_log.write(f"Probability threshold: {_namespace_get(segmenter_cfg, 'probability_threshold', 0.5)}\n")
        run_log.write(f"Min component voxels: {_namespace_get(segmenter_cfg, 'min_component_voxels', 0)}\n")

    wandb_run = _init_wandb(config, output_dir, fold, train_infos, val_positive)

    max_epochs = int(config.training.max_epochs)
    for epoch in range(max_epochs):
        stage = _stage_for_epoch(epoch, config)
        trainable_counts = _apply_stage_freezing(model, stage, head_patterns, partial_patterns)
        lr_summary = scheduler.step(epoch)
        model.train()

        running_loss = 0.0
        running_dice = 0.0
        train_batches = 0
        for batch in tqdm(train_loader, desc=f"SegmenterA fold {fold} epoch {epoch + 1}/{max_epochs}", leave=False):
            images = batch["image"].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last_3d)
            masks = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                loss = dice_ce_loss(logits, masks, class_weights=class_weights)
            scaler.scale(loss).backward()
            if config.training.grad_clip_norm and config.training.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.training.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()

            predictions = torch.argmax(logits.detach(), dim=1)
            batch_scores = [
                dice_score(prediction.cpu().numpy(), target.cpu().numpy())
                for prediction, target in zip(predictions, masks.detach())
            ]
            running_loss += float(loss.detach().cpu())
            running_dice += float(np.mean(batch_scores)) if batch_scores else 0.0
            train_batches += 1

        train_loss = running_loss / max(1, train_batches)
        train_dice = running_dice / max(1, train_batches)
        val_loss = float("nan")
        val_metrics = {
            "dice_all_positive": float("nan"),
            "dice_gt_over_selection_min": float("nan"),
            "dice_very_tiny": float("nan"),
            "dice_tiny": float("nan"),
            "dice_small": float("nan"),
            "dice_large": float("nan"),
            "n_missed_positive": float("nan"),
        }
        val_rows: list[dict[str, object]] = []
        should_validate = validation_interval > 0 and (epoch + 1) % validation_interval == 0
        if should_validate:
            val_loss, val_metrics, val_rows = evaluate_segmenter_a(
                model=model,
                loader=val_loader,
                val_infos=val_positive,
                config=config,
                segmenter_cfg=segmenter_cfg,
                device=device,
                use_amp=amp_enabled,
                class_weights=class_weights,
            )

        selection_value = float(val_metrics.get(selection_metric, float("nan")))

        lr_head = float(lr_summary.get("head", 0.0))
        lr_partial = float(lr_summary.get("partial", 0.0))
        lr_full = float(lr_summary.get("full", 0.0))

        payload = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "lr_summary": lr_summary,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "monitor_metric": selection_metric,
            "monitor_value": selection_value,
            "init_checkpoint": str(init_checkpoint_path),
            "config": json.loads(json.dumps(config, default=lambda value: getattr(value, "__dict__", str(value)))),
        }
        torch.save(payload, output_dir / "last.pt")
        for checkpoint_name, (metric_name, mode) in checkpoint_specs.items():
            metric_value = float(val_metrics.get(metric_name, float("nan")))
            if _is_better_metric(metric_value, best_values[checkpoint_name], mode):
                best_values[checkpoint_name] = metric_value
                torch.save(payload, output_dir / f"{checkpoint_name}.pt")
                _write_val_rows(output_dir / f"{checkpoint_name}_val_cases.csv", val_rows)

        _append_history(
            history_path,
            [
                epoch + 1,
                stage,
                train_loss,
                train_dice,
                val_loss,
                val_metrics.get("dice_all_positive", float("nan")),
                val_metrics.get("dice_gt_over_selection_min", float("nan")),
                val_metrics.get("dice_very_tiny", float("nan")),
                val_metrics.get("dice_tiny", float("nan")),
                val_metrics.get("dice_small", float("nan")),
                val_metrics.get("dice_large", float("nan")),
                val_metrics.get("n_missed_positive", float("nan")),
                selection_metric,
                selection_value,
                lr_head,
                lr_partial,
                lr_full,
            ],
        )

        print(
            f"Epoch {epoch + 1:03d} [{stage}] "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
            f"val_gt50={val_metrics.get('dice_gt_over_selection_min', float('nan')):.4f} "
            f"very_tiny={val_metrics.get('dice_very_tiny', float('nan')):.4f} "
            f"tiny={val_metrics.get('dice_tiny', float('nan')):.4f} "
            f"small={val_metrics.get('dice_small', float('nan')):.4f} "
            f"large={val_metrics.get('dice_large', float('nan')):.4f} "
            f"missed={val_metrics.get('n_missed_positive', float('nan')):.0f} "
            f"lr={lr_head:.2e}/{lr_partial:.2e}/{lr_full:.2e} "
            f"trainable={trainable_counts}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/dice": train_dice,
                    "val/loss": val_loss,
                    "segmenter_a/dice_all_positive": val_metrics.get("dice_all_positive", float("nan")),
                    "segmenter_a/dice_gt50": val_metrics.get("dice_gt_over_selection_min", float("nan")),
                    "segmenter_a/dice_very_tiny": val_metrics.get("dice_very_tiny", float("nan")),
                    "segmenter_a/dice_tiny": val_metrics.get("dice_tiny", float("nan")),
                    "segmenter_a/dice_small": val_metrics.get("dice_small", float("nan")),
                    "segmenter_a/dice_large": val_metrics.get("dice_large", float("nan")),
                    "segmenter_a/n_missed_positive": val_metrics.get("n_missed_positive", float("nan")),
                    "segmenter_a/selection_value": selection_value,
                    "lr/head": lr_head,
                    "lr/partial": lr_partial,
                    "lr/full": lr_full,
                },
                step=epoch + 1,
            )

    if wandb_run is not None:
        wandb_run.finish()

    print("Best checkpoints:")
    for checkpoint_name, (metric_name, _) in checkpoint_specs.items():
        print(f"  {checkpoint_name}.pt ({metric_name}): {best_values[checkpoint_name]:.6f}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
