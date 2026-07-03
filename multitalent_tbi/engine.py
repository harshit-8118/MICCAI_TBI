from __future__ import annotations

import csv
import json
import inspect
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from .config import ensure_parent, load_config, resolve_path
from .data import TBIDataset, discover_cases, make_stratification_labels
from .infer import predict_logits
from .losses import dice_ce_loss
from .model import build_backbone, load_pretrained_weights
from .splits import load_splits, split_records
from .lr_scheduler import StagedPolyLRScheduler

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch_for_speed() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def dice_score(prediction: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    pred = prediction.astype(bool)
    tgt = target.astype(bool)
    intersection = np.logical_and(pred, tgt).sum()
    denominator = pred.sum() + tgt.sum()
    if denominator == 0:
        return 1.0
    return float((2.0 * intersection + smooth) / (denominator + smooth))


def _size_from_label(label: str) -> str:
    """'tiny_t1' -> 'tiny',  'empty_dmri' -> 'empty', etc."""
    return label.split("_")[0]


def build_optimizer(model: torch.nn.Module, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def _lr_scale_for_epoch(epoch: int, max_epochs: int, warmup_epochs: int, poly_power: float) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(epoch + 1) / float(max(1, warmup_epochs))
    if max_epochs <= warmup_epochs:
        return 1.0
    progress = float(epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
    progress = min(max(progress, 0.0), 1.0)
    return (1.0 - progress) ** poly_power


def _apply_lr_scale(optimizer: torch.optim.Optimizer, scale: float) -> dict[str, float]:
    summary: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        base_lr = float(group.setdefault("base_lr", group["lr"]))
        current_lr = base_lr * scale
        group["lr"] = current_lr
        group_name = str(group.get("group_name", f"group_{index}"))
        summary[group_name] = current_lr
    return summary


def _set_nested_attr(namespace, key_path: str, value) -> None:
    parts = key_path.split(".")
    current = namespace
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], value)


def apply_overrides(config, overrides: dict[str, object] | None) -> None:
    if not overrides:
        return
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, dict):
            apply_overrides(getattr(config, key), value)
        else:
            _set_nested_attr(config, key, value)


def _get_staged_tuning(config):
    return getattr(config.training, "staged_tuning", None)


def _normalize_patterns(patterns, fallback):
    values = patterns if patterns is not None else fallback
    return [str(pattern).lower() for pattern in values]


def _categorize_parameter(name: str, head_patterns: list[str], partial_patterns: list[str]) -> str:
    lower_name = name.lower()
    if any(pattern in lower_name for pattern in head_patterns):
        return "head"
    if any(pattern in lower_name for pattern in partial_patterns):
        return "partial"
    return "full"


def _apply_stage_freezing(model: torch.nn.Module, stage: str, head_patterns: list[str], partial_patterns: list[str]) -> tuple[int, int, int]:
    counts = {"head": 0, "partial": 0, "full": 0}
    for name, parameter in model.named_parameters():
        category = _categorize_parameter(name, head_patterns, partial_patterns)
        counts[category] += 1
        if stage == "head":
            parameter.requires_grad = category == "head"
        elif stage == "partial":
            parameter.requires_grad = category in {"head", "partial"}
        else:
            parameter.requires_grad = True
    return counts["head"], counts["partial"], counts["full"]


def _stage_for_epoch(epoch: int, config) -> str:
    staged = _get_staged_tuning(config)
    if staged is None or not bool(staged.enabled):
        return "full"
    head_epochs = int(staged.head_only_epochs)
    partial_epochs = int(staged.partial_tune_epochs)
    if epoch < head_epochs:
        return "head"
    if epoch < head_epochs + partial_epochs:
        return "partial"
    return "full"


def _stage_learning_rates(config) -> dict[str, float]:
    staged = _get_staged_tuning(config)
    if staged is None or not bool(staged.enabled):
        base_lr = float(config.training.base_lr)
        return {"head": base_lr, "partial": base_lr, "full": base_lr}
    return {
        "head": float(staged.head_lr),
        "partial": float(staged.partial_lr),
        "full": float(staged.full_lr),
    }


def build_stage_optimizer(model: torch.nn.Module, config) -> torch.optim.Optimizer:
    staged = _get_staged_tuning(config)
    head_patterns = _normalize_patterns(getattr(staged, "head_patterns", None) if staged is not None else None, ["seg", "final", "classifier", "output"])
    partial_patterns = _normalize_patterns(
        getattr(staged, "partial_patterns", None) if staged is not None else None,
        ["decoder", "up", "localization", "stages.4", "stages.5", "stages.6"],
    )
    learning_rates = _stage_learning_rates(config)
    groups = {"head": [], "partial": [], "full": []}
    for name, parameter in model.named_parameters():
        category = _categorize_parameter(name, head_patterns, partial_patterns)
        groups[category].append(parameter)

    head_only_epochs = int(getattr(staged, "head_only_epochs", 0)) if staged else 0    
    partial_tune_epochs = int(getattr(staged, "partial_tune_epochs", 0)) if staged else 0

    param_groups = []    

    for category in ("head", "partial", "full"):
        if not groups[category]:
            continue
        if category == "head" and head_only_epochs == 0:
            groups["full"].extend(groups["head"])
            continue
        if category == "partial" and partial_tune_epochs == 0:
            groups["full"].extend(groups["partial"])
            continue
        param_groups.append(
            {
                "params": groups[category],
                "lr": learning_rates[category],
                "base_lr": learning_rates[category],
                "group_name": category,
            }
        )
    if not param_groups:
        raise RuntimeError("No trainable parameters were found for the current tuning configuration.")
    if not groups["head"]:
        print("Warning: no parameters matched head patterns; adjust training.staged_tuning.head_patterns if needed.")
    if not groups["partial"]:
        print("Warning: no parameters matched partial patterns; adjust training.staged_tuning.partial_patterns if needed.")
    print(
        "Stage groups: "
        f"head={len(groups['head'])}, partial={len(groups['partial'])}, full={len(groups['full'])}; "
        f"lrs={learning_rates}"
    )
    return build_optimizer_from_param_groups(config.training.optimizer, param_groups, float(config.training.weight_decay))


def build_optimizer_from_param_groups(name: str, param_groups: list[dict[str, object]], weight_decay: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(param_groups, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def _next_run_log_path(output_dir: Path) -> Path:
    existing = []
    for path in output_dir.glob("run_*.txt"):
        try:
            existing.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    next_index = max(existing, default=0) + 1
    return output_dir / f"run_{next_index}.txt"


def _count_lesion_voxels(record) -> int:
    from .data import load_nifti_robust

    lesion = np.asarray(load_nifti_robust(record.lesion_path).dataobj)
    return int(np.count_nonzero(lesion > 0))


def _summarize_records(records) -> dict[str, int]:
    lesion_positive = 0
    lesion_empty = 0
    dmri = 0
    for record in records:
        lesion_voxels = _count_lesion_voxels(record)
        if lesion_voxels > 0:
            lesion_positive += 1
        else:
            lesion_empty += 1
        if record.has_dmri:
            dmri += 1
    return {
        "total": len(records),
        "lesion_positive": lesion_positive,
        "lesion_empty": lesion_empty,
        "dmri": dmri,
    }


def _namespace_to_dict(value):
    if isinstance(value, dict):
        return {key: _namespace_to_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_namespace_to_dict(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: _namespace_to_dict(item) for key, item in value.__dict__.items()}
    return value


def _lesion_positive_flags(records) -> list[bool]:
    return [_count_lesion_voxels(record) > 0 for record in records]


def _build_case_sampler(records, config, fold: int) -> tuple[WeightedRandomSampler | None, dict[str, float | int | bool]]:
    sampling = getattr(config.training, "case_sampling", None)
    summary = {
        "enabled": False,
        "target_positive_fraction": float("nan"),
        "target_empty_fraction": float("nan"),
        "positive_weight": float("nan"),
        "empty_weight": float("nan"),
        "num_samples": len(records),
        "replacement": True,
        "size_weights": {},
        "label_counts": {},
    }
    if sampling is None or not bool(getattr(sampling, "enabled", False)):
        return None, summary

    positive_fraction = float(getattr(sampling, "positive_fraction", 0.5))
    positive_fraction = min(max(positive_fraction, 0.0), 1.0)
    replacement = bool(getattr(sampling, "replacement", True))
    epoch_length_multiplier = float(getattr(sampling, "epoch_length_multiplier", 1.0))
    num_samples = max(1, int(round(len(records) * epoch_length_multiplier)))

    _default_size_weights = {"empty": 1.0, "tiny": 3.0, "small": 2.0, "large": 1.5}
    cfg_sw = getattr(sampling, "size_weights", None)
    if cfg_sw is None:
        size_weights = _default_size_weights.copy()
    elif isinstance(cfg_sw, dict):
        size_weights = {**_default_size_weights, **{k: float(v) for k, v in cfg_sw.items()}}
    else:
        # namespace object
        size_weights = {**_default_size_weights,
                        **{k: float(v) for k, v in vars(cfg_sw).items()}}

    labels = make_stratification_labels(records)   # e.g. ["tiny_t1", "empty_dmri", ...]
    import collections
    label_counts = dict(collections.Counter(labels))

    weights = []
    for label in labels:
        size = _size_from_label(label)
        weights.append(size_weights.get(size, 1.0))
 
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=replacement,
        generator=torch.Generator().manual_seed(int(config.training.seed) + int(fold)),
    )
    positive_labels = {"tiny", "small", "large"}
    total_w  = sum(weights)
    pos_w = sum(w for w, lbl in zip(weights, labels)
                   if _size_from_label(lbl) in positive_labels)
    effective_pos_fraction = pos_w / total_w if total_w > 0 else float("nan")
 
    summary = {
        "enabled": True,
        "target_positive_fraction": positive_fraction,   # from config (reference)
        "effective_positive_fraction": effective_pos_fraction,
        "target_empty_fraction": 1.0 - positive_fraction,
        "positive_weight": float("nan"),   # n/a — per-size now
        "empty_weight": size_weights.get("empty", 1.0),
        "num_samples": num_samples,
        "replacement": replacement,
        "size_weights": size_weights,
        "label_counts": label_counts,
    }
    return sampler, summary


def _batch_dice(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=1)
    scores = []
    for prediction, target in zip(predictions.detach().cpu().numpy(), targets.detach().cpu().numpy()):
        scores.append(dice_score(prediction, target))
    return float(np.mean(scores)) if scores else 0.0


def load_case_records_for_fold(config, fold: int, base_dir: Path):
    records = discover_cases(resolve_path(base_dir, config.paths.dataset_dir))
    splits = load_splits(resolve_path(base_dir, config.paths.splits_file))
    train_records, val_records = split_records(records, splits[fold])
    return train_records, val_records


def build_dataloaders(config, fold: int, base_dir: Path):
    train_records, val_records = load_case_records_for_fold(config, fold, base_dir)
    train_sampler, sampling_summary = _build_case_sampler(train_records, config, fold)
    train_dataset = TBIDataset(
        records=train_records,
        patch_size=config.data.patch_size,
        target_spacing=config.data.target_spacing,
        include_dmri=config.data.include_dmri,
        dmri_reduce=config.data.dmri_reduce,
        dmri_b0_threshold=config.data.dmri_b0_threshold,
        normalize_foreground_only=config.data.normalize_foreground_only,
        oversample_foreground_prob=config.data.oversample_foreground_prob,
        training=True,
        cache_dir=resolve_path(base_dir, config.paths.cache_dir) if config.data.cache_preprocessed else None,
    )
    val_dataset = TBIDataset(
        records=val_records,
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
    loader_kwargs = {
        "pin_memory": True,
    }
    if config.training.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(getattr(config.training, "prefetch_factor", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.training.num_workers,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, val_loader, train_records, val_records, sampling_summary


def _init_wandb_run(config, fold: int, output_dir: Path, dataset_summary: dict[str, int], train_summary: dict[str, int], val_summary: dict[str, int], sampling_summary: dict[str, float | int | bool]):
    wandb_config = getattr(config, "wandb", None)
    if wandb_config is None or not bool(getattr(wandb_config, "enabled", False)):
        return None
    try:
        import wandb
    except ImportError:
        print("Warning: wandb is enabled in config but the package is not installed; skipping W&B logging.")
        return None

    run_name = getattr(wandb_config, "name", None) or f"fold_{fold}_{output_dir.name}"
    mode = str(getattr(wandb_config, "mode", "offline"))
    tags = getattr(wandb_config, "tags", None)
    notes = getattr(wandb_config, "notes", None)
    run = wandb.init(
        project=getattr(wandb_config, "project", "AIMS-TBI-MultiTalentV2"),
        entity=getattr(wandb_config, "entity", None),
        name=run_name,
        dir=str(output_dir / "wandb"),
        mode=mode,
        config=_namespace_to_dict(config),
        tags=tags,
        notes=notes,
        reinit=True,
    )
    run.summary["dataset/total"] = dataset_summary["total"]
    run.summary["dataset/lesion_positive"] = dataset_summary["lesion_positive"]
    run.summary["dataset/lesion_empty"] = dataset_summary["lesion_empty"]
    run.summary["dataset/dmri"] = dataset_summary["dmri"]
    run.summary["train/total"] = train_summary["total"]
    run.summary["val/total"] = val_summary["total"]
    run.summary["sampling/enabled"] = sampling_summary["enabled"]
    run.summary["sampling/target_positive_fraction"] = sampling_summary["target_positive_fraction"]
    run.summary["sampling/target_empty_fraction"] = sampling_summary["target_empty_fraction"]
    return run


def build_model(config, base_dir: Path):
    plans_path = resolve_path(base_dir, config.paths.pretrained_plans)
    model = build_backbone(
        plans_path=plans_path,
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        deep_supervision=config.model.deep_supervision,
    )
    # with open('model.txt', 'w') as f: 
    #     f.write(str(model))
    if config.model.load_pretrained:
        skipped = load_pretrained_weights(
            model=model,
            checkpoint_path=resolve_path(base_dir, config.paths.pretrained_checkpoint),
            strict=config.model.strict_load,
        )
        print(f"Loaded pretrained backbone. Skipped {len(skipped)} incompatible keys.")
    return model.to(memory_format=torch.channels_last_3d)


def train_one_fold(config, fold: int, base_dir: Path) -> dict[str, float]:
    set_seed(int(config.training.seed) + fold)
    configure_torch_for_speed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, train_records, val_records, sampling_summary = build_dataloaders(config, fold, base_dir)
    model = build_model(config, base_dir).to(device)

    staged = _get_staged_tuning(config)
    head_patterns = _normalize_patterns(getattr(staged, "head_patterns", None) if staged is not None else None, ["seg", "final", "classifier", "output"])
    partial_patterns = _normalize_patterns(
        getattr(staged, "partial_patterns", None) if staged is not None else None,
        ["decoder", "up", "localization", "stages.4", "stages.5", "stages.6"],
    )
    optimizer = build_stage_optimizer(model, config)
    scheduler = StagedPolyLRScheduler(optimizer, config)
    amp_enabled = bool(config.training.use_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    class_weights = torch.tensor(config.training.class_weights, dtype=torch.float32, device=device)
    validation_interval = int(getattr(config.training, "validation_interval_epochs", 0))
    final_validation = bool(getattr(config.training, "final_validation", True))
    best_checkpoint_metric = str(getattr(config.training, "best_checkpoint_metric", "train_dice")).lower()
    best_checkpoint_mode = str(getattr(config.training, "best_checkpoint_mode", "max")).lower()
    if validation_interval <= 0 and best_checkpoint_metric == "val_dice":
        print("Warning: validation is disabled during training, so best_checkpoint_metric=val_dice cannot be tracked. Using train_dice.")
        best_checkpoint_metric = "train_dice"
    if best_checkpoint_mode not in {"max", "min"}:
        raise ValueError(f"Unsupported best_checkpoint_mode: {best_checkpoint_mode}")
    if best_checkpoint_metric not in {"train_dice", "val_dice", "train_loss", "val_loss"}:
        raise ValueError(f"Unsupported best_checkpoint_metric: {best_checkpoint_metric}")

    output_dir = resolve_path(base_dir, config.paths.work_dir) / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = _next_run_log_path(output_dir)
    dataset_summary = _summarize_records(train_records + val_records)
    train_summary = _summarize_records(train_records)
    val_summary = _summarize_records(val_records)
    wandb_run = _init_wandb_run(config, fold, output_dir, dataset_summary, train_summary, val_summary, sampling_summary)
    best_monitor_value = -math.inf if best_checkpoint_mode == "max" else math.inf
    best_epoch = 0
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    history_path = output_dir / "history.csv"

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "phase",
                "epoch",
                "train_loss",
                "train_dice",
                "val_loss",
                "val_dice",
                "monitor_metric",
                "monitor_value",
                "lr_head",
                "lr_partial",
                "lr_full",
            ]
        )

    with run_log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Run log for fold {fold}\n")
        handle.write(f"Device: {device}\n")
        handle.write(f"AMP enabled: {amp_enabled}\n")
        handle.write(f"Staged tuning: {bool(staged and staged.enabled)}\n")
        handle.write(f"Validation interval epochs: {validation_interval}\n")
        handle.write(f"Final validation: {final_validation}\n")
        handle.write(f"Best checkpoint metric: {best_checkpoint_metric} ({best_checkpoint_mode})\n")
        handle.write(f"Output dir: {output_dir}\n")
        handle.write(
            "Dataset summary: "
            f"total={dataset_summary['total']}, "
            f"lesion_positive={dataset_summary['lesion_positive']}, "
            f"lesion_empty={dataset_summary['lesion_empty']}, "
            f"dmri={dataset_summary['dmri']}\n"
        )
        handle.write(
            "Train split summary: "
            f"total={train_summary['total']}, "
            f"lesion_positive={train_summary['lesion_positive']}, "
            f"lesion_empty={train_summary['lesion_empty']}, "
            f"dmri={train_summary['dmri']}\n"
        )
        handle.write(
            "Val split summary: "
            f"total={val_summary['total']}, "
            f"lesion_positive={val_summary['lesion_positive']}, "
            f"lesion_empty={val_summary['lesion_empty']}, "
            f"dmri={val_summary['dmri']}\n"
        )
        handle.write(
            "Case sampling: "
            f"enabled={sampling_summary['enabled']}, "
            f"target_positive_fraction={sampling_summary['target_positive_fraction']}, "
            f"target_empty_fraction={sampling_summary['target_empty_fraction']}, "
            f"positive_weight={sampling_summary['positive_weight']}, "
            f"empty_weight={sampling_summary['empty_weight']}, "
            f"num_samples={sampling_summary['num_samples']}, "
            f"replacement={sampling_summary['replacement']}\n"
        )
        handle.write("phase\tepoch\tstage\ttrain_loss\ttrain_dice\tval_loss\tval_dice\tmonitor_metric\tmonitor_value\tlr_head\tlr_partial\tlr_full\n")

    if wandb_run is not None:
        wandb_run.log(
            {
                "data/total": dataset_summary["total"],
                "data/lesion_positive": dataset_summary["lesion_positive"],
                "data/lesion_empty": dataset_summary["lesion_empty"],
                "data/dmri": dataset_summary["dmri"],
                "split/train_total": train_summary["total"],
                "split/train_lesion_positive": train_summary["lesion_positive"],
                "split/train_lesion_empty": train_summary["lesion_empty"],
                "split/val_total": val_summary["total"],
                "split/val_lesion_positive": val_summary["lesion_positive"],
                "split/val_lesion_empty": val_summary["lesion_empty"],
                "sampling/enabled": sampling_summary["enabled"],
                "sampling/target_positive_fraction": sampling_summary["target_positive_fraction"],
                "sampling/target_empty_fraction": sampling_summary["target_empty_fraction"],
                "sampling/positive_weight": sampling_summary["positive_weight"],
                "sampling/empty_weight": sampling_summary["empty_weight"],
                "sampling/num_samples": sampling_summary["num_samples"],
            },
            step=0,
        )

    for epoch in range(int(config.training.max_epochs)):
        stage = _stage_for_epoch(epoch, config)
        trainable_counts = _apply_stage_freezing(model, stage, head_patterns, partial_patterns)
        # lr_scale = _lr_scale_for_epoch(
        #     epoch=epoch,
        #     max_epochs=int(config.training.max_epochs),
        #     warmup_epochs=int(config.training.warmup_epochs),
        #     poly_power=float(config.training.poly_power),
        # )
        # lr_summary = _apply_lr_scale(optimizer, lr_scale)
        lr_summary = scheduler.step(epoch)
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        train_batches = 0
        train_bar = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch + 1}/{config.training.max_epochs}", leave=False)
        for batch in train_bar:
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
            running_loss += float(loss.detach().cpu())
            running_dice += _batch_dice(logits.detach(), masks.detach())
            train_batches += 1
            train_bar.set_postfix(loss=float(loss.detach().cpu()))

        average_loss = running_loss / max(1, len(train_loader))
        average_dice = running_dice / max(1, train_batches)
        val_loss = float("nan")
        val_dice = float("nan")
        if validation_interval > 0 and (epoch + 1) % validation_interval == 0:
            val_loss, val_dice, per_size_dice = evaluate_fold(
                model=model,
                loader=val_loader,
                config=config,
                device=device,
                use_amp=amp_enabled,
                class_weights=class_weights,
                val_records=val_records,
            )
        if best_checkpoint_metric == "train_dice":
            monitor_value = average_dice
        elif best_checkpoint_metric == "train_loss":
            monitor_value = average_loss
        elif best_checkpoint_metric == "val_dice":
            monitor_value = val_dice
        else:
            monitor_value = val_loss
        should_update_best = False
        if math.isfinite(monitor_value):
            if best_checkpoint_mode == "max":
                should_update_best = monitor_value > best_monitor_value
            else:
                should_update_best = monitor_value < best_monitor_value
        lr_head = float(lr_summary.get("head", 0.0))
        lr_partial = float(lr_summary.get("partial", 0.0))
        lr_full = float(lr_summary.get("full", 0.0))
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "train",
                    epoch + 1,
                    average_loss,
                    average_dice,
                    val_loss,
                    val_dice,
                    best_checkpoint_metric,
                    monitor_value,
                    lr_head,
                    lr_partial,
                    lr_full,
                ]
            )

        with run_log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"train\t{epoch + 1}\t{stage}\t{average_loss:.6f}\t{average_dice:.6f}\t"
                f"{val_loss:.6f}\t{val_dice:.6f}\t{best_checkpoint_metric}\t{monitor_value:.6f}\t"
                f"{lr_head:.8f}\t{lr_partial:.8f}\t{lr_full:.8f}\n"
            )
        print(
            f"Epoch {epoch + 1:03d} [{stage}] | train_loss={average_loss:.4f} | train_dice={average_dice:.4f} | "
            f"val_loss={val_loss:.4f} | val_dice={val_dice:.4f} | "
            f"lr(head/partial/full)={lr_head:.8f}/{lr_partial:.8f}/{lr_full:.8f} | "
            f"trainable(head/partial/full)={trainable_counts[0]}/{trainable_counts[1]}/{trainable_counts[2]}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": average_loss,
                    "train/dice": average_dice,
                    "val/loss": val_loss,
                    "val/dice": val_dice,
                    "monitor/value": monitor_value,
                    "monitor/metric": best_checkpoint_metric,
                    "lr/head": lr_head,
                    "lr/partial": lr_partial,
                    "lr/full": lr_full,
                    "stage": stage,
                    "stage_trainable/head": trainable_counts[0],
                    "stage_trainable/partial": trainable_counts[1],
                    "stage_trainable/full": trainable_counts[2],
                    "val/dice_tiny":  per_size_dice["tiny"],
                    "val/dice_small": per_size_dice["small"],
                    "val/dice_large": per_size_dice["large"],
                    "val/dice_empty": per_size_dice["empty"],
                    "val/n_tiny":     per_size_dice["n_tiny"],
                    "val/n_small":    per_size_dice["n_small"],
                    "val/n_large":    per_size_dice["n_large"],
                },
                step=epoch + 1,
            )

        payload = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "lr_summary": lr_summary,
            "monitor_metric": best_checkpoint_metric,
            "monitor_value": monitor_value,
            "train_loss": average_loss,
            "train_dice": average_dice,
            "val_dice": val_dice,
            "val_loss": val_loss,
            "config": json.loads(json.dumps(config, default=lambda value: value.__dict__)),
        }
        torch.save(payload, last_path)
        if should_update_best:
            best_monitor_value = monitor_value
            best_epoch = epoch + 1
            torch.save(payload, best_path)
            with run_log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"saved\t{epoch + 1}\t{stage}\t{average_loss:.6f}\t{average_dice:.6f}\t"
                    f"{val_loss:.6f}\t{val_dice:.6f}\n"
                )
        if config.training.save_every_epoch:
            torch.save(payload, output_dir / f"epoch_{epoch + 1:04d}.pt")

    final_val_loss = float("nan")
    final_val_dice = float("nan")
    best_checkpoint_path = best_path if best_path.exists() else last_path
    best_payload = torch.load(str(best_checkpoint_path), map_location="cpu", weights_only=False)
    model.load_state_dict(best_payload["model_state"])
    if final_validation:
        final_val_loss, final_val_dice = evaluate_fold(
            model=model,
            loader=val_loader,
            config=config,
            device=device,
            use_amp=amp_enabled,
            class_weights=class_weights,
        )
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "final_validation",
                    best_payload.get("epoch", best_epoch),
                    best_payload.get("train_loss", float("nan")),
                    best_payload.get("train_dice", float("nan")),
                    final_val_loss,
                    final_val_dice,
                    best_checkpoint_metric,
                    best_payload.get("monitor_value", best_monitor_value),
                    best_payload.get("lr_summary", {}).get("head", 0.0),
                    best_payload.get("lr_summary", {}).get("partial", 0.0),
                    best_payload.get("lr_summary", {}).get("full", 0.0),
                ]
            )
        with run_log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"final_validation\t{best_payload.get('epoch', best_epoch)}\tbest\t"
                f"{best_payload.get('train_loss', float('nan')):.6f}\t{best_payload.get('train_dice', float('nan')):.6f}\t"
                f"{final_val_loss:.6f}\t{final_val_dice:.6f}\t{best_checkpoint_metric}\t"
                f"{best_payload.get('monitor_value', best_monitor_value):.6f}\t"
                f"{best_payload.get('lr_summary', {}).get('head', 0.0):.8f}\t"
                f"{best_payload.get('lr_summary', {}).get('partial', 0.0):.8f}\t"
                f"{best_payload.get('lr_summary', {}).get('full', 0.0):.8f}\n"
            )
        print(
            f"Final validation on best checkpoint (epoch {best_payload.get('epoch', best_epoch):03d}) | "
            f"val_loss={final_val_loss:.4f} | val_dice={final_val_dice:.4f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "final_validation/loss": final_val_loss,
                    "final_validation/dice": final_val_dice,
                    "final_validation/best_epoch": int(best_payload.get("epoch", best_epoch)),
                    "final_validation/best_monitor": float(best_payload.get("monitor_value", best_monitor_value)),
                },
                step=int(best_payload.get("epoch", best_epoch)),
            )

    if wandb_run is not None:
        wandb_run.finish()

    return {
        "fold": fold,
        "best_epoch": int(best_payload.get("epoch", best_epoch)),
        "best_monitor": float(best_payload.get("monitor_value", best_monitor_value)),
        "final_val_loss": final_val_loss,
        "final_val_dice": final_val_dice,
    }


@torch.no_grad()
def evaluate_fold(
    model: torch.nn.Module,
    loader: DataLoader,
    config,
    device: torch.device,
    use_amp: bool,
    class_weights: torch.Tensor,
    val_records: None,
) -> tuple[float, float, dict[str, float]]:
    model.eval()
    scores: list[float] = []
    losses: list[float] = []
    size_scores:   dict[str, list[float]] = {"empty": [], "tiny": [], "small": [], "large": []}
    for batch_idx, batch in enumerate(tqdm(loader, desc="Validation", leave=False)):
        image = batch["image"].numpy()[0]   # (C, D, H, W)
        mask  = batch["mask"].numpy()[0]    # (D, H, W)
 
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
        mask_tensor   = torch.from_numpy(mask.copy()).unsqueeze(0).to(device=device, dtype=torch.long)
        loss = dice_ce_loss(logits_tensor, mask_tensor, class_weights=class_weights)
        losses.append(float(loss.detach().cpu()))
 
        prediction = np.argmax(logits, axis=0).astype(np.uint8)
        score = dice_score(prediction, mask)
        scores.append(score)
 
        # per-size bucketing
        if val_records is not None and batch_idx < len(val_records):
            record = val_records[batch_idx]
            # derive size from voxel count directly (no dependency on cached labels)
            lesion_voxels = int(np.count_nonzero(mask > 0))
            if lesion_voxels == 0:
                size = "empty"
            else:
                # rough thresholds matching make_stratification_labels
                if lesion_voxels < 500:
                    size = "tiny"
                elif lesion_voxels < 10_000:
                    size = "small"
                else:
                    size = "large"
            size_scores[size].append(score)
 
    mean_loss  = float(np.mean(losses)) if losses else 0.0
    mean_dice  = float(np.mean(scores)) if scores else 0.0
    per_size   = {
        size: float(np.mean(vals)) if vals else float("nan")
        for size, vals in size_scores.items()
    }
    per_size["n_empty"] = float(len(size_scores["empty"]))
    per_size["n_tiny"]  = float(len(size_scores["tiny"]))
    per_size["n_small"] = float(len(size_scores["small"]))
    per_size["n_large"] = float(len(size_scores["large"]))
 
    return mean_loss, mean_dice, per_size


def train_from_config(
    config_path: str | Path,
    fold: int | None = None,
    all_folds: bool = False,
    overrides: dict[str, object] | None = None,
) -> list[dict[str, float]]:
    config = load_config(config_path)
    apply_overrides(config, overrides)
    base_dir = Path(config_path).expanduser().resolve().parent
    splits_file = resolve_path(base_dir, config.paths.splits_file)
    if not Path(splits_file).exists():
        split_name = Path(splits_file).name.lower()
        if "train_val_test" in split_name or "test" in split_name:
            raise FileNotFoundError(
                f"Configured split file does not exist: {splits_file}. "
                "This filename looks like a fixed train/val/test split, so it will not be auto-created. "
                "Create/copy the intended split JSON first, or pass --splits-file to train.py."
            )
        from .splits import build_splits

        build_splits(
            dataset_dir=resolve_path(base_dir, config.paths.dataset_dir),
            num_folds=int(config.training.num_folds),
            seed=int(config.training.seed),
            output_path=splits_file,
        )

    if all_folds:
        folds = list(range(int(config.training.num_folds)))
    else:
        folds = [int(config.training.fold if fold is None else fold)]

    results = []
    for current_fold in folds:
        results.append(train_one_fold(config, current_fold, base_dir))
    return results
