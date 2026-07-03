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

from multitalent_tbi.case_filters import CaseInfo, build_case_infos, lesion_category
from multitalent_tbi.config import load_config, resolve_path
from multitalent_tbi.data import (
    TBIDataset,
    _crop_with_center,
    _pad_to_shape,
    _random_center,
    _random_flip,
    _random_intensity,
    discover_cases,
    load_case_cached,
)
from multitalent_tbi.engine import (
    _apply_stage_freezing,
    _normalize_patterns,
    _stage_for_epoch,
    build_model,
    build_stage_optimizer,
    configure_torch_for_speed,
    dice_score,
    set_seed,
)
from multitalent_tbi.infer import predict_logits
from multitalent_tbi.losses import dice_ce_loss
from multitalent_tbi.lr_scheduler import StagedPolyLRScheduler
from multitalent_tbi.splits import split_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean hierarchical branch trainer for Multi-Patched TBI experiments."
    )
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--branch", default="micro128", help="Branch under hierarchical_clean.branches.")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="Override full/base LR.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--resume-checkpoint", default=None, help="Resume exactly from a saved last.pt checkpoint.")
    parser.add_argument("--dry-run", action="store_true", help="Print split/sampling summary and exit.")
    return parser.parse_args()


def _get(namespace, name: str, default=None):
    return getattr(namespace, name, default) if namespace is not None else default


def _as_dict(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return vars(value)


def _to_jsonable(value):
    if isinstance(value, SimpleNamespace):
        return {key: _to_jsonable(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _load_clean_split(split_path: Path, fold: int) -> dict[str, list[str]]:
    with split_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    folds = payload.get("folds", [])
    for item in folds:
        if int(item.get("fold", -1)) == int(fold):
            return {"train": list(item["train"]), "val": list(item["val"])}
    raise KeyError(f"Fold {fold} not found in {split_path}")


def _write_case_manifest(path: Path, infos: list[CaseInfo]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "gt_voxels", "category", "has_dmri", "t1_path", "lesion_mask_path"],
        )
        writer.writeheader()
        for info in infos:
            writer.writerow(
                {
                    "case_id": info.record.case_id,
                    "gt_voxels": info.gt_voxels,
                    "category": info.category,
                    "has_dmri": int(info.record.has_dmri),
                    "t1_path": str(info.record.t1_path),
                    "lesion_mask_path": str(info.record.lesion_path),
                }
            )


def _category_counts(infos: list[CaseInfo]) -> dict[str, int]:
    counts = {category: 0 for category in ["empty", "very_tiny", "tiny", "small", "large"]}
    for info in infos:
        counts[info.category] = counts.get(info.category, 0) + 1
    return counts


def _sample_category_counts(samples: list[dict[str, object]]) -> dict[str, int]:
    counts = {category: 0 for category in ["empty", "very_tiny", "tiny", "small", "large", "context"]}
    for sample in samples:
        category = str(sample["sample_category"])
        counts[category] = counts.get(category, 0) + 1
    return counts


def _build_sample_sampler(
    samples: list[dict[str, object]],
    fractions_cfg,
    epoch_length_multiplier: float,
    seed: int,
    base_epoch_size: int | None = None,
) -> tuple[WeightedRandomSampler, dict[str, float | int | dict[str, float]]]:
    fractions = {key: float(value) for key, value in _as_dict(fractions_cfg).items()}
    counts = _sample_category_counts(samples)
    available = {category: value for category, value in fractions.items() if counts.get(category, 0) > 0 and value > 0}
    total_fraction = sum(available.values())
    if total_fraction <= 0:
        raise ValueError("No available category fractions for sampler.")
    normalized = {category: value / total_fraction for category, value in available.items()}
    weights = []
    for sample in samples:
        category = str(sample["sample_category"])
        target_fraction = normalized.get(category, 0.0)
        count = max(1, counts.get(category, 0))
        weights.append(target_fraction / float(count))
    epoch_base = int(base_epoch_size) if base_epoch_size is not None else len(samples)
    num_samples = max(1, int(round(epoch_base * float(epoch_length_multiplier))))
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    summary = {
        "num_samples_per_epoch": num_samples,
        "epoch_base_size": epoch_base,
        "epoch_length_multiplier": float(epoch_length_multiplier),
        "virtual_sample_counts": counts,
        "target_category_fractions": normalized,
    }
    return sampler, summary


def _component_category(component_voxels: int) -> str:
    return lesion_category(int(component_voxels))


def _component_center(component_mask: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(component_mask)
    if coords.size == 0:
        return tuple(size // 2 for size in component_mask.shape)
    return tuple(int(value) for value in np.round(coords.mean(axis=0)))


def _random_component_center(component_mask: np.ndarray, rng: random.Random) -> tuple[int, int, int]:
    coords = np.argwhere(component_mask)
    if coords.size == 0:
        return tuple(size // 2 for size in component_mask.shape)
    chosen = coords[rng.randrange(len(coords))]
    return tuple(int(value) for value in chosen)


def _samples_per_component(category: str, component_sampling_cfg) -> int:
    configured = _as_dict(_get(component_sampling_cfg, "samples_per_component", {}))
    default = {"very_tiny": 4, "tiny": 3, "small": 2, "large": 1}
    merged = {**default, **{key: int(value) for key, value in configured.items()}}
    return max(1, int(merged.get(category, 1)))


def _build_virtual_samples(
    train_infos: list[CaseInfo],
    config,
    branch_cfg,
    base_dir: Path,
    cache_dir: Path | None,
    seed: int,
) -> list[dict[str, object]]:
    component_cfg = _get(branch_cfg, "component_sampling", SimpleNamespace(enabled=False))
    if not bool(_get(component_cfg, "enabled", False)):
        return [
            {
                "record": info.record,
                "case_id": info.record.case_id,
                "sample_category": info.category,
                "sample_kind": "case_random",
                "component_index": -1,
                "component_voxels": info.gt_voxels,
                "center": None,
            }
            for info in train_infos
        ]

    rng = random.Random(int(seed))
    min_component_voxels = int(_get(component_cfg, "min_component_voxels", 1))
    max_components_per_case = int(_get(component_cfg, "max_components_per_case", 0))
    max_virtual_samples_per_case = int(_get(component_cfg, "max_virtual_samples_per_case", 0))
    context_samples_per_case = int(_get(component_cfg, "context_samples_per_case", 1))
    empty_samples_per_case = int(_get(component_cfg, "empty_samples_per_case", 1))
    samples: list[dict[str, object]] = []
    for info in tqdm(train_infos, desc="Building component-aware samples", leave=False):
        record = info.record
        if info.gt_voxels == 0:
            for _ in range(max(1, empty_samples_per_case)):
                samples.append(
                    {
                        "record": record,
                        "case_id": record.case_id,
                        "sample_category": "empty",
                        "sample_kind": "empty_random",
                        "component_index": -1,
                        "component_voxels": 0,
                        "center": None,
                    }
                )
            continue

        _, mask, _, _ = load_case_cached(
            case=record,
            target_spacing=tuple(config.data.target_spacing),
            include_dmri=bool(config.data.include_dmri),
            dmri_reduce=str(config.data.dmri_reduce),
            dmri_b0_threshold=float(config.data.dmri_b0_threshold),
            normalize_foreground_only=bool(config.data.normalize_foreground_only),
            cache_dir=cache_dir,
        )
        components, n_components = label(mask > 0)
        case_samples: list[dict[str, object]] = []
        component_items: list[tuple[int, int, str, np.ndarray]] = []
        for component_index in range(1, n_components + 1):
            component_mask = components == component_index
            component_voxels = int(component_mask.sum())
            if component_voxels < min_component_voxels:
                continue
            category = _component_category(component_voxels)
            component_items.append((component_index, component_voxels, category, component_mask))

        rng.shuffle(component_items)
        if max_components_per_case > 0:
            component_items = component_items[:max_components_per_case]

        for component_index, component_voxels, category, component_mask in component_items:
            repeats = _samples_per_component(category, component_cfg)
            centers = [_component_center(component_mask)]
            while len(centers) < repeats:
                centers.append(_random_component_center(component_mask, rng))
            for repeat_index, center in enumerate(centers):
                case_samples.append(
                    {
                        "record": record,
                        "case_id": record.case_id,
                        "sample_category": category,
                        "sample_kind": "component_centroid" if repeat_index == 0 else "component_random",
                        "component_index": component_index,
                        "component_voxels": component_voxels,
                        "center": center,
                    }
                )

        for _ in range(max(0, context_samples_per_case)):
            case_samples.append(
                {
                    "record": record,
                    "case_id": record.case_id,
                    "sample_category": "context",
                    "sample_kind": "context_random",
                    "component_index": -1,
                    "component_voxels": info.gt_voxels,
                    "center": None,
                }
            )

        if max_virtual_samples_per_case > 0 and len(case_samples) > max_virtual_samples_per_case:
            rng.shuffle(case_samples)
            case_samples = case_samples[:max_virtual_samples_per_case]
        samples.extend(case_samples)

    if not samples:
        raise RuntimeError("No virtual training samples were created.")
    return samples


def _write_sample_manifest(path: Path, samples: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "sample_category",
                "sample_kind",
                "component_index",
                "component_voxels",
                "center_i",
                "center_j",
                "center_k",
            ],
        )
        writer.writeheader()
        for sample in samples:
            center = sample.get("center")
            if center is None:
                center_i = center_j = center_k = ""
            else:
                center_i, center_j, center_k = center
            writer.writerow(
                {
                    "case_id": sample["case_id"],
                    "sample_category": sample["sample_category"],
                    "sample_kind": sample["sample_kind"],
                    "component_index": sample["component_index"],
                    "component_voxels": sample["component_voxels"],
                    "center_i": center_i,
                    "center_j": center_j,
                    "center_k": center_k,
                }
            )


class ComponentAwarePatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: list[dict[str, object]],
        patch_size,
        target_spacing,
        include_dmri: bool,
        dmri_reduce: str,
        dmri_b0_threshold: float,
        normalize_foreground_only: bool,
        cache_dir: str | Path | None,
        center_jitter_voxels: int,
        random_intensity_prob: float,
        default_foreground_prob: float,
    ) -> None:
        self.samples = samples
        self.patch_size = tuple(int(value) for value in patch_size)
        self.target_spacing = tuple(float(value) for value in target_spacing)
        self.include_dmri = include_dmri
        self.dmri_reduce = dmri_reduce
        self.dmri_b0_threshold = float(dmri_b0_threshold)
        self.normalize_foreground_only = normalize_foreground_only
        self.cache_dir = cache_dir
        self.center_jitter_voxels = int(center_jitter_voxels)
        self.random_intensity_prob = float(random_intensity_prob)
        self.default_foreground_prob = float(default_foreground_prob)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        record = sample["record"]
        image, mask, image_affine, mask_affine = load_case_cached(
            case=record,
            target_spacing=self.target_spacing,
            include_dmri=self.include_dmri,
            dmri_reduce=self.dmri_reduce,
            dmri_b0_threshold=self.dmri_b0_threshold,
            normalize_foreground_only=self.normalize_foreground_only,
            cache_dir=self.cache_dir,
        )
        image, mask = self._sample_patch(image, mask, sample)
        image, mask = _random_flip(image, mask)
        if random.random() < self.random_intensity_prob:
            image = _random_intensity(image)
        return {
            "image": torch.from_numpy(image.copy()).float(),
            "mask": torch.from_numpy(mask.copy()).long(),
            "case_id": str(sample["case_id"]),
            "sample_category": str(sample["sample_category"]),
            "sample_kind": str(sample["sample_kind"]),
            "image_affine": torch.from_numpy(np.asarray(image_affine)),
            "mask_affine": torch.from_numpy(np.asarray(mask_affine)),
        }

    def _sample_patch(self, image: np.ndarray, mask: np.ndarray, sample: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
        padded_image, pads = _pad_to_shape(image, self.patch_size, fill_value=0.0)
        padded_mask, _ = _pad_to_shape(mask, self.patch_size, fill_value=0)
        center = sample.get("center")
        if center is None:
            if str(sample.get("sample_kind")) == "case_random":
                raw_center = _random_center(padded_mask, self.default_foreground_prob)
            else:
                raw_center = _random_center(padded_mask, 0.0)
        else:
            raw_center = tuple(
                int(center[axis]) + pads[axis][0] + random.randint(-self.center_jitter_voxels, self.center_jitter_voxels)
                for axis in range(3)
            )
        clamped_center = tuple(
            max(0, min(int(raw_center[axis]), padded_mask.shape[axis] - 1))
            for axis in range(3)
        )
        return (
            _crop_with_center(padded_image, clamped_center, self.patch_size),
            _crop_with_center(padded_mask, clamped_center, self.patch_size),
        )


def _filter_components(mask: np.ndarray, min_component_voxels: int) -> np.ndarray:
    if min_component_voxels <= 1 or not mask.any():
        return mask.astype(np.uint8)
    components, n_components = label(mask.astype(bool))
    filtered = np.zeros_like(mask, dtype=np.uint8)
    for component_index in range(1, n_components + 1):
        component = components == component_index
        if int(component.sum()) >= int(min_component_voxels):
            filtered[component] = 1
    return filtered


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _is_better(value: float, best: float, mode: str) -> bool:
    if not math.isfinite(value):
        return False
    return value < best if mode == "min" else value > best


@torch.no_grad()
def evaluate_branch(
    model: torch.nn.Module,
    loader: DataLoader,
    val_infos: list[CaseInfo],
    config,
    branch_cfg,
    device: torch.device,
    use_amp: bool,
    class_weights: torch.Tensor,
) -> tuple[float, dict[str, float], list[dict[str, object]]]:
    model.eval()
    threshold = float(_get(branch_cfg.validation, "threshold", 0.5))
    min_component_voxels = int(_get(branch_cfg.validation, "min_component_voxels", 0))
    losses: list[float] = []
    rows: list[dict[str, object]] = []

    for index, batch in enumerate(tqdm(loader, desc="Validation", leave=False)):
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

        probabilities = torch.softmax(torch.from_numpy(logits).float(), dim=0).numpy()
        prediction = (probabilities[1] >= threshold).astype(np.uint8)
        prediction = _filter_components(prediction, min_component_voxels)
        score = dice_score(prediction, mask)
        pred_voxels = int(np.count_nonzero(prediction > 0))
        rows.append(
            {
                "case_id": info.record.case_id,
                "gt_voxels": info.gt_voxels,
                "gt_category": info.category,
                "pred_voxels": pred_voxels,
                "pred_category": lesion_category(pred_voxels),
                "dice": score,
                "missed_positive": int(info.gt_voxels > 0 and pred_voxels == 0),
                "empty_false_positive": int(info.gt_voxels == 0 and pred_voxels > 0),
            }
        )

    positive_rows = [row for row in rows if int(row["gt_voxels"]) > 0]
    gt50_rows = [row for row in rows if int(row["gt_voxels"]) >= 50]
    micro_rows = [row for row in rows if row["gt_category"] in {"very_tiny", "tiny"}]
    metrics = {
        "val_dice_all": _mean([float(row["dice"]) for row in rows]),
        "val_dice_positive": _mean([float(row["dice"]) for row in positive_rows]),
        "val_dice_gt50": _mean([float(row["dice"]) for row in gt50_rows]),
        "val_dice_micro": _mean([float(row["dice"]) for row in micro_rows]),
        "val_dice_empty": _mean([float(row["dice"]) for row in rows if row["gt_category"] == "empty"]),
        "val_dice_very_tiny": _mean([float(row["dice"]) for row in rows if row["gt_category"] == "very_tiny"]),
        "val_dice_tiny": _mean([float(row["dice"]) for row in rows if row["gt_category"] == "tiny"]),
        "val_dice_small": _mean([float(row["dice"]) for row in rows if row["gt_category"] == "small"]),
        "val_dice_large": _mean([float(row["dice"]) for row in rows if row["gt_category"] == "large"]),
        "n_missed_positive": float(sum(int(row["missed_positive"]) for row in rows)),
        "n_empty_false_positive": float(sum(int(row["empty_false_positive"]) for row in rows)),
        "n_empty": float(sum(1 for row in rows if row["gt_category"] == "empty")),
        "n_very_tiny": float(sum(1 for row in rows if row["gt_category"] == "very_tiny")),
        "n_tiny": float(sum(1 for row in rows if row["gt_category"] == "tiny")),
        "n_small": float(sum(1 for row in rows if row["gt_category"] == "small")),
        "n_large": float(sum(1 for row in rows if row["gt_category"] == "large")),
    }
    empty = metrics["val_dice_empty"]
    positive = metrics["val_dice_positive"]
    metrics["val_dice_balanced"] = (
        float(np.mean([value for value in [empty, positive] if math.isfinite(value)]))
        if any(math.isfinite(value) for value in [empty, positive])
        else float("nan")
    )
    return _mean(losses), metrics, rows


def _apply_branch_config(config, branch_cfg, args: argparse.Namespace) -> None:
    patch_size = list(_get(branch_cfg, "patch_size", config.data.patch_size))
    config.data.patch_size = patch_size
    config.data.oversample_foreground_prob = float(_get(branch_cfg, "oversample_foreground_prob", config.data.oversample_foreground_prob))
    config.inference.roi_size = list(_get(branch_cfg, "roi_size", patch_size))
    config.inference.overlap = float(_get(branch_cfg, "overlap", config.inference.overlap))
    config.inference.sw_batch_size = int(_get(branch_cfg, "sw_batch_size", config.inference.sw_batch_size))

    train_cfg = _get(branch_cfg, "training", SimpleNamespace())
    config.training.max_epochs = int(args.epochs if args.epochs is not None else _get(train_cfg, "max_epochs", config.training.max_epochs))
    config.training.batch_size = int(args.batch_size if args.batch_size is not None else _get(train_cfg, "batch_size", config.training.batch_size))
    config.training.num_workers = int(args.num_workers if args.num_workers is not None else _get(train_cfg, "num_workers", config.training.num_workers))
    config.training.prefetch_factor = int(_get(train_cfg, "prefetch_factor", _get(config.training, "prefetch_factor", 2)))
    lr = float(args.lr if args.lr is not None else _get(train_cfg, "base_lr", config.training.base_lr))
    config.training.base_lr = lr
    config.training.weight_decay = float(_get(train_cfg, "weight_decay", config.training.weight_decay))
    config.training.poly_power = float(_get(train_cfg, "poly_power", config.training.poly_power))
    config.training.min_lr = float(_get(train_cfg, "min_lr", _get(config.training, "min_lr", 1e-7)))
    config.training.use_amp = bool(_get(train_cfg, "use_amp", config.training.use_amp))
    config.training.optimizer = str(_get(train_cfg, "optimizer", config.training.optimizer))
    config.training.class_weights = list(_get(train_cfg, "class_weights", config.training.class_weights))
    config.training.grad_clip_norm = float(_get(train_cfg, "grad_clip_norm", config.training.grad_clip_norm))
    config.training.validation_interval_epochs = 1
    config.training.save_every_epoch = False
    config.training.final_validation = False
    config.training.staged_tuning.enabled = bool(_get(train_cfg, "staged_tuning_enabled", False))
    config.training.staged_tuning.head_only_epochs = int(_get(train_cfg, "head_only_epochs", 0))
    config.training.staged_tuning.partial_tune_epochs = int(_get(train_cfg, "partial_tune_epochs", 0))
    config.training.staged_tuning.head_lr = float(_get(train_cfg, "head_lr", lr))
    config.training.staged_tuning.partial_lr = float(_get(train_cfg, "partial_lr", lr))
    config.training.staged_tuning.full_lr = float(_get(train_cfg, "full_lr", lr))
    config.training.head_warmup_epochs = int(_get(train_cfg, "head_warmup_epochs", 0))
    config.training.partial_warmup_epochs = int(_get(train_cfg, "partial_warmup_epochs", 0))
    config.training.full_warmup_epochs = int(_get(train_cfg, "full_warmup_epochs", _get(train_cfg, "warmup_epochs", 5)))
    config.training.warmup_epochs = int(_get(train_cfg, "warmup_epochs", config.training.full_warmup_epochs))


def _init_wandb(config, branch: str, fold: int, output_dir: Path, args: argparse.Namespace, sampler_summary: dict):
    if args.no_wandb:
        return None
    wandb_cfg = _get(config, "wandb", None)
    if wandb_cfg is None or not bool(_get(wandb_cfg, "enabled", False)):
        return None
    try:
        import wandb
    except ImportError:
        print("[WARN] wandb not installed; continuing without W&B.")
        return None
    if not hasattr(wandb, "init"):
        print("[WARN] imported wandb has no init(); continuing without W&B.")
        return None
    name = args.wandb_name or _get(wandb_cfg, "name", None) or f"hierarchical_{branch}_f{fold}"
    return wandb.init(
        project=_get(wandb_cfg, "project", "AIMS-TBI-Hierarchical"),
        entity=_get(wandb_cfg, "entity", None),
        name=name,
        dir=str(output_dir / "wandb"),
        mode=str(_get(wandb_cfg, "mode", "offline")),
        tags=list(_get(wandb_cfg, "tags", []) or []) + ["hierarchical-clean", branch],
        config={
            "branch": branch,
            "fold": fold,
            "sampler": sampler_summary,
        },
    )


def _write_history_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "stage",
                "train_loss",
                "train_dice",
                "val_loss",
                "val_dice_all",
                "val_dice_positive",
                "val_dice_gt50",
                "val_dice_micro",
                "val_dice_empty",
                "val_dice_very_tiny",
                "val_dice_tiny",
                "val_dice_small",
                "val_dice_large",
                "n_empty",
                "n_very_tiny",
                "n_tiny",
                "n_small",
                "n_large",
                "n_missed_positive",
                "n_empty_false_positive",
                "primary_metric",
                "primary_value",
                "balanced_value",
                "lr_head",
                "lr_partial",
                "lr_full",
            ]
        )


def _append_history(path: Path, row: list[object]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(row)


def _history_best(path: Path, metric: str, mode: str = "max") -> float:
    if not path.exists():
        return -math.inf if mode == "max" else math.inf
    best = -math.inf if mode == "max" else math.inf
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                value = float(row[metric])
            except (KeyError, TypeError, ValueError):
                continue
            if _is_better(value, best, mode):
                best = value
    return best


def _torch_load_checkpoint(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _capture_rng_state(sampler: WeightedRandomSampler) -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "sampler": sampler.generator.get_state() if sampler.generator is not None else None,
    }


def _restore_rng_state(rng_state: dict[str, object], sampler: WeightedRandomSampler) -> None:
    if not rng_state:
        return
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "torch" in rng_state:
        torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng_state["cuda"])
    if sampler.generator is not None and rng_state.get("sampler") is not None:
        sampler.generator.set_state(rng_state["sampler"])


def main() -> None:
    args = parse_args()
    base_dir = Path(args.config).expanduser().resolve().parent
    config = load_config(args.config)
    hierarchical_cfg = _get(config, "hierarchical_clean", None)
    if hierarchical_cfg is None:
        raise ValueError("config.yml needs a hierarchical_clean section.")
    branch_cfg = getattr(hierarchical_cfg.branches, args.branch)
    fold = int(args.fold if args.fold is not None else _get(config.training, "fold", 0))
    _apply_branch_config(config, branch_cfg, args)

    set_seed(int(config.training.seed) + fold)
    configure_torch_for_speed()

    split_path = resolve_path(base_dir, hierarchical_cfg.splits_file)
    split = _load_clean_split(split_path, fold)
    records = discover_cases(resolve_path(base_dir, config.paths.dataset_dir))
    train_records, val_records = split_records(records, split)
    train_infos = build_case_infos(train_records, split="train")
    val_infos = build_case_infos(val_records, split="val")

    output_root = resolve_path(base_dir, args.output_dir or hierarchical_cfg.output_root)
    output_dir = output_root / str(_get(branch_cfg, "output_subdir", args.branch)) / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_case_manifest(output_dir / "train_cases.csv", train_infos)
    _write_case_manifest(output_dir / "val_cases.csv", val_infos)

    cache_dir = resolve_path(base_dir, config.paths.cache_dir) if bool(config.data.cache_preprocessed) else None
    train_samples = _build_virtual_samples(
        train_infos=train_infos,
        config=config,
        branch_cfg=branch_cfg,
        base_dir=base_dir,
        cache_dir=cache_dir,
        seed=int(config.training.seed) + fold,
    )
    _write_sample_manifest(output_dir / "train_virtual_samples.csv", train_samples)

    sampler_cfg = branch_cfg.sampling
    sampler, sampler_summary = _build_sample_sampler(
        train_samples,
        fractions_cfg=sampler_cfg.category_fractions,
        epoch_length_multiplier=float(_get(sampler_cfg, "epoch_length_multiplier", 1.0)),
        seed=int(config.training.seed) + fold,
        base_epoch_size=len(train_infos),
    )

    print(f"[INFO] Branch={args.branch} fold={fold}")
    print(f"[INFO] Split file: {split_path}")
    print(f"[INFO] Train counts: {_category_counts(train_infos)}")
    print(f"[INFO] Val counts  : {_category_counts(val_infos)}")
    print(f"[INFO] Virtual sample counts: {_sample_category_counts(train_samples)}")
    print(f"[INFO] Sampler    : {sampler_summary}")
    print(f"[INFO] Output dir : {output_dir}")
    if args.dry_run:
        return

    component_cfg = _get(branch_cfg, "component_sampling", SimpleNamespace())
    train_dataset = ComponentAwarePatchDataset(
        train_samples,
        patch_size=tuple(config.data.patch_size),
        target_spacing=tuple(config.data.target_spacing),
        include_dmri=bool(config.data.include_dmri),
        dmri_reduce=str(config.data.dmri_reduce),
        dmri_b0_threshold=float(config.data.dmri_b0_threshold),
        normalize_foreground_only=bool(config.data.normalize_foreground_only),
        cache_dir=cache_dir,
        center_jitter_voxels=int(_get(component_cfg, "center_jitter_voxels", 8)),
        random_intensity_prob=float(_get(component_cfg, "random_intensity_prob", 0.2)),
        default_foreground_prob=float(config.data.oversample_foreground_prob),
    )
    val_dataset = TBIDataset(
        val_records,
        patch_size=tuple(config.data.patch_size),
        target_spacing=tuple(config.data.target_spacing),
        include_dmri=bool(config.data.include_dmri),
        dmri_reduce=str(config.data.dmri_reduce),
        dmri_b0_threshold=float(config.data.dmri_b0_threshold),
        normalize_foreground_only=bool(config.data.normalize_foreground_only),
        oversample_foreground_prob=0.0,
        training=False,
        cache_dir=cache_dir,
    )
    loader_kwargs = {
        "num_workers": int(config.training.num_workers),
        "pin_memory": torch.cuda.is_available(),
    }
    if int(config.training.num_workers) > 0:
        loader_kwargs["prefetch_factor"] = int(_get(config.training, "prefetch_factor", 2))
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.training.batch_size),
        sampler=sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, base_dir).to(device)
    optimizer = build_stage_optimizer(model, config)
    scheduler = StagedPolyLRScheduler(optimizer, config)
    amp_enabled = bool(config.training.use_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    class_weights = torch.tensor(config.training.class_weights, dtype=torch.float32, device=device)
    staged = _get(config.training, "staged_tuning", None)
    head_patterns = _normalize_patterns(_get(staged, "head_patterns", None), ["seg", "final", "classifier", "output"])
    partial_patterns = _normalize_patterns(_get(staged, "partial_patterns", None), ["decoder", "transpconvs", "seg_layers"])

    primary_metric = str(_get(branch_cfg, "primary_metric", "val_dice_micro"))
    primary_mode = str(_get(branch_cfg, "primary_mode", "max"))
    best_primary = -math.inf if primary_mode == "max" else math.inf
    best_balanced = -math.inf
    history_path = output_dir / "history.csv"

    start_epoch = 0
    if args.resume_checkpoint is not None:
        resume_path = resolve_path(base_dir, args.resume_checkpoint)
        resume_payload = _torch_load_checkpoint(resume_path, map_location=device)
        if str(resume_payload.get("branch", args.branch)) != str(args.branch):
            raise ValueError(f"Resume checkpoint branch={resume_payload.get('branch')} does not match --branch {args.branch}.")
        if int(resume_payload.get("fold", fold)) != int(fold):
            raise ValueError(f"Resume checkpoint fold={resume_payload.get('fold')} does not match --fold {fold}.")
        model.load_state_dict(resume_payload["model_state"])
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        if "scaler_state" in resume_payload:
            scaler.load_state_dict(resume_payload["scaler_state"])
        _restore_rng_state(resume_payload.get("rng_state", {}), sampler)
        start_epoch = int(resume_payload["epoch"])
        best_primary = _history_best(history_path, primary_metric, primary_mode)
        best_balanced = _history_best(history_path, "balanced_value", "max")
        if not math.isfinite(best_primary):
            best_primary = float(resume_payload.get("primary_value", best_primary))
        if not math.isfinite(best_balanced):
            best_balanced = float(resume_payload.get("balanced_value", best_balanced))
        print(
            f"[INFO] Resumed from {resume_path} at epoch {start_epoch}. "
            f"best_primary={best_primary:.6f}, best_balanced={best_balanced:.6f}"
        )
    else:
        _write_history_header(history_path)

    wandb_run = _init_wandb(config, args.branch, fold, output_dir, args, sampler_summary)

    max_epochs = int(config.training.max_epochs)
    if start_epoch >= max_epochs:
        print(f"[DONE] Resume epoch {start_epoch} already >= max_epochs {max_epochs}. Nothing to train.")
        if wandb_run is not None:
            wandb_run.finish()
        return

    for epoch in range(start_epoch, max_epochs):
        stage = _stage_for_epoch(epoch, config)
        trainable_counts = _apply_stage_freezing(model, stage, head_patterns, partial_patterns)
        lr_summary = scheduler.step(epoch)
        model.train()

        running_loss = 0.0
        running_dice = 0.0
        train_batches = 0
        for batch in tqdm(train_loader, desc=f"{args.branch} f{fold} epoch {epoch + 1}/{max_epochs}", leave=False):
            images = batch["image"].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last_3d)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                loss = dice_ce_loss(logits, masks, class_weights=class_weights)
            scaler.scale(loss).backward()
            if float(config.training.grad_clip_norm) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.training.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()

            predictions = torch.argmax(logits.detach(), dim=1)
            scores = [
                dice_score(pred.cpu().numpy(), target.cpu().numpy())
                for pred, target in zip(predictions, masks.detach())
            ]
            running_loss += float(loss.detach().cpu())
            running_dice += float(np.mean(scores)) if scores else 0.0
            train_batches += 1

        train_loss = running_loss / max(1, train_batches)
        train_dice = running_dice / max(1, train_batches)
        val_loss, val_metrics, val_rows = evaluate_branch(
            model=model,
            loader=val_loader,
            val_infos=val_infos,
            config=config,
            branch_cfg=branch_cfg,
            device=device,
            use_amp=amp_enabled,
            class_weights=class_weights,
        )
        primary_value = float(val_metrics.get(primary_metric, float("nan")))
        balanced_value = float(val_metrics.get("val_dice_balanced", float("nan")))
        payload = {
            "epoch": epoch + 1,
            "branch": args.branch,
            "fold": fold,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "rng_state": _capture_rng_state(sampler),
            "lr_summary": lr_summary,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "primary_metric": primary_metric,
            "primary_value": primary_value,
            "balanced_value": balanced_value,
            "split_file": str(split_path),
            "pretrained_checkpoint": str(resolve_path(base_dir, config.paths.pretrained_checkpoint)),
            "config": _to_jsonable(config),
        }
        torch.save(payload, output_dir / "last.pt")
        if _is_better(primary_value, best_primary, primary_mode):
            best_primary = primary_value
            torch.save(payload, output_dir / "best_primary.pt")
            _write_case_manifest(output_dir / "best_primary_val_cases.csv", val_infos)
            with (output_dir / "best_primary_val_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(val_rows[0].keys()) if val_rows else ["case_id"])
                writer.writeheader()
                writer.writerows(val_rows)
        if _is_better(balanced_value, best_balanced, "max"):
            best_balanced = balanced_value
            torch.save(payload, output_dir / "best_balanced.pt")

        _append_history(
            history_path,
            [
                epoch + 1,
                stage,
                train_loss,
                train_dice,
                val_loss,
                val_metrics.get("val_dice_all", float("nan")),
                val_metrics.get("val_dice_positive", float("nan")),
                val_metrics.get("val_dice_gt50", float("nan")),
                val_metrics.get("val_dice_micro", float("nan")),
                val_metrics.get("val_dice_empty", float("nan")),
                val_metrics.get("val_dice_very_tiny", float("nan")),
                val_metrics.get("val_dice_tiny", float("nan")),
                val_metrics.get("val_dice_small", float("nan")),
                val_metrics.get("val_dice_large", float("nan")),
                val_metrics.get("n_empty", float("nan")),
                val_metrics.get("n_very_tiny", float("nan")),
                val_metrics.get("n_tiny", float("nan")),
                val_metrics.get("n_small", float("nan")),
                val_metrics.get("n_large", float("nan")),
                val_metrics.get("n_missed_positive", float("nan")),
                val_metrics.get("n_empty_false_positive", float("nan")),
                primary_metric,
                primary_value,
                balanced_value,
                float(lr_summary.get("head", 0.0)),
                float(lr_summary.get("partial", 0.0)),
                float(lr_summary.get("full", 0.0)),
            ],
        )
        print(
            f"Epoch {epoch + 1:03d} [{stage}] "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
            f"val_micro={val_metrics['val_dice_micro']:.4f} "
            f"val_pos={val_metrics['val_dice_positive']:.4f} "
            f"val_empty={val_metrics['val_dice_empty']:.4f} "
            f"missed={val_metrics['n_missed_positive']:.0f} "
            f"empty_fp={val_metrics['n_empty_false_positive']:.0f} "
            f"lr={float(lr_summary.get('full', 0.0)):.2e} "
            f"trainable={trainable_counts}"
        )
        if wandb_run is not None:
            bucket_logs = {
                "val_bucket/empty_dice": val_metrics.get("val_dice_empty", float("nan")),
                "val_bucket/very_tiny_dice": val_metrics.get("val_dice_very_tiny", float("nan")),
                "val_bucket/tiny_dice": val_metrics.get("val_dice_tiny", float("nan")),
                "val_bucket/small_dice": val_metrics.get("val_dice_small", float("nan")),
                "val_bucket/large_dice": val_metrics.get("val_dice_large", float("nan")),
                "val_bucket/micro_dice": val_metrics.get("val_dice_micro", float("nan")),
                "val_bucket/empty_n": val_metrics.get("n_empty", float("nan")),
                "val_bucket/very_tiny_n": val_metrics.get("n_very_tiny", float("nan")),
                "val_bucket/tiny_n": val_metrics.get("n_tiny", float("nan")),
                "val_bucket/small_n": val_metrics.get("n_small", float("nan")),
                "val_bucket/large_n": val_metrics.get("n_large", float("nan")),
            }
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/dice": train_dice,
                    "val/loss": val_loss,
                    **{f"val/{key.replace('val_', '')}": value for key, value in val_metrics.items()},
                    **bucket_logs,
                    "monitor/primary": primary_value,
                    "monitor/balanced": balanced_value,
                    "lr/full": float(lr_summary.get("full", 0.0)),
                },
                step=epoch + 1,
            )

    if wandb_run is not None:
        wandb_run.finish()
    print(f"[DONE] best_primary={best_primary:.6f} ({primary_metric}) best_balanced={best_balanced:.6f}")
    print(f"[DONE] Output: {output_dir}")


if __name__ == "__main__":
    main()
