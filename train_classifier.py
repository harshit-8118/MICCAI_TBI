from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from multitalent_tbi.config import load_config, resolve_path
from multitalent_tbi.data import (
    CaseRecord,
    _crop_with_center,
    _pad_to_shape,
    _random_center,
    _random_flip,
    _random_intensity,
    center_crop_or_pad,
    discover_cases,
    load_case_cached,
)
from multitalent_tbi.engine import build_model, configure_torch_for_speed, set_seed
from multitalent_tbi.infer import _pad_volume, _sliding_positions
from multitalent_tbi.model import _adapt_conv_weight, _extract_state_dict, _strip_prefix
from multitalent_tbi.splits import load_splits, split_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frozen-encoder Depth Vector classifier.")
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--splits-file", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--init-checkpoint", default=None, help="Checkpoint used to initialize the encoder.")
    parser.add_argument("--seg-checkpoint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--encoder-lr", type=float, default=None, help="LR for unfrozen encoder parameters.")
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--head-hidden-dim", type=int, default=None, help="Use a 2-layer MLP head when >0. Default is linear head.")
    parser.add_argument("--patch-size", nargs=3, type=int, default=None)
    parser.add_argument("--samples-per-epoch", type=int, default=None, help="0 means one sampled patch per training case per epoch.")
    parser.add_argument("--positive-fraction", type=float, default=None, help="Target fraction of positive cases sampled for train patches.")
    parser.add_argument("--foreground-prob", type=float, default=None, help="For positive cases, probability of sampling near lesion voxels.")
    parser.add_argument("--no-class-weights", action="store_true", help="Use unweighted cross entropy. Useful when sampling is already balanced.")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--validation-overlap", type=float, default=None)
    parser.add_argument("--validation-batch-size", type=int, default=None)
    parser.add_argument("--validation-aggregation", choices=["max", "topk_mean", "mean"], default=None)
    parser.add_argument("--validation-top-k", type=int, default=None)
    parser.add_argument("--unfreeze-encoder", action="store_true", help="Fine-tune encoder too. Default keeps it frozen.")
    parser.add_argument("--unfreeze-last-encoder-stage", action="store_true", help="Fine-tune only the deepest encoder stage with --encoder-lr.")
    parser.add_argument("--unfreeze-last-encoder-block", action="store_true", help="Fine-tune only the deepest block inside the deepest encoder stage.")
    parser.add_argument("--staged-tuning", action="store_true", help="Use head-only -> last-block -> last-stage -> full-encoder tuning.")
    parser.add_argument("--no-staged-tuning", action="store_true", help="Disable config classifier_detection.staged_tuning.")
    parser.add_argument("--head-only-epochs", type=int, default=None)
    parser.add_argument("--last-block-epochs", type=int, default=None)
    parser.add_argument("--last-stage-epochs", type=int, default=None)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--last-block-lr", type=float, default=None)
    parser.add_argument("--last-stage-lr", type=float, default=None)
    parser.add_argument("--full-encoder-lr", type=float, default=None)
    parser.add_argument(
        "--scheduler",
        choices=["poly", "cosine", "none"],
        default=None,
        help="Learning-rate schedule. Default poly decay, matching the segmentation training style.",
    )
    parser.add_argument("--min-lr-ratio", type=float, default=None, help="Final LR as a fraction of each parameter group's initial LR.")
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _torch_load(path: Path, map_location):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def _extract_init_state_dict(payload) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("network_weights", "state_dict", "model", "model_state", "model_state_dict"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(key, str) for key in payload.keys()):
            return payload
    return _extract_state_dict(payload)


def _build_model_architecture_only(config, base_dir: Path) -> nn.Module:
    original_load_pretrained = bool(config.model.load_pretrained)
    config.model.load_pretrained = False
    try:
        return build_model(config, base_dir)
    finally:
        config.model.load_pretrained = original_load_pretrained


def _load_segmentation_checkpoint(model: nn.Module, checkpoint_path: Path) -> dict[str, int]:
    payload = _torch_load(checkpoint_path, map_location="cpu")
    state = _strip_prefix(_extract_init_state_dict(payload))
    state = {key[len("seg_model."):] if key.startswith("seg_model.") else key: value for key, value in state.items()}
    current_state = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    ignored_extra = 0
    skipped_shape = 0
    adapted_shape = 0

    for key, value in state.items():
        if key not in current_state:
            ignored_extra += 1
            continue
        if not torch.is_tensor(value):
            ignored_extra += 1
            continue
        if current_state[key].shape == value.shape:
            filtered[key] = value
            continue
        adapted = _adapt_conv_weight(value, current_state[key].shape)
        if adapted is not None:
            filtered[key] = adapted
            adapted_shape += 1
            continue
        skipped_shape += 1

    result = model.load_state_dict(filtered, strict=False)
    loaded_encoder = sum(1 for key in filtered if key.startswith("encoder."))
    loaded_decoder = sum(1 for key in filtered if key.startswith(("decoder.", "transpconvs.", "seg_layers.")))
    missing_encoder = sum(1 for key in result.missing_keys if key.startswith("encoder."))
    print(
        f"Loaded classifier init checkpoint once: {checkpoint_path}\n"
        f"  loaded={len(filtered)} encoder_loaded={loaded_encoder} decoder_or_seg_loaded={loaded_decoder}\n"
        f"  missing_after_filter={len(result.missing_keys)} missing_encoder={missing_encoder} "
        f"ignored_extra={ignored_extra} skipped_shape={skipped_shape} adapted_shape={adapted_shape}"
    )
    if loaded_encoder == 0:
        raise RuntimeError(
            "No encoder weights were loaded from the init checkpoint. "
            "Check that pretrained_plans and init_checkpoint belong to the same architecture."
        )
    return {
        "loaded": len(filtered),
        "encoder_loaded": loaded_encoder,
        "decoder_or_seg_loaded": loaded_decoder,
        "missing_after_filter": len(result.missing_keys),
        "missing_encoder": missing_encoder,
        "ignored_extra": ignored_extra,
        "skipped_shape": skipped_shape,
        "adapted_shape": adapted_shape,
    }


def encoder_feature(seg_model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    encoder = getattr(seg_model, "encoder", None)
    if encoder is None:
        raise AttributeError("Segmentation model has no '.encoder'. This classifier expects an encoder-style nnU-Net model.")
    features = encoder(image)
    if isinstance(features, dict):
        features = list(features.values())
    if isinstance(features, (list, tuple)):
        return features[-1]
    return features


class DepthVectorPooling(nn.Module):
    def __init__(self, depth_bins: int) -> None:
        super().__init__()
        self.depth_bins = int(depth_bins)
        self.depth_logits = nn.Parameter(torch.zeros(self.depth_bins, dtype=torch.float32))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 5:
            raise ValueError(f"Expected encoder features with shape [B, C, D, H, W], got {tuple(feature.shape)}")
        depth = int(feature.shape[2])
        logits = self.depth_logits
        if logits.numel() != depth:
            logits = F.interpolate(
                logits.view(1, 1, -1),
                size=depth,
                mode="linear",
                align_corners=False,
            ).reshape(-1)
        weights = torch.softmax(logits, dim=0)
        return torch.einsum("bcdhw,d->bchw", feature, weights)


class DepthVectorClassifier(nn.Module):
    def __init__(
        self,
        seg_model: nn.Module,
        feature_channels: int,
        feature_depth: int,
        dropout: float,
        head_hidden_dim: int = 0,
        freeze_encoder: bool = True,
        encoder_eval: bool = True,
    ) -> None:
        super().__init__()
        self.seg_model = seg_model
        self.freeze_encoder = bool(freeze_encoder)
        self.encoder_eval = bool(encoder_eval)
        self.head_hidden_dim = int(head_hidden_dim)
        self.architecture_name = "Depth Vector"
        if self.freeze_encoder:
            for parameter in self.seg_model.parameters():
                parameter.requires_grad = False
        self.depth_pool = DepthVectorPooling(feature_depth)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        hidden_dim = int(self.head_hidden_dim)
        if hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(feature_channels), hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, 2),
            )
        else:
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(feature_channels), 2),
            )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder_eval:
            self.seg_model.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                feature = encoder_feature(self.seg_model, image)
        else:
            feature = encoder_feature(self.seg_model, image)
        feature = self.depth_pool(feature)
        return self.head(self.spatial_pool(feature))


EncoderGAPClassifier = DepthVectorClassifier


def _last_encoder_stage(seg_model: nn.Module) -> tuple[str, nn.Module]:
    encoder = getattr(seg_model, "encoder", None)
    if encoder is None:
        raise AttributeError("Segmentation model has no '.encoder'; cannot unfreeze the last encoder stage.")
    for attr_name in ("stages", "encoder_stages", "blocks"):
        stages = getattr(encoder, attr_name, None)
        if isinstance(stages, (nn.ModuleList, nn.Sequential, list, tuple)) and len(stages) > 0:
            return f"encoder.{attr_name}.{len(stages) - 1}", stages[-1]
    children = list(encoder.named_children())
    if not children:
        raise AttributeError("Encoder has no child modules; cannot identify the last encoder stage.")
    name, module = children[-1]
    return f"encoder.{name}", module


def _last_encoder_block(seg_model: nn.Module) -> tuple[str, nn.Module]:
    stage_name, stage = _last_encoder_stage(seg_model)
    for attr_name in ("blocks", "layers"):
        blocks = getattr(stage, attr_name, None)
        if isinstance(blocks, (nn.ModuleList, nn.Sequential, list, tuple)) and len(blocks) > 0:
            return f"{stage_name}.{attr_name}.{len(blocks) - 1}", blocks[-1]
    if isinstance(stage, (nn.ModuleList, nn.Sequential, list, tuple)) and len(stage) > 0:
        return f"{stage_name}.{len(stage) - 1}", stage[-1]
    return stage_name, stage


def configure_encoder_trainability(seg_model: nn.Module, mode: str) -> dict[str, object]:
    for parameter in seg_model.parameters():
        parameter.requires_grad = False
    if mode == "frozen":
        return {"mode": mode, "trainable_encoder_params": 0, "unfrozen_module": ""}
    if mode == "full":
        for parameter in seg_model.parameters():
            parameter.requires_grad = True
        return {
            "mode": mode,
            "trainable_encoder_params": sum(parameter.numel() for parameter in seg_model.parameters() if parameter.requires_grad),
            "unfrozen_module": "seg_model",
        }
    if mode == "last_stage":
        module_name, module = _last_encoder_stage(seg_model)
        for parameter in module.parameters():
            parameter.requires_grad = True
        return {
            "mode": mode,
            "trainable_encoder_params": sum(parameter.numel() for parameter in seg_model.parameters() if parameter.requires_grad),
            "unfrozen_module": module_name,
        }
    if mode == "last_block":
        module_name, module = _last_encoder_block(seg_model)
        for parameter in module.parameters():
            parameter.requires_grad = True
        return {
            "mode": mode,
            "trainable_encoder_params": sum(parameter.numel() for parameter in seg_model.parameters() if parameter.requires_grad),
            "unfrozen_module": module_name,
        }
    raise ValueError(f"Unsupported encoder trainability mode: {mode}")


def _classifier_config(config):
    return getattr(config, "classifier_detection", getattr(config, "classifier", None))


def _cfg_get(namespace, key: str, default):
    if namespace is None:
        return default
    return getattr(namespace, key, default)


def apply_classifier_defaults(args: argparse.Namespace, config) -> None:
    classifier_config = _classifier_config(config)
    training_config = getattr(config, "training", None)
    defaults = {
        "splits_file": getattr(config.paths, "splits_file", "checkpoints/tbi_splits_5fold.json"),
        "output_dir": f"checkpoints/tbi_classifier/depth_vector/fold_{int(args.fold)}",
        "epochs": 50,
        "batch_size": 1,
        "num_workers": 4,
        "lr": 3e-4,
        "encoder_lr": 1e-5,
        "weight_decay": 1e-4,
        "dropout": 0.2,
        "head_hidden_dim": 0,
        "samples_per_epoch": 0,
        "positive_fraction": 0.7,
        "foreground_prob": 0.85,
        "threshold": 0.5,
        "validation_overlap": 0.5,
        "validation_batch_size": 4,
        "validation_aggregation": "topk_mean",
        "validation_top_k": 5,
        "scheduler": "cosine",
        "min_lr_ratio": 0.05,
        "warmup_epochs": 3,
        "seed": _cfg_get(training_config, "seed", 42),
    }
    for key, fallback in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, _cfg_get(classifier_config, key, fallback))
    if args.init_checkpoint is None:
        args.init_checkpoint = _cfg_get(classifier_config, "init_checkpoint", None)


def classifier_stage_settings(args: argparse.Namespace, config) -> dict[str, object]:
    classifier_config = _classifier_config(config)
    staged_config = _cfg_get(classifier_config, "staged_tuning", None)
    config_enabled = bool(_cfg_get(staged_config, "enabled", False))
    enabled = bool(args.staged_tuning or (config_enabled and not args.no_staged_tuning))
    settings = {
        "enabled": enabled,
        "head_only_epochs": int(args.head_only_epochs if args.head_only_epochs is not None else _cfg_get(staged_config, "head_only_epochs", 15)),
        "last_block_epochs": int(args.last_block_epochs if args.last_block_epochs is not None else _cfg_get(staged_config, "last_block_epochs", 20)),
        "last_stage_epochs": int(args.last_stage_epochs if args.last_stage_epochs is not None else _cfg_get(staged_config, "last_stage_epochs", 15)),
        "head_lr": float(args.head_lr if args.head_lr is not None else _cfg_get(staged_config, "head_lr", args.lr)),
        "last_block_lr": float(args.last_block_lr if args.last_block_lr is not None else _cfg_get(staged_config, "last_block_lr", args.encoder_lr)),
        "last_stage_lr": float(args.last_stage_lr if args.last_stage_lr is not None else _cfg_get(staged_config, "last_stage_lr", args.encoder_lr * 0.5)),
        "full_encoder_lr": float(args.full_encoder_lr if args.full_encoder_lr is not None else _cfg_get(staged_config, "full_encoder_lr", args.encoder_lr * 0.1)),
        "encoder_eval_during_partial": bool(_cfg_get(staged_config, "encoder_eval_during_partial", True)),
    }
    return settings


def classifier_stage_for_epoch(epoch_index: int, settings: dict[str, object]) -> str:
    if not settings["enabled"]:
        return "manual"
    epoch = int(epoch_index)
    head_end = int(settings["head_only_epochs"])
    block_end = head_end + int(settings["last_block_epochs"])
    stage_end = block_end + int(settings["last_stage_epochs"])
    if epoch < head_end:
        return "head_only"
    if epoch < block_end:
        return "last_block"
    if epoch < stage_end:
        return "last_stage"
    return "full_encoder"


def apply_classifier_stage(model: DepthVectorClassifier, stage: str, settings: dict[str, object]) -> dict[str, object]:
    for parameter in model.depth_pool.parameters():
        parameter.requires_grad = True
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    if stage == "manual":
        trainable_encoder_params = sum(parameter.numel() for parameter in model.seg_model.parameters() if parameter.requires_grad)
        return {"stage": stage, "trainable_encoder_params": trainable_encoder_params, "unfrozen_module": "manual"}

    for parameter in model.seg_model.parameters():
        parameter.requires_grad = False

    unfrozen_module = ""
    if stage == "last_block":
        unfrozen_module, module = _last_encoder_block(model.seg_model)
        for parameter in module.parameters():
            parameter.requires_grad = True
    elif stage == "last_stage":
        unfrozen_module, module = _last_encoder_stage(model.seg_model)
        for parameter in module.parameters():
            parameter.requires_grad = True
    elif stage == "full_encoder":
        unfrozen_module = "seg_model"
        for parameter in model.seg_model.parameters():
            parameter.requires_grad = True
    elif stage != "head_only":
        raise ValueError(f"Unsupported classifier stage: {stage}")

    model.freeze_encoder = stage == "head_only"
    model.encoder_eval = stage == "head_only" or (
        bool(settings["encoder_eval_during_partial"]) and stage in {"last_block", "last_stage"}
    )
    trainable_encoder_params = sum(parameter.numel() for parameter in model.seg_model.parameters() if parameter.requires_grad)
    return {"stage": stage, "trainable_encoder_params": trainable_encoder_params, "unfrozen_module": unfrozen_module}


def _unique_params(parameters):
    seen = set()
    result = []
    for parameter in parameters:
        if id(parameter) not in seen:
            seen.add(id(parameter))
            result.append(parameter)
    return result


def _count_params(parameters) -> int:
    return sum(parameter.numel() for parameter in parameters)


def _count_trainable_params(parameters) -> int:
    return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)


def print_trainable_parameter_summary(model: DepthVectorClassifier, optimizer: torch.optim.Optimizer, label: str) -> None:
    total_params = _count_params(model.parameters())
    trainable_params = _count_trainable_params(model.parameters())
    encoder_total = _count_params(model.seg_model.parameters())
    encoder_trainable = _count_trainable_params(model.seg_model.parameters())
    head_total = _count_params(model.head.parameters()) + _count_params(model.depth_pool.parameters())
    head_trainable = _count_trainable_params(model.head.parameters()) + _count_trainable_params(model.depth_pool.parameters())

    print(
        f"[PARAMS] {label}: "
        f"trainable={trainable_params:,}/{total_params:,} "
        f"({100.0 * trainable_params / max(total_params, 1):.2f}%), "
        f"encoder={encoder_trainable:,}/{encoder_total:,}, "
        f"head={head_trainable:,}/{head_total:,}"
    )
    for group in optimizer.param_groups:
        group_name = str(group.get("group_name", "unnamed"))
        group_total = _count_params(group["params"])
        group_trainable = _count_trainable_params(group["params"])
        print(
            f"[PARAMS]   group={group_name:<12} "
            f"trainable={group_trainable:,}/{group_total:,} "
            f"lr={float(group['lr']):.8f}"
        )


def build_optimizer_and_scheduler(
    model: EncoderGAPClassifier,
    args: argparse.Namespace,
    stage_settings: dict[str, object] | None = None,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    head_params = _unique_params(list(model.depth_pool.parameters()) + list(model.head.parameters()))
    staged_enabled = stage_settings is not None and bool(stage_settings["enabled"])
    head_lr = float(stage_settings["head_lr"]) if staged_enabled else float(args.lr)
    param_groups = [{"params": head_params, "lr": head_lr, "group_name": "head"}]

    if staged_enabled:
        _, last_stage = _last_encoder_stage(model.seg_model)
        _, last_block = _last_encoder_block(model.seg_model)
        last_block_ids = {id(parameter) for parameter in last_block.parameters()}
        last_stage_ids = {id(parameter) for parameter in last_stage.parameters()}
        last_block_params = _unique_params(last_block.parameters())
        last_stage_params = _unique_params(parameter for parameter in last_stage.parameters() if id(parameter) not in last_block_ids)
        other_encoder_params = _unique_params(parameter for parameter in model.seg_model.parameters() if id(parameter) not in last_stage_ids)
        if last_block_params:
            param_groups.append({"params": last_block_params, "lr": float(stage_settings["last_block_lr"]), "group_name": "last_block"})
        if last_stage_params:
            param_groups.append({"params": last_stage_params, "lr": float(stage_settings["last_stage_lr"]), "group_name": "last_stage"})
        if other_encoder_params:
            param_groups.append({"params": other_encoder_params, "lr": float(stage_settings["full_encoder_lr"]), "group_name": "full_encoder"})
    else:
        encoder_params = [
            parameter
            for parameter in model.seg_model.parameters()
            if parameter.requires_grad
        ]
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": float(args.encoder_lr), "group_name": "encoder"})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(args.weight_decay))

    if args.scheduler == "none":
        return optimizer, None

    max_epochs = max(1, int(args.epochs))
    warmup_epochs = max(0, int(args.warmup_epochs))
    min_ratio = max(0.0, min(float(args.min_lr_ratio), 1.0))

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        if args.scheduler == "cosine":
            progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
            return min_ratio + (1.0 - min_ratio) * cosine
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        poly = (1.0 - min(max(progress, 0.0), 1.0)) ** 0.9
        return max(min_ratio, poly)

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


class PatchPresenceDataset(Dataset):
    def __init__(
        self,
        records: list[CaseRecord],
        config,
        base_dir: Path,
        patch_size: tuple[int, int, int],
        training: bool,
        samples_per_epoch: int,
        positive_fraction: float,
        foreground_prob: float,
    ) -> None:
        self.records = records
        self.config = config
        self.base_dir = base_dir
        self.patch_size = tuple(int(value) for value in patch_size)
        self.training = bool(training)
        self.samples_per_epoch = int(samples_per_epoch)
        self.positive_fraction = float(positive_fraction)
        self.foreground_prob = float(foreground_prob)
        self.positive_records = [record for record in records if self._case_has_lesion(record)]
        self.negative_records = [record for record in records if record not in set(self.positive_records)]

    def _case_has_lesion(self, record: CaseRecord) -> bool:
        image, mask, _, _ = load_case_cached(
            case=record,
            target_spacing=self.config.data.target_spacing,
            include_dmri=self.config.data.include_dmri,
            dmri_reduce=self.config.data.dmri_reduce,
            dmri_b0_threshold=self.config.data.dmri_b0_threshold,
            normalize_foreground_only=self.config.data.normalize_foreground_only,
            cache_dir=resolve_path(self.base_dir, self.config.paths.cache_dir) if self.config.data.cache_preprocessed else None,
        )
        del image
        return bool(np.any(mask > 0))

    def __len__(self) -> int:
        return self.samples_per_epoch if self.training and self.samples_per_epoch > 0 else len(self.records)

    def _choose_record(self, index: int) -> CaseRecord:
        if not self.training:
            return self.records[index]
        if self.positive_records and self.negative_records:
            source = self.positive_records if random.random() < self.positive_fraction else self.negative_records
            return random.choice(source)
        return self.records[index % len(self.records)]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self._choose_record(index)
        image, mask, _, _ = load_case_cached(
            case=record,
            target_spacing=self.config.data.target_spacing,
            include_dmri=self.config.data.include_dmri,
            dmri_reduce=self.config.data.dmri_reduce,
            dmri_b0_threshold=self.config.data.dmri_b0_threshold,
            normalize_foreground_only=self.config.data.normalize_foreground_only,
            cache_dir=resolve_path(self.base_dir, self.config.paths.cache_dir) if self.config.data.cache_preprocessed else None,
        )
        if self.training:
            padded_image, pads = _pad_to_shape(image, self.patch_size, fill_value=0.0)
            padded_mask, _ = _pad_to_shape(mask, self.patch_size, fill_value=0)
            center = _random_center(padded_mask, self.foreground_prob)
            image = _crop_with_center(padded_image, center, self.patch_size)
            mask = _crop_with_center(padded_mask, center, self.patch_size)
            image, mask = _random_flip(image, mask)
            if random.random() < 0.2:
                image = _random_intensity(image)
        else:
            image, mask = center_crop_or_pad(image, mask, self.patch_size)
        label = int(np.any(mask > 0))
        return {
            "image": torch.from_numpy(image.copy()).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "case_id": record.case_id,
        }


def _confusion_counts(y_true: list[int], y_pred: list[int]) -> dict[str, int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _classification_metrics(y_true: list[int], y_prob: list[float], threshold: float) -> dict[str, float]:
    y_pred = [int(prob >= threshold) for prob in y_prob]
    counts = _confusion_counts(y_true, y_pred)
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
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
        "mean_prob_positive_cases": float(np.mean(pos_probs)) if pos_probs else float("nan"),
        "mean_prob_negative_cases": float(np.mean(neg_probs)) if neg_probs else float("nan"),
    }


@torch.no_grad()
def evaluate_patch_loader(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool, threshold: float) -> dict[str, float]:
    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for batch in tqdm(loader, desc="Patch validation", leave=False):
        images = batch["image"].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last_3d)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        probs = torch.softmax(logits.float(), dim=1)[:, 1]
        total_loss += float(loss.detach().cpu())
        y_true.extend(int(value) for value in labels.detach().cpu().tolist())
        y_prob.extend(float(value) for value in probs.detach().cpu().tolist())
    metrics = _classification_metrics(y_true, y_prob, threshold)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def _aggregate_probabilities(values: list[float], mode: str, top_k: int) -> float:
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
def _predict_sliding_patch_batch(
    model: nn.Module,
    patches: list[np.ndarray],
    device: torch.device,
    use_amp: bool,
) -> list[float]:
    tensor = torch.from_numpy(np.stack(patches)).to(device=device, dtype=torch.float32).contiguous(memory_format=torch.channels_last_3d)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        logits = model(tensor)
    return [float(value) for value in torch.softmax(logits.float(), dim=1)[:, 1].detach().cpu().tolist()]


@torch.no_grad()
def predict_case_probability_sliding(
    model: nn.Module,
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

    probabilities: list[float] = []
    patches: list[np.ndarray] = []
    for z in z_positions:
        for y in y_positions:
            for x in x_positions:
                patches.append(padded[:, z : z + patch_size[0], y : y + patch_size[1], x : x + patch_size[2]])
                if len(patches) >= int(batch_size):
                    probabilities.extend(_predict_sliding_patch_batch(model, patches, device, use_amp))
                    patches = []
    if patches:
        probabilities.extend(_predict_sliding_patch_batch(model, patches, device, use_amp))
    return _aggregate_probabilities(probabilities, aggregation, top_k), len(probabilities)


def evaluate_sliding_window_records(
    model: nn.Module,
    records: list[CaseRecord],
    config,
    base_dir: Path,
    patch_size: tuple[int, int, int],
    overlap: float,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    threshold: float,
    aggregation: str,
    top_k: int,
) -> dict[str, float]:
    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []
    losses: list[float] = []
    patch_counts: list[int] = []
    cache_dir = resolve_path(base_dir, config.paths.cache_dir) if config.data.cache_preprocessed else None
    for record in tqdm(records, desc="Sliding validation", leave=False):
        image, mask, _, _ = load_case_cached(
            case=record,
            target_spacing=config.data.target_spacing,
            include_dmri=config.data.include_dmri,
            dmri_reduce=config.data.dmri_reduce,
            dmri_b0_threshold=config.data.dmri_b0_threshold,
            normalize_foreground_only=config.data.normalize_foreground_only,
            cache_dir=cache_dir,
        )
        label = int(np.any(mask > 0))
        probability, n_patches = predict_case_probability_sliding(
            model=model,
            image=image,
            patch_size=patch_size,
            overlap=overlap,
            batch_size=batch_size,
            device=device,
            use_amp=use_amp,
            aggregation=aggregation,
            top_k=top_k,
        )
        clipped = min(max(float(probability), 1e-6), 1.0 - 1e-6)
        losses.append(float(-(label * math.log(clipped) + (1 - label) * math.log(1.0 - clipped))))
        y_true.append(label)
        y_prob.append(float(probability))
        patch_counts.append(int(n_patches))
    metrics = _classification_metrics(y_true, y_prob, threshold)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["mean_n_patches"] = float(np.mean(patch_counts)) if patch_counts else 0.0
    metrics["total_n_patches"] = float(np.sum(patch_counts)) if patch_counts else 0.0
    metrics["validation_mode"] = "sliding_window"
    metrics["validation_overlap"] = float(overlap)
    metrics["validation_aggregation"] = str(aggregation)
    metrics["validation_top_k"] = float(top_k)
    return metrics


def _write_metrics(output_dir: Path, name: str, metrics: dict[str, float]) -> None:
    with (output_dir / f"{name}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (output_dir / f"{name}_confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "pred_negative", "pred_positive"])
        writer.writerow(["true_negative", int(metrics["tn"]), int(metrics["fp"])])
        writer.writerow(["true_positive", int(metrics["fn"]), int(metrics["tp"])])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_classifier_defaults(args, config)
    set_seed(int(args.seed))
    configure_torch_for_speed()
    base_dir = Path(args.config).expanduser().resolve().parent
    patch_size = tuple(args.patch_size or config.data.patch_size)
    output_dir = Path(str(args.output_dir).format(fold=int(args.fold))).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = discover_cases(resolve_path(base_dir, config.paths.dataset_dir))
    splits = load_splits(resolve_path(base_dir, args.splits_file))
    train_records, val_records = split_records(records, splits[int(args.fold)])
    print(f"Fold {args.fold}: train={len(train_records)} val={len(val_records)} patch_size={patch_size}")
    print(
        "Validation mode: sliding_window "
        f"overlap={float(args.validation_overlap)} "
        f"aggregation={args.validation_aggregation} "
        f"top_k={int(args.validation_top_k)} "
        f"batch_size={int(args.validation_batch_size)}"
    )

    train_dataset = PatchPresenceDataset(
        train_records,
        config,
        base_dir,
        patch_size,
        training=True,
        samples_per_epoch=args.samples_per_epoch,
        positive_fraction=args.positive_fraction,
        foreground_prob=args.foreground_prob,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=int(args.num_workers) > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_model = _build_model_architecture_only(config, base_dir).to(device).eval()
    init_checkpoint = args.init_checkpoint or args.seg_checkpoint or config.paths.pretrained_checkpoint
    init_checkpoint_path = resolve_path(base_dir, init_checkpoint)
    init_load_stats = _load_segmentation_checkpoint(seg_model, init_checkpoint_path)
    if args.unfreeze_encoder:
        encoder_mode = "full"
    elif args.unfreeze_last_encoder_block:
        encoder_mode = "last_block"
    elif args.unfreeze_last_encoder_stage:
        encoder_mode = "last_stage"
    else:
        encoder_mode = "frozen"
    trainability = configure_encoder_trainability(seg_model, encoder_mode)
    print(
        "Encoder trainability: "
        f"mode={trainability['mode']} "
        f"module={trainability['unfrozen_module']} "
        f"trainable_params={trainability['trainable_encoder_params']}"
    )
    with torch.no_grad():
        dummy = torch.zeros((1, int(config.model.in_channels), *patch_size), device=device)
        dummy_feature = encoder_feature(seg_model, dummy)
        feature_channels = int(dummy_feature.shape[1])
        feature_depth = int(dummy_feature.shape[2])
    model = DepthVectorClassifier(
        seg_model=seg_model,
        feature_channels=feature_channels,
        feature_depth=feature_depth,
        dropout=float(args.dropout),
        head_hidden_dim=int(args.head_hidden_dim),
        freeze_encoder=encoder_mode == "frozen",
        encoder_eval=encoder_mode in {"frozen", "last_stage", "last_block"},
    ).to(device)
    print(f"Architecture: {model.architecture_name}")
    stage_settings = classifier_stage_settings(args, config)
    if stage_settings["enabled"]:
        first_stage = classifier_stage_for_epoch(0, stage_settings)
        trainability = apply_classifier_stage(model, first_stage, stage_settings)
        print(
            "Classifier staged tuning: "
            f"head_only={stage_settings['head_only_epochs']}ep, "
            f"last_block={stage_settings['last_block_epochs']}ep, "
            f"last_stage={stage_settings['last_stage_epochs']}ep, "
            "then full_encoder"
        )

    train_labels = [int(record in train_dataset.positive_records) for record in train_records]
    n_pos = max(sum(train_labels), 1)
    n_neg = max(len(train_labels) - sum(train_labels), 1)
    class_weights = torch.tensor([len(train_labels) / (2 * n_neg), len(train_labels) / (2 * n_pos)], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=None if args.no_class_weights else class_weights)
    optimizer, scheduler = build_optimizer_and_scheduler(model, args, stage_settings)
    print_trainable_parameter_summary(model, optimizer, label="initial")
    use_amp = bool(not args.no_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history_path = output_dir / "history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "stage",
                "trainable_encoder_params",
                "train_loss",
                "val_loss",
                "val_balanced_accuracy",
                "val_tpr",
                "val_tnr",
                "val_tp",
                "val_fp",
                "val_fn",
                "val_tn",
                "val_mean_n_patches",
                "val_total_n_patches",
                "lr_head",
                "lr_encoder",
                "lr_last_block",
                "lr_last_stage",
                "lr_full_encoder",
            ]
        )

    best_metric = -math.inf
    previous_stage = None
    for epoch in range(int(args.epochs)):
        stage = classifier_stage_for_epoch(epoch, stage_settings)
        trainability = apply_classifier_stage(model, stage, stage_settings)
        if stage != previous_stage:
            print_trainable_parameter_summary(model, optimizer, label=f"epoch={epoch + 1} stage={stage}")
            previous_stage = stage
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False):
            images = batch["image"].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last_3d)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())

        train_loss = running_loss / max(len(train_loader), 1)
        val_metrics = evaluate_sliding_window_records(
            model=model,
            records=val_records,
            config=config,
            base_dir=base_dir,
            patch_size=patch_size,
            overlap=float(args.validation_overlap),
            batch_size=int(args.validation_batch_size),
            device=device,
            use_amp=use_amp,
            threshold=float(args.threshold),
            aggregation=str(args.validation_aggregation),
            top_k=int(args.validation_top_k),
        )
        monitor = float(val_metrics["balanced_accuracy"])
        lr_by_group = {str(group.get("group_name", index)): float(group["lr"]) for index, group in enumerate(optimizer.param_groups)}
        with history_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    epoch + 1,
                    trainability["stage"],
                    trainability["trainable_encoder_params"],
                    train_loss,
                    val_metrics["loss"],
                    val_metrics["balanced_accuracy"],
                    val_metrics["tpr"],
                    val_metrics["tnr"],
                    val_metrics["tp"],
                    val_metrics["fp"],
                    val_metrics["fn"],
                    val_metrics["tn"],
                    val_metrics["mean_n_patches"],
                    val_metrics["total_n_patches"],
                    lr_by_group.get("head", 0.0),
                    lr_by_group.get("encoder", 0.0),
                    lr_by_group.get("last_block", 0.0),
                    lr_by_group.get("last_stage", 0.0),
                    lr_by_group.get("full_encoder", 0.0),
                ]
            )
        payload = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "architecture": "Depth Vector",
            "architecture_name": model.architecture_name,
            "init_checkpoint": str(init_checkpoint),
            "seg_checkpoint": str(init_checkpoint),
            "patch_size": list(patch_size),
            "feature_channels": feature_channels,
            "feature_depth": feature_depth,
            "threshold": float(args.threshold),
            "args": vars(args),
            "encoder_trainability": trainability,
            "init_load_stats": init_load_stats,
            "classifier_stage_settings": stage_settings,
            "scheduler": str(args.scheduler),
            "optimizer_state": optimizer.state_dict(),
            "val_metrics": val_metrics,
        }
        torch.save(payload, output_dir / "last.pt")
        if monitor > best_metric:
            best_metric = monitor
            torch.save(payload, output_dir / "best.pt")
            _write_metrics(output_dir, "best_patch_val", val_metrics)
        print(
            f"epoch={epoch + 1:03d} stage={trainability['stage']} train_loss={train_loss:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"TPR={val_metrics['tpr']:.4f} TNR={val_metrics['tnr']:.4f} "
            f"lr(head/block/stage/full)={lr_by_group.get('head', 0.0):.8f}/"
            f"{lr_by_group.get('last_block', lr_by_group.get('encoder', 0.0)):.8f}/"
            f"{lr_by_group.get('last_stage', 0.0):.8f}/"
            f"{lr_by_group.get('full_encoder', 0.0):.8f} "
            f"cm=[[tn={int(val_metrics['tn'])}, fp={int(val_metrics['fp'])}], "
            f"[fn={int(val_metrics['fn'])}, tp={int(val_metrics['tp'])}]]"
        )
        if scheduler is not None:
            scheduler.step()

    final_metrics = evaluate_sliding_window_records(
        model=model,
        records=val_records,
        config=config,
        base_dir=base_dir,
        patch_size=patch_size,
        overlap=float(args.validation_overlap),
        batch_size=int(args.validation_batch_size),
        device=device,
        use_amp=use_amp,
        threshold=float(args.threshold),
        aggregation=str(args.validation_aggregation),
        top_k=int(args.validation_top_k),
    )
    _write_metrics(output_dir, "final_patch_val", final_metrics)
    print(f"Saved classifier to {output_dir}")


if __name__ == "__main__":
    main()


'''
python train_classifier.py \
  --config config.yml \
  --splits-file checkpoints/train_val_test_10fold_seed42.json \
  --fold 0 \
  --seg-checkpoint checkpoints/trained_models/best_ddp_fft_finetuned_kpcyjb66_data_leaked_0.54_rank1_leaderboard.pt \
  --output-dir checkpoints/tbi_classifier/fold_0_last_encoder \
  --epochs 50 \
  --batch-size 2 \
  --lr 0.0001 \
  --encoder-lr 0.000005 \
  --scheduler poly \
  --patch-size 160 160 160 \
  --samples-per-epoch 1000 \
  --positive-fraction 0.5 \
  --unfreeze-last-encoder-stage

  ===========

  python train_classifier.py \
  --config config.yml \
  --splits-file checkpoints/train_val_test_10fold_seed42.json \
  --fold 0 \
  --seg-checkpoint checkpoints/trained_models/best_ddp_fft_finetuned_kpcyjb66_data_leaked_0.54_rank1_leaderboard.pt \
  --output-dir checkpoints/tbi_classifier/fold_0_last_block_balanced \
  --epochs 50 \
  --batch-size 2 \
  --lr 0.001 \
  --encoder-lr 0.00001 \
  --scheduler poly \
  --patch-size 160 160 160 \
  --samples-per-epoch 1200 \
  --positive-fraction 0.5 \
  --foreground-prob 0.85 \
  --threshold 0.5 \
  --unfreeze-last-encoder-block \
  --no-class-weights \
  --head-hidden-dim 128
'''