from __future__ import annotations

import inspect
import importlib
import json
from pathlib import Path
from typing import Any

import torch


def _resolve_object(spec: Any) -> Any:
    if spec is None:
        return None
    if isinstance(spec, str):
        module_name, attr_name = spec.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    return spec


def load_plans(plans_path: str | Path) -> dict[str, Any]:
    with Path(plans_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_backbone(
    plans_path: str | Path,
    in_channels: int,
    out_channels: int,
    deep_supervision: bool = False,
) -> torch.nn.Module:
    plans = load_plans(plans_path)
    config = plans["configurations"]["3d_fullres"]
    architecture = config["architecture"]
    cls = _resolve_object(architecture["network_class_name"])
    if cls is None:
        raise RuntimeError(f"Could not import network class: {architecture['network_class_name']}")

    arch_kwargs = dict(architecture["arch_kwargs"])
    for key in ("conv_op", "norm_op", "dropout_op", "nonlin"):
        arch_kwargs[key] = _resolve_object(arch_kwargs.get(key))
    aliases = {
        "input_channels": in_channels,
        "in_channels": in_channels,
        "num_classes": out_channels,
        "out_channels": out_channels,
        "deep_supervision": deep_supervision,
    }
    signature = inspect.signature(cls)
    kwargs: dict[str, Any] = {}
    for parameter_name in signature.parameters:
        if parameter_name in arch_kwargs:
            kwargs[parameter_name] = arch_kwargs[parameter_name]
        elif parameter_name in aliases:
            kwargs[parameter_name] = aliases[parameter_name]
    try:
        return cls(**kwargs)
    except TypeError as error:
        fallback_kwargs = dict(arch_kwargs)
        fallback_kwargs.update({"input_channels": in_channels, "num_classes": out_channels, "deep_supervision": deep_supervision})
        fallback_kwargs.update({"in_channels": in_channels, "out_channels": out_channels})
        try:
            return cls(**fallback_kwargs)
        except TypeError as fallback_error:
            raise RuntimeError(f"Unable to instantiate {cls.__name__}") from fallback_error


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("network_weights", "state_dict", "model", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(key, str) for key in checkpoint.keys()):
            return checkpoint  # type: ignore[return-value]
    raise RuntimeError("Unsupported checkpoint format")


def _strip_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    stripped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        stripped[new_key] = value
    return stripped


def _adapt_conv_weight(source: torch.Tensor, target_shape: torch.Size) -> torch.Tensor | None:
    if source.ndim != 5 or len(target_shape) != 5:
        return None
    if source.shape[0] != target_shape[0] or source.shape[2:] != target_shape[2:]:
        return None
    source_channels = source.shape[1]
    target_channels = target_shape[1]
    if source_channels == target_channels:
        return source
    if source_channels == 1 and target_channels > 1:
        return source.repeat(1, target_channels, 1, 1, 1) / float(target_channels)
    if target_channels == 1:
        return source.mean(dim=1, keepdim=True)
    repeats = (target_channels + source_channels - 1) // source_channels
    return source.repeat(1, repeats, 1, 1, 1)[:, :target_channels] / float(repeats)


def _load_checkpoint(checkpoint_path: str | Path) -> Any:
    try:
        return torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)


def load_pretrained_weights(model: torch.nn.Module, checkpoint_path: str | Path, strict: bool = False) -> list[str]:
    checkpoint = _load_checkpoint(checkpoint_path)
    state_dict = _strip_prefix(_extract_state_dict(checkpoint))
    current_state = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for key, value in state_dict.items():
        if key not in current_state:
            continue
        if current_state[key].shape == value.shape:
            filtered[key] = value
            continue
        adapted = _adapt_conv_weight(value, current_state[key].shape)
        if adapted is not None:
            filtered[key] = adapted
            continue
        skipped.append(key)

    model.load_state_dict(filtered, strict=strict)
    return skipped
