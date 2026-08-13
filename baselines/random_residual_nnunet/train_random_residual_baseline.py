"""Launch the matched scratch residual control without modifying project code.

This process-local launcher temporarily supplies a Kaiming-initialized model to
the existing Model-B training loop. The monkey patch exists only in this Python
process; no original project or nnU-Net package source is edited. It refuses
pretrained, init, and resume checkpoint paths and records the exact initial
state hash before the first optimizer update.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multitalent_tbi import engine
from multitalent_tbi.config import load_config, resolve_path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=SCRIPT_DIR / "model_b_matched_random_init.yaml",
        help="Prepared matched random-residual YAML configuration.",
    )
    parser.add_argument("--fold", type=int, default=0, choices=(0,))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def require_scratch_config(config, config_path: Path, fold: int) -> Path:
    if fold != 0 or int(config.training.fold) != 0:
        raise ValueError("This pre-registered control may only train fold 0.")
    if bool(config.model.load_pretrained):
        raise ValueError("Random-residual control requires model.load_pretrained=false.")
    if str(getattr(config.model, "initialization", "")) != "kaiming_normal_leaky_relu_0.01":
        raise ValueError("Unexpected model.initialization; use the pre-registered Kaiming-normal initializer.")
    if getattr(config.training, "init_checkpoint", None) not in (None, ""):
        raise ValueError("init_checkpoint is prohibited for the random-residual control.")
    if getattr(config.training, "resume_checkpoint", None) not in (None, ""):
        raise ValueError("resume_checkpoint is prohibited for the first random-residual control run.")

    work_dir = resolve_path(config_path.parent, config.paths.work_dir)
    manifest_path = work_dir / "preflight_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run prepare_random_residual_baseline.py before launching training."
        )
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    current_config_sha = sha256(config_path)
    if manifest.get("config_sha256") != current_config_sha:
        raise RuntimeError("Config changed after preflight. Re-run prepare_random_residual_baseline.py before training.")
    initialization = manifest.get("initialization", {})
    if initialization.get("load_pretrained") is not False or initialization.get("pretrained_checkpoint") is not None:
        raise RuntimeError("Preflight manifest does not prove a scratch initialization.")
    output_dir = work_dir / "fold_0"
    if any((output_dir / name).exists() for name in ("history.csv", "last.pt", "best.pt")):
        raise FileExistsError(
            f"Output directory already contains a training run: {output_dir}. "
            "Do not resume or overwrite this pre-registered control."
        )
    return output_dir


def initialize_kaiming_normal(model: torch.nn.Module) -> int:
    convolution_types = (torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)
    count = 0
    for module in model.modules():
        if isinstance(module, convolution_types):
            torch.nn.init.kaiming_normal_(module.weight, a=0.01, mode="fan_out", nonlinearity="leaky_relu")
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            count += 1
    return count


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = load_config(config_path)
    output_dir = require_scratch_config(config, config_path, args.fold)
    initial_audit: dict[str, Any] = {}
    original_build_model = engine.build_model

    def build_random_model(active_config, base_dir: Path):
        if bool(active_config.model.load_pretrained):
            raise ValueError("The patched random-residual builder refuses pretrained tensors.")
        model = original_build_model(active_config, base_dir)
        convolution_count = initialize_kaiming_normal(model)
        initial_audit.update(
            {
                "protocol": "matched_random_residual_model_b_control",
                "load_pretrained": False,
                "pretrained_checkpoint": None,
                "init_checkpoint": None,
                "resume_checkpoint": None,
                "initializer": "kaiming_normal_leaky_relu_0.01",
                "convolution_modules_reinitialized": convolution_count,
                "initial_state_sha256": state_sha256(model),
                "seed": int(active_config.training.seed) + args.fold,
                "fold": args.fold,
                "config_path": str(config_path),
                "config_sha256": sha256(config_path),
                "torch_version": torch.__version__,
                "dynamic_network_architectures_version": package_version("dynamic-network-architectures"),
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "initialization_audit.json").open("w", encoding="utf-8") as handle:
            json.dump(initial_audit, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(
            "[RANDOM-INIT AUDIT] "
            f"Kaiming-normal reset {convolution_count} convolution modules; "
            f"initial SHA-256={initial_audit['initial_state_sha256']}"
        )
        return model

    engine.build_model = build_random_model
    try:
        results = engine.train_from_config(str(config_path), fold=args.fold, all_folds=False)
    finally:
        engine.build_model = original_build_model
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
