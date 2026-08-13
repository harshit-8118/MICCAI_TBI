"""Record the exact planned standard nnU-Net and Model-B architecture sizes.

Run this after ``nnUNetv2_plan_and_preprocess``. It instantiates the network
described by each plan without loading a checkpoint, then records the actual
number of parameters and planning choices. It does not train, alter, or
overwrite any model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, required=True, help="Standard nnU-Net nnUNetPlans.json.")
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument(
        "--reference-plans",
        type=Path,
        default=None,
        help="Optional upstream MultiTalentV2 pretraining plan JSON used only to instantiate Model B's residual architecture.",
    )
    parser.add_argument("--reference-configuration", default="3d_fullres")
    parser.add_argument("--reference-input-channels", type=int, default=1)
    parser.add_argument("--reference-output-channels", type=int, default=2)
    parser.add_argument(
        "--reference-deep-supervision",
        choices=("true", "false"),
        default="false",
        help="Model B used false; use its historical setting when comparing parameter counts.",
    )
    parser.add_argument(
        "--model-b-reference",
        type=Path,
        default=SCRIPT_DIR / "model_b_reference.yaml",
        help="Immutable historical Model-B fine-tuning provenance YAML.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return payload


def installed_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def summarize_plan(
    plans_path: Path,
    configuration_name: str,
    input_channels: int,
    output_channels: int,
    deep_supervision: bool,
) -> dict[str, Any]:
    plans = PlansManager(str(plans_path))
    configuration = plans.get_configuration(configuration_name)
    architecture = configuration.configuration["architecture"]
    network = get_network_from_plans(
        configuration.network_arch_class_name,
        configuration.network_arch_init_kwargs,
        configuration.network_arch_init_kwargs_req_import,
        input_channels=input_channels,
        output_channels=output_channels,
        allow_init=False,
        deep_supervision=deep_supervision,
    )
    arch_kwargs = architecture["arch_kwargs"]
    parameters = list(network.parameters())
    return {
        "plans_path": str(plans_path.resolve()),
        "plans_sha256": sha256(plans_path),
        "plans_name": plans.plans_name,
        "experiment_planner": plans.experiment_planner_name,
        "configuration": configuration_name,
        "network_class": configuration.network_arch_class_name,
        "input_channels": input_channels,
        "output_channels": output_channels,
        "deep_supervision": deep_supervision,
        "total_parameters": sum(parameter.numel() for parameter in parameters),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
        "patch_size": configuration.patch_size,
        "batch_size": configuration.batch_size,
        "spacing": configuration.spacing,
        "median_image_size_in_voxels": configuration.median_image_size_in_voxels,
        "feature_widths": arch_kwargs.get("features_per_stage"),
        "number_of_stages": arch_kwargs.get("n_stages"),
        "encoder_blocks_per_stage": arch_kwargs.get("n_conv_per_stage", arch_kwargs.get("n_blocks_per_stage")),
        "decoder_blocks_per_stage": arch_kwargs.get("n_conv_per_stage_decoder"),
        "architecture_kwargs": arch_kwargs,
    }


def summarize_historical_model_b(reference_path: Path, architecture_source: dict[str, Any]) -> dict[str, Any]:
    """Combine historical fine-tuning settings with its inherited architecture.

    Model B inherited the residual architecture from the upstream MultiTalent
    plan, but did not inherit that plan's batch size or patch size. Those are
    properties of upstream pretraining and must not be reported as Model B
    fine-tuning settings.
    """
    reference = load_yaml(reference_path)
    data = reference["data"]
    training = reference["training"]
    return {
        "role": reference.get("role"),
        "reference_path": str(reference_path.resolve()),
        "architecture_inherited_from_pretraining_plan": {
            "plans_path": architecture_source["plans_path"],
            "plans_sha256": architecture_source["plans_sha256"],
            "network_class": architecture_source["network_class"],
            "feature_widths": architecture_source["feature_widths"],
            "number_of_stages": architecture_source["number_of_stages"],
            "encoder_blocks_per_stage": architecture_source["encoder_blocks_per_stage"],
            "decoder_blocks_per_stage": architecture_source["decoder_blocks_per_stage"],
            "total_parameters": architecture_source["total_parameters"],
            "trainable_parameters": architecture_source["trainable_parameters"],
        },
        "historical_fine_tuning": {
            "target_spacing_mm": data["target_spacing_mm"],
            "patch_size_voxels": data["patch_size_voxels"],
            "batch_size": training["batch_size"],
            "deep_supervision": False,
            "epochs": training["epochs"],
        },
        "note": (
            "The 192^3 patch and batch size 24 in the upstream MultiTalentV2 plan are pretraining-plan values, "
            "not Model B fine-tuning settings."
        ),
    }


def main() -> None:
    args = parse_args()
    if not args.plans.is_file():
        raise FileNotFoundError(args.plans)
    if not args.dataset_json.is_file():
        raise FileNotFoundError(args.dataset_json)
    dataset_json = load_json(args.dataset_json)
    channels = dataset_json.get("channel_names", dataset_json.get("modality"))
    labels = dataset_json.get("labels")
    if not isinstance(channels, dict) or not isinstance(labels, dict):
        raise ValueError("dataset.json needs mapping-valued channel_names and labels.")
    standard = summarize_plan(
        args.plans,
        args.configuration,
        input_channels=len(channels),
        output_channels=len(labels),
        deep_supervision=True,
    )
    payload: dict[str, Any] = {
        "purpose": "Measured architecture capacity only; not an isolated causal comparison of pretraining.",
        "nnunetv2_version": installed_version("nnunetv2"),
        "dynamic_network_architectures_version": installed_version("dynamic-network-architectures"),
        "standard_nnunet": standard,
    }
    if args.reference_plans is not None:
        if not args.reference_plans.is_file():
            raise FileNotFoundError(args.reference_plans)
        if not args.model_b_reference.is_file():
            raise FileNotFoundError(args.model_b_reference)
        pretraining_plan = summarize_plan(
            args.reference_plans,
            args.reference_configuration,
            input_channels=args.reference_input_channels,
            output_channels=args.reference_output_channels,
            deep_supervision=args.reference_deep_supervision == "true",
        )
        payload["multitalentv2_pretraining_plan"] = pretraining_plan
        payload["model_b_historical_fine_tuning"] = summarize_historical_model_b(
            args.model_b_reference,
            pretraining_plan,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote architecture report: {args.output}")


if __name__ == "__main__":
    main()
