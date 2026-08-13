"""Log standard nnU-Net split/planning provenance to a separate W&B run.

This is intentionally separate from nnU-Net's native training W&B run. It
uploads only a redacted split audit (case IDs, never source image paths),
configuration hashes, and optional plans metadata. It does not train, select a
checkpoint, evaluate any Phase-2 case, or alter the active training process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("checkpoints/paper_baselines/standard_nnunet_model_b_matched"),
        help="Baseline run root created by prepare_standard_nnunet_baseline.py.",
    )
    parser.add_argument("--plans", type=Path, default=None, help="Optional completed nnUNetPlans.json.")
    parser.add_argument("--project", default="AIMS-TBI-standard-nnUNet")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--mode", default="online", choices=("online", "offline", "disabled"))
    parser.add_argument("--name", default="standard_nnunet_fold0_provenance")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload


def selected_fold_audit(splits: list[dict[str, list[str]]], fold: int) -> dict[str, Any]:
    if fold < 0 or fold >= len(splits):
        raise ValueError(f"Selected fold {fold} is outside the {len(splits)} recorded folds.")
    selected = splits[fold]
    train_ids = [str(case_id) for case_id in selected["train"]]
    val_ids = [str(case_id) for case_id in selected["val"]]
    if set(train_ids) & set(val_ids):
        raise ValueError("Recorded nnU-Net fold has train/validation overlap.")
    return {
        "fold": fold,
        "n_train": len(train_ids),
        "n_validation": len(val_ids),
        "train_case_ids": train_ids,
        "validation_case_ids": val_ids,
        "all_fold_sizes": [
            {"fold": index, "n_train": len(item["train"]), "n_validation": len(item["val"])}
            for index, item in enumerate(splits)
        ],
    }


def plan_summary(plans_path: Path | None) -> dict[str, Any] | None:
    if plans_path is None:
        return None
    if not plans_path.is_file():
        raise FileNotFoundError(plans_path)
    plans = load_json(plans_path)
    if not isinstance(plans, dict):
        raise ValueError(f"Expected plans mapping in {plans_path}.")
    configuration = plans.get("configurations", {}).get("3d_fullres", {})
    architecture = configuration.get("architecture", {})
    return {
        "plans_path": str(plans_path.resolve()),
        "plans_sha256": sha256(plans_path),
        "plans_name": plans.get("plans_name"),
        "planner": plans.get("experiment_planner_used"),
        "configuration": "3d_fullres",
        "spacing": configuration.get("spacing"),
        "patch_size": configuration.get("patch_size"),
        "batch_size": configuration.get("batch_size"),
        "network_class": architecture.get("network_class_name"),
    }


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    manifest_path = run_root / "baseline_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing baseline manifest: {manifest_path}")
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected baseline manifest mapping in {manifest_path}.")
    dataset_dir = Path(str(manifest["baseline_dataset_dir"])).resolve()
    splits_path = dataset_dir / "splits_final.json"
    if not splits_path.is_file():
        raise FileNotFoundError(f"Missing generated split file: {splits_path}")
    splits = load_json(splits_path)
    if not isinstance(splits, list):
        # nnU-Net's file is a top-level array, but accept a mapping only if a
        # future tool writes an explicit `splits` list.
        raise ValueError(f"Expected a top-level split list in {splits_path}.")
    selected_fold = int(manifest["selected_fold"])
    split_audit = selected_fold_audit(splits, selected_fold)
    plan_audit = plan_summary(args.plans)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Training-data split and planning provenance; no Phase-2 evaluation.",
        "protocol": manifest["protocol"],
        "baseline_manifest_sha256": sha256(manifest_path),
        "generated_nnunet_split_sha256": sha256(splits_path),
        "source_split_sha256": manifest["source_split_sha256"],
        "cohort": {
            "n_full_cohort": manifest["n_full_cohort"],
            "n_development": manifest["n_development"],
            "n_excluded": manifest["n_excluded"],
            "released_validation_cases_accessed": manifest["external_validation_input_cases"],
        },
        "fold": split_audit,
        "planning": plan_audit,
    }
    audit_path = run_root / "wandb_provenance_payload.json"
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("wandb is required to upload provenance. Install it in the nnU-Net environment.") from error
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name,
        mode=args.mode,
        job_type="provenance",
        tags=["standard-nnunet", "fold-0", "split-audit", "no-phase2-training"],
        config={
            "fold": selected_fold,
            "n_train": split_audit["n_train"],
            "n_validation": split_audit["n_validation"],
            "n_excluded": manifest["n_excluded"],
            "source_split_sha256": manifest["source_split_sha256"],
            "generated_nnunet_split_sha256": payload["generated_nnunet_split_sha256"],
            "released_validation_cases_accessed": 0,
        },
        reinit=True,
    )
    if plan_audit is not None:
        run.summary.update(
            {
                "plan/spacing": plan_audit["spacing"],
                "plan/patch_size": plan_audit["patch_size"],
                "plan/batch_size": plan_audit["batch_size"],
                "plan/network_class": plan_audit["network_class"],
            }
        )
    artifact = wandb.Artifact("standard-nnunet-fold0-provenance", type="split-audit")
    artifact.add_file(str(audit_path), name="provenance_payload.json")
    artifact.add_file(str(splits_path), name="splits_final.json")
    artifact.add_file(str(run_root / "baseline_config_snapshot.yaml"), name="baseline_config_snapshot.yaml")
    artifact.add_file(str(run_root / "model_b_reference_snapshot.yaml"), name="model_b_reference_snapshot.yaml")
    run.log_artifact(artifact)
    run.finish()
    print(f"[DONE] Logged redacted split/planning provenance to W&B: {audit_path}")


if __name__ == "__main__":
    main()
