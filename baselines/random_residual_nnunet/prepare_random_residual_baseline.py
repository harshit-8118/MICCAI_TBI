"""Audit the matched random-residual control before it can be trained.

The script creates provenance files only. It does not build a GPU model, train,
or access the released Phase-2 validation data. The actual trainer records a
second initialization audit in its fold output immediately after it creates
the randomly initialized model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multitalent_tbi.data import CaseRecord, discover_cases, load_nifti_robust
from multitalent_tbi.splits import split_records


SCRIPT_DIR = Path(__file__).resolve().parent
EXPECTED_FULL_COHORT = 552
EXPECTED_DEVELOPMENT_POOL = 522
EXPECTED_EXCLUDED = 30
EXPECTED_FOLDS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=SCRIPT_DIR / "model_b_matched_random_init.yaml",
        help="Matched random-residual YAML configuration.",
    )
    parser.add_argument(
        "--model-b-reference",
        type=Path,
        default=PROJECT_ROOT / "baselines" / "standard_nnunet" / "model_b_reference.yaml",
        help="Immutable historical Model-B provenance YAML.",
    )
    parser.add_argument(
        "--skip-nifti-preflight",
        action="store_true",
        help="Skip the full local NIfTI readability check (not recommended).",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(base_dir: Path, value: Any, label: str) -> Path:
    if value in (None, ""):
        raise ValueError(f"{label} must be set.")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} must be {expected!r}, found {actual!r}.")


def validate_random_control_config(config: dict[str, Any], base_dir: Path) -> tuple[Path, Path, Path]:
    paths = config.get("paths")
    data = config.get("data")
    model = config.get("model")
    training = config.get("training")
    if not all(isinstance(section, dict) for section in (paths, data, model, training)):
        raise ValueError("Configuration requires mapping-valued paths, data, model, and training sections.")
    if "external_validation" in config:
        raise ValueError("The training configuration must not contain an external_validation section.")

    require_equal(data.get("target_spacing"), [1.0, 1.0, 1.0], "data.target_spacing")
    require_equal(data.get("patch_size"), [160, 160, 160], "data.patch_size")
    require_equal(data.get("input_modalities"), ["T1"], "data.input_modalities")
    require_equal(data.get("include_dmri"), False, "data.include_dmri")
    require_equal(data.get("normalize_foreground_only"), True, "data.normalize_foreground_only")
    require_equal(data.get("oversample_foreground_prob"), 0.33, "data.oversample_foreground_prob")

    require_equal(model.get("in_channels"), 1, "model.in_channels")
    require_equal(model.get("out_channels"), 2, "model.out_channels")
    require_equal(model.get("deep_supervision"), False, "model.deep_supervision")
    require_equal(model.get("load_pretrained"), False, "model.load_pretrained")
    require_equal(model.get("initialization"), "kaiming_normal_leaky_relu_0.01", "model.initialization")
    if paths.get("pretrained_checkpoint") not in (None, ""):
        raise ValueError("paths.pretrained_checkpoint must be null for the scratch control.")

    expected_training = {
        "seed": 42,
        "fold": 0,
        "batch_size": 2,
        "max_epochs": 300,
        "base_lr": 0.0001,
        "weight_decay": 0.00002,
        "poly_power": 0.9,
        "min_lr": 0.00000001,
        "optimizer": "adamw",
        "class_weights": [1.0, 2.0],
        "grad_clip_norm": 8.0,
        "validation_interval_epochs": 1,
        "final_validation": False,
        "best_checkpoint_metric": "val_dice",
        "best_checkpoint_mode": "max",
        "full_warmup_epochs": 2,
    }
    for key, expected in expected_training.items():
        require_equal(training.get(key), expected, f"training.{key}")
    staged = training.get("staged_tuning")
    if not isinstance(staged, dict):
        raise ValueError("training.staged_tuning must be a mapping.")
    require_equal(staged.get("enabled"), False, "training.staged_tuning.enabled")
    require_equal(staged.get("head_only_epochs"), 0, "training.staged_tuning.head_only_epochs")
    require_equal(staged.get("partial_tune_epochs"), 0, "training.staged_tuning.partial_tune_epochs")
    sampling = training.get("case_sampling")
    if not isinstance(sampling, dict):
        raise ValueError("training.case_sampling must be a mapping.")
    require_equal(sampling.get("enabled"), False, "training.case_sampling.enabled")
    if training.get("init_checkpoint") not in (None, "") or training.get("resume_checkpoint") not in (None, ""):
        raise ValueError("The scratch control cannot specify an init_checkpoint or resume_checkpoint.")

    dataset_dir = resolve(base_dir, paths.get("dataset_dir"), "paths.dataset_dir")
    split_path = resolve(base_dir, paths.get("splits_file"), "paths.splits_file")
    plan_path = resolve(base_dir, paths.get("pretrained_plans"), "paths.pretrained_plans")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(dataset_dir)
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    return dataset_dir, split_path, plan_path


def validate_plan(plan_path: Path) -> dict[str, Any]:
    with plan_path.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    try:
        architecture = plan["configurations"]["3d_fullres"]["architecture"]
        network_class = architecture["network_class_name"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Could not find a 3d_fullres architecture in {plan_path}.") from error
    if "ResidualEncoderUNet" not in str(network_class):
        raise ValueError(f"Expected ResidualEncoderUNet architecture, found {network_class!r}.")
    return {
        "network_class": network_class,
        "architecture": architecture,
        "plans_name": plan.get("plans_name"),
    }


def validate_splits(records: list[CaseRecord], split_path: Path) -> dict[str, int]:
    with split_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    splits = payload.get("splits", payload.get("folds"))
    if not isinstance(splits, list) or len(splits) != EXPECTED_FOLDS:
        raise ValueError(f"Expected exactly {EXPECTED_FOLDS} folds in {split_path}.")
    excluded_ids = {str(case_id) for case_id in payload.get("test", [])}
    if len(excluded_ids) != EXPECTED_EXCLUDED:
        raise ValueError(f"Expected {EXPECTED_EXCLUDED} excluded IDs, found {len(excluded_ids)}.")

    all_ids = {record.case_id for record in records}
    resolved_excluded, _ = split_records(records, {"train": list(excluded_ids), "val": []})
    excluded_record_ids = {record.case_id for record in resolved_excluded}
    if len(excluded_record_ids) != EXPECTED_EXCLUDED:
        raise ValueError("Some excluded IDs could not be resolved uniquely.")

    development_ids: set[str] | None = None
    selected_train = selected_val = 0
    for index, split in enumerate(splits):
        if not isinstance(split, dict) or not isinstance(split.get("train"), list) or not isinstance(split.get("val"), list):
            raise ValueError(f"Fold {index} lacks list-valued train/val IDs.")
        train_records, val_records = split_records(records, split)
        train_ids = {record.case_id for record in train_records}
        val_ids = {record.case_id for record in val_records}
        if train_ids & val_ids:
            raise ValueError(f"Fold {index} has train/validation overlap.")
        fold_ids = train_ids | val_ids
        if fold_ids & excluded_record_ids:
            raise ValueError(f"Fold {index} leaks an excluded case into train/validation.")
        if len(fold_ids) != EXPECTED_DEVELOPMENT_POOL:
            raise ValueError(f"Fold {index} has {len(fold_ids)} cases, expected {EXPECTED_DEVELOPMENT_POOL}.")
        if development_ids is None:
            development_ids = fold_ids
        elif fold_ids != development_ids:
            raise ValueError(f"Fold {index} does not reproduce the shared 522-case development pool.")
        if index == 0:
            selected_train, selected_val = len(train_records), len(val_records)
    if development_ids is None or development_ids | excluded_record_ids != all_ids:
        raise ValueError("Development and excluded IDs do not reproduce the full local cohort.")
    return {
        "n_full_cohort": len(records),
        "n_development": len(development_ids),
        "n_excluded": len(excluded_record_ids),
        "n_fold0_train": selected_train,
        "n_fold0_val": selected_val,
    }


def validate_nifti(record: CaseRecord) -> None:
    try:
        image = load_nifti_robust(record.t1_path)
        label = load_nifti_robust(record.lesion_path)
        image_data = np.asarray(image.dataobj)
        label_data = np.asarray(label.dataobj)
    except Exception as error:  # noqa: BLE001 - aggregate all malformed cases below.
        raise RuntimeError(f"could not fully read image/label data: {error}") from error
    if image_data.ndim != 3 or label_data.ndim != 3:
        raise ValueError(f"expected 3-D T1/label, got image={image_data.shape}, label={label_data.shape}")
    if image.shape[:3] != label.shape[:3]:
        raise ValueError(f"image/label shape mismatch: {image.shape} vs {label.shape}")
    if not np.allclose(image.affine, label.affine, rtol=0.0, atol=1e-4):
        raise ValueError("image/label affine mismatch exceeds 1e-4")
    if not np.isfinite(image_data).all() or not np.isfinite(label_data).all():
        raise ValueError("image or label contains NaN/infinite voxels")
    if np.any(label_data < 0) or not np.allclose(label_data, np.rint(label_data), rtol=0.0, atol=1e-5):
        raise ValueError("label must use non-negative integer codes")


def preflight_niftis(records: list[CaseRecord]) -> None:
    failures: list[str] = []
    for index, record in enumerate(sorted(records, key=lambda item: item.case_id), start=1):
        try:
            validate_nifti(record)
        except Exception as error:  # noqa: BLE001
            failures.append(f"{record.case_id}: {error}")
        if index % 25 == 0 or index == len(records):
            print(f"[preflight] complete local cohort: read {index}/{len(records)} image/label pairs")
    if failures:
        preview = "\n".join(f"  - {failure}" for failure in failures[:20])
        more = "" if len(failures) <= 20 else f"\n  ... and {len(failures) - 20} more"
        raise RuntimeError(f"NIfTI preflight failed for {len(failures)} case(s):\n{preview}{more}")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    reference_path = args.model_b_reference.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    config = load_yaml(config_path)
    model_b_reference = load_yaml(reference_path)
    base_dir = config_path.parent

    dataset_dir, split_path, plan_path = validate_random_control_config(config, base_dir)
    records = discover_cases(dataset_dir)
    if len(records) != EXPECTED_FULL_COHORT:
        raise ValueError(f"Expected {EXPECTED_FULL_COHORT} local cases, discovered {len(records)}.")
    split_summary = validate_splits(records, split_path)
    if not args.skip_nifti_preflight:
        preflight_niftis(records)
    plan_summary = validate_plan(plan_path)

    work_dir = resolve(base_dir, config["paths"]["work_dir"], "paths.work_dir")
    fold_dir = work_dir / "fold_0"
    existing_training_outputs = [fold_dir / name for name in ("history.csv", "last.pt", "best.pt")]
    if any(path.exists() for path in existing_training_outputs):
        raise FileExistsError(
            f"Refusing to prepare over an existing random-control training run: {fold_dir}. "
            "Use a new work_dir; never overwrite or resume this first-run control."
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, work_dir / "training_config_snapshot.yaml")
    shutil.copy2(reference_path, work_dir / "model_b_reference_snapshot.yaml")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": config["protocol"]["purpose"],
        "protocol": config["protocol"],
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "model_b_reference_path": str(reference_path),
        "model_b_reference_sha256": sha256(reference_path),
        "model_b_historical_settings": {
            "data": model_b_reference.get("data"),
            "training": model_b_reference.get("training"),
        },
        "split_path": str(split_path),
        "split_sha256": sha256(split_path),
        "split_summary": split_summary,
        "architecture_plan_path": str(plan_path),
        "architecture_plan_sha256": sha256(plan_path),
        "architecture_plan": plan_summary,
        "initialization": {
            "load_pretrained": False,
            "pretrained_checkpoint": None,
            "init_checkpoint": None,
            "resume_checkpoint": None,
            "initializer": config["model"]["initialization"],
            "training_audit_path": str(fold_dir / "initialization_audit.json"),
        },
        "released_validation_cases_accessed": 0,
        "nifti_read_preflight": {
            "complete_local_cohort_cases": 0 if args.skip_nifti_preflight else len(records),
            "released_validation_cases": 0,
            "checks": ["full_decompression", "3d_shape", "image_label_geometry", "finite_voxels", "nonnegative_integer_lesion_codes"],
        },
    }
    with (work_dir / "preflight_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[DONE] Random-residual control is ready: {work_dir}")
    print(
        "[DONE] Fold 0: "
        f"train={split_summary['n_fold0_train']}, val={split_summary['n_fold0_val']}, "
        f"excluded={split_summary['n_excluded']}"
    )
    print(f"[DONE] Preflight manifest: {work_dir / 'preflight_manifest.json'}")
    print("[NEXT] Launch only after the active standard nnU-Net GPU job finishes.")


if __name__ == "__main__":
    main()
