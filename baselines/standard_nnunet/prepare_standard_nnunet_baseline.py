"""Prepare a conventional nnU-Net v2 T1 baseline using Model B's exact fold.

This script creates a new nnU-Net raw dataset without changing the submitted
MultiTalentV2 models. It refuses to create the dataset unless the supplied
split proves the expected 552 = 522 development + 30 holdout partition. The
released Phase-2 validation data are intentionally out of scope here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nibabel as nib
import numpy as np
import yaml

from multitalent_tbi.data import CaseRecord, discover_cases, load_nifti_robust
from multitalent_tbi.splits import split_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="baselines/standard_nnunet/model_b_matched_baseline.yaml",
        help="Baseline YAML configuration.",
    )
    parser.add_argument("--source-dataset-dir", type=Path, default=None)
    parser.add_argument("--model-b-split-file", type=Path, default=None)
    parser.add_argument("--raw-root", type=Path, default=None, help="Value for nnUNet_raw.")
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return payload


def require_path(value: Any, label: str) -> Path:
    if value in (None, ""):
        raise ValueError(f"{label} must be set in the YAML or provided on the command line.")
    path = Path(str(value)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path.resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nnunet_case_id(case_id: str, prefix: str) -> str:
    normalized = str(case_id).strip()
    return normalized if normalized.startswith(prefix) else f"{prefix}{normalized}"


def _copy_or_link(source: Path, destination: Path, mode: str) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
    if not source.name.lower().endswith(".nii.gz"):
        raise ValueError(
            f"nnU-Net baseline preparation requires .nii.gz inputs, got {source}. "
            "Convert the source file explicitly rather than giving an uncompressed file a .nii.gz name."
        )
    # A few legacy source files have an uncompressed NIfTI payload despite a
    # .nii.gz suffix. nnU-Net's Nibabel reader rightly rejects such a file, so
    # materialize a valid gzip NIfTI instead of propagating the bad suffix.
    if source.name.lower().endswith(".nii.gz") and not _looks_like_gzip(source):
        repaired_image = load_nifti_robust(source)
        nib.save(repaired_image, str(destination))
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def _looks_like_gzip(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


def _validate_and_describe_lesion_values(case_id: str, label_data: np.ndarray) -> list[float]:
    """Validate source coding before collapsing it to Model B's binary target.

    The original TBI masks have several positive annotation codes. Model B
    maps them to one lesion class with ``lesion > 0.5`` after nearest-neighbour
    resampling. Applying that exact definition here avoids accidentally giving
    standard nnU-Net a multi-class target.
    """
    if not np.isfinite(label_data).all():
        raise ValueError(f"{case_id}: lesion label contains NaN or infinite voxels.")
    if np.any(label_data < 0):
        raise ValueError(f"{case_id}: lesion label contains negative values.")
    if not np.allclose(label_data, np.rint(label_data), rtol=0.0, atol=1e-5):
        raise ValueError(f"{case_id}: lesion label is not integer-coded.")
    return np.unique(label_data).astype(float).tolist()


def _write_model_b_binary_label(source: Path, reference_image_path: Path, destination: Path) -> list[float]:
    """Write ``source > 0.5`` on the corresponding T1's exact voxel grid."""
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
    source_image = load_nifti_robust(source)
    reference_image = load_nifti_robust(reference_image_path)
    if source_image.shape[:3] != reference_image.shape[:3]:
        raise ValueError(
            f"Cannot export label: source label/T1 shape mismatch: {source_image.shape} vs {reference_image.shape}"
        )
    if not np.allclose(source_image.affine, reference_image.affine, rtol=0.0, atol=1e-4):
        raise ValueError("Cannot export label: source label/T1 affine mismatch exceeds 1e-4.")
    source_data = np.asarray(source_image.dataobj)
    source_values = _validate_and_describe_lesion_values(source.stem, source_data)
    binary_data = (source_data > 0.5).astype(np.uint8)
    # nnU-Net requires image and segmentation headers to agree exactly. The
    # verified image grid is authoritative; some source label headers differ
    # only by floating-point rounding in qform/sform fields.
    header = reference_image.header.copy()
    header.set_data_dtype(np.uint8)
    output_image = nib.Nifti1Image(binary_data, reference_image.affine, header=header)
    qform, qform_code = reference_image.get_qform(coded=True)
    sform, sform_code = reference_image.get_sform(coded=True)
    output_image.set_qform(qform, int(qform_code))
    output_image.set_sform(sform, int(sform_code))
    nib.save(output_image, str(destination))
    return source_values


def _validate_case_readability(record: CaseRecord) -> None:
    """Fully read a T1/lesion pair before training can start.

    ``nib.load`` alone is lazy for compressed NIfTI data. Materializing both
    arrays catches truncated gzip members and malformed voxel data now, rather
    than after an expensive nnU-Net job has begun.
    """
    try:
        image = load_nifti_robust(record.t1_path)
        label = load_nifti_robust(record.lesion_path)
        image_data = np.asarray(image.dataobj)
        label_data = np.asarray(label.dataobj)
    except Exception as error:  # noqa: BLE001 - aggregate the affected case IDs below.
        raise RuntimeError(f"could not fully read image/label data: {error}") from error
    if image_data.ndim != 3:
        raise ValueError(f"T1 image must be 3D for this baseline, got shape {image_data.shape}.")
    if label_data.ndim != 3:
        raise ValueError(f"Lesion label must be 3D, got shape {label_data.shape}.")
    if image.shape[:3] != label.shape[:3]:
        raise ValueError(f"{record.case_id}: image/label shape mismatch: {image.shape} vs {label.shape}")
    if not np.allclose(image.affine, label.affine, rtol=0.0, atol=1e-4):
        raise ValueError(f"{record.case_id}: image/label affine mismatch")
    if not np.isfinite(image_data).all():
        raise ValueError(f"{record.case_id}: T1 image contains NaN or infinite voxels.")
    _validate_and_describe_lesion_values(record.case_id, label_data)


def _preflight_records(records: list[CaseRecord], label: str) -> None:
    """Read every NIfTI pair and report every failure before any baseline data are written."""
    failures: list[str] = []
    for index, record in enumerate(sorted(records, key=lambda item: item.case_id), start=1):
        try:
            _validate_case_readability(record)
        except Exception as error:  # noqa: BLE001 - a combined report is more useful than repeated job failures.
            failures.append(f"{record.case_id}: {error}")
        if index % 25 == 0 or index == len(records):
            print(f"[preflight] {label}: read {index}/{len(records)} NIfTI image/label pairs")
    if failures:
        preview = "\n".join(f"  - {failure}" for failure in failures[:20])
        more = "" if len(failures) <= 20 else f"\n  ... and {len(failures) - 20} more"
        raise RuntimeError(
            f"NIfTI preflight failed for {len(failures)} {label} case(s). Fix these files before training:\n{preview}{more}"
        )


def _parse_split(split_path: Path, expected_folds: int, selected_fold: int) -> tuple[list[dict[str, list[str]]], set[str]]:
    payload = load_yaml(split_path)
    splits = payload.get("splits", payload.get("folds"))
    if not isinstance(splits, list) or not splits:
        raise ValueError(f"{split_path} must contain a non-empty 'splits' or 'folds' list.")
    if len(splits) != expected_folds:
        raise ValueError(f"Expected {expected_folds} folds in {split_path}, found {len(splits)}.")
    if selected_fold < 0 or selected_fold >= len(splits):
        raise ValueError(f"Requested fold {selected_fold}, but only folds 0..{len(splits) - 1} exist.")
    normalized: list[dict[str, list[str]]] = []
    for index, split in enumerate(splits):
        if not isinstance(split, dict) or not isinstance(split.get("train"), list) or not isinstance(split.get("val"), list):
            raise ValueError(f"Fold {index} lacks list-valued train/val IDs.")
        train_ids = [str(case_id) for case_id in split["train"]]
        val_ids = [str(case_id) for case_id in split["val"]]
        if set(train_ids) & set(val_ids):
            raise ValueError(f"Fold {index} has train/validation overlap.")
        normalized.append({"train": train_ids, "val": val_ids})
    test_ids = {str(case_id) for case_id in payload.get("test", [])}
    return normalized, test_ids


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "nnunet_case_id",
        "source_case_id",
        "role",
        "t1_path",
        "lesion_path",
        "source_label_values",
        "nnunet_label_transform",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _require_builtin_wandb_support(enabled: bool) -> str | None:
    """Fail before data preparation if requested standard nnU-Net W&B logging is unavailable."""
    if not enabled:
        return None
    try:
        logger_module = importlib.import_module("nnunetv2.training.logging.nnunet_logger")
    except ImportError as error:
        raise RuntimeError(
            "The installed nnU-Net v2 package has no importable built-in logger. "
            "Install the pinned nnU-Net version before preparing this W&B-enabled baseline."
        ) from error
    if not hasattr(logger_module, "WandbLogger"):
        raise RuntimeError(
            "The installed nnU-Net v2 version does not expose its built-in WandbLogger. "
            "Use an nnU-Net v2 version with MetaLogger/WandbLogger support, or disable wandb.enabled explicitly."
        )
    try:
        return version("wandb")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "wandb.enabled is true but the W&B package is not installed. Install a pinned wandb package "
            "in the nnU-Net environment before running this baseline."
        ) from error


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    data_cfg = config["data"]
    nnunet_cfg = config["nnunet"]
    output_cfg = config["outputs"]
    wandb_cfg = config.get("wandb", {})

    source_dataset_dir = require_path(args.source_dataset_dir or data_cfg.get("source_dataset_dir"), "source_dataset_dir")
    split_path = require_path(args.model_b_split_file or data_cfg.get("model_b_split_file"), "model_b_split_file")
    raw_root = require_path(args.raw_root or nnunet_cfg.get("raw_root"), "raw_root")
    link_mode = args.link_mode or str(output_cfg.get("link_mode", "hardlink"))
    selected_fold = int(data_cfg["fold"])
    expected_folds = int(data_cfg["expected_number_of_folds"])
    prefix = str(data_cfg.get("case_prefix", "scan_"))
    dataset_name = f"Dataset{int(nnunet_cfg['dataset_id']):03d}_{nnunet_cfg['dataset_name']}"
    dataset_dir = raw_root / dataset_name
    if dataset_dir.exists():
        raise FileExistsError(
            f"Refusing to reuse {dataset_dir}. Choose a new dataset ID/name or manually inspect and remove the old baseline dataset."
        )
    try:
        installed_nnunet_version = version("nnunetv2")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "nnunetv2 is not installed in this environment. Prepare and train the baseline in the pinned nnU-Net v2 environment."
        ) from error
    installed_wandb_version = _require_builtin_wandb_support(bool(wandb_cfg.get("enabled", False)))

    records = discover_cases(source_dataset_dir)
    if len(records) != int(data_cfg["expected_full_cohort_cases"]):
        raise ValueError(f"Expected {data_cfg['expected_full_cohort_cases']} local cases, discovered {len(records)}.")
    # Read all local pairs up front, including the 30 holdout records. This
    # proves that malformed/truncated local NIfTI files cannot interrupt later
    # preparation or preprocessing. No released-validation files are accessed.
    _preflight_records(records, "complete local cohort")
    splits, excluded_ids = _parse_split(split_path, expected_folds, selected_fold)
    train_records, val_records = split_records(records, splits[selected_fold])
    development_records = {record.case_id: record for record in [*train_records, *val_records]}
    if len(development_records) != int(data_cfg["expected_development_pool_cases"]):
        raise ValueError(
            f"Fold {selected_fold} contains {len(development_records)} unique development cases; "
            f"expected {data_cfg['expected_development_pool_cases']}."
        )
    if len(excluded_ids) != int(data_cfg["expected_excluded_cases"]):
        raise ValueError(
            f"Split declares {len(excluded_ids)} excluded/test IDs; expected {data_cfg['expected_excluded_cases']}."
        )
    all_records = {record.case_id: record for record in records}
    resolved_excluded, _ = split_records(records, {"train": list(excluded_ids), "val": []})
    excluded_record_ids = {record.case_id for record in resolved_excluded}
    if excluded_record_ids & set(development_records):
        raise ValueError("Excluded/test cases overlap Model B's development pool.")
    if len(excluded_record_ids) != len(excluded_ids):
        raise ValueError("Some excluded/test split IDs could not be resolved uniquely.")
    if set(development_records) | excluded_record_ids != set(all_records):
        raise ValueError("Development plus excluded cases do not reproduce the complete local cohort.")
    for index, split in enumerate(splits):
        split_train, split_val = split_records(records, split)
        split_ids = {record.case_id for record in [*split_train, *split_val]}
        if split_ids & excluded_record_ids:
            raise ValueError(f"Fold {index} includes excluded/test cases.")
        if split_ids != set(development_records):
            raise ValueError(f"Fold {index} does not use exactly the Model-B development pool.")

    dataset_dir.mkdir(parents=True)
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_tr.mkdir()
    labels_tr.mkdir()
    manifest_rows: list[dict[str, str]] = []
    train_ids = {record.case_id for record in train_records}
    val_ids = {record.case_id for record in val_records}
    for record in sorted(development_records.values(), key=lambda item: item.case_id):
        case_name = nnunet_case_id(record.case_id, prefix)
        _copy_or_link(record.t1_path, images_tr / f"{case_name}_0000.nii.gz", link_mode)
        source_label_values = _write_model_b_binary_label(
            record.lesion_path,
            record.t1_path,
            labels_tr / f"{case_name}.nii.gz",
        )
        role = "train" if record.case_id in train_ids else "val"
        manifest_rows.append(
            {
                "nnunet_case_id": case_name,
                "source_case_id": record.case_id,
                "role": role,
                "t1_path": str(record.t1_path),
                "lesion_path": str(record.lesion_path),
                "source_label_values": json.dumps(source_label_values),
                "nnunet_label_transform": "uint8(source_label > 0.5) on verified T1 voxel grid",
            }
        )

    converted_splits: list[dict[str, list[str]]] = []
    for split in splits:
        split_train, split_val = split_records(records, split)
        converted_splits.append(
            {
                "train": [nnunet_case_id(record.case_id, prefix) for record in split_train],
                "val": [nnunet_case_id(record.case_id, prefix) for record in split_val],
            }
        )
    _write_json(
        dataset_dir / "dataset.json",
        {
            "channel_names": {"0": "T1w"},
            "labels": {"background": 0, "lesion": 1},
            "numTraining": len(development_records),
            "file_ending": ".nii.gz",
            "overwrite_image_reader_writer": str(nnunet_cfg["image_reader_writer"]),
        },
    )
    _write_json(dataset_dir / "splits_final.json", converted_splits)
    _write_manifest(dataset_dir / "case_manifest.csv", manifest_rows)

    run_root = Path(str(output_cfg["run_root"]))
    run_root.mkdir(parents=True, exist_ok=True)
    model_b_reference_path = Path(str(config["model_b_reference"]))
    if not model_b_reference_path.is_absolute():
        model_b_reference_path = config_path.parent / model_b_reference_path
    model_b_reference_path = model_b_reference_path.resolve()
    if not model_b_reference_path.exists():
        raise FileNotFoundError(f"Model-B provenance file not found: {model_b_reference_path}")
    shutil.copy2(config_path, run_root / "baseline_config_snapshot.yaml")
    shutil.copy2(model_b_reference_path, run_root / "model_b_reference_snapshot.yaml")
    manifest = {
        "protocol": config["protocol"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_dataset_dir": str(dataset_dir),
        "source_dataset_dir": str(source_dataset_dir),
        "source_split_file": str(split_path),
        "source_split_sha256": sha256(split_path),
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "model_b_reference_path": str(model_b_reference_path),
        "model_b_reference_sha256": sha256(model_b_reference_path),
        "selected_fold": selected_fold,
        "n_full_cohort": len(records),
        "n_development": len(development_records),
        "n_train": len(train_records),
        "n_val": len(val_records),
        "n_excluded": len(excluded_record_ids),
        "label_transform": "uint8(source_label > 0.5) on the verified T1 voxel grid, matching Model B's binary lesion definition after nearest-neighbour resampling",
        "nifti_read_preflight": {
            "complete_local_cohort_cases": len(records),
            "released_validation_cases": 0,
            "checks": ["full_decompression", "3d_shape", "image_label_geometry", "finite_voxels", "nonnegative_integer_lesion_codes"],
            "released_validation_access": "prohibited during training preparation",
        },
        "nnunet": {**nnunet_cfg, "installed_version": installed_nnunet_version},
        "wandb": {**wandb_cfg, "installed_version": installed_wandb_version},
        "link_mode": link_mode,
        "external_validation_input_cases": 0,
        "released_validation_policy": "Not accessed or staged by this training-preparation script.",
        "model_b_reference": config["model_b_reference"],
    }
    _write_json(run_root / "baseline_manifest.json", manifest)
    print(f"[DONE] Prepared {dataset_dir}")
    print(f"[DONE] Fold {selected_fold}: train={len(train_records)}, val={len(val_records)}, excluded={len(excluded_record_ids)}")
    print(f"[DONE] Manifest: {run_root / 'baseline_manifest.json'}")


if __name__ == "__main__":
    main()
