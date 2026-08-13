"""Create an inference-only input folder after standard nnU-Net training.

This script deliberately runs separately from training preparation. It fully
reads released-validation image/label pairs, then links or copies only T1
images to a new inference directory with nnU-Net channel suffixes. It never
writes into nnUNet_raw, nnUNet_preprocessed, or the training dataset.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multitalent_tbi.data import load_nifti_robust


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--released-validation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    return parser.parse_args()


def strip_nii_suffix(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return name[:-7]
    if lower.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def discover_pairs(dataset_dir: Path) -> list[tuple[str, Path, Path]]:
    images: dict[str, Path] = {}
    labels: dict[str, Path] = {}
    for path in sorted(dataset_dir.glob("scan_*.nii*")):
        stem = strip_nii_suffix(path.name)
        lower = stem.lower()
        if lower.endswith("_lesion"):
            labels[stem[:-7]] = path
        elif lower.endswith("_dmri"):
            continue
        else:
            case_id = stem[:-3] if lower.endswith("_t1") else stem
            images[case_id] = path
    case_ids = sorted(set(images) | set(labels))
    if not case_ids:
        raise FileNotFoundError(f"No scan_*.nii* files found in {dataset_dir}.")
    missing = [case_id for case_id in case_ids if case_id not in images or case_id not in labels]
    if missing:
        raise RuntimeError(f"Missing image/label partners for released-validation cases: {missing[:20]}")
    return [(case_id, images[case_id], labels[case_id]) for case_id in case_ids]


def validate_pair(case_id: str, image_path: Path, label_path: Path) -> str | None:
    """Fully read one source pair and return a non-fatal affine warning, if any.

    The staged directory contains only the T1 image.  The released-validation
    evaluator independently resamples the T1-derived prediction and the
    source label to its canonical 1-mm grid, matching the historical Model
    A/B protocol.  Therefore a source affine discrepancy must be recorded for
    audit, but must not silently change the T1 or block T1-only inference when
    the pair is otherwise readable and has matching voxel-array dimensions.
    """
    try:
        image = load_nifti_robust(image_path)
        label = load_nifti_robust(label_path)
        image_data = np.asarray(image.dataobj)
        label_data = np.asarray(label.dataobj)
    except Exception as error:  # noqa: BLE001 - report all failed files together.
        raise RuntimeError(f"could not fully read NIfTI data: {error}") from error
    if image_data.ndim != 3 or label_data.ndim != 3:
        raise ValueError(f"expected 3-D T1/label, got image={image_data.shape}, label={label_data.shape}")
    if image_data.shape != label_data.shape:
        raise ValueError(f"image/label shape mismatch: {image_data.shape} vs {label_data.shape}")
    if not np.isfinite(image_data).all() or not np.isfinite(label_data).all():
        raise ValueError("image or label contains NaN/infinite voxels")
    # Model B defines the binary target as ``label > 0.5`` after nearest-
    # neighbour resampling. Accept its non-negative integer source codes here;
    # labels are not staged and this script never alters them.
    if np.any(label_data < 0) or not np.allclose(label_data, np.rint(label_data), rtol=0.0, atol=1e-5):
        raise ValueError("lesion label must use non-negative integer codes")
    if not np.allclose(image.affine, label.affine, rtol=0.0, atol=1e-4):
        max_affine_difference = float(np.max(np.abs(image.affine - label.affine)))
        origin_displacement_mm = float(np.linalg.norm(image.affine[:3, 3] - label.affine[:3, 3]))
        return (
            "image/label affine mismatch retained from source "
            f"(max_abs={max_affine_difference:.6f}, origin_displacement_mm={origin_displacement_mm:.6f})"
        )
    return None


def copy_or_link(source: Path, destination: Path, mode: str) -> None:
    if not source.name.lower().endswith(".nii.gz"):
        raise ValueError(f"Only .nii.gz inputs are supported, got {source}")
    if mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def ensure_outside(parent: Path, candidate: Path, message: str) -> None:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return
    raise ValueError(message)


def main() -> None:
    args = parse_args()
    dataset_dir = args.released_validation_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not dataset_dir.is_dir():
        raise NotADirectoryError(dataset_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse or overwrite inference input directory: {output_dir}")
    ensure_outside(
        dataset_dir,
        output_dir,
        "The inference directory must be outside the released-validation dataset so it remains untouched.",
    )
    raw_root = os.environ.get("nnUNet_raw")
    if raw_root:
        ensure_outside(
            Path(raw_root).expanduser().resolve(),
            output_dir,
            "The inference directory must be outside nnUNet_raw to prevent released-validation data entering planning/training.",
        )
    pairs = discover_pairs(dataset_dir)
    failures: list[str] = []
    affine_warnings: dict[str, str] = {}
    for index, (case_id, image_path, label_path) in enumerate(pairs, start=1):
        try:
            warning = validate_pair(case_id, image_path, label_path)
            if warning is not None:
                affine_warnings[case_id] = warning
        except Exception as error:  # noqa: BLE001
            failures.append(f"{case_id}: {error}")
        if index % 25 == 0 or index == len(pairs):
            print(f"[preflight] released validation: read {index}/{len(pairs)} image/label pairs")
    if failures:
        raise RuntimeError("Released-validation preflight failed:\n  - " + "\n  - ".join(failures[:20]))
    for case_id, warning in affine_warnings.items():
        print(f"[preflight warning] {case_id}: {warning}")

    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "released_validation_input_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case_id",
                "source_t1_path",
                "source_label_path",
                "inference_t1_path",
                "source_affine_warning",
            ),
        )
        writer.writeheader()
        for case_id, image_path, label_path in pairs:
            destination = output_dir / f"{case_id}_0000.nii.gz"
            copy_or_link(image_path, destination, args.link_mode)
            writer.writerow(
                {
                    "case_id": case_id,
                    "source_t1_path": str(image_path),
                    "source_label_path": str(label_path),
                    "inference_t1_path": str(destination),
                    "source_affine_warning": affine_warnings.get(case_id, ""),
                }
            )
    print(f"[DONE] Staged {len(pairs)} released-validation T1 inputs after full preflight.")
    print(f"[DONE] This directory is inference-only and is outside nnUNet_raw: {output_dir}")


if __name__ == "__main__":
    main()
