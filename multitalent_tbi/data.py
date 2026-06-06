from __future__ import annotations

import shutil
import json
import math
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import torch
from nibabel.filebasedimages import ImageFileError
from nibabel.processing import resample_from_to, resample_to_output


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    t1_path: Path
    lesion_path: Path
    dmri_path: Path | None = None
    bval_path: Path | None = None
    bvec_path: Path | None = None

    @property
    def has_dmri(self) -> bool:
        return self.dmri_path is not None and self.dmri_path.exists()


def _scan_id_from_name(name: str) -> str:
    stem = name
    for suffix in (".nii.gz", ".nii", ".img", ".hdr"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if stem.startswith("scan_"):
        stem = stem[len("scan_") :]
    for suffix in ("_T1", "_t1", "_Lesion", "_lesion", "_dMRI", "_dmri"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def _find_existing_file(dataset_dir: Path, case_id: str, stem: str) -> Path | None:
    candidates = [
        dataset_dir / f"scan_{case_id}{stem}.nii.gz",
        dataset_dir / f"scan_{case_id}{stem}.nii",
        dataset_dir / f"scan_{case_id}{stem}.NII.GZ",
        dataset_dir / f"scan_{case_id}{stem}.NII",
        dataset_dir / f"scan_{case_id}{stem}.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    lower_name = f"scan_{case_id}{stem}".lower()
    for path in dataset_dir.iterdir():
        if path.is_file() and path.name.lower().startswith(lower_name):
            return path
    return None


def discover_cases(dataset_dir: str | Path) -> list[CaseRecord]:
    dataset_dir = Path(dataset_dir)
    records: list[CaseRecord] = []
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    candidate_files = [path for path in dataset_dir.iterdir() if path.is_file() and path.name.lower().startswith("scan_")]
    for t1_path in sorted(candidate_files):
        lower_name = t1_path.name.lower()
        if "_t1" not in lower_name:
            continue
        case_id = _scan_id_from_name(t1_path.name)
        lesion_path = _find_existing_file(dataset_dir, case_id, "_Lesion")
        if lesion_path is None:
            continue
        dmri_path = _find_existing_file(dataset_dir, case_id, "_dMRI")
        bval_path = next((path for path in dataset_dir.glob(f"scan_{case_id}_bval.*") if path.is_file()), None)
        bvec_path = next((path for path in dataset_dir.glob(f"scan_{case_id}_bvec.*") if path.is_file()), None)
        records.append(
            CaseRecord(
                case_id=case_id,
                t1_path=t1_path,
                lesion_path=lesion_path,
                dmri_path=dmri_path if dmri_path is not None else None,
                bval_path=bval_path if bval_path is not None else None,
                bvec_path=bvec_path if bvec_path is not None else None,
            )
        )
    if not records:
        preview = [path.name for path in candidate_files[:20]]
        raise FileNotFoundError(
            f"No scan cases were discovered in {dataset_dir}. "
            f"Found {len(candidate_files)} scan-like files. Preview: {preview}"
        )
    return records


def _load_canonical_nifti(path: Path) -> nib.Nifti1Image:
    return nib.as_closest_canonical(load_nifti_robust(path))


def _looks_like_gzip(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


def load_nifti_robust(path: str | Path, repair_dir: str | Path | None = None) -> nib.Nifti1Image:
    path = Path(path)
    try:
        return nib.load(str(path))
    except ImageFileError as error:
        if path.suffix.lower() != ".gz" and not path.name.endswith(".nii.gz"):
            raise
        if _looks_like_gzip(path):
            raise

        repair_root = Path(repair_dir) if repair_dir is not None else Path(tempfile.gettempdir()) / "tbi_nifti_repaired"
        repair_root.mkdir(parents=True, exist_ok=True)
        repaired_path = repair_root / path.with_suffix("").name
        if not repaired_path.exists():
            shutil.copyfile(path, repaired_path)
        return nib.load(str(repaired_path))


def _load_array(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = _load_canonical_nifti(path)
    return np.asarray(image.dataobj), image.affine


def _reduce_dmri(case: CaseRecord, strategy: str, b0_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    image = _load_canonical_nifti(case.dmri_path)  # type: ignore[arg-type]
    data = np.asarray(image.dataobj)
    if data.ndim == 4:
        if strategy == "first":
            data = data[..., 0]
        elif strategy == "b0" and case.bval_path is not None:
            bvals = np.loadtxt(case.bval_path)
            if np.ndim(bvals) == 0:
                bvals = np.array([float(bvals)])
            indices = np.where(np.asarray(bvals) <= float(b0_threshold))[0]
            if len(indices) == 0:
                data = data.mean(axis=-1)
            else:
                data = data[..., indices].mean(axis=-1)
        else:
            data = data.mean(axis=-1)
    return np.asarray(data, dtype=np.float32), image.affine


def _resample(array: np.ndarray, affine: np.ndarray, spacing: Iterable[float], order: int) -> tuple[np.ndarray, np.ndarray]:
    image = nib.Nifti1Image(array, affine)
    resampled = resample_to_output(image, voxel_sizes=tuple(spacing), order=order)
    return np.asarray(resampled.dataobj), resampled.affine


def zscore_foreground(array: np.ndarray) -> np.ndarray:
    mask = np.isfinite(array) & (array != 0)
    if not np.any(mask):
        mask = np.isfinite(array)
    values = array[mask]
    mean = float(values.mean()) if values.size else 0.0
    std = float(values.std()) if values.size else 1.0
    if std < 1e-8:
        std = 1.0
    normalized = (array - mean) / std
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized.astype(np.float32)


def _pad_to_shape(array: np.ndarray, target_shape: tuple[int, int, int], fill_value: float = 0.0) -> tuple[np.ndarray, list[tuple[int, int]]]:
    pads: list[tuple[int, int]] = []
    for size, target in zip(array.shape[-3:], target_shape):
        total = max(target - size, 0)
        before = total // 2
        after = total - before
        pads.append((before, after))
    padded = np.pad(array, [(0, 0)] * (array.ndim - 3) + pads, mode="constant", constant_values=fill_value)
    return padded, pads


def _crop_with_center(array: np.ndarray, center: tuple[int, int, int], patch_size: tuple[int, int, int]) -> np.ndarray:
    starts = []
    for c, size, dim in zip(center, patch_size, array.shape[-3:]):
        start = int(c - size // 2)
        start = max(start, 0)
        start = min(start, max(dim - size, 0))
        starts.append(start)
    slices = tuple(slice(start, start + size) for start, size in zip(starts, patch_size))
    if array.ndim == 4:
        return array[:, slices[0], slices[1], slices[2]]
    return array[slices]


def center_crop_or_pad(image: np.ndarray, mask: np.ndarray, patch_size: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    padded_image, _ = _pad_to_shape(image, patch_size, fill_value=0.0)
    padded_mask, _ = _pad_to_shape(mask, patch_size, fill_value=0)
    center = tuple(size // 2 for size in padded_mask.shape[-3:])
    return _crop_with_center(padded_image, center, patch_size), _crop_with_center(padded_mask, center, patch_size)


def _random_center(mask: np.ndarray, oversample_foreground_prob: float) -> tuple[int, int, int]:
    if random.random() < oversample_foreground_prob and np.any(mask > 0):
        foreground = np.argwhere(mask > 0)
        chosen = foreground[random.randrange(len(foreground))]
        return tuple(int(x) for x in chosen)
    return tuple(random.randrange(size) for size in mask.shape)


def _random_flip(image: np.ndarray, mask: np.ndarray, axes: tuple[int, int, int] = (0, 1, 2)) -> tuple[np.ndarray, np.ndarray]:
    for axis in axes:
        if random.random() < 0.5:
            image = np.flip(image, axis=axis + 1).copy()
            mask = np.flip(mask, axis=axis).copy()
    return image, mask


def _random_intensity(image: np.ndarray) -> np.ndarray:
    scale = random.uniform(0.9, 1.1)
    shift = random.uniform(-0.1, 0.1)
    noise = np.random.normal(0.0, 0.01, size=image.shape).astype(np.float32)
    return (image * scale + shift + noise).astype(np.float32)


def load_case(
    case: CaseRecord,
    target_spacing: Iterable[float],
    include_dmri: bool,
    dmri_reduce: str,
    dmri_b0_threshold: float,
    normalize_foreground_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t1_image = _load_canonical_nifti(case.t1_path)
    t1 = np.asarray(t1_image.dataobj, dtype=np.float32)
    t1, t1_affine = _resample(t1, t1_image.affine, target_spacing, order=1)

    channels = [zscore_foreground(t1) if normalize_foreground_only else t1.astype(np.float32)]

    if include_dmri:
        if case.has_dmri:
            dmri, dmri_affine = _reduce_dmri(case, dmri_reduce, dmri_b0_threshold)
            dmri, _ = _resample(dmri, dmri_affine, target_spacing, order=1)
            channels.append(zscore_foreground(dmri) if normalize_foreground_only else dmri.astype(np.float32))
        else:
            channels.append(np.zeros_like(channels[0], dtype=np.float32))

    image = np.stack(channels, axis=0)

    lesion_image = _load_canonical_nifti(case.lesion_path)
    lesion = np.asarray(lesion_image.dataobj, dtype=np.float32)
    lesion, lesion_affine = _resample(lesion, lesion_image.affine, target_spacing, order=0)
    lesion = (lesion > 0.5).astype(np.uint8)
    return image, lesion, t1_affine, lesion_affine


def load_case_cached(
    case: CaseRecord,
    target_spacing: Iterable[float],
    include_dmri: bool,
    dmri_reduce: str,
    dmri_b0_threshold: float,
    normalize_foreground_only: bool,
    cache_dir: str | Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{case.case_id}_c{int(include_dmri)}_{'_'.join(str(x) for x in target_spacing)}.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            return cached["image"], cached["mask"], cached["image_affine"], cached["mask_affine"]

    image, mask, image_affine, mask_affine = load_case(
        case=case,
        target_spacing=target_spacing,
        include_dmri=include_dmri,
        dmri_reduce=dmri_reduce,
        dmri_b0_threshold=dmri_b0_threshold,
        normalize_foreground_only=normalize_foreground_only,
    )
    if cache_path is not None:
        np.savez(
            cache_path,
            image=image,
            mask=mask,
            image_affine=image_affine,
            mask_affine=mask_affine,
        )
    return image, mask, image_affine, mask_affine


class TBIDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records: list[CaseRecord],
        patch_size: Iterable[int],
        target_spacing: Iterable[float],
        include_dmri: bool,
        dmri_reduce: str,
        dmri_b0_threshold: float,
        normalize_foreground_only: bool,
        oversample_foreground_prob: float,
        training: bool,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.records = records
        self.patch_size = tuple(int(x) for x in patch_size)
        self.target_spacing = tuple(float(x) for x in target_spacing)
        self.include_dmri = include_dmri
        self.dmri_reduce = dmri_reduce
        self.dmri_b0_threshold = float(dmri_b0_threshold)
        self.normalize_foreground_only = normalize_foreground_only
        self.oversample_foreground_prob = float(oversample_foreground_prob)
        self.training = training
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        image, mask, image_affine, mask_affine = load_case_cached(
            case=record,
            target_spacing=self.target_spacing,
            include_dmri=self.include_dmri,
            dmri_reduce=self.dmri_reduce,
            dmri_b0_threshold=self.dmri_b0_threshold,
            normalize_foreground_only=self.normalize_foreground_only,
            cache_dir=self.cache_dir,
        )

        if self.training:
            image, mask = self._sample_patch(image, mask)
            image, mask = self._augment(image, mask)

        return {
            "image": torch.from_numpy(image.copy()).float(),
            "mask": torch.from_numpy(mask.copy()).long(),
            "case_id": record.case_id,
            "image_affine": torch.from_numpy(np.asarray(image_affine)),
            "mask_affine": torch.from_numpy(np.asarray(mask_affine)),
        }

    def _sample_patch(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        padded_image, pads = _pad_to_shape(image, self.patch_size, fill_value=0.0)
        padded_mask, _ = _pad_to_shape(mask, self.patch_size, fill_value=0)
        center = _random_center(padded_mask, self.oversample_foreground_prob)
        return _crop_with_center(padded_image, center, self.patch_size), _crop_with_center(padded_mask, center, self.patch_size)

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image, mask = _random_flip(image, mask)
        if random.random() < 0.2:
            image = _random_intensity(image)
        return image, mask


def make_stratification_labels(records: list[CaseRecord]) -> list[str]:
    labels: list[str] = []
    for record in records:
        lesion = np.asarray(load_nifti_robust(record.lesion_path).dataobj)
        lesion_voxels = int(np.count_nonzero(lesion > 0))
        if lesion_voxels == 0:
            lesion_bucket = "empty"
        elif lesion_voxels < 1000:
            lesion_bucket = "tiny"
        elif lesion_voxels < 5000:
            lesion_bucket = "small"
        else:
            lesion_bucket = "large"
        labels.append(f"{lesion_bucket}_{'dmri' if record.has_dmri else 't1'}")
    return labels
