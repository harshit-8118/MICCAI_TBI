from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import CaseRecord, load_nifti_robust


@dataclass(frozen=True)
class CaseInfo:
    record: CaseRecord
    gt_voxels: int
    category: str
    split: str | None = None


def lesion_category(gt_voxels: int) -> str:
    if gt_voxels == 0:
        return "empty"
    if gt_voxels <= 50:
        return "very_tiny"
    if gt_voxels < 1000:
        return "tiny"
    if gt_voxels < 5000:
        return "small"
    return "large"


def count_lesion_voxels(record: CaseRecord) -> int:
    lesion = np.asarray(load_nifti_robust(record.lesion_path).dataobj)
    return int(np.count_nonzero(lesion > 0))


def build_case_infos(records: list[CaseRecord], split: str | None = None) -> list[CaseInfo]:
    infos: list[CaseInfo] = []
    for record in records:
        gt_voxels = count_lesion_voxels(record)
        infos.append(
            CaseInfo(
                record=record,
                gt_voxels=gt_voxels,
                category=lesion_category(gt_voxels),
                split=split,
            )
        )
    return infos


def positive_infos(infos: list[CaseInfo], min_lesion_voxels: int = 1) -> list[CaseInfo]:
    return [info for info in infos if info.gt_voxels >= min_lesion_voxels]


def empty_infos(infos: list[CaseInfo]) -> list[CaseInfo]:
    return [info for info in infos if info.gt_voxels == 0]


def write_case_manifest(path: str | Path, infos: list[CaseInfo]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "split",
                "gt_voxels",
                "category",
                "has_dmri",
                "t1_path",
                "lesion_mask_path",
                "dmri_path",
            ],
        )
        writer.writeheader()
        for info in infos:
            record = info.record
            writer.writerow(
                {
                    "case_id": record.case_id,
                    "split": info.split or "",
                    "gt_voxels": info.gt_voxels,
                    "category": info.category,
                    "has_dmri": int(record.has_dmri),
                    "t1_path": str(record.t1_path),
                    "lesion_mask_path": str(record.lesion_path),
                    "dmri_path": str(record.dmri_path or ""),
                }
            )


def write_case_id_list(path: str | Path, infos: list[CaseInfo]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for info in infos:
            handle.write(f"{info.record.case_id}\n")


def write_mask_path_list(path: str | Path, infos: list[CaseInfo]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for info in infos:
            handle.write(f"{info.record.lesion_path}\n")
