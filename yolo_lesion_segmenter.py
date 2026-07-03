from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import yaml

from multitalent_tbi.case_filters import CaseInfo, build_case_infos, lesion_category
from multitalent_tbi.config import load_yaml, resolve_path
from multitalent_tbi.data import CaseRecord, discover_cases, load_case_cached
from multitalent_tbi.splits import load_splits, split_records


AXIS_TO_INDEX = {"sagittal": 0, "coronal": 1, "axial": 2}


@dataclass(frozen=True)
class SliceItem:
    case_id: str
    split: str
    slice_index: int
    image_path: Path
    label_path: Path
    positive: bool
    category: str
    lesion_pixels: int


class TextLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        stamped = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(stamped, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(stamped + "\n")


def _namespace_to_dict(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {key: _namespace_to_dict(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {key: _namespace_to_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_namespace_to_dict(item) for item in value]
    return value


def _cfg_get(mapping: dict[str, Any], key: str, default: Any = None) -> Any:
    return mapping[key] if key in mapping and mapping[key] is not None else default


def _resolve_config_path(base_dir: Path, value: str | Path | None, fallback: str | Path | None = None) -> Path:
    chosen = value if value is not None else fallback
    if chosen is None:
        raise ValueError("Missing required path in config.")
    return resolve_path(base_dir, chosen)


def _case_mapping(records: list[CaseRecord]) -> dict[str, CaseRecord]:
    mapping: dict[str, CaseRecord] = {}
    for record in records:
        aliases = {record.case_id, record.case_id.lstrip("0")}
        if record.case_id.isdigit():
            aliases.add(str(int(record.case_id)))
            aliases.add(record.case_id.zfill(4))
        for alias in aliases:
            if alias:
                mapping.setdefault(alias, record)
    return mapping


def _select_train_records(
    train_records: list[CaseRecord],
    target_categories: set[str],
    limit: int | None,
    seed: int,
    logger: TextLogger,
) -> list[CaseRecord]:
    infos = build_case_infos(train_records, split="train")
    by_category: dict[str, list[CaseInfo]] = {}
    for info in infos:
        by_category.setdefault(info.category, []).append(info)

    rng = random.Random(seed)
    for values in by_category.values():
        rng.shuffle(values)

    target_infos: list[CaseInfo] = []
    for category in sorted(target_categories):
        target_infos.extend(by_category.get(category, []))
    rng.shuffle(target_infos)

    other_infos = [info for info in infos if info.category not in target_categories]
    rng.shuffle(other_infos)

    selected_infos = target_infos + other_infos
    if limit is not None and limit > 0:
        selected_infos = selected_infos[:limit]

    counts: dict[str, int] = {}
    for info in selected_infos:
        counts[info.category] = counts.get(info.category, 0) + 1
    logger.write(f"Selected {len(selected_infos)} train MRIs for YOLO from fold train set: {counts}")
    return [info.record for info in selected_infos]


def _slice_2d(array: np.ndarray, axis_index: int, slice_index: int) -> np.ndarray:
    if axis_index == 0:
        return array[slice_index, :, :]
    if axis_index == 1:
        return array[:, slice_index, :]
    return array[:, :, slice_index]


def _normalize_to_uint8(slice_image: np.ndarray, clip_range: tuple[float, float]) -> np.ndarray:
    low, high = clip_range
    clipped = np.clip(slice_image.astype(np.float32), low, high)
    scaled = (clipped - low) / max(high - low, 1e-6)
    return np.round(scaled * 255.0).astype(np.uint8)


def _slice_has_content(slice_image: np.ndarray, min_foreground_fraction: float, min_std: float) -> bool:
    finite = np.asarray(slice_image[np.isfinite(slice_image)], dtype=np.float32)
    if finite.size == 0:
        return False
    foreground_fraction = float(np.mean(np.abs(finite) > 1e-4))
    slice_std = float(finite.std())
    return foreground_fraction >= min_foreground_fraction and slice_std >= min_std


def _save_png(path: Path, slice_image: np.ndarray, clip_range: tuple[float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gray = _normalize_to_uint8(slice_image, clip_range)
    rgb = np.stack([gray, gray, gray], axis=-1)
    mpimg.imsave(path, rgb)


def _try_cv2_contours(mask2d: np.ndarray, min_area: float) -> list[np.ndarray] | None:
    try:
        import cv2  # type: ignore
    except Exception:
        return None

    contours, _ = cv2.findContours(mask2d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[np.ndarray] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        epsilon = max(0.5, 0.002 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if approx.shape[0] >= 3:
            polygons.append(approx.astype(np.float32))
    return polygons


def _bbox_polygons(mask2d: np.ndarray, min_area: float) -> list[np.ndarray]:
    try:
        from scipy import ndimage
    except Exception:
        ys, xs = np.where(mask2d > 0)
        if len(xs) < min_area:
            return []
        return [np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()]], dtype=np.float32)]

    labeled, num = ndimage.label(mask2d > 0)
    polygons: list[np.ndarray] = []
    for component_id in range(1, num + 1):
        ys, xs = np.where(labeled == component_id)
        if len(xs) < min_area:
            continue
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        if x1 <= x0 or y1 <= y0:
            continue
        polygons.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32))
    return polygons


def mask_to_yolo_polygons(mask2d: np.ndarray, min_area: float) -> list[np.ndarray]:
    polygons = _try_cv2_contours(mask2d, min_area)
    if polygons is not None:
        return polygons
    return _bbox_polygons(mask2d, min_area)


def _write_yolo_label(path: Path, polygons: list[np.ndarray], width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for polygon in polygons:
        polygon = polygon.astype(np.float32)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        if polygon.shape[0] < 3:
            continue
        values: list[str] = ["0"]
        for x, y in polygon:
            values.append(f"{x / max(width - 1, 1):.6f}")
            values.append(f"{y / max(height - 1, 1):.6f}")
        lines.append(" ".join(values))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _sample_negative_slices(
    available: list[int],
    positive_count: int,
    negative_ratio: float,
    max_per_case: int,
    rng: random.Random,
) -> set[int]:
    if not available:
        return set()
    desired = int(round(max(positive_count, 1) * negative_ratio))
    desired = min(desired, max_per_case, len(available))
    return set(rng.sample(available, desired))


def _prepare_split_slices(
    records: list[CaseRecord],
    split_name: str,
    cfg: dict[str, Any],
    paths: dict[str, Path],
    target_categories: set[str],
    logger: TextLogger,
) -> list[SliceItem]:
    axis_name = str(cfg["axis"]).lower()
    axis_index = AXIS_TO_INDEX[axis_name]
    rng = random.Random(int(cfg["seed"]) + (0 if split_name == "train" else 10_000))
    clip_range = tuple(float(x) for x in cfg["clip_range"])
    margin = int(cfg["positive_slice_margin"])
    min_area = float(cfg["min_contour_area_px"])
    negative_ratio = float(cfg["negative_slice_ratio"])
    max_negatives = int(cfg["max_negative_slices_per_case"])
    min_foreground_fraction = float(cfg.get("min_slice_foreground_fraction", 0.0))
    min_slice_std = float(cfg.get("min_slice_std", 0.0))

    items: list[SliceItem] = []
    case_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        image, mask, _, _ = load_case_cached(
            case=record,
            target_spacing=cfg["target_spacing"],
            include_dmri=False,
            dmri_reduce="mean",
            dmri_b0_threshold=50.0,
            normalize_foreground_only=True,
            cache_dir=paths["cache_dir"],
        )
        image3d = image[0]
        gt_voxels = int(np.count_nonzero(mask > 0))
        category = lesion_category(gt_voxels)
        min_case_voxels = int(cfg.get("min_component_voxels_3d", 0))
        is_target_case = category in target_categories and gt_voxels >= min_case_voxels
        num_slices = mask.shape[axis_index]
        content_slices = {
            slice_index
            for slice_index in range(num_slices)
            if _slice_has_content(_slice_2d(image3d, axis_index, slice_index), min_foreground_fraction, min_slice_std)
        }

        positive_slices: set[int] = set()
        positive_core: set[int] = set()
        if is_target_case:
            for slice_index in range(num_slices):
                mask2d = _slice_2d(mask, axis_index, slice_index)
                if int(np.count_nonzero(mask2d > 0)) >= min_area:
                    polygons = mask_to_yolo_polygons(mask2d > 0, min_area)
                    if polygons:
                        positive_core.add(slice_index)
            for slice_index in positive_core:
                start = max(0, slice_index - margin)
                end = min(num_slices - 1, slice_index + margin)
                positive_slices.add(slice_index)
                positive_slices.update(slice_id for slice_id in range(start, end + 1) if slice_id in content_slices)

        negative_candidates = [
            slice_index
            for slice_index in range(num_slices)
            if slice_index in content_slices
            and slice_index not in positive_slices
            and int(np.count_nonzero(_slice_2d(mask, axis_index, slice_index) > 0)) == 0
        ]
        if split_name == "train":
            negative_slices = _sample_negative_slices(
                negative_candidates,
                positive_count=len(positive_core),
                negative_ratio=negative_ratio,
                max_per_case=max_negatives,
                rng=rng,
            )
            selected_slices = positive_slices | negative_slices
        else:
            selected_slices = positive_slices | set(negative_candidates[:: max(1, num_slices // max_negatives)])

        for slice_index in sorted(selected_slices):
            mask2d = _slice_2d(mask, axis_index, slice_index)
            image2d = _slice_2d(image3d, axis_index, slice_index)
            polygons: list[np.ndarray] = []
            if is_target_case:
                polygons = mask_to_yolo_polygons(mask2d > 0, min_area)
            positive = bool(polygons)

            stem = f"{record.case_id}_{axis_name}_{slice_index:03d}"
            image_path = paths["dataset_dir"] / "images" / split_name / f"{stem}.png"
            label_path = paths["dataset_dir"] / "labels" / split_name / f"{stem}.txt"
            _save_png(image_path, image2d, clip_range)  # type: ignore[arg-type]
            _write_yolo_label(label_path, polygons, width=mask2d.shape[1], height=mask2d.shape[0])
            items.append(
                SliceItem(
                    case_id=record.case_id,
                    split=split_name,
                    slice_index=slice_index,
                    image_path=image_path,
                    label_path=label_path,
                    positive=positive,
                    category=category,
                    lesion_pixels=int(np.count_nonzero(mask2d > 0)),
                )
            )

        case_rows.append(
            {
                "case_id": record.case_id,
                "split": split_name,
                "category": category,
                "gt_voxels": gt_voxels,
                "selected_slices": len(selected_slices),
                "positive_core_slices": len(positive_core),
                "positive_or_margin_slices": len(positive_slices),
                "content_slices": len(content_slices),
            }
        )
        if index % 25 == 0:
            logger.write(f"Prepared {split_name} slices for {index}/{len(records)} cases.")

    _write_dict_csv(paths["output_dir"] / f"{split_name}_case_slice_summary.csv", case_rows)
    return items


def _write_slice_manifest(path: Path, items: list[SliceItem]) -> None:
    rows = [
        {
            "case_id": item.case_id,
            "split": item.split,
            "slice_index": item.slice_index,
            "positive": int(item.positive),
            "category": item.category,
            "lesion_pixels": item.lesion_pixels,
            "image_path": str(item.image_path),
            "label_path": str(item.label_path),
        }
        for item in items
    ]
    _write_dict_csv(path, rows)


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prepare_yolo_dataset(config_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    base_dir = config_path.parent.resolve()
    full_cfg = load_yaml(config_path)
    yolo_cfg = dict(full_cfg.get("yolo", {}))
    data_cfg = dict(full_cfg.get("data", {}))
    paths_cfg = dict(full_cfg.get("paths", {}))

    yolo_cfg.setdefault("fold", full_cfg.get("training", {}).get("fold", 0))
    yolo_cfg.setdefault("splits_file", None)
    yolo_cfg.setdefault("seed", full_cfg.get("training", {}).get("seed", 42))
    yolo_cfg.setdefault("axis", "axial")
    yolo_cfg.setdefault("target_categories", ["small", "large"])
    yolo_cfg.setdefault("train_mri_limit", 100)
    yolo_cfg.setdefault("val_mri_limit", None)
    yolo_cfg.setdefault("positive_slice_margin", 2)
    yolo_cfg.setdefault("negative_slice_ratio", 1.0)
    yolo_cfg.setdefault("max_negative_slices_per_case", 12)
    yolo_cfg.setdefault("min_slice_foreground_fraction", 0.03)
    yolo_cfg.setdefault("min_slice_std", 0.01)
    yolo_cfg.setdefault("min_contour_area_px", 8)
    yolo_cfg.setdefault("clip_range", [-4.0, 4.0])
    yolo_cfg.setdefault("image_size", 512)
    yolo_cfg["target_spacing"] = tuple(float(x) for x in data_cfg.get("target_spacing", [1.0, 1.0, 1.0]))

    output_dir = _resolve_config_path(base_dir, yolo_cfg.get("output_dir"), "checkpoints/tbi_yolo_seg/fold_0_100mri")
    log_file = _resolve_config_path(base_dir, yolo_cfg.get("log_file"), output_dir / "finetune_log.txt")
    logger = TextLogger(log_file)
    logger.write("Starting YOLO segmentation dataset preparation.")

    split_file = _resolve_config_path(base_dir, yolo_cfg.get("splits_file"), paths_cfg.get("splits_file"))
    dataset_root = _resolve_config_path(base_dir, paths_cfg.get("dataset_dir"))
    cache_dir = _resolve_config_path(base_dir, paths_cfg.get("cache_dir"), output_dir / "cache")
    yolo_dataset_dir = output_dir / "dataset"
    paths = {
        "output_dir": output_dir,
        "log_file": log_file,
        "split_file": split_file,
        "source_dataset_dir": dataset_root,
        "cache_dir": cache_dir,
        "dataset_dir": yolo_dataset_dir,
    }

    if not split_file.exists():
        raise FileNotFoundError(f"Split file does not exist: {split_file}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")

    if yolo_dataset_dir.exists():
        shutil.rmtree(yolo_dataset_dir)
    for split_name in ("train", "val"):
        (yolo_dataset_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (yolo_dataset_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    records = discover_cases(dataset_root)
    splits = load_splits(split_file)
    fold = int(yolo_cfg["fold"])
    train_records, val_records = split_records(records, splits[fold])
    train_ids = {record.case_id for record in train_records}
    val_ids = {record.case_id for record in val_records}
    overlap_ids = sorted(train_ids & val_ids)
    if overlap_ids:
        raise ValueError(f"Fold {fold} has train/val overlap: {overlap_ids[:20]}")
    target_categories = set(str(item) for item in yolo_cfg["target_categories"])

    train_records = _select_train_records(
        train_records=train_records,
        target_categories=target_categories,
        limit=yolo_cfg.get("train_mri_limit"),
        seed=int(yolo_cfg["seed"]),
        logger=logger,
    )
    val_limit = yolo_cfg.get("val_mri_limit")
    if val_limit is not None and int(val_limit) > 0:
        val_records = val_records[: int(val_limit)]

    logger.write(f"Fold {fold}: train MRIs={len(train_records)}, val MRIs={len(val_records)}")
    logger.write(f"Target categories for positive YOLO labels: {sorted(target_categories)}")
    logger.write(f"Using {yolo_cfg['axis']} slices, image_size={yolo_cfg['image_size']}, min_contour_area_px={yolo_cfg['min_contour_area_px']}")

    train_items = _prepare_split_slices(train_records, "train", yolo_cfg, paths, target_categories, logger)
    val_items = _prepare_split_slices(val_records, "val", yolo_cfg, paths, target_categories, logger)
    _write_slice_manifest(output_dir / "slice_manifest.csv", train_items + val_items)

    data_yaml = {
        "path": str(yolo_dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "lesion"},
    }
    data_yaml_path = output_dir / "data.yaml"
    with data_yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data_yaml, handle, sort_keys=False)

    logger.write(
        "Prepared YOLO dataset: "
        f"train_slices={len(train_items)} train_positive={sum(item.positive for item in train_items)} "
        f"val_slices={len(val_items)} val_positive={sum(item.positive for item in val_items)}"
    )
    logger.write(f"YOLO data yaml: {data_yaml_path}")
    return yolo_cfg, paths


def train_yolo(config_path: Path) -> Path:
    base_dir = config_path.parent.resolve()
    full_cfg = load_yaml(config_path)
    yolo_cfg = dict(full_cfg.get("yolo", {}))
    output_dir = _resolve_config_path(base_dir, yolo_cfg.get("output_dir"), "checkpoints/tbi_yolo_seg/fold_0_100mri")
    log_file = _resolve_config_path(base_dir, yolo_cfg.get("log_file"), output_dir / "finetune_log.txt")
    logger = TextLogger(log_file)
    data_yaml = output_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing YOLO data yaml. Run --prepare first: {data_yaml}")

    try:
        from ultralytics import YOLO
    except Exception as error:
        raise RuntimeError("Ultralytics is required for --train. Install with: pip install ultralytics") from error

    logger.write("Starting YOLO fine-tuning.")
    logger.write(f"Ultralytics model: {yolo_cfg.get('model', 'yolo26m-seg.pt')}")
    logger.write(f"Training data: {data_yaml}")

    model = YOLO(str(yolo_cfg.get("model", "yolo11n-seg.pt")))
    results = model.train(
        data=str(data_yaml),
        imgsz=int(yolo_cfg.get("image_size", 512)),
        epochs=int(yolo_cfg.get("epochs", 50)),
        batch=int(yolo_cfg.get("batch_size", 16)),
        workers=int(yolo_cfg.get("workers", 8)),
        device=yolo_cfg.get("device", 0),
        patience=int(yolo_cfg.get("patience", 20)),
        optimizer=str(yolo_cfg.get("optimizer", "auto")),
        lr0=float(yolo_cfg.get("lr0", 0.001)),
        weight_decay=float(yolo_cfg.get("weight_decay", 0.0005)),
        cache=bool(yolo_cfg.get("cache", False)),
        pretrained=bool(yolo_cfg.get("pretrained", True)),
        project=str(output_dir / "runs"),
        name="train",
        exist_ok=True,
    )
    save_dir = Path(getattr(results, "save_dir", output_dir / "runs" / "train"))
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    logger.write(f"YOLO fine-tuning finished. best={best} exists={best.exists()} last={last} exists={last.exists()}")
    return best if best.exists() else last


def _fill_polygon_mask(mask: np.ndarray, polygons: list[np.ndarray]) -> np.ndarray:
    if not polygons:
        return mask
    try:
        import cv2  # type: ignore

        cv_polys = [np.round(poly).astype(np.int32).reshape(-1, 1, 2) for poly in polygons if len(poly) >= 3]
        if cv_polys:
            cv2.fillPoly(mask, cv_polys, 1)
        return mask
    except Exception:
        pass

    try:
        from PIL import Image, ImageDraw

        image = Image.fromarray(mask.astype(np.uint8), mode="L")
        draw = ImageDraw.Draw(image)
        for polygon in polygons:
            if len(polygon) >= 3:
                draw.polygon([tuple(point) for point in polygon.tolist()], fill=1)
        return np.asarray(image, dtype=np.uint8)
    except Exception:
        return mask


def _predict_case_yolo(
    model: Any,
    record: CaseRecord,
    cfg: dict[str, Any],
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis_index = AXIS_TO_INDEX[str(cfg.get("axis", "axial")).lower()]
    clip_range = tuple(float(x) for x in cfg.get("clip_range", [-4.0, 4.0]))
    image, mask, _, mask_affine = load_case_cached(
        case=record,
        target_spacing=tuple(float(x) for x in cfg.get("target_spacing", [1.0, 1.0, 1.0])),
        include_dmri=False,
        dmri_reduce="mean",
        dmri_b0_threshold=50.0,
        normalize_foreground_only=True,
        cache_dir=cache_dir,
    )
    image3d = image[0]
    pred = np.zeros_like(mask, dtype=np.uint8)
    num_slices = mask.shape[axis_index]

    for slice_index in range(num_slices):
        image2d = _slice_2d(image3d, axis_index, slice_index)
        gray = _normalize_to_uint8(image2d, clip_range)
        rgb = np.stack([gray, gray, gray], axis=-1)
        results = model.predict(
            source=rgb,
            imgsz=int(cfg.get("image_size", 512)),
            conf=float(cfg.get("prediction_conf", 0.15)),
            iou=float(cfg.get("prediction_iou", 0.50)),
            max_det=int(cfg.get("max_det", 50)),
            verbose=False,
        )
        result = results[0]
        polygons: list[np.ndarray] = []
        if getattr(result, "masks", None) is not None and result.masks is not None:
            for polygon in result.masks.xy:
                poly = np.asarray(polygon, dtype=np.float32)
                if poly.shape[0] >= 3:
                    polygons.append(poly)
        mask2d = np.zeros_like(_slice_2d(mask, axis_index, slice_index), dtype=np.uint8)
        mask2d = _fill_polygon_mask(mask2d, polygons)
        if axis_index == 0:
            pred[slice_index, :, :] = mask2d
        elif axis_index == 1:
            pred[:, slice_index, :] = mask2d
        else:
            pred[:, :, slice_index] = mask2d

    return pred, mask.astype(np.uint8), mask_affine


def _dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred_bool = pred > 0
    gt_bool = gt > 0
    intersection = int(np.logical_and(pred_bool, gt_bool).sum())
    pred_sum = int(pred_bool.sum())
    gt_sum = int(gt_bool.sum())
    union = int(np.logical_or(pred_bool, gt_bool).sum())
    dice = (2.0 * intersection) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 1.0
    iou = intersection / union if union > 0 else 1.0
    return float(dice), float(iou)


def _find_default_weights(config_path: Path) -> Path:
    full_cfg = load_yaml(config_path)
    yolo_cfg = dict(full_cfg.get("yolo", {}))
    output_dir = _resolve_config_path(config_path.parent.resolve(), yolo_cfg.get("output_dir"), "checkpoints/tbi_yolo_seg/fold_0_100mri")
    for candidate in [output_dir / "runs" / "train" / "weights" / "best.pt", output_dir / "runs" / "train" / "weights" / "last.pt"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find trained YOLO weights under {output_dir / 'runs' / 'train' / 'weights'}")


def predict_and_evaluate(config_path: Path, weights_path: Path | None = None) -> None:
    base_dir = config_path.parent.resolve()
    full_cfg = load_yaml(config_path)
    yolo_cfg = dict(full_cfg.get("yolo", {}))
    data_cfg = dict(full_cfg.get("data", {}))
    paths_cfg = dict(full_cfg.get("paths", {}))
    yolo_cfg["target_spacing"] = tuple(float(x) for x in data_cfg.get("target_spacing", [1.0, 1.0, 1.0]))

    output_dir = _resolve_config_path(base_dir, yolo_cfg.get("output_dir"), "checkpoints/tbi_yolo_seg/fold_0_100mri")
    log_file = _resolve_config_path(base_dir, yolo_cfg.get("log_file"), output_dir / "finetune_log.txt")
    logger = TextLogger(log_file)
    split_file = _resolve_config_path(base_dir, yolo_cfg.get("splits_file"), paths_cfg.get("splits_file"))
    dataset_root = _resolve_config_path(base_dir, paths_cfg.get("dataset_dir"))
    cache_dir = _resolve_config_path(base_dir, paths_cfg.get("cache_dir"), output_dir / "cache")
    predictions_dir = output_dir / "val_predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    if weights_path is None:
        weights_path = _find_default_weights(config_path)
    try:
        from ultralytics import YOLO
    except Exception as error:
        raise RuntimeError("Ultralytics is required for prediction. Install with: pip install ultralytics") from error

    logger.write(f"Starting YOLO validation prediction/evaluation with weights: {weights_path}")
    records = discover_cases(dataset_root)
    splits = load_splits(split_file)
    fold = int(yolo_cfg.get("fold", 0))
    _, val_records = split_records(records, splits[fold])
    val_limit = yolo_cfg.get("val_mri_limit")
    if val_limit is not None and int(val_limit) > 0:
        val_records = val_records[: int(val_limit)]
    eval_categories = set(str(item) for item in yolo_cfg.get("eval_categories", ["small", "large"]))

    model = YOLO(str(weights_path))
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(val_records, start=1):
        pred, gt, affine = _predict_case_yolo(model, record, yolo_cfg, cache_dir)
        gt_voxels = int(np.count_nonzero(gt > 0))
        pred_voxels = int(np.count_nonzero(pred > 0))
        category = lesion_category(gt_voxels)
        dice, iou = _dice_iou(pred, gt)
        row = {
            "case_id": record.case_id,
            "gt_category": category,
            "gt_voxels": gt_voxels,
            "pred_voxels": pred_voxels,
            "dice": dice,
            "iou": iou,
            "evaluated_target_category": int(category in eval_categories),
        }
        rows.append(row)
        if bool(yolo_cfg.get("save_val_predictions", True)):
            nib.save(nib.Nifti1Image(pred.astype(np.uint8), affine), str(predictions_dir / f"scan_{record.case_id}_YOLO_Lesion.nii.gz"))
        logger.write(
            f"Val {index}/{len(val_records)} case={record.case_id} category={category} "
            f"gt={gt_voxels} pred={pred_voxels} dice={dice:.4f} iou={iou:.4f}"
        )

    metrics_path = output_dir / "val_yolo_case_metrics.csv"
    _write_dict_csv(metrics_path, rows)
    summary_rows = _summarize_metrics(rows, eval_categories)
    summary_path = output_dir / "val_yolo_summary_metrics.csv"
    _write_dict_csv(summary_path, summary_rows)
    _plot_metrics(rows, eval_categories, output_dir)
    logger.write(f"Wrote validation case metrics: {metrics_path}")
    logger.write(f"Wrote validation summary metrics: {summary_path}")


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _summarize_metrics(rows: list[dict[str, Any]], eval_categories: set[str]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for category in ["empty", "very_tiny", "tiny", "small", "large", "target_small_large"]:
        if category == "target_small_large":
            selected = [row for row in rows if row["gt_category"] in eval_categories]
        else:
            selected = [row for row in rows if row["gt_category"] == category]
        summary.append(
            {
                "category": category,
                "n": len(selected),
                "mean_dice": _mean([float(row["dice"]) for row in selected]),
                "mean_iou": _mean([float(row["iou"]) for row in selected]),
                "mean_gt_voxels": _mean([float(row["gt_voxels"]) for row in selected]),
                "mean_pred_voxels": _mean([float(row["pred_voxels"]) for row in selected]),
            }
        )
    return summary


def _plot_metrics(rows: list[dict[str, Any]], eval_categories: set[str], output_dir: Path) -> None:
    categories = [category for category in ["small", "large"] if category in eval_categories]
    if not categories:
        categories = sorted(eval_categories)
    data = [[float(row["iou"]) for row in rows if row["gt_category"] == category] for category in categories]

    plt.figure(figsize=(7, 5))
    if any(data):
        plt.boxplot(data, labels=[f"{category}\n(n={len(values)})" for category, values in zip(categories, data)], showmeans=True)
    plt.ylabel("3D IoU")
    plt.title("YOLO-seg validation IoU by lesion size")
    plt.ylim(-0.02, 1.02)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "val_iou_by_small_large.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    for category in categories:
        selected = [row for row in rows if row["gt_category"] == category]
        plt.scatter(
            [float(row["gt_voxels"]) for row in selected],
            [float(row["iou"]) for row in selected],
            label=f"{category} (n={len(selected)})",
            alpha=0.75,
        )
    plt.xscale("log")
    plt.xlabel("Ground-truth lesion voxels")
    plt.ylabel("3D IoU")
    plt.title("YOLO-seg IoU vs lesion volume")
    plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "val_iou_vs_gt_voxels.png", dpi=160)
    plt.close()


def dump_run_environment(output_dir: Path, logger: TextLogger) -> None:
    env_path = output_dir / "run_environment.json"
    payload = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": sys.platform,
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        payload["pip_freeze"] = completed.stdout.splitlines()
    except Exception as error:
        payload["pip_freeze_error"] = str(error)
    env_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.write(f"Wrote run environment: {env_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare, train, and evaluate 2D YOLO-seg for AIMS-TBI lesion contours.")
    parser.add_argument("--config", default="config.yml", help="Path to config.yml.")
    parser.add_argument("--prepare", action="store_true", help="Create YOLO segmentation PNG/label dataset.")
    parser.add_argument("--train", action="store_true", help="Fine-tune Ultralytics YOLO segmentation.")
    parser.add_argument("--predict", action="store_true", help="Run validation prediction with YOLO weights.")
    parser.add_argument("--evaluate", action="store_true", help="Alias for --predict; prediction includes evaluation.")
    parser.add_argument("--all", action="store_true", help="Run prepare, train, and validation prediction/evaluation.")
    parser.add_argument("--weights", default=None, help="YOLO weights for --predict/--evaluate. Defaults to output runs/train/weights/best.pt.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    if args.all:
        args.prepare = True
        args.train = True
        args.predict = True
    if args.evaluate:
        args.predict = True
    if not any([args.prepare, args.train, args.predict]):
        args.prepare = True

    if args.prepare:
        _, paths = prepare_yolo_dataset(config_path)
        dump_run_environment(paths["output_dir"], TextLogger(paths["log_file"]))
    if args.train:
        train_yolo(config_path)
    if args.predict:
        weights_path = Path(args.weights).resolve() if args.weights else None
        predict_and_evaluate(config_path, weights_path)


if __name__ == "__main__":
    main()


# python yolo_lesion_segmenter.py --config config.yml --prepare
# python yolo_lesion_segmenter.py --config config.yml --train
# python yolo_lesion_segmenter.py --config config.yml --predict