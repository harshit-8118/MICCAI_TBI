from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, find_objects, label
from sklearn.ensemble import RandomForestClassifier

from .case_filters import lesion_category


FEATURE_COLUMNS = [
    "component_voxels",
    "bbox_d",
    "bbox_h",
    "bbox_w",
    "bbox_volume",
    "fill_ratio",
    "centroid_d_norm",
    "centroid_h_norm",
    "centroid_w_norm",
    "center_d_norm",
    "center_h_norm",
    "center_w_norm",
    "mean_prob",
    "max_prob",
    "p75_prob",
    "p90_prob",
    "std_prob",
    "sum_prob",
    "mean_intensity",
    "std_intensity",
    "min_intensity",
    "max_intensity",
    "local_contrast",
    "local_abs_contrast",
    "edge_distance_mean",
    "edge_distance_min",
    "edge_distance_max",
]


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = prediction.astype(bool)
    tgt = target.astype(bool)
    tp = int(np.logical_and(pred, tgt).sum())
    fp = int(np.logical_and(pred, ~tgt).sum())
    fn = int(np.logical_and(~pred, tgt).sum())
    denominator = 2 * tp + fp + fn
    return float((2 * tp) / denominator) if denominator > 0 else 1.0


def binary_counts(prediction: np.ndarray, target: np.ndarray) -> tuple[int, int, int]:
    pred = prediction.astype(bool)
    tgt = target.astype(bool)
    tp = int(np.logical_and(pred, tgt).sum())
    fp = int(np.logical_and(pred, ~tgt).sum())
    fn = int(np.logical_and(~pred, tgt).sum())
    return tp, fp, fn


def summarize_case_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[float, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["calibrator_threshold"]), int(row["min_component_voxels"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    categories = ["empty", "very_tiny", "tiny", "small", "large"]
    for (calibrator_threshold, min_component_voxels), combo_rows in sorted(grouped.items()):
        positive_rows = [row for row in combo_rows if int(row["gt_voxels"]) > 0]
        gt50_rows = [row for row in combo_rows if int(row["gt_voxels"]) >= 50]
        summary = {
            "calibrator_threshold": calibrator_threshold,
            "min_component_voxels": min_component_voxels,
            "n_cases": len(combo_rows),
            "mean_dice_all": _mean([float(row["dice"]) for row in combo_rows]),
            "mean_dice_positive": _mean([float(row["dice"]) for row in positive_rows]),
            "mean_dice_gt50": _mean([float(row["dice"]) for row in gt50_rows]),
            "n_positive": len(positive_rows),
            "n_gt50": len(gt50_rows),
            "n_missed_positive": sum(int(row["missed_positive"]) for row in combo_rows),
            "n_empty_false_positive": sum(int(row["empty_false_positive"]) for row in combo_rows),
        }
        for category in categories:
            category_rows = [row for row in combo_rows if row["gt_category"] == category]
            summary[f"mean_dice_{category}"] = _mean([float(row["dice"]) for row in category_rows])
            summary[f"n_{category}"] = len(category_rows)
        summary_rows.append(summary)
    return summary_rows


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def case_metric_row(
    case_id: str,
    prediction: np.ndarray,
    target: np.ndarray,
    gt_voxels: int,
    candidate_threshold: float,
    calibrator_threshold: float,
    min_component_voxels: int,
    kept_components: int,
    rejected_components: int,
) -> dict[str, object]:
    pred_voxels = int(np.count_nonzero(prediction > 0))
    tp, fp, fn = binary_counts(prediction, target)
    precision = float(tp / (tp + fp)) if tp + fp > 0 else 0.0
    recall = float(tp / (tp + fn)) if tp + fn > 0 else 0.0
    return {
        "candidate_threshold": candidate_threshold,
        "calibrator_threshold": calibrator_threshold,
        "min_component_voxels": min_component_voxels,
        "case_id": case_id,
        "dice": dice_score(prediction, target),
        "precision": precision,
        "recall": recall,
        "gt_voxels": gt_voxels,
        "gt_category": lesion_category(gt_voxels),
        "pred_voxels": pred_voxels,
        "pred_category": lesion_category(pred_voxels),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "kept_components": kept_components,
        "rejected_components": rejected_components,
        "missed_positive": int(gt_voxels > 0 and pred_voxels == 0),
        "empty_false_positive": int(gt_voxels == 0 and pred_voxels > 0),
    }


def build_brain_mask(image: np.ndarray) -> np.ndarray:
    if image.ndim == 4:
        base = image[0]
    else:
        base = image
    finite = np.isfinite(base)
    nonzero = finite & (base != 0)
    if int(nonzero.sum()) > 100:
        return nonzero
    return finite


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return label(mask.astype(bool))


def extract_component_features(
    *,
    case_id: str,
    lesion_probability: np.ndarray,
    image: np.ndarray,
    target: np.ndarray | None,
    candidate_threshold: float,
    min_component_voxels: int,
    min_true_overlap_voxels: int,
) -> list[dict[str, object]]:
    candidate_mask = lesion_probability >= float(candidate_threshold)
    components, n_components = connected_components(candidate_mask)
    if n_components == 0:
        return []

    brain_mask = build_brain_mask(image)
    edge_distance = distance_transform_edt(brain_mask)
    slices = find_objects(components)
    rows: list[dict[str, object]] = []
    for component_index, component_slice in enumerate(slices, start=1):
        if component_slice is None:
            continue
        local_labels = components[component_slice]
        local_component = local_labels == component_index
        component_voxels = int(local_component.sum())
        if component_voxels < int(min_component_voxels):
            continue

        global_component = components == component_index
        rows.append(
            component_feature_row(
                case_id=case_id,
                component_index=component_index,
                component_mask=global_component,
                component_slice=component_slice,
                lesion_probability=lesion_probability,
                image=image,
                target=target,
                edge_distance=edge_distance,
                min_true_overlap_voxels=min_true_overlap_voxels,
            )
        )
    return rows


def component_feature_row(
    *,
    case_id: str,
    component_index: int,
    component_mask: np.ndarray,
    component_slice: tuple[slice, slice, slice],
    lesion_probability: np.ndarray,
    image: np.ndarray,
    target: np.ndarray | None,
    edge_distance: np.ndarray,
    min_true_overlap_voxels: int,
) -> dict[str, object]:
    coords = np.argwhere(component_mask)
    component_voxels = int(coords.shape[0])
    shape = np.asarray(component_mask.shape, dtype=np.float32)
    centroid = coords.mean(axis=0)
    probs = lesion_probability[component_mask].astype(np.float32)
    image3d = image[0] if image.ndim == 4 else image
    intensities = image3d[component_mask].astype(np.float32)
    edge_values = edge_distance[component_mask].astype(np.float32)
    bbox_sizes = [int(s.stop - s.start) for s in component_slice]
    bbox_volume = int(np.prod(bbox_sizes))
    max_coord = coords[int(np.argmax(probs))]

    dilated = binary_dilation(component_mask, iterations=2)
    shell = dilated & ~component_mask
    shell_values = image3d[shell & np.isfinite(image3d)]
    local_mean = float(shell_values.mean()) if shell_values.size else float(np.mean(image3d[np.isfinite(image3d)]))
    mean_intensity = float(np.mean(intensities)) if intensities.size else 0.0
    local_contrast = mean_intensity - local_mean

    overlap_voxels = 0
    overlap_fraction = 0.0
    component_dice = 0.0
    label_true = 0
    if target is not None:
        target_bool = target.astype(bool)
        overlap_voxels = int(np.logical_and(component_mask, target_bool).sum())
        overlap_fraction = float(overlap_voxels / max(component_voxels, 1))
        component_dice = dice_score(component_mask.astype(np.uint8), target_bool.astype(np.uint8))
        label_true = int(overlap_voxels >= int(min_true_overlap_voxels))

    row = {
        "case_id": case_id,
        "component_index": component_index,
        "label": label_true,
        "overlap_voxels": overlap_voxels,
        "overlap_fraction": overlap_fraction,
        "component_dice": component_dice,
        "component_voxels": component_voxels,
        "bbox_d": bbox_sizes[0],
        "bbox_h": bbox_sizes[1],
        "bbox_w": bbox_sizes[2],
        "bbox_volume": bbox_volume,
        "fill_ratio": float(component_voxels / max(bbox_volume, 1)),
        "centroid_d_norm": float(centroid[0] / max(shape[0] - 1.0, 1.0)),
        "centroid_h_norm": float(centroid[1] / max(shape[1] - 1.0, 1.0)),
        "centroid_w_norm": float(centroid[2] / max(shape[2] - 1.0, 1.0)),
        "center_d_norm": float(max_coord[0] / max(shape[0] - 1.0, 1.0)),
        "center_h_norm": float(max_coord[1] / max(shape[1] - 1.0, 1.0)),
        "center_w_norm": float(max_coord[2] / max(shape[2] - 1.0, 1.0)),
        "mean_prob": float(np.mean(probs)),
        "max_prob": float(np.max(probs)),
        "p75_prob": float(np.percentile(probs, 75)),
        "p90_prob": float(np.percentile(probs, 90)),
        "std_prob": float(np.std(probs)),
        "sum_prob": float(np.sum(probs)),
        "mean_intensity": mean_intensity,
        "std_intensity": float(np.std(intensities)) if intensities.size else 0.0,
        "min_intensity": float(np.min(intensities)) if intensities.size else 0.0,
        "max_intensity": float(np.max(intensities)) if intensities.size else 0.0,
        "local_contrast": local_contrast,
        "local_abs_contrast": abs(local_contrast),
        "edge_distance_mean": float(np.mean(edge_values)) if edge_values.size else 0.0,
        "edge_distance_min": float(np.min(edge_values)) if edge_values.size else 0.0,
        "edge_distance_max": float(np.max(edge_values)) if edge_values.size else 0.0,
    }
    return row


def feature_matrix(rows: Iterable[dict[str, object]]) -> np.ndarray:
    matrix = []
    for row in rows:
        matrix.append([float(row[column]) for column in FEATURE_COLUMNS])
    return np.asarray(matrix, dtype=np.float32)


def train_random_forest_calibrator(
    rows: list[dict[str, object]],
    *,
    positive_weight: float,
    n_estimators: int,
    seed: int,
) -> RandomForestClassifier:
    if not rows:
        raise ValueError("No component rows found; cannot train calibrator.")
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    if len(np.unique(labels)) < 2:
        raise ValueError(f"Calibrator needs both positive and negative components. Found labels={sorted(set(labels))}.")
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=None,
        min_samples_leaf=2,
        class_weight={0: 1.0, 1: float(positive_weight)},
        n_jobs=-1,
        random_state=int(seed),
    )
    model.fit(feature_matrix(rows), labels)
    return model


def save_calibrator(
    path: str | Path,
    model: RandomForestClassifier,
    *,
    candidate_threshold: float,
    min_component_voxels: int,
    min_true_overlap_voxels: int,
    metadata: dict[str, object],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "candidate_threshold": float(candidate_threshold),
        "min_component_voxels": int(min_component_voxels),
        "min_true_overlap_voxels": int(min_true_overlap_voxels),
        "metadata": metadata,
    }
    joblib.dump(payload, path)
    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata_path.write_text(json.dumps({k: v for k, v in payload.items() if k != "model"}, indent=2), encoding="utf-8")


def load_calibrator(path: str | Path) -> dict[str, object]:
    payload = joblib.load(path)
    columns = payload.get("feature_columns")
    if list(columns) != FEATURE_COLUMNS:
        raise ValueError(f"Feature column mismatch in {path}.")
    return payload


def predict_component_probabilities(payload: dict[str, object], rows: list[dict[str, object]]) -> np.ndarray:
    if not rows:
        return np.asarray([], dtype=np.float32)
    model = payload["model"]
    probabilities = model.predict_proba(feature_matrix(rows))[:, 1]
    return np.asarray(probabilities, dtype=np.float32)


def apply_component_decisions(
    *,
    candidate_mask: np.ndarray,
    component_rows: list[dict[str, object]],
    component_probabilities: np.ndarray,
    calibrator_threshold: float,
    min_component_voxels: int,
) -> tuple[np.ndarray, int, int]:
    components, _ = connected_components(candidate_mask)
    prediction = np.zeros_like(candidate_mask, dtype=np.uint8)
    kept = 0
    rejected = 0
    for row, component_probability in zip(component_rows, component_probabilities):
        component_index = int(row["component_index"])
        component_voxels = int(row["component_voxels"])
        if component_voxels < int(min_component_voxels):
            rejected += 1
            continue
        if float(component_probability) >= float(calibrator_threshold):
            prediction[components == component_index] = 1
            kept += 1
        else:
            rejected += 1
    return prediction, kept, rejected
