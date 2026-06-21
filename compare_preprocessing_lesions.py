from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output
from scipy.ndimage import binary_dilation

from multitalent_tbi.case_filters import build_case_infos
from multitalent_tbi.config import load_config, resolve_path
from multitalent_tbi.data import discover_cases, load_nifti_robust, zscore_foreground
from multitalent_tbi.splits import load_splits, split_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare lesion visibility before/after preprocessing. Produces lesion-centric "
            "PNG panels and a CSV with lesion/ring contrast metrics."
        )
    )
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--case-ids", nargs="*", default=None, help="Optional exact case ids to visualize.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["very_tiny", "tiny"],
        choices=["empty", "very_tiny", "tiny", "small", "large"],
        help="GT lesion categories to include when --case-ids is not supplied.",
    )
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["raw_zscore", "clip_1_99_zscore"],
        choices=[
            "raw",
            "raw_zscore",
            "clip_1_99",
            "clip_1_99_zscore",
            "clip_0_5_99_5_zscore",
            "clip_2_98_zscore",
        ],
    )
    parser.add_argument("--target-spacing", nargs=3, type=float, default=None)
    parser.add_argument("--zoom-size", type=int, default=56)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _select_records(config, base_dir: Path, fold: int, split_name: str):
    records = discover_cases(resolve_path(base_dir, config.paths.dataset_dir))
    if split_name == "all":
        return records
    splits = load_splits(resolve_path(base_dir, config.paths.splits_file))
    train_records, val_records = split_records(records, splits[fold])
    return train_records if split_name == "train" else val_records


def _resample_image(path: Path, spacing: tuple[float, float, float], order: int) -> tuple[np.ndarray, np.ndarray]:
    image = nib.as_closest_canonical(load_nifti_robust(path))
    resampled = resample_to_output(image, voxel_sizes=spacing, order=order)
    return np.asarray(resampled.dataobj, dtype=np.float32), resampled.affine


def _foreground_mask(array: np.ndarray) -> np.ndarray:
    mask = np.isfinite(array) & (array != 0)
    if int(mask.sum()) < 100:
        mask = np.isfinite(array)
    return mask


def _clip_foreground(array: np.ndarray, lower: float, upper: float) -> np.ndarray:
    mask = _foreground_mask(array)
    values = array[mask]
    if values.size == 0:
        return array.astype(np.float32)
    lo, hi = np.percentile(values, [lower, upper])
    clipped = np.clip(array, lo, hi)
    return clipped.astype(np.float32)


def apply_method(array: np.ndarray, method: str) -> np.ndarray:
    if method == "raw":
        return array.astype(np.float32)
    if method == "raw_zscore":
        return zscore_foreground(array)
    if method == "clip_1_99":
        return _clip_foreground(array, 1.0, 99.0)
    if method == "clip_1_99_zscore":
        return zscore_foreground(_clip_foreground(array, 1.0, 99.0))
    if method == "clip_0_5_99_5_zscore":
        return zscore_foreground(_clip_foreground(array, 0.5, 99.5))
    if method == "clip_2_98_zscore":
        return zscore_foreground(_clip_foreground(array, 2.0, 98.0))
    raise ValueError(f"Unknown method: {method}")


def _largest_lesion_slice(mask: np.ndarray) -> int:
    if not np.any(mask > 0):
        return mask.shape[2] // 2
    counts = (mask > 0).sum(axis=(0, 1))
    return int(np.argmax(counts))


def _lesion_center(mask: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return tuple(size // 2 for size in mask.shape)
    return tuple(int(x) for x in np.round(coords.mean(axis=0)))


def _window_for_display(array: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    if mask is not None and np.any(mask > 0):
        dilated = binary_dilation(mask > 0, iterations=8)
        values = array[dilated & np.isfinite(array)]
        if values.size >= 20:
            lo, hi = np.percentile(values, [1, 99])
            if hi > lo:
                return float(lo), float(hi)
    foreground = _foreground_mask(array)
    values = array[foreground]
    if values.size == 0:
        return float(np.nanmin(array)), float(np.nanmax(array))
    lo, hi = np.percentile(values, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _crop2d(array2d: np.ndarray, center_xy: tuple[int, int], size: int) -> np.ndarray:
    half = int(size) // 2
    x, y = center_xy
    x0 = max(x - half, 0)
    y0 = max(y - half, 0)
    x1 = min(x0 + int(size), array2d.shape[0])
    y1 = min(y0 + int(size), array2d.shape[1])
    x0 = max(x1 - int(size), 0)
    y0 = max(y1 - int(size), 0)
    return array2d[x0:x1, y0:y1]


def _ring_mask(mask: np.ndarray, iterations: int = 4) -> np.ndarray:
    lesion = mask > 0
    if not lesion.any():
        return np.zeros_like(lesion, dtype=bool)
    return binary_dilation(lesion, iterations=iterations) & ~lesion


def compute_metrics(case_id: str, method: str, array: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    lesion = mask > 0
    ring = _ring_mask(mask)
    foreground = _foreground_mask(array)
    lesion_values = array[lesion & np.isfinite(array)]
    ring_values = array[ring & foreground & np.isfinite(array)]
    if lesion_values.size == 0:
        lesion_mean = lesion_std = lesion_min = lesion_max = float("nan")
    else:
        lesion_mean = float(lesion_values.mean())
        lesion_std = float(lesion_values.std())
        lesion_min = float(lesion_values.min())
        lesion_max = float(lesion_values.max())
    if ring_values.size == 0:
        ring_mean = ring_std = float("nan")
    else:
        ring_mean = float(ring_values.mean())
        ring_std = float(ring_values.std())
    cnr = float(abs(lesion_mean - ring_mean) / max(ring_std, 1e-6)) if np.isfinite(ring_std) else float("nan")
    return {
        "case_id": case_id,
        "method": method,
        "gt_voxels": int(lesion.sum()),
        "lesion_mean": lesion_mean,
        "lesion_std": lesion_std,
        "lesion_min": lesion_min,
        "lesion_max": lesion_max,
        "ring_mean": ring_mean,
        "ring_std": ring_std,
        "lesion_ring_abs_contrast": float(abs(lesion_mean - ring_mean))
        if np.isfinite(lesion_mean) and np.isfinite(ring_mean)
        else float("nan"),
        "lesion_ring_cnr": cnr,
    }


def save_case_figure(
    *,
    case_id: str,
    category: str,
    arrays: dict[str, np.ndarray],
    mask: np.ndarray,
    output_path: Path,
    zoom_size: int,
) -> None:
    z = _largest_lesion_slice(mask)
    center = _lesion_center(mask)
    method_names = list(arrays.keys())
    figure, axes = plt.subplots(2, len(method_names), figsize=(4.2 * len(method_names), 7.8))
    if len(method_names) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col, method in enumerate(method_names):
        array = arrays[method]
        lo, hi = _window_for_display(array, mask)
        image_slice = array[:, :, z].T
        mask_slice = (mask[:, :, z] > 0).T

        axes[0, col].imshow(image_slice, cmap="gray", origin="lower", vmin=lo, vmax=hi)
        if mask_slice.any():
            overlay = np.ma.masked_where(~mask_slice, mask_slice)
            axes[0, col].imshow(overlay, cmap="autumn", alpha=0.55, origin="lower", vmin=0, vmax=1)
        axes[0, col].set_title(f"{method}\nfull slice z={z}", fontsize=9)
        axes[0, col].axis("off")

        zoom_image = _crop2d(array[:, :, z], (center[0], center[1]), zoom_size).T
        zoom_mask = _crop2d(mask[:, :, z] > 0, (center[0], center[1]), zoom_size).T
        axes[1, col].imshow(zoom_image, cmap="gray", origin="lower", vmin=lo, vmax=hi)
        if zoom_mask.any():
            overlay = np.ma.masked_where(~zoom_mask, zoom_mask)
            axes[1, col].imshow(overlay, cmap="autumn", alpha=0.6, origin="lower", vmin=0, vmax=1)
        axes[1, col].set_title("lesion zoom", fontsize=9)
        axes[1, col].axis("off")

    figure.suptitle(f"scan_{case_id} | {category} | GT voxels={int((mask > 0).sum())}", fontsize=12)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    base_dir = Path(args.config).expanduser().resolve().parent
    fold = int(args.fold if args.fold is not None else config.training.fold)
    spacing = tuple(args.target_spacing or config.data.target_spacing)
    output_dir = (
        resolve_path(base_dir, args.output_dir)
        if args.output_dir
        else resolve_path(base_dir, config.paths.work_dir) / "preprocessing_qa" / f"fold_{fold}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _select_records(config, base_dir, fold, args.split)
    infos = build_case_infos(records, split=args.split)
    if args.case_ids:
        wanted = set(args.case_ids)
        infos = [info for info in infos if info.record.case_id in wanted or f"scan_{info.record.case_id}" in wanted]
    else:
        categories = set(args.categories)
        infos = [info for info in infos if info.category in categories]
    if args.max_cases is not None:
        infos = infos[: int(args.max_cases)]

    print(f"[INFO] Cases: {len(infos)} | split={args.split} | fold={fold} | spacing={spacing}")
    print(f"[INFO] Methods: {args.methods}")
    rows: list[dict[str, object]] = []
    for info in infos:
        record = info.record
        t1, _ = _resample_image(record.t1_path, spacing, order=1)
        lesion, _ = _resample_image(record.lesion_path, spacing, order=0)
        lesion = (lesion > 0).astype(np.uint8)
        arrays = {method: apply_method(t1, method) for method in args.methods}
        for method, array in arrays.items():
            metric = compute_metrics(record.case_id, method, array, lesion)
            metric["category"] = info.category
            rows.append(metric)
        save_case_figure(
            case_id=record.case_id,
            category=info.category,
            arrays=arrays,
            mask=lesion,
            output_path=output_dir / "figures" / f"scan_{record.case_id}_{info.category}.png",
            zoom_size=int(args.zoom_size),
        )

    metrics_path = output_dir / "preprocessing_lesion_metrics.csv"
    write_csv(metrics_path, rows)
    print(f"[INFO] Metrics: {metrics_path}")
    print(f"[INFO] Figures: {output_dir / 'figures'}")


if __name__ == "__main__":
    main()
