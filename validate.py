from __future__ import annotations

import argparse
import csv
from math import isnan
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to
from scipy.ndimage import distance_transform_edt

from multitalent_tbi.config import load_config, resolve_path
from multitalent_tbi.data import load_case_cached, load_nifti_robust
from multitalent_tbi.engine import build_dataloaders, build_model, dice_score
from multitalent_tbi.infer import predict_logits
from multitalent_tbi.losses import dice_ce_loss


# ── voxel size thresholds (must match make_stratification_labels in data.py) ──
def _voxel_category(n_voxels: int) -> str:
    if n_voxels == 0:
        return "empty"
    elif n_voxels < 1000:
        return "tiny"
    elif n_voxels < 5000:
        return "small"
    else:
        return "large"


# ── boundary distance metrics ─────────────────────────────────────────────────

def _surface_distances(pred: np.ndarray, target: np.ndarray,
                       voxel_spacing: tuple[float, ...]) -> tuple[float, float, float] | None:
    """
    Compute HD95 and ASSD between binary pred and target masks.
    voxel_spacing: (sx, sy, sz) in mm — taken from the NIfTI affine diagonal.
    Returns (HD95, ASSD, mean_surface_distance) or None if either mask is empty.
    """
    pred_bool   = pred.astype(bool)
    target_bool = target.astype(bool)

    if not pred_bool.any() and not target_bool.any():
        return (0.0, 0.0, 0.0)          # both empty → perfect agreement
    if not pred_bool.any() or not target_bool.any():
        return None                      # one empty → undefined boundary distance

    # distance transform: distance from every voxel to the nearest True voxel
    # sampling= applies voxel spacing so distances are in mm
    dist_pred_to_target = distance_transform_edt(~target_bool, sampling=voxel_spacing)
    dist_target_to_pred = distance_transform_edt(~pred_bool,   sampling=voxel_spacing)

    # surface voxels = foreground voxels that border the background
    def surface_voxels(mask: np.ndarray) -> np.ndarray:
        from scipy.ndimage import binary_erosion
        return mask & ~binary_erosion(mask)

    pred_surface   = surface_voxels(pred_bool)
    target_surface = surface_voxels(target_bool)

    # distances from each surface to the other surface
    d_p2t = dist_pred_to_target[pred_surface]    # each pred surface voxel → nearest target voxel
    d_t2p = dist_target_to_pred[target_surface]  # each target surface voxel → nearest pred voxel

    all_distances = np.concatenate([d_p2t, d_t2p])

    hd95 = float(np.percentile(all_distances, 95))
    assd = float(np.mean(all_distances))

    return hd95, assd


# ── visualisation ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a trained MultiTalentV2 checkpoint on the fold split."
    )
    parser.add_argument("--config",       default="config.yml")
    parser.add_argument("--fold",         type=int, default=0)
    parser.add_argument("--checkpoint",   default=None)
    parser.add_argument("--output-dir",   default=None)
    parser.add_argument("--metrics-file", default=None)
    parser.add_argument("--no-save-predictions", action="store_true")
    return parser.parse_args()


def _select_slice_index(image, target, prediction) -> int:
    combined = target.sum(axis=(0, 1)) + prediction.sum(axis=(0, 1))
    if combined.max() > 0:
        return int(np.argmax(combined))
    return int(image.shape[2] // 2)


def _save_side_by_side_preview(output_path, image, target, prediction, case_id,
                                gt_voxels, pred_voxels, gt_cat, pred_cat,
                                dice, hd95, assd) -> None:
    slice_index = _select_slice_index(image, target, prediction)
    image_slice      = image[:, :, slice_index]
    target_slice     = target[:, :, slice_index]
    prediction_slice = prediction[:, :, slice_index]

    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    panels = [
        ("T1", image_slice, None),
        (f"Ground Truth\n{gt_voxels} voxels ({gt_cat})", image_slice, target_slice),
        (f"Prediction\n{pred_voxels} voxels ({pred_cat})", image_slice, prediction_slice),
    ]
    for axis, (title, base_slice, overlay_slice) in zip(axes, panels):
        axis.imshow(base_slice.T, cmap="gray", origin="lower")
        if overlay_slice is not None and np.any(overlay_slice > 0):
            axis.imshow(
                np.ma.masked_where(overlay_slice.T <= 0, overlay_slice.T),
                cmap="autumn", alpha=0.45, origin="lower",
            )
        axis.set_title(title, fontsize=9)
        axis.axis("off")

    metrics_str = (
        f"Case: {case_id} | z={slice_index} | "
        f"Dice={dice:.3f}  HD95={hd95:.1f}mm  ASSD={assd:.1f}mm"
        if hd95 is not None
        else f"Case: {case_id} | z={slice_index} | Dice={dice:.3f}  HD95=N/A  ASSD=N/A"
    )
    figure.suptitle(metrics_str, fontsize=9)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_category_summary_plot(output_path: Path, rows: list[dict]) -> None:
    """Bar chart: mean Dice per size category."""
    categories = ["empty", "tiny", "small", "large"]
    cat_dice: dict[str, list[float]] = {c: [] for c in categories}
    cat_hd95: dict[str, list[float]] = {c: [] for c in categories}
    cat_assd: dict[str, list[float]] = {c: [] for c in categories}

    for r in rows:
        cat = r["gt_category"]
        cat_dice[cat].append(r["dice"])
        if r["hd95"] is not None:
            cat_hd95[cat].append(r["hd95"])
            cat_assd[cat].append(r["assd"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, (metric_name, cat_data) in zip(axes, [
        ("Mean Dice",  cat_dice),
        ("Mean HD95 (mm)", cat_hd95),
        ("Mean ASSD (mm)", cat_assd),
    ]):
        means  = [float(np.mean(cat_data[c])) if cat_data[c] else 0.0 for c in categories]
        counts = [len(cat_data[c]) for c in categories]
        colors = ["#aaaaaa", "#e06c75", "#e5c07b", "#61afef"]
        bars = ax.bar(categories, means, color=colors, edgecolor="white", linewidth=0.5)
        for bar, n, m in zip(bars, counts, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{m:.3f}\n(n={n})",
                ha="center", va="bottom", fontsize=8,
            )
        ax.set_title(metric_name, fontsize=10)
        ax.set_ylim(0, max(means) * 1.3 + 0.05 if any(means) else 1.0)
        ax.set_ylabel(metric_name)

    fig.suptitle("Per-category validation metrics", fontsize=11)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    config = load_config(args.config)
    base_dir = Path(args.config).expanduser().resolve().parent

    _, val_loader, _, _, _ = build_dataloaders(config, args.fold, base_dir)
    records = list(val_loader.dataset.records)

    checkpoint_path = resolve_path(
        base_dir,
        args.checkpoint or (
            resolve_path(base_dir, config.paths.work_dir)
            / f"fold_{args.fold}" / "best.pt"
        ),
    )

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model       = build_model(config, base_dir).to(device)
    payload     = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=False)
    amp_enabled = bool(config.inference.use_amp) and device.type == "cuda"
    class_weights = torch.tensor(
        config.training.class_weights, dtype=torch.float32, device=device
    )

    export_dir = (
        resolve_path(base_dir, args.output_dir)
        if args.output_dir
        else resolve_path(base_dir, config.paths.work_dir)
             / f"fold_{args.fold}" / "validation_exports"
    )
    predictions_dir = export_dir / "nifti"
    previews_dir    = export_dir / "previews"
    export_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_save_predictions:
        predictions_dir.mkdir(parents=True, exist_ok=True)
        previews_dir.mkdir(parents=True, exist_ok=True)

    # ── per-case results ──────────────────────────────────────────────────────
    rows:        list[dict]  = []
    dice_scores: list[float] = []
    losses:      list[float] = []

    for record in records:
        original_image = load_nifti_robust(record.t1_path)

        image, mask, image_affine, _ = load_case_cached(
            case=record,
            target_spacing=tuple(config.data.target_spacing),
            include_dmri=bool(config.data.include_dmri),
            dmri_reduce=str(config.data.dmri_reduce),
            dmri_b0_threshold=float(config.data.dmri_b0_threshold),
            normalize_foreground_only=bool(config.data.normalize_foreground_only),
            cache_dir=resolve_path(base_dir, config.paths.cache_dir)
                      if config.data.cache_preprocessed else None,
        )

        logits = predict_logits(
            model=model,
            image=image,
            roi_size=tuple(config.inference.roi_size),
            overlap=float(config.inference.overlap),
            batch_size=int(config.inference.sw_batch_size),
            device=device,
            use_amp=amp_enabled,
        )

        logits_tensor = torch.from_numpy(logits).unsqueeze(0).to(device=device, dtype=torch.float32)
        mask_tensor   = torch.from_numpy(mask.copy()).unsqueeze(0).to(device=device, dtype=torch.long)
        loss = dice_ce_loss(logits_tensor, mask_tensor, class_weights=class_weights)

        probabilities = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
        prediction    = np.argmax(probabilities, axis=0).astype(np.uint8)

        # resample back to original space
        lesion_nifti        = nib.Nifti1Image(prediction, affine=np.asarray(image_affine))
        restored_prediction = resample_from_to(lesion_nifti, original_image, order=0)
        restored_pred_data  = np.asarray(restored_prediction.dataobj, dtype=np.uint8)

        restored_target     = resample_from_to(
            nib.Nifti1Image(mask.astype(np.uint8), affine=np.asarray(image_affine)),
            original_image, order=0,
        )
        restored_target_data = np.asarray(restored_target.dataobj, dtype=np.uint8)
        restored_image_data  = np.asarray(original_image.dataobj, dtype=np.float32)

        # ── voxel counts & categories ─────────────────────────────────────────
        gt_voxels   = int(np.count_nonzero(restored_target_data > 0))
        pred_voxels = int(np.count_nonzero(restored_pred_data > 0))
        gt_cat      = _voxel_category(gt_voxels)
        pred_cat    = _voxel_category(pred_voxels)

        # ── Dice ─────────────────────────────────────────────────────────────
        case_dice = dice_score(restored_pred_data, restored_target_data)

        # ── HD95 + ASSD in original voxel spacing (mm) ───────────────────────
        affine    = original_image.affine
        spacing   = tuple(abs(float(affine[i, i])) for i in range(3))   # (sx, sy, sz) mm
        boundary  = _surface_distances(restored_pred_data, restored_target_data, spacing)
        hd95_val  = boundary[0] if boundary is not None else None
        assd_val  = boundary[1] if boundary is not None else None

        losses.append(float(loss.detach().cpu()))
        dice_scores.append(case_dice)

        rows.append({
            "case_id":      record.case_id,
            "loss":         float(loss.detach().cpu()),
            "dice":         case_dice,
            "gt_voxels":    gt_voxels,
            "pred_voxels":  pred_voxels,
            "gt_category":  gt_cat,
            "pred_category": pred_cat,
            "hd95":         hd95_val,
            "assd":         assd_val,
        })

        status = (
            f"HD95={hd95_val:.1f}mm  ASSD={assd_val:.1f}mm"
            if hd95_val is not None else "HD95=N/A (one mask empty)"
        )
        print(
            f"[{record.case_id}]  Dice={case_dice:.4f}  {status}  "
            f"GT={gt_voxels}vx({gt_cat})  Pred={pred_voxels}vx({pred_cat})"
        )

        if not args.no_save_predictions:
            nib.save(
                restored_prediction,
                str(predictions_dir / f"scan_{record.case_id}_Lesion.nii.gz"),
            )
            _save_side_by_side_preview(
                output_path=previews_dir / f"scan_{record.case_id}_preview.png",
                image=restored_image_data,
                target=restored_target_data,
                prediction=restored_pred_data,
                case_id=record.case_id,
                gt_voxels=gt_voxels,
                pred_voxels=pred_voxels,
                gt_cat=gt_cat,
                pred_cat=pred_cat,
                dice=case_dice,
                hd95=hd95_val,
                assd=assd_val,
            )

    # ── aggregates ────────────────────────────────────────────────────────────
    val_loss = float(np.mean(losses)) if losses else float("nan")
    val_dice = float(np.mean(dice_scores)) if dice_scores else float("nan")

    valid_hd95 = [r["hd95"] for r in rows if r["hd95"] is not None]
    valid_assd = [r["assd"] for r in rows if r["assd"] is not None]
    mean_hd95  = float(np.mean(valid_hd95)) if valid_hd95 else float("nan")
    mean_assd  = float(np.mean(valid_assd)) if valid_assd else float("nan")

    # per-category aggregates
    categories = ["empty", "tiny", "small", "large"]
    cat_summary: dict[str, dict] = {}
    for cat in categories:
        cat_rows  = [r for r in rows if r["gt_category"] == cat]
        cat_dice  = [r["dice"] for r in cat_rows]
        cat_hd95  = [r["hd95"] for r in cat_rows if r["hd95"] is not None]
        cat_assd  = [r["assd"] for r in cat_rows if r["assd"] is not None]
        cat_summary[cat] = {
            "n":         len(cat_rows),
            "mean_dice": float(np.mean(cat_dice)) if cat_dice else float("nan"),
            "mean_hd95": float(np.mean(cat_hd95)) if cat_hd95 else float("nan"),
            "mean_assd": float(np.mean(cat_assd)) if cat_assd else float("nan"),
        }

    # ── save category summary chart ───────────────────────────────────────────
    if not args.no_save_predictions:
        _save_category_summary_plot(export_dir / "category_summary.png", rows)
        print(f"Saved category summary plot: {export_dir / 'category_summary.png'}")

    # ── write CSV ─────────────────────────────────────────────────────────────
    metrics_file = (
        resolve_path(base_dir, args.metrics_file)
        if args.metrics_file
        else export_dir / "validation_metrics.csv"
    )
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    with metrics_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)

        # ── overall summary ───────────────────────────────────────────────────
        writer.writerow(["=== OVERALL ==="])
        writer.writerow(["fold", "checkpoint", "n_cases",
                         "mean_val_loss", "mean_val_dice",
                         "mean_hd95_mm", "mean_assd_mm",
                         "n_hd95_computed"])
        writer.writerow([
            args.fold, str(checkpoint_path), len(rows),
            f"{val_loss:.6f}", f"{val_dice:.6f}",
            f"{mean_hd95:.3f}", f"{mean_assd:.3f}",
            len(valid_hd95),
        ])
        writer.writerow([])

        # ── per-category summary ──────────────────────────────────────────────
        writer.writerow(["=== PER CATEGORY (GT lesion size) ==="])
        writer.writerow(["category", "n", "mean_dice", "mean_hd95_mm", "mean_assd_mm"])
        for cat in categories:
            s = cat_summary[cat]
            writer.writerow([
                cat, s["n"],
                f"{s['mean_dice']:.4f}" if not isnan(s["mean_dice"]) else "N/A",
                f"{s['mean_hd95']:.3f}" if not isnan(s["mean_hd95"]) else "N/A",
                f"{s['mean_assd']:.3f}" if not isnan(s["mean_assd"]) else "N/A",
            ])
        writer.writerow([])

        # ── per-case detail ───────────────────────────────────────────────────
        writer.writerow(["=== PER CASE ==="])
        writer.writerow([
            "case_id", "loss", "dice",
            "gt_voxels", "gt_category",
            "pred_voxels", "pred_category",
            "hd95_mm", "assd_mm",
        ])
        for r in rows:
            writer.writerow([
                r["case_id"],
                f"{r['loss']:.6f}",
                f"{r['dice']:.6f}",
                r["gt_voxels"],
                r["gt_category"],
                r["pred_voxels"],
                r["pred_category"],
                f"{r['hd95']:.3f}" if r["hd95"] is not None else "N/A",
                f"{r['assd']:.3f}" if r["assd"] is not None else "N/A",
            ])

    # ── terminal summary ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Val loss   : {val_loss:.6f}")
    print(f"Val Dice   : {val_dice:.6f}")
    print(f"Mean HD95  : {mean_hd95:.3f} mm  (n={len(valid_hd95)})")
    print(f"Mean ASSD  : {mean_assd:.3f} mm")
    print()
    print(f"{'Category':<10} {'N':>5} {'Dice':>8} {'HD95(mm)':>10} {'ASSD(mm)':>10}")
    print("-" * 47)
    for cat in categories:
        s = cat_summary[cat]
        dice_str = f"{s['mean_dice']:.4f}" if not isnan(s["mean_dice"]) else "  N/A"
        hd95_str = f"{s['mean_hd95']:.3f}" if not isnan(s["mean_hd95"]) else "     N/A"
        assd_str = f"{s['mean_assd']:.3f}" if not isnan(s["mean_assd"]) else "     N/A"
        print(f"{cat:<10} {s['n']:>5} {dice_str:>8} {hd95_str:>10} {assd_str:>10}")
    print("=" * 60)
    print(f"Metrics CSV : {metrics_file}")


if __name__ == "__main__":
    main()

# python3 validate.py --config config.yml --fold 1 \
#   --checkpoint "/data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning/checkpoints/trained_models/best_val_f1_q8mwzqms.pt" \
#   --output-dir "/data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning/checkpoints/tbi_multitalentv2/fold_1/validation_exports_tr_f1_val_f1_q8mwzqms"