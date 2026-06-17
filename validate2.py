"""
AIMS-TBI 2026 — Segmentation Task (Ensemble)
=============================================
Runs sequentially: load → predict → unload → next checkpoint

LOCAL MODE  (default):
    python inference.py --input /path/to/scan_XXXX_T1.nii.gz
    python inference.py --input /path/to/scan_XXXX_T1.nii.gz \
                        --lesion /path/to/scan_XXXX_Lesion.nii.gz

GRAND CHALLENGE MODE (Docker):
    python inference.py
    Reads  from /input/images/t1-brain-mri/<uuid>.mha
    Writes to   /output/images/tbi-segmentation/<uuid>.mha
    Models from /opt/ml/model/*.pt + plans.json
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import tempfile
import time
from pathlib import Path

# torch first — avoids torch.utils AttributeError on some systems
import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from nibabel.processing import resample_from_to
from scipy.ndimage import distance_transform_edt, binary_erosion

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Local paths ───────────────────────────────────────────────────────────────
LOCAL_MODEL_DIR  = Path(
    "/data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning"
    "/checkpoints/trained_models"
)
LOCAL_PLANS_PATH = Path(
    "/data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning"
    "/MultiTalentV2_pretrained/Dataset617_nativect"
    "/MultiTalent_trainer_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres"
    "/fold_all/nnUNetResEncUNetL1x1x1_Plans_znorm_bs24.json"
)
LOCAL_OUTPUT_DIR = Path(
    "/data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning/predictions/ensemble"
)

# ── Grand Challenge I/O paths (Docker) ───────────────────────────────────────
GC_INPUT_DIR  = Path("/input/images/t1-brain-mri")
GC_OUTPUT_DIR = Path("/output/images/tbi-segmentation")
GC_MODEL_DIR  = Path("/opt/ml/model")

# ── Inference parameters (from config.yml) ────────────────────────────────────
TARGET_SPACING = (1.0, 1.0, 1.0)
ROI_SIZE       = (192, 192, 192)
OVERLAP        = 0.5
SW_BATCH_SIZE  = 2
IN_CHANNELS    = 1
OUT_CHANNELS   = 2
NORMALIZE_FG   = True
INCLUDE_DMRI   = False
DMRI_REDUCE    = "mean"
DMRI_B0_THRESH = 50.0
USE_AMP        = True


# ── Voxel size categories ─────────────────────────────────────────────────────
def _voxel_category(n: int) -> str:
    if n == 0:   return "empty"
    if n < 1000: return "tiny"
    if n < 5000: return "small"
    return "large"


# ── Metrics ───────────────────────────────────────────────────────────────────

def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = (pred > 0).astype(np.uint8)
    gt_b   = (gt   > 0).astype(np.uint8)
    tp  = int((pred_b & gt_b).sum())
    fp  = int((pred_b & ~gt_b.astype(bool)).sum())
    fn  = int((~pred_b.astype(bool) & gt_b).sum())
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 1.0   # both empty → perfect


def _surface_distances(pred: np.ndarray,
                        gt:   np.ndarray,
                        spacing: tuple[float, ...]) -> tuple[float, float] | None:
    """Returns (HD95_mm, ASSD_mm) or None if one mask is empty."""
    pred_b = pred.astype(bool)
    gt_b   = gt.astype(bool)

    if not pred_b.any() and not gt_b.any():
        return (0.0, 0.0)
    if not pred_b.any() or not gt_b.any():
        return None

    d_p2g = distance_transform_edt(~gt_b,   sampling=spacing)
    d_g2p = distance_transform_edt(~pred_b, sampling=spacing)

    def _surface(m: np.ndarray) -> np.ndarray:
        return m & ~binary_erosion(m)

    all_d = np.concatenate([
        d_p2g[_surface(pred_b)],
        d_g2p[_surface(gt_b)],
    ])
    return float(np.percentile(all_d, 95)), float(np.mean(all_d))


def _compute_metrics(pred:    np.ndarray,
                     gt:      np.ndarray,
                     spacing: tuple[float, ...]) -> dict:
    pred_b = (pred > 0).astype(np.uint8)
    gt_b   = (gt   > 0).astype(np.uint8)

    tp = int((pred_b & gt_b.astype(bool)).sum())
    fp = int((pred_b.astype(bool) & ~gt_b.astype(bool)).sum())
    fn = int((~pred_b.astype(bool) & gt_b.astype(bool)).sum())
    tn = int((~pred_b.astype(bool) & ~gt_b.astype(bool)).sum())

    denom_dice = 2 * tp + fp + fn
    dice  = (2 * tp / denom_dice) if denom_dice > 0 else 1.0
    prec  = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec   = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    gt_vox   = int(gt_b.sum())
    pred_vox = int(pred_b.sum())

    boundary = _surface_distances(pred_b, gt_b, spacing)
    hd95 = boundary[0] if boundary is not None else None
    assd = boundary[1] if boundary is not None else None

    return {
        "dice":       dice,
        "precision":  prec,
        "recall":     rec,
        "tp":         tp,
        "fp":         fp,
        "fn":         fn,
        "tn":         tn,
        "gt_voxels":  gt_vox,
        "pred_voxels": pred_vox,
        "gt_category":   _voxel_category(gt_vox),
        "pred_category": _voxel_category(pred_vox),
        "hd95_mm":    hd95,
        "assd_mm":    assd,
    }


def _print_metrics(m: dict, scan_id: str) -> None:
    hd95_str = f"{m['hd95_mm']:.2f}" if m["hd95_mm"] is not None else "N/A"
    assd_str = f"{m['assd_mm']:.2f}" if m["assd_mm"] is not None else "N/A"
    print()
    print("╔══════════════════════════════════════════════════╗")
    print(f"║  Metrics — {scan_id:<38}║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Dice        : {m['dice']:.4f}                            ║")
    print(f"║  Precision   : {m['precision']:.4f}                            ║")
    print(f"║  Recall      : {m['recall']:.4f}                            ║")
    print(f"║  HD95        : {hd95_str:<36}║")
    print(f"║  ASSD        : {assd_str:<36}║")
    print(f"║  GT voxels   : {m['gt_voxels']:>10,}  [{m['gt_category']:<6}]          ║")
    print(f"║  Pred voxels : {m['pred_voxels']:>10,}  [{m['pred_category']:<6}]          ║")
    print(f"║  TP={m['tp']:>8,}  FP={m['fp']:>8,}  FN={m['fn']:>8,}         ║")
    print("╚══════════════════════════════════════════════════╝")


def _save_metrics_csv(m: dict, scan_id: str, output_dir: Path) -> Path:
    csv_path = output_dir / f"{scan_id}_ensemble_metrics.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        writer.writerow(["scan_id",       scan_id])
        writer.writerow(["dice",          f"{m['dice']:.6f}"])
        writer.writerow(["precision",     f"{m['precision']:.6f}"])
        writer.writerow(["recall",        f"{m['recall']:.6f}"])
        writer.writerow(["hd95_mm",       f"{m['hd95_mm']:.3f}" if m["hd95_mm"] is not None else "N/A"])
        writer.writerow(["assd_mm",       f"{m['assd_mm']:.3f}" if m["assd_mm"] is not None else "N/A"])
        writer.writerow(["gt_voxels",     m["gt_voxels"]])
        writer.writerow(["pred_voxels",   m["pred_voxels"]])
        writer.writerow(["gt_category",   m["gt_category"]])
        writer.writerow(["pred_category", m["pred_category"]])
        writer.writerow(["tp",            m["tp"]])
        writer.writerow(["fp",            m["fp"]])
        writer.writerow(["fn",            m["fn"]])
        writer.writerow(["tn",            m["tn"]])
    print(f"[INFO] Metrics CSV  : {csv_path}")
    return csv_path


def _save_preview(t1:          np.ndarray,
                  pred:        np.ndarray,
                  gt:          np.ndarray | None,
                  scan_id:     str,
                  metrics:     dict | None,
                  output_dir:  Path) -> None:
    """3-column preview: T1 | GT overlay | Pred overlay (axial best slice)."""

    # Pick slice with most content
    combined = pred.sum(axis=(0, 1))
    if gt is not None:
        combined = combined + gt.sum(axis=(0, 1))
    z = int(np.argmax(combined)) if combined.max() > 0 else t1.shape[2] // 2

    lo  = np.percentile(t1[t1 > 0], 1)  if (t1 > 0).any() else 0
    hi  = np.percentile(t1[t1 > 0], 99) if (t1 > 0).any() else 1

    n_cols = 3 if gt is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5),
                             facecolor="#111111")

    def _show(ax, base, overlay, title):
        ax.imshow(base[:, :, z].T, cmap="gray", origin="lower",
                  vmin=lo, vmax=hi)
        if overlay is not None and overlay[:, :, z].any():
            masked = np.ma.masked_where(overlay[:, :, z].T == 0,
                                         overlay[:, :, z].T)
            ax.imshow(masked, cmap="autumn", alpha=0.5,
                      origin="lower", vmin=0, vmax=1)
        ax.set_title(title, color="white", fontsize=9)
        ax.axis("off")

    _show(axes[0], t1,   None, f"T1  z={z}")
    if gt is not None:
        _show(axes[1], t1, gt,   f"Ground Truth\n{(gt>0).sum():,} vox [{_voxel_category(int((gt>0).sum()))}]")
        _show(axes[2], t1, pred, f"Ensemble Pred\n{(pred>0).sum():,} vox [{_voxel_category(int((pred>0).sum()))}]")
    else:
        _show(axes[1], t1, pred, f"Ensemble Pred\n{(pred>0).sum():,} vox [{_voxel_category(int((pred>0).sum()))}]")

    # Metrics subtitle
    if metrics is not None:
        hd95_str = f"{metrics['hd95_mm']:.1f}mm" if metrics["hd95_mm"] is not None else "N/A"
        title_str = (
            f"{scan_id}  |  Dice={metrics['dice']:.4f}  "
            f"Prec={metrics['precision']:.3f}  Rec={metrics['recall']:.3f}  "
            f"HD95={hd95_str}  ASSD="
            + (f"{metrics['assd_mm']:.1f}mm" if metrics["assd_mm"] is not None else "N/A")
        )
        fig.suptitle(title_str, color="white", fontsize=8, y=1.01)

    fig.tight_layout()
    out_path = output_dir / f"{scan_id}_ensemble_preview.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Preview PNG  : {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AIMS-TBI ensemble inference — local or Grand Challenge mode"
    )
    parser.add_argument(
        "--input", "-i", default=None,
        help="Path to T1 .nii.gz / .mha (local mode). Omit for Grand Challenge.",
    )
    parser.add_argument(
        "--lesion", "-l", default=None,
        help=(
            "Path to ground-truth lesion mask .nii.gz (optional, local only). "
            "When provided: computes Dice, HD95, ASSD, precision, recall "
            "and saves a side-by-side preview PNG."
        ),
    )
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help=f"Output directory (default: {LOCAL_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model-dir", "-m", default=None,
        help=f"Directory with *.pt checkpoints (default: {LOCAL_MODEL_DIR})",
    )
    parser.add_argument(
        "--plans", "-p", default=None,
        help=f"Path to plans.json (default: {LOCAL_PLANS_PATH})",
    )
    parser.add_argument(
        "--overlap", type=float, default=OVERLAP,
        help=f"Sliding window overlap (default: {OVERLAP})",
    )
    parser.add_argument(
        "--sw-batch-size", type=int, default=SW_BATCH_SIZE,
        help=f"Sliding window batch size (default: {SW_BATCH_SIZE})",
    )
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable AMP (mixed precision)",
    )
    parser.add_argument(
        "--no-preview", action="store_true",
        help="Skip saving the preview PNG",
    )
    return parser.parse_args()


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _find_input_file(directory: Path) -> Path:
    for pattern in ("*.mha", "*.tif", "*.nii.gz", "*.nii"):
        files = sorted(directory.glob(pattern))
        if files:
            return files[0]
    raise FileNotFoundError(f"No input image found in {directory}")


def _find_checkpoints(model_dir: Path) -> list[Path]:
    pts = sorted(model_dir.glob("*.pt"))
    if not pts:
        raise FileNotFoundError(
            f"No *.pt checkpoints found in {model_dir}.\n"
            "Expected: best_tr_f0_hpzvfifp.pt, best_val_f2_dk0lavnm.pt …"
        )
    print(f"[INFO] Found {len(pts)} checkpoint(s) in {model_dir}:")
    for p in pts:
        print(f"         {p.name}  ({p.stat().st_size / 1e6:.0f} MB)")
    return pts


def _read_as_nifti_tmp(input_path: Path, tmp_dir: Path) -> Path:
    suffix = "".join(input_path.suffixes).lower()
    if suffix in (".nii.gz", ".nii"):
        return input_path
    nifti_path = tmp_dir / "t1.nii.gz"
    print(f"[INFO] Converting {input_path.suffix} → NIfTI …")
    sitk.WriteImage(sitk.ReadImage(str(input_path)), str(nifti_path))
    return nifti_path


def _load_gt_mask(lesion_path: Path,
                  original_nifti: nib.Nifti1Image) -> np.ndarray:
    """Load GT mask, resample to original T1 space, return binary uint8 array."""
    gt_nifti  = nib.load(str(lesion_path))
    gt_res    = resample_from_to(gt_nifti, original_nifti, order=0)
    gt_arr    = np.asarray(gt_res.dataobj, dtype=np.uint8)
    gt_arr    = (gt_arr > 0).astype(np.uint8)
    print(f"[INFO] GT mask      : {lesion_path.name}  "
          f"({int(gt_arr.sum()):,} lesion voxels  "
          f"[{_voxel_category(int(gt_arr.sum()))}])")
    return gt_arr


def _write_mask(mask_arr:       np.ndarray,
                mask_affine:    np.ndarray,
                original_nifti: nib.Nifti1Image,
                input_file:     Path,
                output_dir:     Path,
                gc_mode:        bool) -> tuple[Path, np.ndarray]:
    """Returns (output_path, restored_array_in_original_space)."""
    pred_nifti   = nib.Nifti1Image(mask_arr, affine=mask_affine)
    restored     = resample_from_to(pred_nifti, original_nifti, order=0)
    restored_arr = np.asarray(restored.dataobj, dtype=np.int8)

    pred_voxels = int((restored_arr > 0).sum())
    print(f"[INFO] Pred voxels (original space) : "
          f"{pred_voxels:,}  [{_voxel_category(pred_voxels)}]")

    out_stem = input_file.name.split(".")[0]

    if gc_mode:
        out_path = output_dir / f"{out_stem}.mha"
        ref      = sitk.ReadImage(str(input_file))
        data_t   = restored_arr.transpose(2, 1, 0)
        out_img  = sitk.GetImageFromArray(data_t)
        out_img.CopyInformation(ref)
        sitk.WriteImage(out_img, str(out_path), useCompression=False)
    else:
        out_path = output_dir / f"{out_stem}_ensemble_lesion.nii.gz"
        nib.save(nib.Nifti1Image(restored_arr, restored.affine), str(out_path))

    print(f"[INFO] Mask saved   : {out_path}")
    return out_path, restored_arr


# ── Model helpers ─────────────────────────────────────────────────────────────

def _build_model(plans_path: Path, device: torch.device) -> torch.nn.Module:
    from multitalent_tbi.model import build_backbone
    model = build_backbone(
        plans_path=plans_path, in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS, deep_supervision=False,
    )
    return model.to(memory_format=torch.channels_last_3d).to(device)


def _load_weights(model: torch.nn.Module, ckpt_path: Path) -> torch.nn.Module:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    cleaned = {}
    for k, v in state_dict.items():
        for prefix in ("module.", "model."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        cleaned[k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"         [WARN] Missing keys    : {len(missing)}  {missing[:3]}")
    if unexpected:
        print(f"         [WARN] Unexpected keys : {len(unexpected)}  {unexpected[:3]}")
    return model


def _unload_model(model: torch.nn.Module) -> None:
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess(nifti_t1: Path, tmp_dir: Path) -> tuple:
    from multitalent_tbi.data import CaseRecord, load_case_cached

    dummy_lesion = tmp_dir / "dummy_lesion.nii.gz"
    ref = nib.load(str(nifti_t1))
    nib.save(
        nib.Nifti1Image(np.zeros(ref.shape[:3], dtype=np.uint8), ref.affine),
        str(dummy_lesion),
    )
    record = CaseRecord(
        case_id="case", t1_path=nifti_t1, lesion_path=dummy_lesion,
    )
    image, mask, image_affine, _ = load_case_cached(
        case=record,
        target_spacing=TARGET_SPACING,
        include_dmri=INCLUDE_DMRI,
        dmri_reduce=DMRI_REDUCE,
        dmri_b0_threshold=DMRI_B0_THRESH,
        normalize_foreground_only=NORMALIZE_FG,
        cache_dir=None,
    )
    return image, mask, image_affine, ref


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args    = parse_args()
    gc_mode = args.input is None

    if gc_mode:
        input_file = _find_input_file(GC_INPUT_DIR)
        output_dir = GC_OUTPUT_DIR
        model_dir  = GC_MODEL_DIR
        plans_path = GC_MODEL_DIR / "plans.json"
        print("[INFO] Mode: Grand Challenge")
    else:
        input_file = Path(args.input).expanduser().resolve()
        output_dir = Path(args.output_dir) if args.output_dir else LOCAL_OUTPUT_DIR
        model_dir  = Path(args.model_dir)  if args.model_dir  else LOCAL_MODEL_DIR
        plans_path = Path(args.plans)      if args.plans       else LOCAL_PLANS_PATH
        print("[INFO] Mode: Local")

    output_dir.mkdir(parents=True, exist_ok=True)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and not args.no_amp
    print(f"[INFO] Device : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"[INFO] GPU    : {props.name}  ({props.total_memory/1e9:.1f} GB)")
    print(f"[INFO] AMP    : {use_amp}")

    if not input_file.exists():
        sys.exit(f"[ERROR] Input not found: {input_file}")
    if not plans_path.exists():
        sys.exit(f"[ERROR] plans.json not found: {plans_path}")

    checkpoints = _find_checkpoints(model_dir)
    print(f"[INFO] Input  : {input_file}")
    print(f"[INFO] Plans  : {plans_path}")
    print(f"[INFO] Output : {output_dir}")

    t_start = time.time()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        nifti_t1 = _read_as_nifti_tmp(input_file, tmp_dir)

        print("\n[INFO] Preprocessing …")
        image, _, image_affine, original_nifti = _preprocess(nifti_t1, tmp_dir)
        print(f"[INFO] Shape : {image.shape}  "
              f"range=[{image.min():.2f}, {image.max():.2f}]")

        # ── Sequential ensemble ───────────────────────────────────────────────
        accumulated_probs: np.ndarray | None = None

        for idx, ckpt_path in enumerate(checkpoints, 1):
            print(f"\n{'─'*55}")
            print(f"[{idx}/{len(checkpoints)}] Loading    : {ckpt_path.name}")
            t0 = time.time()

            model = _build_model(plans_path, device)
            model = _load_weights(model, ckpt_path)
            model.eval()

            print(f"[{idx}/{len(checkpoints)}] Predicting …")
            from multitalent_tbi.infer import predict_logits
            with torch.no_grad():
                logits = predict_logits(
                    model=model, image=image, roi_size=ROI_SIZE,
                    overlap=args.overlap, batch_size=args.sw_batch_size,
                    device=device, use_amp=use_amp and device.type == "cuda",
                )

            probs = torch.softmax(
                torch.from_numpy(logits).float(), dim=0
            ).numpy()

            elapsed = time.time() - t0
            print(f"[{idx}/{len(checkpoints)}] Done  {elapsed:.1f}s | "
                  f"lesion>0.5: {int((probs[1]>0.5).sum()):,} vox | "
                  f"peak: {probs[1].max():.3f}")

            accumulated_probs = probs.copy() if accumulated_probs is None \
                                else accumulated_probs + probs

            print(f"[{idx}/{len(checkpoints)}] Unloading …")
            _unload_model(model)

        print(f"\n{'─'*55}")

        # ── Average + argmax ──────────────────────────────────────────────────
        mean_probs = accumulated_probs / len(checkpoints)
        prediction = np.argmax(mean_probs, axis=0).astype(np.uint8)
        print(f"[INFO] Ensemble complete  ({len(checkpoints)} models)")

        # ── Write mask ────────────────────────────────────────────────────────
        _, restored_pred = _write_mask(
            mask_arr=prediction,
            mask_affine=np.asarray(image_affine),
            original_nifti=original_nifti,
            input_file=input_file,
            output_dir=output_dir,
            gc_mode=gc_mode,
        )

        # ── Metrics (only if --lesion given and local mode) ───────────────────
        scan_id = input_file.name.split(".")[0]
        metrics = None
        gt_arr  = None

        if args.lesion and not gc_mode:
            lesion_path = Path(args.lesion).expanduser().resolve()
            if not lesion_path.exists():
                print(f"[WARN] Lesion file not found: {lesion_path} — skipping metrics")
            else:
                gt_arr = _load_gt_mask(lesion_path, original_nifti)

                # spacing from original NIfTI affine diagonal (mm)
                affine  = original_nifti.affine
                spacing = tuple(abs(float(affine[i, i])) for i in range(3))

                metrics = _compute_metrics(restored_pred, gt_arr, spacing)
                _print_metrics(metrics, scan_id)
                _save_metrics_csv(metrics, scan_id, output_dir)

        elif args.lesion and gc_mode:
            print("[WARN] --lesion ignored in Grand Challenge mode")

        # ── Preview PNG ───────────────────────────────────────────────────────
        if not args.no_preview and not gc_mode:
            t1_data = np.asarray(original_nifti.dataobj, dtype=np.float32)
            _save_preview(
                t1=t1_data,
                pred=restored_pred.astype(np.uint8),
                gt=gt_arr,
                scan_id=scan_id,
                metrics=metrics,
                output_dir=output_dir,
            )

    total = time.time() - t_start
    print(f"\n[INFO] Total time : {total:.1f}s  ({total/60:.1f} min)")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()

# ── Usage examples ─────────────────────────────────────────────────────────────
#
# No ground truth (prediction only):
#   python inference.py --input MICCAI_AIMS_TBI/scan_0440_T1.nii.gz
#
# With ground truth (computes Dice, HD95, ASSD, saves CSV + preview PNG):
#   python inference.py \
#       --input   MICCAI_AIMS_TBI/scan_0440_T1.nii.gz \
#       --lesion  MICCAI_AIMS_TBI/scan_0440_Lesion.nii.gz
#
# Higher overlap:
#   python inference.py --input scan.nii.gz --lesion mask.nii.gz --overlap 0.75
#
# Custom output dir:
#   python inference.py --input scan.nii.gz --lesion mask.nii.gz \
#       --output-dir ./my_results