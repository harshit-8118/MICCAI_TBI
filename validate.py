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

from multitalent_tbi.config import load_config, resolve_path
from multitalent_tbi.data import load_case_cached, load_nifti_robust
from multitalent_tbi.engine import build_dataloaders, build_model, dice_score
from multitalent_tbi.infer import predict_logits
from multitalent_tbi.losses import dice_ce_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a trained MultiTalentV2 checkpoint on the fold split.")
    parser.add_argument("--config", default="config.yml", help="Path to the YAML config.")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to validate.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path to evaluate. Defaults to fold_<n>/best.pt.")
    parser.add_argument("--output-dir", default=None, help="Directory to save predictions and visualizations.")
    parser.add_argument("--metrics-file", default=None, help="Optional CSV file to write summary metrics to.")
    parser.add_argument("--no-save-predictions", action="store_true", help="Only compute metrics; do not save masks or PNGs.")
    return parser.parse_args()


def _select_slice_index(image: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> int:
    combined = target.sum(axis=(0, 1)) + prediction.sum(axis=(0, 1))
    if combined.max() > 0:
        return int(np.argmax(combined))
    return int(image.shape[2] // 2)


def _save_side_by_side_preview(
    output_path: Path,
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    case_id: str,
) -> None:
    slice_index = _select_slice_index(image, target, prediction)
    image_slice = image[:, :, slice_index]
    target_slice = target[:, :, slice_index]
    prediction_slice = prediction[:, :, slice_index]

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    panels = [
        ("T1", image_slice, None),
        ("Ground Truth", image_slice, target_slice),
        ("Prediction", image_slice, prediction_slice),
    ]
    for axis, (title, base_slice, overlay_slice) in zip(axes, panels):
        axis.imshow(base_slice.T, cmap="gray", origin="lower")
        if overlay_slice is not None and np.any(overlay_slice > 0):
            axis.imshow(np.ma.masked_where(overlay_slice.T <= 0, overlay_slice.T), cmap="autumn", alpha=0.45, origin="lower")
        axis.set_title(f"{title} | {case_id} | z={slice_index}")
        axis.axis("off")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    base_dir = Path(args.config).expanduser().resolve().parent

    _, val_loader = build_dataloaders(config, args.fold, base_dir)
    val_dataset = val_loader.dataset
    records = list(val_dataset.records)

    checkpoint_path = resolve_path(
        base_dir,
        args.checkpoint or (resolve_path(base_dir, config.paths.work_dir) / f"fold_{args.fold}" / "best.pt"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, base_dir).to(device)
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=False)

    amp_enabled = bool(config.inference.use_amp) and device.type == "cuda"
    class_weights = torch.tensor(config.training.class_weights, dtype=torch.float32, device=device)
    export_dir = (
        resolve_path(base_dir, args.output_dir)
        if args.output_dir
        else resolve_path(base_dir, config.paths.work_dir) / f"fold_{args.fold}" / "validation_exports"
    )
    predictions_dir = export_dir / "nifti"
    previews_dir = export_dir / "previews"
    export_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_save_predictions:
        predictions_dir.mkdir(parents=True, exist_ok=True)
        previews_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[list[object]] = []
    dice_scores: list[float] = []
    losses: list[float] = []
    for record in records:
        original_image = load_nifti_robust(record.t1_path)
        image, mask, image_affine, _ = load_case_cached(
            case=record,
            target_spacing=tuple(config.data.target_spacing),
            include_dmri=bool(config.data.include_dmri),
            dmri_reduce=str(config.data.dmri_reduce),
            dmri_b0_threshold=float(config.data.dmri_b0_threshold),
            normalize_foreground_only=bool(config.data.normalize_foreground_only),
            cache_dir=resolve_path(base_dir, config.paths.cache_dir) if config.data.cache_preprocessed else None,
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
        mask_tensor = torch.from_numpy(mask.copy()).unsqueeze(0).to(device=device, dtype=torch.long)
        loss = dice_ce_loss(logits_tensor, mask_tensor, class_weights=class_weights)
        probabilities = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
        prediction = np.argmax(probabilities, axis=0).astype(np.uint8)

        lesion_image = nib.Nifti1Image(prediction, affine=np.asarray(image_affine))
        restored_prediction = resample_from_to(lesion_image, original_image, order=0)
        restored_prediction_data = np.asarray(restored_prediction.dataobj, dtype=np.uint8)

        restored_target = resample_from_to(nib.Nifti1Image(mask.astype(np.uint8), affine=np.asarray(image_affine)), original_image, order=0)
        restored_target_data = np.asarray(restored_target.dataobj, dtype=np.uint8)
        restored_image = np.asarray(original_image.dataobj, dtype=np.float32)

        case_dice = dice_score(restored_prediction_data, restored_target_data)
        losses.append(float(loss.detach().cpu()))
        dice_scores.append(case_dice)
        summary_rows.append([record.case_id, float(loss.detach().cpu()), case_dice])

        if not args.no_save_predictions:
            prediction_path = predictions_dir / f"scan_{record.case_id}_Lesion.nii.gz"
            nib.save(restored_prediction, str(prediction_path))
            preview_path = previews_dir / f"scan_{record.case_id}_preview.png"
            _save_side_by_side_preview(
                output_path=preview_path,
                image=restored_image,
                target=restored_target_data,
                prediction=restored_prediction_data,
                case_id=record.case_id,
            )
            print(f"Saved prediction: {prediction_path}")
            print(f"Saved preview: {preview_path}")

    val_loss = float(np.mean(losses)) if losses else float("nan")
    val_dice = float(np.mean(dice_scores)) if dice_scores else float("nan")

    metrics_file = (
        resolve_path(base_dir, args.metrics_file)
        if args.metrics_file
        else export_dir / "validation_metrics.csv"
    )
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with metrics_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fold", "checkpoint", "mean_val_loss", "mean_val_dice"])
        writer.writerow([args.fold, str(checkpoint_path), val_loss, val_dice])
        writer.writerow([])
        writer.writerow(["case_id", "loss", "dice"])
        writer.writerows(summary_rows)

    print(f"Validation checkpoint: {checkpoint_path}")
    print(f"Validation loss: {val_loss:.6f}")
    print(f"Validation dice: {val_dice:.6f}")
    print(f"Saved metrics to: {metrics_file}")
    if not args.no_save_predictions:
        print(f"Saved predictions to: {predictions_dir}")
        print(f"Saved previews to: {previews_dir}")


if __name__ == "__main__":
    main()
