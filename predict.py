from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to

from multitalent_tbi.config import load_config, resolve_path
from multitalent_tbi.data import discover_cases, load_case_cached, load_nifti_robust
from multitalent_tbi.engine import build_model
from multitalent_tbi.infer import predict_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with one or more MultiTalentV2 checkpoints.")
    parser.add_argument("--config", default="config.yml", help="Path to the YAML config.")
    parser.add_argument("--input-dir", default=None, help="Override input dataset directory.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--checkpoints", nargs="+", default=None, help="One or more checkpoint paths for ensembling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    base_dir = Path(args.config).expanduser().resolve().parent
    dataset_dir = resolve_path(base_dir, args.input_dir or config.paths.dataset_dir)
    output_dir = resolve_path(base_dir, args.output_dir or config.paths.predictions_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = discover_cases(dataset_dir)

    checkpoint_paths = [resolve_path(base_dir, path) for path in (args.checkpoints or [resolve_path(base_dir, config.paths.work_dir) / "fold_0" / "best.pt"])]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    for checkpoint_path in checkpoint_paths:
        model = build_model(config, base_dir).to(device)
        payload = torch.load(str(checkpoint_path), map_location="cpu")
        model.load_state_dict(payload["model_state"], strict=False)
        model.eval()
        models.append(model)

    for record in records:
        probability_maps = []
        original_image = load_nifti_robust(record.t1_path)
        image, _, image_affine, _ = load_case_cached(
            case=record,
            target_spacing=tuple(config.data.target_spacing),
            include_dmri=bool(config.data.include_dmri),
            dmri_reduce=str(config.data.dmri_reduce),
            dmri_b0_threshold=float(config.data.dmri_b0_threshold),
            normalize_foreground_only=bool(config.data.normalize_foreground_only),
            cache_dir=resolve_path(base_dir, config.paths.cache_dir) if config.data.cache_preprocessed else None,
        )
        for model in models:
            logits = predict_logits(
                model=model,
                image=image,
                roi_size=tuple(config.inference.roi_size),
                overlap=float(config.inference.overlap),
                batch_size=int(config.inference.sw_batch_size),
                device=device,
                use_amp=bool(config.inference.use_amp) and device.type == "cuda",
            )
            probability_maps.append(torch.softmax(torch.from_numpy(logits), dim=0).numpy())

        averaged = np.mean(probability_maps, axis=0)
        mask = np.argmax(averaged, axis=0).astype(np.uint8)
        output_path = output_dir / f"scan_{record.case_id}_Lesion.nii.gz"
        resampled_mask = nib.Nifti1Image(mask, affine=np.asarray(image_affine))
        restored = resample_from_to(resampled_mask, original_image, order=0)
        nib.save(restored, str(output_path))
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
