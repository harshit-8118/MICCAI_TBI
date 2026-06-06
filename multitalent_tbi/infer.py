from __future__ import annotations

import itertools
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to

from .data import CaseRecord, load_case_cached, load_nifti_robust


def _pad_volume(volume: np.ndarray, roi_size: tuple[int, int, int]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    pads: list[tuple[int, int]] = []
    for size, target in zip(volume.shape[-3:], roi_size):
        total = max(target - size, 0)
        before = total // 2
        after = total - before
        pads.append((before, after))
    padded = np.pad(volume, [(0, 0)] + pads, mode="constant", constant_values=0.0)
    return padded, pads


def _sliding_positions(length: int, roi: int, stride: int) -> list[int]:
    if length <= roi:
        return [0]
    positions = list(range(0, length - roi + 1, stride))
    if positions[-1] != length - roi:
        positions.append(length - roi)
    return positions


@torch.inference_mode()
def predict_logits(
    model: torch.nn.Module,
    image: np.ndarray,
    roi_size: tuple[int, int, int],
    overlap: float,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    model.eval()
    padded, pads = _pad_volume(image, roi_size)
    _, depth, height, width = padded.shape
    stride = [max(1, int(size * (1.0 - overlap))) for size in roi_size]
    z_positions = _sliding_positions(depth, roi_size[0], stride[0])
    y_positions = _sliding_positions(height, roi_size[1], stride[1])
    x_positions = _sliding_positions(width, roi_size[2], stride[2])
    num_classes = None
    logits_sum = None
    counts = None

    coordinates = list(itertools.product(z_positions, y_positions, x_positions))
    for start in range(0, len(coordinates), batch_size):
        batch_coordinates = coordinates[start : start + batch_size]
        patches = []
        for z, y, x in batch_coordinates:
            patches.append(
                padded[
                    :,
                    z : z + roi_size[0],
                    y : y + roi_size[1],
                    x : x + roi_size[2],
                ]
            )
        patch_tensor = torch.from_numpy(np.stack(patches)).to(device=device, dtype=torch.float32).contiguous(memory_format=torch.channels_last_3d)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            batch_logits = model(patch_tensor)
        if isinstance(batch_logits, (tuple, list)):
            batch_logits = batch_logits[0]
        batch_logits = batch_logits.float().cpu().numpy()
        if logits_sum is None:
            num_classes = batch_logits.shape[1]
            logits_sum = np.zeros((num_classes, depth, height, width), dtype=np.float32)
            counts = np.zeros((1, depth, height, width), dtype=np.float32)
        for (z, y, x), sample_logits in zip(batch_coordinates, batch_logits):
            logits_sum[:, z : z + roi_size[0], y : y + roi_size[1], x : x + roi_size[2]] += sample_logits
            counts[:, z : z + roi_size[0], y : y + roi_size[1], x : x + roi_size[2]] += 1.0

    assert logits_sum is not None and counts is not None
    logits_sum /= np.maximum(counts, 1e-6)
    if any(pad[0] or pad[1] for pad in pads):
        z0, z1 = pads[0]
        y0, y1 = pads[1]
        x0, x1 = pads[2]
        logits_sum = logits_sum[:, z0 : logits_sum.shape[1] - z1, y0 : logits_sum.shape[2] - y1, x0 : logits_sum.shape[3] - x1]
    return logits_sum


def predict_case(
    model: torch.nn.Module,
    record: CaseRecord,
    target_spacing: tuple[float, float, float],
    include_dmri: bool,
    dmri_reduce: str,
    dmri_b0_threshold: float,
    normalize_foreground_only: bool,
    cache_dir: str | Path | None,
    roi_size: tuple[int, int, int],
    overlap: float,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> nib.Nifti1Image:
    original_image = load_nifti_robust(record.t1_path)
    image, _, image_affine, _ = load_case_cached(
        case=record,
        target_spacing=target_spacing,
        include_dmri=include_dmri,
        dmri_reduce=dmri_reduce,
        dmri_b0_threshold=dmri_b0_threshold,
        normalize_foreground_only=normalize_foreground_only,
        cache_dir=cache_dir,
    )
    logits = predict_logits(
        model=model,
        image=image,
        roi_size=roi_size,
        overlap=overlap,
        batch_size=batch_size,
        device=device,
        use_amp=use_amp,
    )
    probabilities = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
    mask = np.argmax(probabilities, axis=0).astype(np.uint8)
    prediction = nib.Nifti1Image(mask, affine=np.asarray(image_affine))
    restored = resample_from_to(prediction, original_image, order=0)
    return restored
