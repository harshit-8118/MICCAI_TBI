from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure, label


@dataclass(frozen=True)
class ConditionalM2Stats:
    m1_pred_voxels: int
    m2_pred_voxels: int
    accepted_m2_components: int
    rejected_m2_components: int
    accepted_m2_voxels: int
    rejected_m2_voxels: int


def filter_components(mask: np.ndarray, min_component_voxels: int) -> np.ndarray:
    if int(min_component_voxels) <= 1 or not mask.any():
        return mask.astype(np.uint8)
    components, n_components = label(mask > 0)
    filtered = np.zeros_like(mask, dtype=np.uint8)
    for component_index in range(1, n_components + 1):
        component = components == component_index
        if int(component.sum()) >= int(min_component_voxels):
            filtered[component] = 1
    return filtered


def core_halo_prediction(
    lesion_probability: np.ndarray,
    core_threshold: float,
    halo_threshold: float,
    core_min_component_voxels: int,
    max_growth_ratio: float,
    max_grown_component_voxels: int,
) -> np.ndarray:
    effective_halo_threshold = min(float(halo_threshold), float(core_threshold))
    core = lesion_probability >= float(core_threshold)
    if not core.any():
        return np.zeros_like(lesion_probability, dtype=np.uint8)

    halo = lesion_probability >= effective_halo_threshold
    components, n_components = label(halo)
    prediction = np.zeros_like(halo, dtype=np.uint8)
    for component_index in range(1, n_components + 1):
        component = components == component_index
        core_in_component = core & component
        core_voxels = int(core_in_component.sum())
        if core_voxels <= 0:
            continue

        grown_voxels = int(component.sum())
        allow_growth = core_voxels >= int(core_min_component_voxels)
        if float(max_growth_ratio) > 0:
            allow_growth = allow_growth and (grown_voxels / max(core_voxels, 1) <= float(max_growth_ratio))
        if int(max_grown_component_voxels) > 0:
            allow_growth = allow_growth and (grown_voxels <= int(max_grown_component_voxels))

        if allow_growth:
            prediction[component] = 1
        else:
            prediction[core_in_component] = 1
    return prediction


def dilate_mask(mask: np.ndarray, radius: int, min_component_voxels: int) -> np.ndarray:
    if int(radius) <= 0 or not mask.any():
        return mask.astype(np.uint8)
    structure = generate_binary_structure(rank=3, connectivity=1)
    components, n_components = label(mask > 0)
    dilated = np.zeros_like(mask, dtype=bool)
    for component_index in range(1, n_components + 1):
        component = components == component_index
        if int(component.sum()) < int(min_component_voxels):
            dilated |= component
            continue
        dilated |= binary_dilation(component, structure=structure, iterations=int(radius))
    return dilated.astype(np.uint8)


def postprocess_prediction(
    lesion_probability: np.ndarray,
    threshold: float,
    min_component_voxels: int,
    postprocess: str = "none",
    core_threshold: float | None = None,
    halo_threshold: float = 0.10,
    dilation_radius: int = 0,
    dilate_min_component_voxels: int = 1,
    core_min_component_voxels: int = 1,
    max_growth_ratio: float = 0.0,
    max_grown_component_voxels: int = 0,
) -> np.ndarray:
    if postprocess == "core_halo":
        prediction = core_halo_prediction(
            lesion_probability=lesion_probability,
            core_threshold=float(threshold) if core_threshold is None else float(core_threshold),
            halo_threshold=halo_threshold,
            core_min_component_voxels=core_min_component_voxels,
            max_growth_ratio=max_growth_ratio,
            max_grown_component_voxels=max_grown_component_voxels,
        )
    else:
        prediction = (lesion_probability >= float(threshold)).astype(np.uint8)

    if postprocess == "dilate" or int(dilation_radius) > 0:
        prediction = dilate_mask(
            prediction,
            radius=int(dilation_radius),
            min_component_voxels=int(dilate_min_component_voxels),
        )
    return filter_components(prediction, int(min_component_voxels))


def conditional_m2_component_acceptance(
    m1_probability: np.ndarray,
    m2_probability: np.ndarray,
    m1_threshold: float,
    m1_min_component_voxels: int,
    m2_threshold: float,
    m2_min_component_voxels: int,
    m1_support_threshold: float,
    support_radius: int = 0,
    support_min_overlap_voxels: int = 1,
    support_min_overlap_ratio: float = 0.0,
    final_min_component_voxels: int = 0,
) -> tuple[np.ndarray, ConditionalM2Stats]:
    """Use M1 as safe seed/support and accept only supported M2 components."""
    m1_mask = postprocess_prediction(
        m1_probability,
        threshold=float(m1_threshold),
        min_component_voxels=int(m1_min_component_voxels),
        postprocess="none",
    ).astype(bool)
    m2_mask = postprocess_prediction(
        m2_probability,
        threshold=float(m2_threshold),
        min_component_voxels=int(m2_min_component_voxels),
        postprocess="none",
    ).astype(bool)

    m1_support = m1_probability >= float(m1_support_threshold)
    if int(support_radius) > 0 and m1_support.any():
        structure = generate_binary_structure(rank=3, connectivity=1)
        m1_support = binary_dilation(m1_support, structure=structure, iterations=int(support_radius))

    final_mask = m1_mask.copy()
    components, n_components = label(m2_mask)
    accepted_components = 0
    rejected_components = 0
    accepted_voxels = 0
    rejected_voxels = 0
    min_overlap = max(1, int(support_min_overlap_voxels))
    min_ratio = max(0.0, float(support_min_overlap_ratio))

    for component_index in range(1, n_components + 1):
        component = components == component_index
        component_voxels = int(component.sum())
        if component_voxels <= 0:
            continue
        overlap_voxels = int(np.logical_and(component, m1_support).sum())
        overlap_ratio = overlap_voxels / float(max(component_voxels, 1))
        accept_component = overlap_voxels >= min_overlap and overlap_ratio >= min_ratio
        if accept_component:
            final_mask |= component
            accepted_components += 1
            accepted_voxels += component_voxels
        else:
            rejected_components += 1
            rejected_voxels += component_voxels

    final_mask = filter_components(final_mask.astype(np.uint8), int(final_min_component_voxels)).astype(np.uint8)
    stats = ConditionalM2Stats(
        m1_pred_voxels=int(m1_mask.sum()),
        m2_pred_voxels=int(m2_mask.sum()),
        accepted_m2_components=accepted_components,
        rejected_m2_components=rejected_components,
        accepted_m2_voxels=accepted_voxels,
        rejected_m2_voxels=rejected_voxels,
    )
    return final_mask, stats


def postprocess_setting_values(
    postprocess: str,
    halo_thresholds: list[float],
    max_growth_ratios: list[float],
) -> tuple[list[float], list[float]]:
    if postprocess != "core_halo":
        return [float(halo_thresholds[0])], [float(max_growth_ratios[0])]
    return [float(value) for value in halo_thresholds], [float(value) for value in max_growth_ratios]


def setting_tag(setting: Mapping[str, object]) -> str:
    strategy = str(setting.get("ensemble_strategy", "probability_average"))

    def tag_value(value: object) -> str:
        return f"{float(value):g}".replace("-", "m").replace(".", "p")

    if strategy == "strategy2_conditional_m2":
        parts = [
            "strategy2",
            f"m1thr{tag_value(setting['m1_threshold'])}",
            f"m1cc{int(setting['m1_min_component_voxels'])}",
            f"m2thr{tag_value(setting['m2_threshold'])}",
            f"m2cc{int(setting['m2_min_component_voxels'])}",
            f"sup{tag_value(setting['m1_support_threshold'])}",
            f"rad{int(setting['support_radius'])}",
            f"finalcc{int(setting['final_min_component_voxels'])}",
        ]
        return "_".join(parts)

    tag = f"thr{tag_value(setting['threshold'])}_mincc{int(setting['min_component_voxels'])}"
    if str(setting.get("postprocess", "none")) == "core_halo":
        if "core_threshold" in setting and np.isfinite(float(setting["core_threshold"])):
            tag += f"_core{tag_value(setting['core_threshold'])}"
        tag += f"_halo{tag_value(setting['halo_threshold'])}_grow{tag_value(setting['max_growth_ratio'])}"
        tag += f"_coremin{int(setting['core_min_component_voxels'])}"
        if int(setting.get("max_grown_component_voxels", 0)) > 0:
            tag += f"_maxvox{int(setting['max_grown_component_voxels'])}"
    if int(setting.get("dilation_radius", 0)) > 0:
        tag += f"_dil{int(setting['dilation_radius'])}"
    return tag
