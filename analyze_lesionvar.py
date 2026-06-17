import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd

def robust_zscore_foreground(array: np.ndarray) -> np.ndarray:
    """Robust Z-score normalization with 1st/99th percentile clipping."""
    mask = np.isfinite(array) & (array != 0)
    if not np.any(mask):
        mask = np.isfinite(array)
    values = array[mask]
    if not values.size:
        return np.zeros_like(array, dtype=np.float32)
    
    p1, p99 = np.percentile(values, [1, 99])
    clipped_array = np.clip(array, p1, p99)
    
    clipped_values = clipped_array[mask]
    mean = float(clipped_values.mean())
    std = float(clipped_values.std()) if clipped_values.std() > 1e-8 else 1.0
    
    normalized = (clipped_array - mean) / std
    normalized[~np.isfinite(normalized)] = 0.0
    normalized[~mask] = 0.0
    return normalized.astype(np.float32)

def standard_zscore_foreground(array: np.ndarray) -> np.ndarray:
    """Standard Z-score normalization using full foreground without clipping."""
    mask = np.isfinite(array) & (array != 0)
    if not np.any(mask):
        mask = np.isfinite(array)
    values = array[mask]
    if not values.size:
        return np.zeros_like(array, dtype=np.float32)
    
    mean = float(values.mean())
    std = float(values.std()) if values.std() > 1e-8 else 1.0
    
    normalized = (array - mean) / std
    normalized[~np.isfinite(normalized)] = 0.0
    normalized[~mask] = 0.0
    return normalized.astype(np.float32)

def analyze_lesion_variance(mri_paths: list, mask_paths: list, output_plot_path: str = "lesion_variance_comparison.png"):
    """
    Analyzes and visualizes the statistical variance of voxels strictly inside 
    the true lesion mask across different normalization methods.
    """
    all_results = []
    plot_data_standard = []
    plot_data_robust = []
    case_labels = []

    print("="*80)
    print("  MRI LESION VARIANCE ANALYSIS PIPELINE")
    print("="*80)

    for idx, (mri_p, lesion_p) in enumerate(zip(mri_paths, mask_paths)):
        case_id = os.path.basename(mri_p).split('.')[0]
        print(f"\n[Processing] {case_id}...")

        # 1. Load MRI and the ground truth lesion mask
        mri_img = nib.load(mri_p)
        lesion_img = nib.load(lesion_p)
        
        mri_arr = mri_img.get_fdata().astype(np.float32)
        true_lesion_mask = lesion_img.get_fdata() > 0 

        # 2. Dynamically define the original non-zero tissue foreground mask
        foreground_mask = np.isfinite(mri_arr) & (mri_arr != 0)
        if not np.any(foreground_mask):
            foreground_mask = np.isfinite(mri_arr)

        # --- THE FIX: Only track the label if it passes the safety check ---
        case_labels.append(case_id)

        # Intersection: Ensure we only look at lesion voxels that actually sit inside our valid foreground
        valid_lesion_mask = true_lesion_mask & foreground_mask
        foreground_values = mri_arr[foreground_mask]

        # ----------------------------------------------------
        # Method 1: Standard Z-Score (Calculated on Foreground)
        # ----------------------------------------------------
        mean_std = float(foreground_values.mean())
        std_std = float(foreground_values.std()) if foreground_values.std() > 1e-8 else 1.0
        
        norm_standard = (mri_arr - mean_std) / std_std
        lesion_voxels_standard = norm_standard[valid_lesion_mask]

        # ----------------------------------------------------
        # Method 2: Robust Z-Score (1%-99% Clipped on Foreground)
        # ----------------------------------------------------
        p1, p99 = np.percentile(foreground_values, [1, 99])
        clipped_mri = np.clip(mri_arr, p1, p99)
        
        clipped_foreground_values = clipped_mri[foreground_mask]
        mean_robust = float(clipped_foreground_values.mean())
        std_robust = float(clipped_foreground_values.std()) if clipped_foreground_values.std() > 1e-8 else 1.0
        
        norm_robust = (clipped_mri - mean_robust) / std_robust
        lesion_voxels_robust = norm_robust[valid_lesion_mask]

        # ----------------------------------------------------
        # Save & Compute Metrics
        # ----------------------------------------------------
        plot_data_standard.append(lesion_voxels_standard)
        plot_data_robust.append(lesion_voxels_robust)

        metrics = {
            "Case ID": case_id,
            "Valid Lesion Voxels": int(np.sum(valid_lesion_mask)),
            "Raw Lesion Var": float(np.var(mri_arr[valid_lesion_mask])),
            "Std Z Lesion Var": float(np.var(lesion_voxels_standard)),
            "Std Z Lesion Mean": float(np.mean(lesion_voxels_standard)),
            "Robust Z Lesion Var": float(np.var(lesion_voxels_robust)),
            "Robust Z Lesion Mean": float(np.mean(lesion_voxels_robust))
        }
        all_results.append(metrics)

    # Convert results summary to a clean Pandas DataFrame
    df = pd.DataFrame(all_results)
    print("\n" + "="*80)
    print("  SUMMARY OVERVIEW ACROSS DATASET")
    print("="*80)
    print(df.to_string(index=False))

     # ----------------------------------------------------
    # 6. Comparative Box Plots (Updated to avoid deprecation warnings)
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Changed 'labels=' to 'tick_labels='
    axes[0].boxplot(plot_data_standard, tick_labels=case_labels, patch_artist=True,
                    boxprops=dict(facecolor='crimson', alpha=0.6),
                    medianprops=dict(color='black', linewidth=1.5))
    axes[0].set_title("Lesion Voxel Variance:\nStandard Z-Score (Contextual Foreground)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Normalized Intensity Value")
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Changed 'labels=' to 'tick_labels='
    axes[1].boxplot(plot_data_robust, tick_labels=case_labels, patch_artist=True,
                    boxprops=dict(facecolor='teal', alpha=0.6),
                    medianprops=dict(color='black', linewidth=1.5))
    axes[1].set_title("Lesion Voxel Variance:\nRobust Z-Score (Contextual 1%-99% Clipped)", fontsize=12, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("How Normalization Changes Voxel Distribution Inside True Lesion Mask", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[Success] Distribution plot saved to: {output_plot_path}\n")

# ----------------------------------------------------------------------
# Example execution configuration
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Add your real NIfTI file path configurations here
    import os 

    root_dir = "/data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning/MICCAI_AIMS_TBI/"
    all_mris = os.listdir(root_dir)  # Quick check to confirm paths scan_0020_T1.nii.gz
    t1_images  = [os.path.join(root_dir, f) for f in all_mris if f in ("scan_0020_T1.nii.gz", "scan_0029_T1.nii.gz", "scan_0145_T1.nii.gz")]
    lesion_masks  = [os.path.join(root_dir, f.replace("_T1.nii.gz", "_Lesion.nii.gz")) for f in t1_images]
    
    # Quick sanity validation to guide user before execution
    if os.path.exists(t1_images[0]) and os.path.exists(lesion_masks[0]):
        analyze_lesion_variance(t1_images, lesion_masks)
    else:
        print("[System Note] Please replace dummy array strings with actual file system paths to run.")