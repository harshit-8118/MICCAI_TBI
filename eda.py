"""
AIMS-TBI Dataset EDA
====================
Scans a flat folder of .nii.gz / .txt files, groups them by subject,
and produces:
  1. Console summary  (file counts, shapes, voxel stats)
  2. plots/01_file_counts.png
  3. plots/02_shape_distributions.png
  4. plots/03_lesion_volume_distribution.png
  5. plots/04_t1_intensity_stats.png
  6. plots/05_sample_slices_<scanid>.png  (one figure per unique scan)
  7. eda_summary.csv  (per-file stats table)

Usage
-----
    python eda.py --data_dir /path/to/MICCAI_AIMS_TBI
    python eda.py --data_dir /path/to/MICCAI_AIMS_TBI --max_samples 10

All outputs go into  <data_dir>/eda_outputs/
"""

import os
import re
import argparse
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import nibabel as nib
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── colour palette (matches AIMS-TBI paper style) ───────────────────────────
PAL = {
    "T1":     "#378ADD",
    "Lesion": "#E85D24",
    "dMRI":   "#EF9F27",
    "bvec":   "#1D9E75",
    "bval":   "#7F77DD",
    "other":  "#888780",
}

# ── regex: scan_XXXX_<type>.<ext> ───────────────────────────────────────────
FILE_RE = re.compile(
    r"^(?P<prefix>[a-zA-Z]+)_(?P<scan_id>\d+)_(?P<ftype>[a-zA-Z0-9]+)\.(?P<ext>.+)$"
)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  SCAN ─ parses one file, returns a metadata dict
# ═══════════════════════════════════════════════════════════════════════════
def parse_file(filepath: Path) -> dict | None:
    m = FILE_RE.match(filepath.name)
    if not m:
        return None

    ftype = m.group("ftype")
    scan_id = m.group("scan_id")
    ext = m.group("ext")
    size_bytes = filepath.stat().st_size

    record = {
        "filepath":  str(filepath),
        "filename":  filepath.name,
        "scan_id":   scan_id,
        "ftype":     ftype,
        "ext":       ext,
        "size_mb":   round(size_bytes / 1e6, 3),
        "shape":     None,
        "ndim":      None,
        "dtype":     None,
        "vox_mm":    None,   # voxel size in mm (T1/Lesion/dMRI)
        "min_val":   None,
        "max_val":   None,
        "mean_val":  None,
        "nonzero":   None,   # for Lesion: lesion voxel count
        "n_dirs":    None,   # for dMRI: number of gradient directions
        "error":     None,
    }

    try:
        if ext in ("nii.gz", "nii"):
            img = nib.load(str(filepath))
            data = img.get_fdata(dtype=np.float32)
            hdr  = img.header
            record["shape"]   = tuple(int(x) for x in data.shape)
            record["ndim"]    = data.ndim
            record["dtype"]   = str(data.dtype)
            zooms = hdr.get_zooms()
            record["vox_mm"]  = tuple(round(float(z), 3) for z in zooms[:3])
            record["min_val"] = round(float(data.min()), 4)
            record["max_val"] = round(float(data.max()), 4)
            record["mean_val"]= round(float(data.mean()), 4)
            record["nonzero"] = int(np.count_nonzero(data))
            if ftype == "dMRI" and data.ndim == 4:
                record["n_dirs"] = data.shape[3]

        elif ext == "txt":
            arr = np.loadtxt(str(filepath))
            record["shape"]  = tuple(int(x) for x in arr.shape) if arr.ndim > 1 else (1, int(arr.size))
            record["ndim"]   = arr.ndim
            record["dtype"]  = str(arr.dtype)
            record["min_val"]= round(float(arr.min()), 4)
            record["max_val"]= round(float(arr.max()), 4)

    except Exception as e:
        record["error"] = str(e)

    return record


# ═══════════════════════════════════════════════════════════════════════════
# 2.  COLLECT all files
# ═══════════════════════════════════════════════════════════════════════════
def collect_files(data_dir: Path, max_samples: int | None = None) -> pd.DataFrame:
    files = sorted(data_dir.glob("*"))
    nii_files = [f for f in files if f.suffix in (".gz", ".nii") or f.name.endswith(".nii.gz")]
    txt_files  = [f for f in files if f.suffix == ".txt"]
    all_files  = sorted(set(nii_files + txt_files))

    if max_samples:
        # group by scan_id and cap at max_samples unique subjects
        seen, kept = set(), []
        for f in all_files:
            m = FILE_RE.match(f.name)
            sid = m.group("scan_id") if m else None
            if sid and sid not in seen:
                seen.add(sid)
            if sid in seen and len(seen) <= max_samples:
                kept.append(f)
        all_files = kept

    records = []
    for f in tqdm(all_files, desc="Scanning files", unit="file"):
        r = parse_file(f)
        if r:
            records.append(r)

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  CONSOLE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("  AIMS-TBI  ─  Dataset EDA Summary")
    print("=" * 60)

    print(f"\n{'Total files parsed':.<40} {len(df)}")
    print(f"{'Unique scan IDs':.<40} {df['scan_id'].nunique()}")
    print(f"{'Files with errors':.<40} {df['error'].notna().sum()}")

    print("\n── File counts by type ─────────────────────────")
    for ftype, grp in df.groupby("ftype"):
        print(f"  {ftype:<12} {len(grp):>5} files   "
              f"avg {grp['size_mb'].mean():.1f} MB")

    for ftype in ["T1", "Lesion", "dMRI"]:
        sub = df[(df["ftype"] == ftype) & df["shape"].notna()]
        if sub.empty:
            continue
        shapes = sub["shape"].tolist()
        print(f"\n── {ftype} shapes ───────────────────────────────")
        print(f"  Count      : {len(shapes)}")
        unique_shapes = list(dict.fromkeys(str(s) for s in shapes))
        print(f"  Unique shapes ({len(unique_shapes)}) : {unique_shapes[:5]}"
              f"{'...' if len(unique_shapes)>5 else ''}")

        dims = np.array([list(s[:3]) for s in shapes])
        for i, ax in enumerate(["X", "Y", "Z"]):
            print(f"  {ax} dim  → min={dims[:,i].min():>4}  "
                  f"max={dims[:,i].max():>4}  "
                  f"mean={dims[:,i].mean():.1f}")

        if ftype == "Lesion":
            vols = sub["nonzero"].dropna()
            lesion_cases = (vols > 0).sum()
            clean_cases  = (vols == 0).sum()
            print(f"\n── Lesion statistics ────────────────────────────")
            print(f"  Subjects with lesion  : {lesion_cases}")
            print(f"  Lesion-free subjects  : {clean_cases}")
            print(f"  Lesion voxels  min    : {vols[vols>0].min():.0f}")
            print(f"  Lesion voxels  max    : {vols[vols>0].max():.0f}")
            print(f"  Lesion voxels  median : {vols[vols>0].median():.0f}")

        if ftype == "dMRI":
            n_dirs = sub["n_dirs"].dropna()
            if not n_dirs.empty:
                print(f"  Gradient dirs → min={n_dirs.min():.0f}  "
                      f"max={n_dirs.max():.0f}  "
                      f"mode={n_dirs.mode()[0]:.0f}")

    print("\n" + "=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# 4.  PLOTS
# ═══════════════════════════════════════════════════════════════════════════
def plot_file_counts(df: pd.DataFrame, out_dir: Path):
    counts = df.groupby("ftype").size().reset_index(name="count")
    colors = [PAL.get(t, PAL["other"]) for t in counts["ftype"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("File counts per type", fontsize=14, fontweight="bold", y=1.01)

    # bar
    ax = axes[0]
    bars = ax.bar(counts["ftype"], counts["count"], color=colors, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_xlabel("File type")
    ax.set_ylabel("Count")
    ax.set_title("Number of files per type")
    ax.spines[["top","right"]].set_visible(False)

    # size distribution per type
    ax = axes[1]
    for ftype, grp in df.groupby("ftype"):
        if grp["size_mb"].max() > 0:
            ax.hist(grp["size_mb"], bins=30, alpha=0.6,
                    label=ftype, color=PAL.get(ftype, PAL["other"]))
    ax.set_xlabel("File size (MB)")
    ax.set_ylabel("Count")
    ax.set_title("File size distribution")
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = out_dir / "01_file_counts.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_shape_distributions(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Spatial shape distributions (X / Y / Z dims)", fontsize=14, fontweight="bold")

    ftypes = ["T1", "Lesion", "dMRI"]
    for row, ftype in enumerate(ftypes):
        sub = df[(df["ftype"] == ftype) & df["shape"].notna()]
        if sub.empty:
            for col in range(3):
                axes[row//2*2+row%2, col].set_visible(False) if row < 2 else None
            continue

        shapes = np.array([list(s[:3]) for s in sub["shape"].tolist()])
        col_offset = 0 if row < 2 else 0

        ax_row = row if row < 2 else 1
        for col, (dim_vals, dim_label) in enumerate(zip(shapes.T, ["X", "Y", "Z"])):
            ax = axes[row if row < 2 else 1, col]
            ax.hist(dim_vals, bins=20, color=PAL.get(ftype, PAL["other"]),
                    alpha=0.7, edgecolor="white")
            ax.axvline(dim_vals.mean(), color="black", linestyle="--",
                       linewidth=1, label=f"mean={dim_vals.mean():.0f}")
            ax.set_title(f"{ftype} — {dim_label} dim")
            ax.set_xlabel("Voxels")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)
            ax.spines[["top","right"]].set_visible(False)

    # use all 6 axes: T1 top row, Lesion middle-ish, dMRI bottom
    fig2, axes2 = plt.subplots(len(ftypes), 3, figsize=(16, 4*len(ftypes)))
    fig2.suptitle("Spatial shape distributions (X / Y / Z)", fontsize=14, fontweight="bold")

    for row, ftype in enumerate(ftypes):
        sub = df[(df["ftype"] == ftype) & df["shape"].notna()]
        for col, dim_label in enumerate(["X", "Y", "Z"]):
            ax = axes2[row, col]
            if sub.empty:
                ax.text(0.5, 0.5, f"No {ftype} files found",
                        ha="center", va="center", transform=ax.transAxes,
                        color="gray")
                ax.set_visible(True)
                continue
            shapes = np.array([list(s[:3]) for s in sub["shape"].tolist()])
            dim_vals = shapes[:, col]
            ax.hist(dim_vals, bins=20, color=PAL.get(ftype, PAL["other"]),
                    alpha=0.75, edgecolor="white", linewidth=0.4)
            ax.axvline(dim_vals.mean(), color="#222", linestyle="--",
                       linewidth=1.2, label=f"μ={dim_vals.mean():.0f}")
            ax.axvline(dim_vals.min(), color="#999", linestyle=":",
                       linewidth=1, label=f"min={dim_vals.min()}")
            ax.axvline(dim_vals.max(), color="#999", linestyle=":",
                       linewidth=1, label=f"max={dim_vals.max()}")
            ax.set_title(f"{ftype} — {dim_label}")
            ax.set_xlabel("Voxels")
            ax.set_ylabel("Count")
            ax.legend(fontsize=7)
            ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = out_dir / "02_shape_distributions.png"
    fig2.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_lesion_volume(df: pd.DataFrame, out_dir: Path):
    sub = df[(df["ftype"] == "Lesion") & df["nonzero"].notna()].copy()
    if sub.empty:
        print("  [skip] No Lesion files found for volume plot.")
        return

    sub["lesion_voxels"] = sub["nonzero"].astype(float)
    has_lesion = sub[sub["lesion_voxels"] > 0]["lesion_voxels"]
    no_lesion  = sub[sub["lesion_voxels"] == 0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Lesion volume analysis", fontsize=14, fontweight="bold")

    # pie: lesion vs clean
    ax = axes[0]
    values = [len(has_lesion), len(no_lesion)]
    labels = [f"Has lesion\n(n={len(has_lesion)})", f"Lesion-free\n(n={len(no_lesion)})"]
    colors = [PAL["Lesion"], "#B4B2A9"]
    wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors,
                                       autopct="%1.1f%%", startangle=90,
                                       textprops={"fontsize": 10})
    ax.set_title("Lesion presence")

    # histogram (log scale) for lesion volumes
    ax = axes[1]
    if len(has_lesion) > 0:
        ax.hist(has_lesion, bins=40, color=PAL["Lesion"], alpha=0.75, edgecolor="white")
        ax.set_xlabel("Lesion voxels (count)")
        ax.set_ylabel("Subjects")
        ax.set_title("Lesion size distribution")
        ax.spines[["top","right"]].set_visible(False)

    # log scale version
    ax = axes[2]
    if len(has_lesion) > 0:
        ax.hist(np.log10(has_lesion + 1), bins=40, color=PAL["Lesion"], alpha=0.75, edgecolor="white")
        ax.set_xlabel("log₁₀(lesion voxels + 1)")
        ax.set_ylabel("Subjects")
        ax.set_title("Lesion size (log scale)")
        ax.spines[["top","right"]].set_visible(False)
        stats = has_lesion.describe()
        txt = (f"min  = {stats['min']:.0f}\n"
               f"med  = {stats['50%']:.0f}\n"
               f"mean = {stats['mean']:.0f}\n"
               f"max  = {stats['max']:.0f}")
        ax.text(0.97, 0.97, txt, transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#ccc", alpha=0.9))

    plt.tight_layout()
    out = out_dir / "03_lesion_volume_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_t1_intensity_stats(df: pd.DataFrame, out_dir: Path):
    sub = df[(df["ftype"] == "T1") & df["min_val"].notna()]
    if sub.empty:
        print("  [skip] No T1 files for intensity plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("T1 intensity statistics across subjects", fontsize=14, fontweight="bold")

    for ax, col, label in zip(
        axes,
        ["min_val", "max_val", "mean_val"],
        ["Min intensity", "Max intensity", "Mean intensity"],
    ):
        ax.hist(sub[col], bins=40, color=PAL["T1"], alpha=0.75, edgecolor="white")
        ax.axvline(sub[col].median(), color="#222", linestyle="--",
                   linewidth=1.2, label=f"median={sub[col].median():.1f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Subjects")
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = out_dir / "04_t1_intensity_stats.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_sample_slices(df: pd.DataFrame, out_dir: Path, n_samples: int = 5):
    """
    For up to n_samples scan IDs that have both T1 and Lesion,
    plot axial / coronal / sagittal middle slices side-by-side.
    """
    lesion_ids = set(df[df["ftype"] == "Lesion"]["scan_id"])
    t1_ids     = set(df[df["ftype"] == "T1"]["scan_id"])
    paired_ids = sorted(lesion_ids & t1_ids)[:n_samples]

    if not paired_ids:
        # fall back to just T1
        paired_ids = sorted(t1_ids)[:n_samples]

    t1_map  = dict(zip(df[df["ftype"]=="T1"]["scan_id"],  df[df["ftype"]=="T1"]["filepath"]))
    les_map = dict(zip(df[df["ftype"]=="Lesion"]["scan_id"], df[df["ftype"]=="Lesion"]["filepath"]))

    for sid in tqdm(paired_ids, desc="Rendering slices", unit="subject"):
        t1_path  = t1_map.get(sid)
        les_path = les_map.get(sid)
        if not t1_path:
            continue

        try:
            t1_img  = nib.load(t1_path)
            t1_data = t1_img.get_fdata(dtype=np.float32)
            has_lesion_file = les_path is not None
            if has_lesion_file:
                les_data = nib.load(les_path).get_fdata(dtype=np.float32)
            else:
                les_data = np.zeros_like(t1_data)

            # normalise T1 for display
            p2, p98 = np.percentile(t1_data[t1_data > 0], [2, 98]) if t1_data.max() > 0 else (0, 1)
            t1_norm = np.clip((t1_data - p2) / (p98 - p2 + 1e-8), 0, 1)

            cx, cy, cz = [s // 2 for s in t1_data.shape[:3]]

            views = [
                ("Axial",    t1_norm[:, :, cz],  les_data[:, :, cz]),
                ("Coronal",  t1_norm[:, cy, :],  les_data[:, cy, :]),
                ("Sagittal", t1_norm[cx, :, :],  les_data[cx, :, :]),
            ]

            ncols = 3 if not has_lesion_file else 6
            fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
            fig.suptitle(
                f"Scan {sid}  |  T1 shape: {t1_data.shape}  "
                f"|  Lesion voxels: {int(les_data.sum()):,}",
                fontsize=11, fontweight="bold"
            )

            for i, (view_name, t1_sl, les_sl) in enumerate(views):
                # T1 only
                ax = axes[i] if not has_lesion_file else axes[i * 2]
                ax.imshow(np.rot90(t1_sl), cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"{view_name} — T1", fontsize=9)
                ax.axis("off")

                if has_lesion_file:
                    # T1 + lesion overlay
                    ax2 = axes[i * 2 + 1]
                    ax2.imshow(np.rot90(t1_sl), cmap="gray", vmin=0, vmax=1)
                    if les_sl.max() > 0:
                        masked = np.ma.masked_where(les_sl == 0, les_sl)
                        ax2.imshow(np.rot90(masked), cmap="Reds",
                                   alpha=0.55, vmin=0, vmax=1)
                    ax2.set_title(f"{view_name} — + lesion", fontsize=9)
                    ax2.axis("off")

            plt.tight_layout()
            out = out_dir / f"05_sample_slices_{sid}.png"
            fig.savefig(out, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved → {out}")

        except Exception as e:
            print(f"  [warn] Could not render scan {sid}: {e}")


def plot_voxel_size_distribution(df: pd.DataFrame, out_dir: Path):
    sub = df[(df["ftype"] == "T1") & df["vox_mm"].notna()].copy()
    if sub.empty:
        return

    vox = np.array([list(v) for v in sub["vox_mm"]])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Voxel size distribution (T1, mm)", fontsize=13, fontweight="bold")

    for i, (ax, label) in enumerate(zip(axes, ["X (mm)", "Y (mm)", "Z (mm)"])):
        ax.hist(vox[:, i], bins=30, color=PAL["T1"], alpha=0.75, edgecolor="white")
        ax.axvline(vox[:, i].mean(), color="#222", linestyle="--",
                   linewidth=1.2, label=f"mean={vox[:,i].mean():.2f}")
        ax.set_title(f"Voxel size — {label}")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = out_dir / "06_voxel_size_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 5.  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="AIMS-TBI EDA Script")
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to flat MICCAI_AIMS_TBI folder (all .nii.gz and .txt files)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit to N unique scan IDs (useful for quick testing)"
    )
    parser.add_argument(
        "--slice_samples", type=int, default=5,
        help="Number of subjects to visualise as brain slices (default: 5)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    out_dir = Path("eda_outputs")
    out_dir.mkdir(exist_ok=True)
    print(f"\nData  dir : {data_dir}")
    print(f"Output dir: {out_dir}")

    # ── collect ──────────────────────────────────────────────────────────
    print("\nStep 1/6 — collecting file metadata …")
    df = collect_files(data_dir, max_samples=args.max_samples)

    if df.empty:
        print("No recognised files found. Check --data_dir and filename format.")
        return

    # ── save CSV ─────────────────────────────────────────────────────────
    csv_out = out_dir / "eda_summary.csv"
    df.to_csv(csv_out, index=False)
    print(f"  CSV saved → {csv_out}")

    # ── console summary ───────────────────────────────────────────────────
    print_summary(df)

    # ── plots ─────────────────────────────────────────────────────────────
    print("Step 2/6 — file counts & sizes …")
    plot_file_counts(df, out_dir)

    print("Step 3/6 — shape distributions …")
    plot_shape_distributions(df, out_dir)

    print("Step 4/6 — lesion volume analysis …")
    plot_lesion_volume(df, out_dir)

    print("Step 5/6 — T1 intensity stats …")
    plot_t1_intensity_stats(df, out_dir)

    print("Step 6/6 — voxel size distribution …")
    plot_voxel_size_distribution(df, out_dir)

    print(f"\nStep 6b — sample brain slices (n={args.slice_samples}) …")
    plot_sample_slices(df, out_dir, n_samples=args.slice_samples)

    print(f"\n✓ All outputs written to: {out_dir}\n")
    print("Files generated:")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()