"""
pairing_summary.py
==================
For every scan_XXXX in the dataset folder, checks:
  - Does a T1 file exist?
  - Does a Lesion file exist?
  - Are both present (matched pair)?
  - Is the lesion mask positive (has lesion) or empty (lesion-free)?
  - Do T1 and Lesion shapes match?
  - Are there orphan T1s (no Lesion) or orphan Lesions (no T1)?

Outputs
-------
  pairing_summary/
    00_console_report.txt          full printed report
    01_matched_pairs.csv           all scan_IDs with both T1 + Lesion
    02_positive_lesions.csv        matched pairs where lesion_voxels > 0
    03_empty_lesions.csv           matched pairs where lesion_voxels == 0
    04_orphan_t1_no_lesion.csv     T1 exists but Lesion missing
    05_orphan_lesion_no_t1.csv     Lesion exists but T1 missing
    06_shape_mismatches.csv        pairs where T1 shape != Lesion shape
    07_complete_summary.csv        one row per scan_ID, all columns

Usage
-----
    python pairing_summary.py --data_dir /media/lab/"My Book"/MICCAI_AIMS_TBI/dataset

    # also create bar chart figures
    python pairing_summary.py --data_dir ~/aims_tbi/dataset --plot
"""

import argparse
import io
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

# optional plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

FILE_RE = re.compile(
    r"^(?P<prefix>[a-zA-Z]+)_(?P<scan_id>\d+)_(?P<ftype>[a-zA-Z0-9]+)\.(?P<ext>.+)$"
)

# ── colour palette ────────────────────────────────────────────────────────────
PAL = {
    "T1":       "#378ADD",
    "Lesion":   "#E85D24",
    "positive": "#E85D24",
    "empty":    "#B4B2A9",
    "orphan":   "#EF9F27",
    "mismatch": "#7F77DD",
}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — collect scan IDs and file paths
# ═══════════════════════════════════════════════════════════════════════════════

def collect_id_map(data_dir: Path) -> dict[str, dict[str, Path]]:
    """
    Returns  { scan_id: { 'T1': Path | None, 'Lesion': Path | None, ... } }
    """
    id_map: dict[str, dict] = {}

    for f in sorted(data_dir.iterdir()):
        m = FILE_RE.match(f.name)
        if not m:
            continue
        sid   = m.group("scan_id")
        ftype = m.group("ftype")
        if sid not in id_map:
            id_map[sid] = {"T1": None, "Lesion": None,
                           "dMRI": None, "bvec": None, "bval": None}
        if ftype in id_map[sid]:
            id_map[sid][ftype] = f

    return id_map


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — load Lesion stats for one file
# ═══════════════════════════════════════════════════════════════════════════════

def lesion_stats(path: Path) -> dict:
    """Returns shape, lesion_voxels, error for a Lesion nii file."""
    try:
        img  = nib.load(str(path))
        data = img.get_fdata(dtype=np.float32)
        return {
            "lesion_shape":   tuple(int(x) for x in data.shape),
            "lesion_voxels":  int(data.sum()),
            "lesion_unique":  sorted(np.unique(data).tolist()),
            "lesion_error":   None,
        }
    except Exception as e:
        return {
            "lesion_shape":  None,
            "lesion_voxels": None,
            "lesion_unique": None,
            "lesion_error":  str(e),
        }


def t1_stats(path: Path) -> dict:
    """Returns shape, error for a T1 nii file."""
    try:
        img = nib.load(str(path))
        return {
            "t1_shape": tuple(int(x) for x in img.header.get_data_shape()),
            "t1_error": None,
        }
    except Exception as e:
        return {"t1_shape": None, "t1_error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — build master table
# ═══════════════════════════════════════════════════════════════════════════════

def build_table(id_map: dict) -> pd.DataFrame:
    rows = []
    for sid, files in tqdm(id_map.items(), desc="Loading file metadata", unit="scan"):

        has_t1     = files["T1"]     is not None
        has_lesion = files["Lesion"] is not None
        has_dmri   = files["dMRI"]   is not None
        has_bvec   = files["bvec"]   is not None
        has_bval   = files["bval"]   is not None

        row = {
            "scan_id":   sid,
            "has_t1":    has_t1,
            "has_lesion":has_lesion,
            "has_dmri":  has_dmri,
            "has_bvec":  has_bvec,
            "has_bval":  has_bval,
            "t1_path":   str(files["T1"])     if has_t1     else None,
            "lesion_path": str(files["Lesion"]) if has_lesion else None,
        }

        # T1 shape
        if has_t1:
            row.update(t1_stats(files["T1"]))
        else:
            row.update({"t1_shape": None, "t1_error": None})

        # Lesion stats
        if has_lesion:
            row.update(lesion_stats(files["Lesion"]))
        else:
            row.update({
                "lesion_shape":  None,
                "lesion_voxels": None,
                "lesion_unique": None,
                "lesion_error":  None,
            })

        # derived flags
        matched          = has_t1 and has_lesion
        shape_ok         = (
            matched
            and row["t1_shape"] is not None
            and row["lesion_shape"] is not None
            and row["t1_shape"] == row["lesion_shape"]
        )
        lesion_positive  = (
            matched
            and row["lesion_voxels"] is not None
            and row["lesion_voxels"] > 0
        )
        lesion_empty     = (
            matched
            and row["lesion_voxels"] is not None
            and row["lesion_voxels"] == 0
        )
        has_any_error    = bool(row.get("t1_error") or row.get("lesion_error"))

        row.update({
            "matched":          matched,
            "shape_match":      shape_ok,
            "lesion_positive":  lesion_positive,
            "lesion_empty":     lesion_empty,
            "has_any_error":    has_any_error,
            "pairing_status":   (
                "matched_positive" if lesion_positive else
                "matched_empty"    if lesion_empty    else
                "matched_error"    if (matched and has_any_error) else
                "orphan_t1"        if (has_t1 and not has_lesion) else
                "orphan_lesion"    if (has_lesion and not has_t1) else
                "no_files"
            ),
        })

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("scan_id").reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — console report
# ═══════════════════════════════════════════════════════════════════════════════

def build_report(df: pd.DataFrame) -> str:
    buf = io.StringIO()

    def p(*args, **kw):
        print(*args, **kw, file=buf)

    def hr(title=""):
        p(f"\n{'═'*62}")
        if title:
            p(f"  {title}")
            p(f"{'═'*62}")

    hr("AIMS-TBI  ─  T1 / Lesion Pairing Summary")

    total = len(df)
    p(f"\n{'Total scan IDs found':<40} {total}")

    # file presence
    hr("File Presence")
    p(f"  {'Scans with T1':<38} {df['has_t1'].sum()}")
    p(f"  {'Scans with Lesion mask':<38} {df['has_lesion'].sum()}")
    p(f"  {'Scans with dMRI':<38} {df['has_dmri'].sum()}")
    p(f"  {'Scans with bvec':<38} {df['has_bvec'].sum()}")
    p(f"  {'Scans with bval':<38} {df['has_bval'].sum()}")

    # pairing
    hr("T1 ↔ Lesion Pairing")
    matched    = df["matched"].sum()
    unmatched  = total - matched
    p(f"  {'Matched pairs (T1 + Lesion both exist)':<38} {matched}")
    p(f"  {'Unmatched (missing one or both)':<38} {unmatched}")

    orphan_t1  = (df["pairing_status"] == "orphan_t1").sum()
    orphan_les = (df["pairing_status"] == "orphan_lesion").sum()
    p(f"    ├─ Orphan T1  (Lesion missing)':<34 {orphan_t1}")
    p(f"    └─ Orphan Lesion (T1 missing)':<34 {orphan_les}")

    # shape check
    hr("Shape Consistency (matched pairs only)")
    matched_df    = df[df["matched"]]
    shape_ok      = matched_df["shape_match"].sum()
    shape_bad     = matched_df["matched"].sum() - shape_ok
    p(f"  {'T1 shape == Lesion shape':<38} {shape_ok}")
    p(f"  {'Shape MISMATCH':<38} {shape_bad}")

    # lesion content
    hr("Lesion Content (matched pairs only)")
    pos   = df["lesion_positive"].sum()
    empty = df["lesion_empty"].sum()
    err   = (df["pairing_status"] == "matched_error").sum()
    p(f"  {'Positive lesion  (voxels > 0)':<38} {pos}")
    p(f"  {'Empty / lesion-free (voxels = 0)':<38} {empty}")
    p(f"  {'Load error (corrupt / truncated)':<38} {err}")
    if pos + empty > 0:
        pct_pos   = 100 * pos   / (pos + empty)
        pct_empty = 100 * empty / (pos + empty)
        p(f"\n  Positive rate : {pct_pos:.1f}%   ({pos}/{pos+empty})")
        p(f"  Empty rate    : {pct_empty:.1f}%   ({empty}/{pos+empty})")

    # lesion volume stats
    pos_df = df[df["lesion_positive"]]
    if not pos_df.empty:
        vols = pos_df["lesion_voxels"]
        hr("Lesion Volume (positive cases, in voxels)")
        p(f"  {'Min':<20} {vols.min():>10,.0f}")
        p(f"  {'Max':<20} {vols.max():>10,.0f}")
        p(f"  {'Mean':<20} {vols.mean():>10,.1f}")
        p(f"  {'Median':<20} {vols.median():>10,.1f}")
        p(f"  {'Std dev':<20} {vols.std():>10,.1f}")

        # volume buckets
        hr("Lesion Size Buckets (positive cases)")
        bins   = [0, 10, 100, 500, 1000, 5000, 20000, 50000, np.inf]
        labels = ["1–10", "11–100", "101–500", "501–1K",
                  "1K–5K", "5K–20K", "20K–50K", ">50K"]
        bucket = pd.cut(vols, bins=bins, labels=labels)
        counts = bucket.value_counts().sort_index()
        for lbl, cnt in counts.items():
            bar = "█" * int(cnt / max(counts) * 30)
            p(f"  {lbl:<10} {cnt:>5}  {bar}")

    # dMRI pairing completeness
    hr("dMRI Completeness (for scans that have dMRI)")
    dmri_df = df[df["has_dmri"]]
    if not dmri_df.empty:
        p(f"  Scans with dMRI            : {len(dmri_df)}")
        p(f"  dMRI + bvec + bval (full)  : "
          f"{(dmri_df['has_bvec'] & dmri_df['has_bval']).sum()}")
        p(f"  dMRI missing bvec          : {(~dmri_df['has_bvec']).sum()}")
        p(f"  dMRI missing bval          : {(~dmri_df['has_bval']).sum()}")
        dmri_pos = dmri_df[dmri_df["lesion_positive"]]
        dmri_emp = dmri_df[dmri_df["lesion_empty"]]
        p(f"  dMRI + positive lesion     : {len(dmri_pos)}")
        p(f"  dMRI + empty lesion        : {len(dmri_emp)}")

    # errors
    hr("Files with Load Errors")
    err_df = df[df["has_any_error"]]
    if err_df.empty:
        p("  None  ✓")
    else:
        for _, r in err_df.iterrows():
            p(f"  scan_{r['scan_id']}  T1_err={r['t1_error']}  "
              f"Lesion_err={r['lesion_error']}")

    hr()
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — optional plots
# ═══════════════════════════════════════════════════════════════════════════════

def make_plots(df: pd.DataFrame, out_dir: Path):
    if not HAS_MPL:
        print("  matplotlib not available — skipping plots")
        return

    # ── Plot 1: pairing status overview ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("T1 ↔ Lesion Pairing Overview", fontsize=14, fontweight="bold")

    # pie: pairing status
    ax = axes[0]
    status_counts = df["pairing_status"].value_counts()
    colors_map = {
        "matched_positive": PAL["positive"],
        "matched_empty":    PAL["empty"],
        "matched_error":    PAL["mismatch"],
        "orphan_t1":        PAL["orphan"],
        "orphan_lesion":    PAL["T1"],
        "no_files":         "#dddddd",
    }
    colors = [colors_map.get(s, "#888") for s in status_counts.index]
    wedges, texts, autotexts = ax.pie(
        status_counts.values,
        labels=[f"{s}\n(n={v})" for s, v in status_counts.items()],
        colors=colors, autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 8},
    )
    ax.set_title("Pairing status breakdown")

    # bar: file presence
    ax = axes[1]
    presence = {
        "T1":     df["has_t1"].sum(),
        "Lesion": df["has_lesion"].sum(),
        "dMRI":   df["has_dmri"].sum(),
        "bvec":   df["has_bvec"].sum(),
        "bval":   df["has_bval"].sum(),
    }
    bars = ax.bar(presence.keys(), presence.values(),
                  color=[PAL.get(k, "#888780") for k in presence],
                  edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_ylabel("Count")
    ax.set_title("File presence per type")
    ax.spines[["top","right"]].set_visible(False)
    ax.set_ylim(0, max(presence.values()) * 1.15)

    # bar: matched vs orphan
    ax = axes[2]
    cats   = ["Matched\npairs", "Positive\nlesion", "Empty\nlesion",
              "Orphan T1\n(no Lesion)", "Orphan Lesion\n(no T1)"]
    vals   = [
        df["matched"].sum(),
        df["lesion_positive"].sum(),
        df["lesion_empty"].sum(),
        (df["pairing_status"] == "orphan_t1").sum(),
        (df["pairing_status"] == "orphan_lesion").sum(),
    ]
    colors2 = [PAL["T1"], PAL["positive"], PAL["empty"],
               PAL["orphan"], PAL["mismatch"]]
    bars2 = ax.bar(cats, vals, color=colors2, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars2, padding=3, fontsize=10)
    ax.set_ylabel("Count")
    ax.set_title("Pairing breakdown")
    ax.spines[["top","right"]].set_visible(False)
    ax.set_ylim(0, max(vals) * 1.15)

    plt.tight_layout()
    out = out_dir / "plot_01_pairing_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")

    # ── Plot 2: lesion volume distribution ────────────────────────────────────
    pos_df = df[df["lesion_positive"]]
    if not pos_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Lesion Volume Distribution (positive cases)", fontsize=14, fontweight="bold")

        vols = pos_df["lesion_voxels"].astype(float)

        ax = axes[0]
        ax.hist(vols, bins=50, color=PAL["positive"], alpha=0.8, edgecolor="white")
        ax.axvline(vols.median(), color="#222", linestyle="--",
                   linewidth=1.2, label=f"median={vols.median():.0f}")
        ax.set_xlabel("Lesion voxels")
        ax.set_ylabel("Subjects")
        ax.set_title("Linear scale")
        ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)

        ax = axes[1]
        ax.hist(np.log10(vols + 1), bins=50, color=PAL["positive"], alpha=0.8, edgecolor="white")
        ax.axvline(np.log10(vols.median() + 1), color="#222", linestyle="--",
                   linewidth=1.2, label=f"median={vols.median():.0f}")
        ax.set_xlabel("log₁₀(lesion voxels + 1)")
        ax.set_ylabel("Subjects")
        ax.set_title("Log scale")
        ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)

        ax = axes[2]
        bins   = [0, 10, 100, 500, 1000, 5000, 20000, 50000, np.inf]
        labels = ["1–10", "11–100", "101–500", "501–1K",
                  "1K–5K", "5K–20K", "20K–50K", ">50K"]
        bucket = pd.cut(vols, bins=bins, labels=labels)
        counts = bucket.value_counts().sort_index()
        bars3  = ax.bar(counts.index.astype(str), counts.values,
                        color=PAL["positive"], alpha=0.8, edgecolor="white")
        ax.bar_label(bars3, padding=3, fontsize=9)
        ax.set_xlabel("Lesion size bucket (voxels)")
        ax.set_ylabel("Subjects")
        ax.set_title("Size bucket distribution")
        ax.tick_params(axis="x", rotation=35)
        ax.spines[["top","right"]].set_visible(False)

        plt.tight_layout()
        out = out_dir / "plot_02_lesion_volume.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out}")

    # ── Plot 3: dMRI availability vs lesion status ─────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("dMRI Availability vs Lesion Status", fontsize=13, fontweight="bold")

    categories = [
        ("dMRI + positive lesion",  df["has_dmri"] & df["lesion_positive"],  PAL["positive"]),
        ("dMRI + empty lesion",     df["has_dmri"] & df["lesion_empty"],      PAL["empty"]),
        ("No dMRI + positive",      ~df["has_dmri"] & df["lesion_positive"],  "#F4A261"),
        ("No dMRI + empty",         ~df["has_dmri"] & df["lesion_empty"],     "#ccc"),
    ]
    labels_plot = [c[0] for c in categories]
    vals_plot   = [c[1].sum() for c in categories]
    colors_plot = [c[2] for c in categories]
    bars4 = ax.barh(labels_plot, vals_plot, color=colors_plot, edgecolor="white", height=0.55)
    ax.bar_label(bars4, padding=4, fontsize=10)
    ax.set_xlabel("Number of subjects")
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xlim(0, max(vals_plot) * 1.15)
    plt.tight_layout()
    out = out_dir / "plot_03_dmri_vs_lesion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AIMS-TBI Pairing Summary")
    parser.add_argument("--data_dir", required=True,
                        help="Flat dataset folder with all .nii.gz and .txt files")
    parser.add_argument("--plot", action="store_true",
                        help="Also generate bar/pie chart PNG figures")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        print(f"ERROR: {data_dir} does not exist"); sys.exit(1)

    out_dir = data_dir.parent / "pairing_summary"
    out_dir.mkdir(exist_ok=True)

    print(f"\nData dir  : {data_dir}")
    print(f"Output dir: {out_dir}\n")

    # ── 1. collect ──────────────────────────────────────────────────────────
    print("Step 1/4 — collecting file paths …")
    id_map = collect_id_map(data_dir)
    print(f"  Found {len(id_map)} unique scan IDs")

    # ── 2. build table ──────────────────────────────────────────────────────
    print("\nStep 2/4 — loading metadata (T1 shapes + Lesion stats) …")
    df = build_table(id_map)

    # ── 3. save CSVs ────────────────────────────────────────────────────────
    print("\nStep 3/4 — saving CSV files …")

    def save(sub_df, name, desc):
        path = out_dir / name
        sub_df.to_csv(path, index=False)
        print(f"  [{len(sub_df):>4} rows] {name:<45} {desc}")
        return path

    save(df,                                             "07_complete_summary.csv",       "all scan IDs, all columns")
    save(df[df["matched"]],                              "01_matched_pairs.csv",          "T1 + Lesion both present")
    save(df[df["lesion_positive"]],                      "02_positive_lesions.csv",       "matched, lesion_voxels > 0")
    save(df[df["lesion_empty"]],                         "03_empty_lesions.csv",          "matched, lesion_voxels == 0")
    save(df[df["pairing_status"] == "orphan_t1"],        "04_orphan_t1_no_lesion.csv",    "T1 exists, Lesion missing")
    save(df[df["pairing_status"] == "orphan_lesion"],    "05_orphan_lesion_no_t1.csv",    "Lesion exists, T1 missing")
    save(df[~df["shape_match"] & df["matched"]],         "06_shape_mismatches.csv",       "T1 shape != Lesion shape")

    # ── 4. report ───────────────────────────────────────────────────────────
    print("\nStep 4/4 — generating report …")
    report = build_report(df)
    print(report)

    rpath = out_dir / "00_console_report.txt"
    rpath.write_text(report)
    print(f"  Report saved → {rpath}")

    # ── optional plots ───────────────────────────────────────────────────────
    if args.plot:
        print("\nGenerating plots …")
        make_plots(df, out_dir)

    print(f"\n✓ All outputs in: {out_dir}")
    print("\nFiles generated:")
    for f in sorted(out_dir.iterdir()):
        size = f.stat().st_size
        print(f"  {f.name:<50} {size/1e3:>8.1f} KB")


if __name__ == "__main__":
    main()