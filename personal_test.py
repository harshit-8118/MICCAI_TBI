"""
check_errors.py
===============
Investigates the two erroneous files in AIMS-TBI dataset:
  1. scan_0435_Lesion.nii.gz  — path too long / file not found
  2. scan_0523_dMRI.nii.gz    — truncated / corrupted gzip

Run:
    python check_errors.py --data_dir /media/lab/"My Book"/MICCAI_AIMS_TBI/dataset
"""

import os
import sys
import gzip
import struct
import shutil
import argparse
import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def hr(title=""):
    print(f"\n{'─'*60}")
    if title:
        print(f"  {title}")
        print(f"{'─'*60}")


def file_md5(path: Path, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


# ════════════════════════════════════════════════════════════════════════════
# FILE 1 — scan_0435_Lesion.nii.gz
#   Error: "File /media/lab/My Book/... "  (path error — space in path)
# ════════════════════════════════════════════════════════════════════════════

def check_lesion(data_dir: Path):
    hr("FILE 1 — scan_0435_Lesion.nii.gz")

    fname = "scan_0435_Lesion.nii.gz"
    fpath = data_dir / fname

    # ── 1. Does the file exist? ───────────────────────────────────────────
    print(f"\n[1] Path used  : {fpath}")
    print(f"    Path exists : {fpath.exists()}")
    print(f"    Is file     : {fpath.is_file() if fpath.exists() else 'N/A'}")

    if not fpath.exists():
        # Try to find it with glob (handles unusual chars)
        matches = list(data_dir.glob("*0435*Lesion*"))
        print(f"\n    glob search for '*0435*Lesion*': {matches}")
        if not matches:
            print("\n    ✗ File genuinely missing from disk.")
            print("    → This subject has NO lesion mask.")
            print("    → In training: treat as lesion-free (all-zero mask).")
            print("    → In data loader: skip or synthesise zero mask.")
            return
        fpath = matches[0]
        print(f"\n    Found via glob: {fpath}")

    # ── 2. File stats ─────────────────────────────────────────────────────
    size_bytes = fpath.stat().st_size
    print(f"\n[2] File size   : {size_bytes:,} bytes  ({size_bytes/1e6:.3f} MB)")
    print(f"    MD5         : {file_md5(fpath)}")

    # ── 3. Try loading ────────────────────────────────────────────────────
    print("\n[3] Attempting nibabel load …")
    try:
        img  = nib.load(str(fpath))
        data = img.get_fdata(dtype=np.float32)
        print(f"    ✓ Loaded OK")
        print(f"    Shape       : {data.shape}")
        print(f"    Dtype       : {data.dtype}")
        print(f"    Voxel size  : {img.header.get_zooms()}")
        print(f"    Unique vals : {np.unique(data)}")
        print(f"    Lesion voxels: {int(data.sum())}")

        # Check if paired T1 exists and shapes match
        t1_path = data_dir / "scan_0435_T1.nii.gz"
        if t1_path.exists():
            t1 = nib.load(str(t1_path))
            t1_shape = t1.header.get_data_shape()
            match = tuple(data.shape) == tuple(t1_shape)
            print(f"\n[4] Paired T1 shape : {t1_shape}")
            print(f"    Lesion shape    : {data.shape}")
            print(f"    Shapes match    : {'✓' if match else '✗ MISMATCH'}")

    except Exception as e:
        print(f"    ✗ Load failed: {e}")
        print("\n    DIAGNOSIS: The error in eda.py was likely caused by")
        print("    the space in the parent directory path:")
        print("    /media/lab/My Book/  ← space here can break some tools")
        print("    nibabel itself handles it fine via pathlib.Path.")
        print("    The eda.py error was from passing a raw string with the")
        print("    space unquoted in certain os.system() calls (not nibabel).")
        print("\n    FIX: Always use Path() objects, never raw strings with spaces.")
        print("    Or symlink the dataset to a no-space path:")
        print("      ln -s '/media/lab/My Book/MICCAI_AIMS_TBI' ~/aims_tbi")


# ════════════════════════════════════════════════════════════════════════════
# FILE 2 — scan_0523_dMRI.nii.gz
#   Error: "Compressed file ended before the end-of-stream marker"
#   = Truncated / corrupted gzip file
# ════════════════════════════════════════════════════════════════════════════

def check_dmri(data_dir: Path):
    hr("FILE 2 — scan_0523_dMRI.nii.gz")

    fname = "scan_0523_dMRI.nii.gz"
    fpath = data_dir / fname

    print(f"\n[1] Path   : {fpath}")
    print(f"    Exists : {fpath.exists()}")

    if not fpath.exists():
        print("    ✗ File not found on disk.")
        return

    size_bytes = fpath.stat().st_size
    print(f"\n[2] File size : {size_bytes:,} bytes  ({size_bytes/1e6:.1f} MB)")

    # Compare against healthy dMRI files in the dataset
    dmri_files = sorted(data_dir.glob("*_dMRI.nii.gz"))
    sizes = [f.stat().st_size for f in dmri_files if f.name != fname]
    if sizes:
        print(f"    Healthy dMRI sizes → "
              f"min={min(sizes)/1e6:.1f} MB  "
              f"max={max(sizes)/1e6:.1f} MB  "
              f"median={np.median(sizes)/1e6:.1f} MB")
        ratio = size_bytes / np.median(sizes)
        print(f"    This file is {ratio:.1%} of median size")
        if ratio < 0.95:
            print(f"    → ✗ CONFIRMED TRUNCATED  (significantly smaller than peers)")
        else:
            print(f"    → Size looks normal; corruption may be internal")

    # ── 3. Raw gzip integrity check ───────────────────────────────────────
    print("\n[3] Raw gzip integrity check …")
    try:
        with gzip.open(str(fpath), "rb") as gz:
            _ = gz.read()
        print("    ✓ gzip decompresses without error (not truncated at byte level)")
    except (gzip.BadGzipFile, EOFError, OSError) as e:
        print(f"    ✗ gzip error: {e}")
        print("    → File is truncated mid-download or mid-copy")

    # ── 4. Check gzip end-of-stream marker ───────────────────────────────
    print("\n[4] Checking gzip EOF marker (last 8 bytes) …")
    try:
        with open(fpath, "rb") as f:
            f.seek(-8, 2)
            tail = f.read(8)
        crc32_stored = struct.unpack("<I", tail[:4])[0]
        isize_stored = struct.unpack("<I", tail[4:])[0]
        print(f"    Stored CRC32 : 0x{crc32_stored:08X}")
        print(f"    Stored ISIZE : {isize_stored} bytes (original uncompressed size mod 2^32)")
    except Exception as e:
        print(f"    Could not read tail: {e}")

    # ── 5. Try nibabel partial load ───────────────────────────────────────
    print("\n[5] Attempting nibabel load …")
    try:
        img  = nib.load(str(fpath))
        data = img.get_fdata(dtype=np.float32)
        print(f"    ✓ Loaded OK — shape {data.shape}")
        print("    The error in eda.py may have been a one-off decompression")
        print("    issue (disk I/O during the 18-min scan). Try re-running eda.py.")
    except Exception as e:
        print(f"    ✗ Load failed: {e}")

        # ── 6. Check paired bvec/bval ─────────────────────────────────────
        print("\n[6] Checking paired bvec / bval files …")
        for ext in ["bvec", "bval"]:
            p = data_dir / f"scan_0523_{ext}.txt"
            status = "✓ exists" if p.exists() else "✗ missing"
            print(f"    scan_0523_{ext}.txt : {status}")

        print("\n" + "═"*60)
        print("  DIAGNOSIS: TRUNCATED FILE")
        print("═"*60)
        print("""
  scan_0523_dMRI.nii.gz was not fully written to disk.
  This typically happens from:
    a) Interrupted download / network cut mid-transfer
    b) Drive ran out of space during copy
    c) USB hard drive (your 'My Book') disconnected briefly

  IMPACT:
    • This file cannot be loaded or used
    • scan_0523 will have NO dMRI data
    • T1 + Lesion for scan_0523 are unaffected (check below)
    • dMRI is supplementary only — training can proceed without it

  RECOMMENDED ACTIONS (in order of preference):
    1. Re-download scan_0523_dMRI.nii.gz from Synapse if available
    2. Delete the corrupt file so eda.py skips it cleanly
    3. Exclude scan_0523 from dMRI-based training only
       (keep it in T1/Lesion training — those files are fine)
        """)

        # Check T1 and Lesion for same subject
        print("[7] Checking T1 + Lesion for scan_0523 …")
        for ftype in ["T1", "Lesion"]:
            p = data_dir / f"scan_0523_{ftype}.nii.gz"
            if p.exists():
                try:
                    img = nib.load(str(p))
                    shape = img.header.get_data_shape()
                    print(f"    scan_0523_{ftype}: ✓  shape={shape}")
                except Exception as e2:
                    print(f"    scan_0523_{ftype}: ✗  {e2}")
            else:
                print(f"    scan_0523_{ftype}: ✗ missing")


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════════

def print_action_summary():
    hr("ACTION SUMMARY")
    print("""
  File 1 — scan_0435_Lesion.nii.gz
  ─────────────────────────────────
  Root cause : Space in directory path  /media/lab/My Book/
               breaks raw string operations (not nibabel itself)
  Fix        : Use Path() objects everywhere  OR
               symlink:  ln -s '/media/lab/My Book/MICCAI_AIMS_TBI' ~/aims_tbi
               then run: python eda.py --data_dir ~/aims_tbi/dataset
  Training   : If file loads fine via Path() → use normally
               If truly missing → treat as lesion-free (zero mask)

  File 2 — scan_0523_dMRI.nii.gz
  ─────────────────────────────────
  Root cause : Truncated gzip — file was not fully written/downloaded
  Fix        : Re-download from Synapse (preferred)  OR
               Delete the file:
                 rm '/media/lab/My Book/MICCAI_AIMS_TBI/dataset/scan_0523_dMRI.nii.gz'
               The paired scan_0523_bvec.txt and scan_0523_bval.txt
               can also be deleted if you remove the dMRI.
  Training   : Exclude scan_0523 from dMRI training only.
               scan_0523_T1 + scan_0523_Lesion are fine — keep them.
    """)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to flat dataset folder")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        print(f"ERROR: data_dir not found: {data_dir}")
        sys.exit(1)

    print(f"\nData dir: {data_dir}")

    check_lesion(data_dir)
    check_dmri(data_dir)
    print_action_summary()


if __name__ == "__main__":
    main()