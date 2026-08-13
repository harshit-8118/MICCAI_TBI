"""Stage fold-0 validation images for an isolated full-volume monitor.

It creates only symlinks/copies to the already prepared Dataset501 validation
images and a JSON case-ID list. It does not touch the trainer, consume any
released Phase-2 data, or select a checkpoint. Use it once before running
occasional checkpoint snapshots on an otherwise idle GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dataset-dir", type=Path, required=True, help="Prepared Dataset501_AIMSTBI_T1_Standard directory.")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("checkpoints/paper_baselines/standard_nnunet_model_b_matched"),
    )
    parser.add_argument("--fold", type=int, default=0, choices=(0,))
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dataset_dir = args.raw_dataset_dir.resolve()
    splits_path = raw_dataset_dir / "splits_final.json"
    images_tr = raw_dataset_dir / "imagesTr"
    if not splits_path.is_file() or not images_tr.is_dir():
        raise FileNotFoundError("Expected splits_final.json and imagesTr in --raw-dataset-dir.")
    with splits_path.open(encoding="utf-8") as handle:
        splits = json.load(handle)
    if not isinstance(splits, list) or args.fold >= len(splits):
        raise ValueError("splits_final.json does not contain the requested fold.")
    val_ids = [str(case_id) for case_id in splits[args.fold]["val"]]
    if len(set(val_ids)) != len(val_ids):
        raise ValueError("Validation case IDs are not unique.")
    monitor_root = args.run_root.resolve() / "internal_fold_monitor" / f"fold_{args.fold}"
    input_dir = monitor_root / "images"
    if monitor_root.exists():
        raise FileExistsError(f"Refusing to reuse monitor directory: {monitor_root}")
    input_dir.mkdir(parents=True)
    manifest: list[dict[str, str]] = []
    for case_id in val_ids:
        source = images_tr / f"{case_id}_0000.nii.gz"
        destination = input_dir / source.name
        if not source.is_file():
            raise FileNotFoundError(f"Missing validation image: {source}")
        if args.link_mode == "symlink":
            os.symlink(source, destination)
        else:
            shutil.copy2(source, destination)
        manifest.append({"case_id": case_id, "source_image": str(source), "monitor_image": str(destination)})
    case_ids_path = monitor_root / "fold0_validation_case_ids.json"
    with case_ids_path.open("w", encoding="utf-8") as handle:
        json.dump({"case_ids": val_ids}, handle, indent=2)
        handle.write("\n")
    with (monitor_root / "input_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"fold": args.fold, "n_cases": len(val_ids), "images": manifest}, handle, indent=2)
        handle.write("\n")
    print(f"[DONE] Prepared {len(val_ids)} fold-{args.fold} validation images: {input_dir}")
    print(f"[DONE] Case-ID restriction: {case_ids_path}")


if __name__ == "__main__":
    main()
