from __future__ import annotations

import argparse
import json
from pathlib import Path

from multitalent_tbi.case_filters import (
    build_case_infos,
    empty_infos,
    positive_infos,
    write_case_id_list,
    write_case_manifest,
    write_mask_path_list,
)
from multitalent_tbi.config import load_config, resolve_path
from multitalent_tbi.data import discover_cases
from multitalent_tbi.splits import load_splits, split_records

def flag(f):
    print("\n", "-"*20, f, "\n", "-"*20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create lesion-positive and empty case manifests for AIMS-TBI experiments."
    )
    parser.add_argument("--config", default="config.yml", help="Path to config.yml.")
    parser.add_argument("--fold", type=int, default=None, help="Fold index. Defaults to config training.fold.")
    parser.add_argument("--output-dir", default=None, help="Override output filter directory.")
    parser.add_argument("--min-positive-voxels", type=int, default=1, help="Minimum GT voxels for lesion-positive list.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    base_dir = Path(args.config).expanduser().resolve().parent
    fold = int(args.fold if args.fold is not None else config.training.fold)
    flag(1)
    dataset_dir = resolve_path(base_dir, config.paths.dataset_dir)
    splits_path = resolve_path(base_dir, config.paths.splits_file)
    output_dir = resolve_path(
        base_dir,
        args.output_dir
        or getattr(
            getattr(config, "segmenter_a", object()),
            "filters_dir",
            str(resolve_path(base_dir, config.paths.work_dir) / "case_filters"),
        ),
    )
    flag(2)

    records = discover_cases(dataset_dir)
    splits = load_splits(splits_path)
    train_records, val_records = split_records(records, splits[fold])
    flag(3)

    all_infos = build_case_infos(records)
    train_infos = build_case_infos(train_records, split="train")
    flag(4)
    val_infos = build_case_infos(val_records, split="val")
    fold_infos = train_infos + val_infos

    fold_dir = output_dir / f"fold_{fold}"
    positive_dir = fold_dir / "lesion_positive"
    empty_dir = fold_dir / "empty"

    write_case_manifest(output_dir / "all_cases.csv", all_infos)
    write_case_manifest(fold_dir / "all_fold_cases.csv", fold_infos)
    write_case_manifest(positive_dir / "train_cases.csv", positive_infos(train_infos, args.min_positive_voxels))
    write_case_manifest(positive_dir / "val_cases.csv", positive_infos(val_infos, args.min_positive_voxels))
    write_case_manifest(empty_dir / "train_cases.csv", empty_infos(train_infos))
    write_case_manifest(empty_dir / "val_cases.csv", empty_infos(val_infos))
    flag(5)

    for split_name, infos in [
        ("train", train_infos),
        ("val", val_infos),
    ]:
        pos = positive_infos(infos, args.min_positive_voxels)
        emp = empty_infos(infos)
        write_case_id_list(positive_dir / f"{split_name}_ids.txt", pos)
        write_mask_path_list(positive_dir / f"{split_name}_mask_paths.txt", pos)
        write_case_id_list(empty_dir / f"{split_name}_ids.txt", emp)
        write_mask_path_list(empty_dir / f"{split_name}_mask_paths.txt", emp)

    flag(6)
    summary = {
        "fold": fold,
        "output_dir": str(fold_dir),
        "train_total": len(train_infos),
        "train_positive": len(positive_infos(train_infos, args.min_positive_voxels)),
        "train_empty": len(empty_infos(train_infos)),
        "val_total": len(val_infos),
        "val_positive": len(positive_infos(val_infos, args.min_positive_voxels)),
        "val_empty": len(empty_infos(val_infos)),
    }
    with (fold_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
