from __future__ import annotations

import argparse
from pathlib import Path

from multitalent_tbi.config import load_config, resolve_path
from multitalent_tbi.data import discover_cases
from multitalent_tbi.engine import load_case_records_for_fold, _summarize_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit dataset and fold split counts for AIMS-TBI.")
    parser.add_argument("--config", default="config.yml", help="Path to the YAML config.")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to audit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    base_dir = Path(args.config).expanduser().resolve().parent
    dataset_dir = resolve_path(base_dir, config.paths.dataset_dir)
    records = discover_cases(dataset_dir)
    train_records, val_records = load_case_records_for_fold(config, args.fold, base_dir)
    print('FLag1')
    total_summary = _summarize_records(records)
    print('FLag2')
    train_summary = _summarize_records(train_records)
    print('FLag3')
    val_summary = _summarize_records(val_records)
    print('FLag4')

    print(f"Dataset directory: {dataset_dir}")
    print(
        "All cases: "
        f"total={total_summary['total']}, "
        f"lesion_positive={total_summary['lesion_positive']}, "
        f"lesion_empty={total_summary['lesion_empty']}, "
        f"dmri={total_summary['dmri']}"
    )
    print(
        f"Fold {args.fold} train: "
        f"total={train_summary['total']}, "
        f"lesion_positive={train_summary['lesion_positive']}, "
        f"lesion_empty={train_summary['lesion_empty']}, "
        f"dmri={train_summary['dmri']}"
    )
    print(
        f"Fold {args.fold} val: "
        f"total={val_summary['total']}, "
        f"lesion_positive={val_summary['lesion_positive']}, "
        f"lesion_empty={val_summary['lesion_empty']}, "
        f"dmri={val_summary['dmri']}"
    )


if __name__ == "__main__":
    main()

# python3 audit_dataset.py --config config.yml --fold 0

'''
Dataset directory: /data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning/MICCAI_AIMS_TBI
All cases: total=329, lesion_positive=190, lesion_empty=139, dmri=140
Fold 0 train: total=263, lesion_positive=152, lesion_empty=111, dmri=112
Fold 0 val: total=66, lesion_positive=38, lesion_empty=28, dmri=28
'''