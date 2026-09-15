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
    