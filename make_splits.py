from __future__ import annotations

import argparse
from pathlib import Path

from multitalent_tbi.config import load_config, resolve_path
from multitalent_tbi.splits import build_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified k-fold splits for AIMS-TBI.")
    parser.add_argument("--config", default="config.yml", help="Path to the YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    base_dir = Path(args.config).expanduser().resolve().parent
    build_splits(
        dataset_dir=resolve_path(base_dir, config.paths.dataset_dir),
        num_folds=int(config.training.num_folds),
        seed=int(config.training.seed),
        output_path=resolve_path(base_dir, config.paths.splits_file),
    )
    print(f"Saved splits to {resolve_path(base_dir, config.paths.splits_file)}")


if __name__ == "__main__":
    main()
