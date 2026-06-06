from __future__ import annotations

import argparse

from multitalent_tbi.engine import train_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MultiTalentV2 on AIMS-TBI.")
    parser.add_argument("--config", default="config.yml", help="Path to the YAML config.")
    parser.add_argument("--fold", type=int, default=None, help="Fold index to train.")
    parser.add_argument("--all-folds", action="store_true", help="Train all folds sequentially.")
    parser.add_argument("--epochs", type=int, default=None, help="Override total training epochs.")
    parser.add_argument("--base-lr", type=float, default=None, help="Override the base learning rate.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--use-amp", action="store_true", help="Force AMP on.")
    parser.add_argument("--no-amp", action="store_true", help="Force AMP off.")
    parser.add_argument("--staged-tuning", action="store_true", help="Enable staged tuning.")
    parser.add_argument("--no-staged-tuning", action="store_true", help="Disable staged tuning.")
    parser.add_argument("--head-only-epochs", type=int, default=None, help="Override head-only stage length.")
    parser.add_argument("--partial-tune-epochs", type=int, default=None, help="Override partial-tune stage length.")
    parser.add_argument("--head-lr", type=float, default=None, help="Override head learning rate.")
    parser.add_argument("--partial-lr", type=float, default=None, help="Override partial learning rate.")
    parser.add_argument("--full-lr", type=float, default=None, help="Override full fine-tuning learning rate.")
    parser.add_argument("--validation-interval", type=int, default=None, help="Validate every N epochs during training. Use 0 to disable.")
    parser.add_argument("--final-validation", action="store_true", help="Run validation once at the end on the best checkpoint.")
    parser.add_argument("--no-final-validation", action="store_true", help="Skip the final end-of-training validation.")
    parser.add_argument("--best-checkpoint-metric", type=str, default=None, help="Metric used to select the best checkpoint.")
    parser.add_argument("--best-checkpoint-mode", type=str, choices=["max", "min"], default=None, help="Whether the best checkpoint metric should be maximized or minimized.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides: dict[str, object] = {}
    if args.epochs is not None:
        overrides.setdefault("training", {})["max_epochs"] = args.epochs
    if args.base_lr is not None:
        overrides.setdefault("training", {})["base_lr"] = args.base_lr
    if args.batch_size is not None:
        overrides.setdefault("training", {})["batch_size"] = args.batch_size
    if args.num_workers is not None:
        overrides.setdefault("training", {})["num_workers"] = args.num_workers
    if args.use_amp:
        overrides.setdefault("training", {})["use_amp"] = True
    if args.no_amp:
        overrides.setdefault("training", {})["use_amp"] = False
    if args.staged_tuning or args.no_staged_tuning:
        staged = overrides.setdefault("training", {}).setdefault("staged_tuning", {})
        staged["enabled"] = bool(args.staged_tuning and not args.no_staged_tuning)
    if args.head_only_epochs is not None:
        overrides.setdefault("training", {}).setdefault("staged_tuning", {})["head_only_epochs"] = args.head_only_epochs
    if args.partial_tune_epochs is not None:
        overrides.setdefault("training", {}).setdefault("staged_tuning", {})["partial_tune_epochs"] = args.partial_tune_epochs
    if args.head_lr is not None:
        overrides.setdefault("training", {}).setdefault("staged_tuning", {})["head_lr"] = args.head_lr
    if args.partial_lr is not None:
        overrides.setdefault("training", {}).setdefault("staged_tuning", {})["partial_lr"] = args.partial_lr
    if args.full_lr is not None:
        overrides.setdefault("training", {}).setdefault("staged_tuning", {})["full_lr"] = args.full_lr
    if args.validation_interval is not None:
        overrides.setdefault("training", {})["validation_interval_epochs"] = args.validation_interval
    if args.final_validation or args.no_final_validation:
        overrides.setdefault("training", {})["final_validation"] = bool(args.final_validation and not args.no_final_validation)
    if args.best_checkpoint_metric is not None:
        overrides.setdefault("training", {})["best_checkpoint_metric"] = args.best_checkpoint_metric
    if args.best_checkpoint_mode is not None:
        overrides.setdefault("training", {})["best_checkpoint_mode"] = args.best_checkpoint_mode

    results = train_from_config(args.config, fold=args.fold, all_folds=args.all_folds, overrides=overrides or None)
    for item in results:
        print(item)


if __name__ == "__main__":
    main()



# python3 train.py --config config.yml --fold 0 --validation-interval 10
# python3 validate.py --config config.yml --fold 0