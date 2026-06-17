"""
resume_train.py — continue training from a best.pt (or any .pt) checkpoint.

Usage example:
    python resume_train.py \
        --config /path/to/config.yml \
        --checkpoint /path/to/fold_0/best.pt \
        --output-dir /path/to/fold_0_resumed \
        --fold 0 \
        --extra-epochs 50 \
        --lr-head 1e-5 --lr-partial 5e-6 --lr-full 2e-6 \
        --stage full \
        --validation-interval 5 \
        --wandb --wandb-name resumed_fold0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Re-use all helpers from the existing package
# ---------------------------------------------------------------------------
from multitalent_tbi.config import load_config, resolve_path
from multitalent_tbi.data import TBIDataset, discover_cases
from multitalent_tbi.infer import predict_logits
from multitalent_tbi.losses import dice_ce_loss
from multitalent_tbi.model import build_backbone
from multitalent_tbi.splits import load_splits, split_records
from multitalent_tbi.engine import (
    set_seed,
    configure_torch_for_speed,
    dice_score,
    _batch_dice,
    _apply_stage_freezing,
    _categorize_parameter,
    _normalize_patterns,
    _get_staged_tuning,
    _build_case_sampler,
    _next_run_log_path,
    _summarize_records,
    _namespace_to_dict,
    build_optimizer_from_param_groups,
    evaluate_fold,
)


# ---------------------------------------------------------------------------
# LR injection helpers
# ---------------------------------------------------------------------------

def _inject_lrs(optimizer: torch.optim.Optimizer, lr_map: dict[str, float]) -> None:
    """Overwrite lr and base_lr for named param groups."""
    for group in optimizer.param_groups:
        name = str(group.get("group_name", ""))
        if name in lr_map:
            group["lr"] = lr_map[name]
            group["base_lr"] = lr_map[name]


def _current_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    summary: dict[str, float] = {}
    for group in optimizer.param_groups:
        name = str(group.get("group_name", f"group_{id(group)}"))
        summary[name] = float(group["lr"])
    return summary


def _build_resume_optimizer(
    model: torch.nn.Module,
    config,
    lr_head: float | None,
    lr_partial: float | None,
    lr_full: float | None,
) -> torch.optim.Optimizer:
    staged = _get_staged_tuning(config)
    head_patterns = _normalize_patterns(
        getattr(staged, "head_patterns", None) if staged else None,
        ["seg", "final", "classifier", "output"],
    )
    partial_patterns = _normalize_patterns(
        getattr(staged, "partial_patterns", None) if staged else None,
        ["decoder", "up", "localization", "stages.4", "stages.5", "stages.6"],
    )

    # Use provided LRs, fall back to config values
    def _cfg_lr(key: str, default: float) -> float:
        if staged:
            return float(getattr(staged, key, default))
        return float(getattr(config.training, "base_lr", default))

    lrs = {
        "head":    lr_head    if lr_head    is not None else _cfg_lr("head_lr",    1e-4),
        "partial": lr_partial if lr_partial is not None else _cfg_lr("partial_lr", 1e-4),
        "full":    lr_full    if lr_full    is not None else _cfg_lr("full_lr",    1e-4),
    }

    groups: dict[str, list] = {"head": [], "partial": [], "full": []}
    for name, param in model.named_parameters():
        cat = _categorize_parameter(name, head_patterns, partial_patterns)
        groups[cat].append(param)

    param_groups = [
        {"params": groups[cat], "lr": lrs[cat], "base_lr": lrs[cat], "group_name": cat}
        for cat in ("head", "partial", "full") if groups[cat]
    ]
    return build_optimizer_from_param_groups(
        config.training.optimizer, param_groups, float(config.training.weight_decay)
    )


# ---------------------------------------------------------------------------
# Core resume loop
# ---------------------------------------------------------------------------

def resume_training(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    fold: int,
    extra_epochs: int,
    stage: str,
    lr_head: float | None,
    lr_partial: float | None,
    lr_full: float | None,
    validation_interval: int,
    final_validation: bool,
    best_checkpoint_metric: str,
    best_checkpoint_mode: str,
    wandb_kwargs: dict,
) -> dict:
    config = load_config(config_path)
    base_dir = Path(config_path).expanduser().resolve().parent
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(config.training.seed) + fold)
    configure_torch_for_speed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(config.training.use_amp) and device.type == "cuda"

    # ---- load checkpoint ----
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    resumed_from_epoch = int(ckpt.get("epoch", 0))

    # ---- build model from architecture (no pretrained reload) ----
    plans_path = resolve_path(base_dir, config.paths.pretrained_plans)
    model = build_backbone(
        plans_path=plans_path,
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        deep_supervision=config.model.deep_supervision,
    ).to(memory_format=torch.channels_last_3d)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    # ---- staged freeze ----
    staged = _get_staged_tuning(config)
    head_patterns = _normalize_patterns(
        getattr(staged, "head_patterns", None) if staged else None,
        ["seg", "final", "classifier", "output"],
    )
    partial_patterns = _normalize_patterns(
        getattr(staged, "partial_patterns", None) if staged else None,
        ["decoder", "up", "localization", "stages.4", "stages.5", "stages.6"],
    )
    trainable_counts = _apply_stage_freezing(model, stage, head_patterns, partial_patterns)

    # ---- optimizer: build fresh, inject LRs ----
    optimizer = _build_resume_optimizer(model, config, lr_head, lr_partial, lr_full)

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    class_weights = torch.tensor(config.training.class_weights, dtype=torch.float32, device=device)

    # ---- dataloaders ----
    records = discover_cases(resolve_path(base_dir, config.paths.dataset_dir))
    splits = load_splits(resolve_path(base_dir, config.paths.splits_file))
    train_records, val_records = split_records(records, splits[fold])
    train_sampler, sampling_summary = _build_case_sampler(train_records, config, fold)

    train_dataset = TBIDataset(
        records=train_records,
        patch_size=config.data.patch_size,
        target_spacing=config.data.target_spacing,
        include_dmri=config.data.include_dmri,
        dmri_reduce=config.data.dmri_reduce,
        dmri_b0_threshold=config.data.dmri_b0_threshold,
        normalize_foreground_only=config.data.normalize_foreground_only,
        oversample_foreground_prob=config.data.oversample_foreground_prob,
        training=True,
        cache_dir=resolve_path(base_dir, config.paths.cache_dir) if config.data.cache_preprocessed else None,
    )
    val_dataset = TBIDataset(
        records=val_records,
        patch_size=config.data.patch_size,
        target_spacing=config.data.target_spacing,
        include_dmri=config.data.include_dmri,
        dmri_reduce=config.data.dmri_reduce,
        dmri_b0_threshold=config.data.dmri_b0_threshold,
        normalize_foreground_only=config.data.normalize_foreground_only,
        oversample_foreground_prob=0.0,
        training=False,
        cache_dir=resolve_path(base_dir, config.paths.cache_dir) if config.data.cache_preprocessed else None,
    )
    loader_kw = {"pin_memory": True}
    if config.training.num_workers > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = int(getattr(config.training, "prefetch_factor", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.training.num_workers,
        **loader_kw,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # ---- wandb ----
    wandb_run = None
    if wandb_kwargs.get("enabled"):
        try:
            import wandb
            wandb_run = wandb.init(
                project=wandb_kwargs.get("project", "AIMS-TBI-MultiTalentV2"),
                entity=wandb_kwargs.get("entity"),
                name=wandb_kwargs.get("name") or f"resume_fold{fold}",
                dir=str(output_dir / "wandb"),
                mode=wandb_kwargs.get("mode", "online"),
                config={"resumed_from_epoch": resumed_from_epoch, "extra_epochs": extra_epochs,
                        "stage": stage, "lr_head": lr_head, "lr_partial": lr_partial, "lr_full": lr_full},
                reinit=True,
            )
        except ImportError:
            pass

    # ---- logging setup ----
    run_log_path = _next_run_log_path(output_dir)
    history_path = output_dir / "history_resume.csv"
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    best_monitor_value = float(ckpt.get("monitor_value", -math.inf if best_checkpoint_mode == "max" else math.inf))
    best_epoch = resumed_from_epoch

    with history_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "phase", "epoch", "train_loss", "train_dice",
            "val_loss", "val_dice", "monitor_metric", "monitor_value",
            "lr_head", "lr_partial", "lr_full",
        ])

    with run_log_path.open("w", encoding="utf-8") as f:
        f.write(f"Resume from: {checkpoint_path}  (epoch {resumed_from_epoch})\n")
        f.write(f"Stage: {stage} | extra_epochs: {extra_epochs} | device: {device}\n")
        f.write(f"LRs: head={lr_head} partial={lr_partial} full={lr_full}\n")
        f.write(f"trainable(head/partial/full)={trainable_counts}\n")
        f.write("phase\tepoch\tstage\ttrain_loss\ttrain_dice\tval_loss\tval_dice\tmonitor_metric\tmonitor_value\tlr_head\tlr_partial\tlr_full\n")

    # ---- training loop ----
    for local_epoch in range(extra_epochs):
        global_epoch = resumed_from_epoch + local_epoch + 1
        model.train()
        running_loss = running_dice = 0.0
        train_batches = 0
        lr_summary = _current_lrs(optimizer)

        bar = tqdm(train_loader, desc=f"Epoch {global_epoch}", leave=False)
        for batch in bar:
            images = batch["image"].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last_3d)
            masks = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                loss = dice_ce_loss(logits, masks, class_weights=class_weights)
            scaler.scale(loss).backward()
            if config.training.grad_clip_norm and config.training.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.training.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())
            running_dice += _batch_dice(logits.detach(), masks.detach())
            train_batches += 1
            bar.set_postfix(loss=float(loss.detach().cpu()))

        avg_loss = running_loss / max(1, len(train_loader))
        avg_dice = running_dice / max(1, train_batches)
        val_loss = val_dice = float("nan")

        if validation_interval > 0 and global_epoch % validation_interval == 0:
            val_loss, val_dice = evaluate_fold(
                model=model, loader=val_loader, config=config,
                device=device, use_amp=amp_enabled, class_weights=class_weights,
            )

        monitor_value = avg_dice if best_checkpoint_metric == "train_dice" else (
            avg_loss if best_checkpoint_metric == "train_loss" else (
            val_dice if best_checkpoint_metric == "val_dice" else val_loss))

        should_save_best = math.isfinite(monitor_value) and (
            (best_checkpoint_mode == "max" and monitor_value > best_monitor_value) or
            (best_checkpoint_mode == "min" and monitor_value < best_monitor_value)
        )

        lr_head_v   = float(lr_summary.get("head",    0.0))
        lr_partial_v = float(lr_summary.get("partial", 0.0))
        lr_full_v   = float(lr_summary.get("full",    0.0))

        with history_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "train", global_epoch, avg_loss, avg_dice,
                val_loss, val_dice, best_checkpoint_metric, monitor_value,
                lr_head_v, lr_partial_v, lr_full_v,
            ])
        with run_log_path.open("a", encoding="utf-8") as f:
            f.write(f"train\t{global_epoch}\t{stage}\t{avg_loss:.6f}\t{avg_dice:.6f}\t"
                    f"{val_loss:.6f}\t{val_dice:.6f}\t{best_checkpoint_metric}\t{monitor_value:.6f}\t"
                    f"{lr_head_v:.8f}\t{lr_partial_v:.8f}\t{lr_full_v:.8f}\n")

        print(f"Epoch {global_epoch:04d} [{stage}] | loss={avg_loss:.4f} | dice={avg_dice:.4f} | "
              f"val_loss={val_loss:.4f} | val_dice={val_dice:.4f} | "
              f"lr(h/p/f)={lr_head_v:.2e}/{lr_partial_v:.2e}/{lr_full_v:.2e}")

        if wandb_run is not None:
            wandb_run.log({
                "epoch": global_epoch, "train/loss": avg_loss, "train/dice": avg_dice,
                "val/loss": val_loss, "val/dice": val_dice,
                "monitor/value": monitor_value, "lr/head": lr_head_v,
                "lr/partial": lr_partial_v, "lr/full": lr_full_v,
            }, step=global_epoch)

        payload = {
            "epoch": global_epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "lr_summary": lr_summary,
            "monitor_metric": best_checkpoint_metric, "monitor_value": monitor_value,
            "train_loss": avg_loss, "train_dice": avg_dice,
            "val_dice": val_dice, "val_loss": val_loss,
        }
        torch.save(payload, last_path)
        if should_save_best:
            best_monitor_value = monitor_value
            best_epoch = global_epoch
            torch.save(payload, best_path)

    # ---- final validation on best ----
    if final_validation:
        best_ckpt_path = best_path if best_path.exists() else last_path
        best_payload = torch.load(str(best_ckpt_path), map_location="cpu", weights_only=False)
        model.load_state_dict(best_payload["model_state"])
        fv_loss, fv_dice = evaluate_fold(
            model=model, loader=val_loader, config=config,
            device=device, use_amp=amp_enabled, class_weights=class_weights,
        )
        print(f"Final val (epoch {best_payload.get('epoch', best_epoch)}) | loss={fv_loss:.4f} | dice={fv_dice:.4f}")
        with history_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "final_validation", best_payload.get("epoch", best_epoch),
                best_payload.get("train_loss", float("nan")), best_payload.get("train_dice", float("nan")),
                fv_loss, fv_dice, best_checkpoint_metric, best_payload.get("monitor_value", best_monitor_value),
                0.0, 0.0, 0.0,
            ])
        if wandb_run is not None:
            wandb_run.log({"final_validation/loss": fv_loss, "final_validation/dice": fv_dice,
                           "final_validation/best_epoch": int(best_payload.get("epoch", best_epoch))},
                          step=int(best_payload.get("epoch", best_epoch)))

    if wandb_run is not None:
        wandb_run.finish()

    return {"fold": fold, "best_epoch": best_epoch, "best_monitor": best_monitor_value}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resume training from a best.pt checkpoint.")
    p.add_argument("--config",      required=True,  help="Path to config.yml")
    p.add_argument("--checkpoint",  required=True,  help="Path to the .pt file to resume from")
    p.add_argument("--output-dir",  required=True,  help="Directory to write checkpoints and logs")
    p.add_argument("--fold",        type=int, default=0, help="Fold index (needed for data splits)")
    p.add_argument("--extra-epochs", type=int, required=True, help="How many more epochs to train")
    p.add_argument("--stage",       choices=["head", "partial", "full"], default="full",
                   help="Which parameter group to unfreeze")
    p.add_argument("--lr-head",    type=float, default=None, help="LR for head parameters")
    p.add_argument("--lr-partial", type=float, default=None, help="LR for partial (decoder) parameters")
    p.add_argument("--lr-full",    type=float, default=None, help="LR for all (backbone) parameters")
    p.add_argument("--validation-interval", type=int, default=1,
                   help="Run validation every N epochs (0 = disable mid-run)")
    p.add_argument("--final-validation", action=argparse.BooleanOptionalAction, default=True,
                   help="Run final validation on best checkpoint after loop")
    p.add_argument("--best-checkpoint-metric", default="val_dice",
                   choices=["train_dice", "train_loss", "val_dice", "val_loss"])
    p.add_argument("--best-checkpoint-mode", default="max", choices=["max", "min"])
    p.add_argument("--wandb",         action="store_true")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity",  default=None)
    p.add_argument("--wandb-name",    default=None)
    p.add_argument("--wandb-mode",    default="online", choices=["online", "offline", "disabled"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    result = resume_training(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        fold=args.fold,
        extra_epochs=args.extra_epochs,
        stage=args.stage,
        lr_head=args.lr_head,
        lr_partial=args.lr_partial,
        lr_full=args.lr_full,
        validation_interval=args.validation_interval,
        final_validation=args.final_validation,
        best_checkpoint_metric=args.best_checkpoint_metric,
        best_checkpoint_mode=args.best_checkpoint_mode,
        wandb_kwargs={
            "enabled":  args.wandb,
            "project":  args.wandb_project,
            "entity":   args.wandb_entity,
            "name":     args.wandb_name,
            "mode":     args.wandb_mode,
        },
    )
    print(result)