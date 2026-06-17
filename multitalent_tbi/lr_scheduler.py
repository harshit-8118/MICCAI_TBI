from __future__ import annotations
import math


class _GroupScheduleState:
    def __init__(
        self,
        base_lr: float,
        active_from: int,
        active_total: int,
        warmup_epochs: int,
        poly_power: float,
        min_lr: float,
    ) -> None:
        self.base_lr = base_lr
        self.active_from = active_from
        self.active_total = max(1, active_total)
        self.warmup_epochs = warmup_epochs
        self.poly_power = poly_power
        self.min_lr = min_lr

    def is_active(self, epoch: int) -> bool:
        return epoch >= self.active_from

    def scale(self, epoch: int) -> float:
        local = epoch - self.active_from
        if local < 0:
            return 0.0
        if self.warmup_epochs > 0 and local < self.warmup_epochs:
            return float(local + 1) / float(self.warmup_epochs)
        decay_epochs = self.active_total - self.warmup_epochs
        if decay_epochs <= 0:
            return 1.0
        progress = float(local - self.warmup_epochs) / float(decay_epochs)
        progress = min(max(progress, 0.0), 1.0)
        raw_scale = (1.0 - progress) ** self.poly_power
        min_scale = self.min_lr / max(self.base_lr, 1e-12)
        return max(raw_scale, min_scale)

    def lr(self, epoch: int) -> float:
        return self.base_lr * self.scale(epoch)


class StagedPolyLRScheduler:
    def __init__(self, optimizer, config) -> None:
        import torch
        self._torch = torch
        self.optimizer = optimizer

        max_epochs  = int(config.training.max_epochs)
        head_epochs = int(config.training.staged_tuning.head_only_epochs)
        part_epochs = int(config.training.staged_tuning.partial_tune_epochs)
        full_start  = head_epochs + part_epochs
        full_epochs = max(1, max_epochs - full_start)
        poly_power  = float(config.training.poly_power)
        min_lr      = float(getattr(config.training, "min_lr", 1e-7))

        head_warmup = int(getattr(config.training, "head_warmup_epochs",    5))
        part_warmup = int(getattr(config.training, "partial_warmup_epochs", 3))
        full_warmup = int(getattr(config.training, "full_warmup_epochs",    3))

        stage_cfg: dict[str, tuple[int, int, int]] = {
            "head":    (0,          head_epochs,  head_warmup),
            "partial": (head_epochs, part_epochs, part_warmup),
            "full":    (full_start,  full_epochs, full_warmup),
        }

        self._states: dict[str, _GroupScheduleState] = {}
        for group in optimizer.param_groups:
            name: str = group.get("group_name", "full")
            if name not in self._states:
                active_from, active_total, warmup = stage_cfg[name]
                self._states[name] = _GroupScheduleState(
                    base_lr=float(group["base_lr"]),
                    active_from=active_from,
                    active_total=active_total,
                    warmup_epochs=warmup,
                    poly_power=poly_power,
                    min_lr=min_lr,
                )

        # track which groups have been activated so we reset Adam state once
        self._activated: set[str] = set()
        # activate head immediately (epoch 0)
        self._activated.add("head")

    def _reset_adam_state_for_group(self, group_name: str) -> None:
        """
        Wipe m and v buffers for all params in this group.
        Prevents stale momentum from frozen phase driving a bad first step.
        """
        for group in self.optimizer.param_groups:
            if group.get("group_name") != group_name:
                continue
            for param in group["params"]:
                if param in self.optimizer.state:
                    state = self.optimizer.state[param]
                    # reset first and second moment estimates
                    if "exp_avg" in state:
                        state["exp_avg"].zero_()
                    if "exp_avg_sq" in state:
                        state["exp_avg_sq"].zero_()
                    # reset max exp_avg_sq used by AMSGrad variant
                    if "max_exp_avg_sq" in state:
                        state["max_exp_avg_sq"].zero_()
                    # reset step count so bias correction restarts cleanly
                    if "step" in state:
                        if self._torch.is_tensor(state["step"]):
                            state["step"].zero_()
                        else:
                            state["step"] = 0

    def step(self, epoch: int) -> dict[str, float]:
        summary: dict[str, float] = {}

        for group in self.optimizer.param_groups:
            name: str = group.get("group_name", "full")
            state = self._states[name]

            # first activation: reset Adam buffers to prevent momentum spike
            if state.is_active(epoch) and name not in self._activated:
                self._activated.add(name)
                self._reset_adam_state_for_group(name)
                print(f"[StagedPolyLR] epoch {epoch+1}: activating '{name}' group, Adam state reset.")

            new_lr = state.lr(epoch)
            group["lr"] = new_lr
            summary[name] = new_lr

        return summary
    