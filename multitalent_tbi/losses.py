from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-5,
    include_background: bool = False,
) -> torch.Tensor:
    num_classes = logits.shape[1]
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    if not include_background and num_classes > 1:
        probabilities = probabilities[:, 1:]
        one_hot = one_hot[:, 1:]
    dims = tuple(range(2, probabilities.ndim))
    intersection = torch.sum(probabilities * one_hot, dim=dims)
    denominator = torch.sum(probabilities, dim=dims) + torch.sum(one_hot, dim=dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def dice_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets.long(), weight=class_weights)
    dice = soft_dice_loss(logits, targets, include_background=False)
    return ce + dice

