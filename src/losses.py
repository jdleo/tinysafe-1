"""Focal loss and combined dual-head loss for safety classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss — down-weights easy examples, focuses on hard ones."""

    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DualHeadLoss(nn.Module):
    """Combined loss: focal(binary) + weight * BCE(categories)."""

    def __init__(self, gamma: float = 2.0, category_weight: float = 0.5):
        super().__init__()
        self.binary_loss = FocalLoss(gamma=gamma)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.category_weight = category_weight

    def forward(
        self,
        binary_logits: torch.Tensor,
        category_logits: torch.Tensor,
        binary_targets: torch.Tensor,
        category_targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        l_binary = self.binary_loss(binary_logits.squeeze(-1), binary_targets)
        l_category = self.category_loss(category_logits, category_targets)
        total = l_binary + self.category_weight * l_category

        return {
            "loss": total,
            "binary_loss": l_binary.detach(),
            "category_loss": l_category.detach(),
        }
