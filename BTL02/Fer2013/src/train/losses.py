"""Hàm mất mát: CrossEntropy, Weighted CE, Focal Loss (xử lý mất cân bằng lớp Disgust)."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none", label_smoothing=self.label_smoothing)
        p_t = torch.exp(-F.cross_entropy(logits, target, reduction="none"))
        loss = (1 - p_t) ** self.gamma * ce
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


def build_loss(name: str = "ce", class_weights: Optional[torch.Tensor] = None,
               gamma: float = 2.0, label_smoothing: float = 0.0) -> nn.Module:
    """name: ``ce`` | ``weighted_ce`` | ``focal`` | ``weighted_focal``."""
    name = name.lower()
    if name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    if name == "focal":
        return FocalLoss(gamma=gamma, label_smoothing=label_smoothing)
    if name == "weighted_focal":
        return FocalLoss(gamma=gamma, weight=class_weights, label_smoothing=label_smoothing)
    raise ValueError(f"Loss '{name}' không hỗ trợ")
