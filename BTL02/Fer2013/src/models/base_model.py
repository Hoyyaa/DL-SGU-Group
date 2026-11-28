"""Lớp cơ sở cho mọi mô hình nhận diện cảm xúc."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseEmotionModel(nn.Module, ABC):
    """Mọi mô hình phải: (1) trả logits [B, num_classes]; (2) chỉ ra tầng Conv cuối
    để phục vụ trực quan hóa filter và Grad-CAM."""

    def __init__(self, num_classes: int = 7, in_channels: int = 1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def get_target_layer(self) -> nn.Module:
        """Tầng tích chập cuối cùng (dùng cho Grad-CAM / feature maps)."""

    def last_conv(self) -> nn.Conv2d:
        """Conv2d cuối cùng trong mạng (dùng để vẽ filter của tầng conv cuối)."""
        convs = [m for m in self.modules() if isinstance(m, nn.Conv2d)]
        return convs[-1]

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)
