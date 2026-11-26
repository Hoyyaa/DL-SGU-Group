"""Transfer learning với backbone ImageNet (torchvision): VGG16, ResNet50, EfficientNet-B0.

Lưu ý (fail case trong demo_report): các mạng này có 5 lần giảm chiều (/32), nên ảnh 48x48
chỉ còn feature map 1x1 → nên phóng ảnh lên >=128 (config ``image_size``, ``input_mode: imagenet``).
"""

import torch
import torch.nn as nn
from torchvision import models

from src.models.base_model import BaseEmotionModel

_BACKBONES = {
    "vgg16": (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
}


class PretrainedEmotionModel(BaseEmotionModel):
    """Backbone ImageNet + đầu phân loại mới (Dropout → Linear(7))."""

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.5,
    ) -> None:
        super().__init__(num_classes, in_channels=3)
        if backbone not in _BACKBONES:
            raise ValueError(f"Backbone '{backbone}' không hỗ trợ: {list(_BACKBONES)}")
        fn, weights = _BACKBONES[backbone]
        self.backbone_name = backbone
        self.net = fn(weights=weights if pretrained else None)

        if backbone == "vgg16":
            in_f = self.net.classifier[-1].in_features
            self.net.classifier[-1] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
            self._head = self.net.classifier
            self._target = self.net.features[28]  # conv5_3
        elif backbone == "resnet50":
            in_f = self.net.fc.in_features
            self.net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
            self._head = self.net.fc
            self._target = self.net.layer4[-1]
        else:  # efficientnet_b0
            in_f = self.net.classifier[-1].in_features
            self.net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
            self._head = self.net.classifier
            self._target = self.net.features[-1]

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Chỉ train đầu phân loại (linear probing)."""
        for p in self.net.parameters():
            p.requires_grad = False
        for p in self._head.parameters():
            p.requires_grad = True

    def unfreeze_all(self) -> None:
        for p in self.net.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_target_layer(self) -> nn.Module:
        return self._target
