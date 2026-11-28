"""Custom CNN kiểu VGG cho ảnh 48x48 (huấn luyện từ đầu).

Kiến trúc bám theo các mô hình trong ``demo_report.pdf`` (Small / Medium / Medium1 / Large):
mỗi block = N x [Conv3x3 (+BN) + Activation] → MaxPool 2x2 → Dropout(0.2);
bộ phân loại = Flatten → Dense → Dropout(0.5) → Dense(7).
"""

from typing import List, Sequence

import torch
import torch.nn as nn

from src.models.base_model import BaseEmotionModel

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}

# Cấu hình block: danh sách số filter của từng Conv trong mỗi block.
CNN_SPECS = {
    "small_cnn": {"blocks": [[64, 64], [128, 128, 256, 256]], "fc": 1024},
    "medium_cnn": {"blocks": [[64, 64], [128] * 3, [256] * 3], "fc": 1024},
    "medium1_cnn": {"blocks": [[64] * 3, [128] * 4, [256] * 4], "fc": 1024},
    "large_cnn": {"blocks": [[64] * 4, [128] * 4, [256] * 4, [512] * 4], "fc": 2048},
}


def make_activation(name: str) -> nn.Module:
    if name not in ACTIVATIONS:
        raise ValueError(f"Activation '{name}' không hỗ trợ. Chọn trong {list(ACTIVATIONS)}")
    return ACTIVATIONS[name]()


class VGGStyleCNN(BaseEmotionModel):
    """CNN tùy biến số block / số conv / BN / activation / dropout từ config."""

    def __init__(
        self,
        blocks: Sequence[Sequence[int]],
        fc_dim: int = 1024,
        num_classes: int = 7,
        in_channels: int = 1,
        image_size: int = 48,
        activation: str = "relu",
        batch_norm: bool = False,
        conv_dropout: float = 0.2,
        fc_dropout: float = 0.5,
        global_pool: bool = False,
    ) -> None:
        super().__init__(num_classes, in_channels)
        layers: List[nn.Module] = []
        c_in = in_channels
        for filters in blocks:
            for c_out in filters:
                conv = nn.Conv2d(c_in, c_out, 3, padding=1, bias=not batch_norm)
                layers.append(conv)
                if batch_norm:
                    layers.append(nn.BatchNorm2d(c_out))
                layers.append(make_activation(activation))
                c_in = c_out
            layers.append(nn.MaxPool2d(2))
            if conv_dropout > 0:
                layers.append(nn.Dropout(conv_dropout))
        self.features = nn.Sequential(*layers)
        self._target = [m for m in self.features if isinstance(m, nn.Conv2d)][-1]

        if global_pool:
            self.pool = nn.AdaptiveAvgPool2d(1)
            flat = c_in
        else:
            self.pool = nn.Identity()
            with torch.no_grad():
                flat = self.features(torch.zeros(1, in_channels, image_size, image_size)).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, fc_dim),
            make_activation(activation),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))

    def get_target_layer(self) -> nn.Module:
        return self._target


def build_custom_cnn(name: str, **kwargs) -> VGGStyleCNN:
    spec = CNN_SPECS[name]
    blocks = kwargs.pop("blocks", None) or spec["blocks"]
    fc_dim = kwargs.pop("fc_dim", None) or spec["fc"]
    return VGGStyleCNN(blocks=blocks, fc_dim=fc_dim, **kwargs)
