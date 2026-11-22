"""Mô hình và hàm tạo mô hình.

Mọi mô hình trả về từ build_model nhận ảnh (B,3,64,64) trong [0,1] và tự chuẩn hóa bên trong.
Thêm mô hình mới: viết lớp trong file riêng rồi thêm một dòng vào NETS bên dưới.
"""
import json

import torch
import torch.nn as nn

from gtsrb.models.small_cnn import SmallCNN
from gtsrb.utils.config import ROOT

# B thêm resnet18 và plain18 vào đây, ví dụ:
#   from gtsrb.models.resnet18 import ResNet18
#   NETS["resnet18"] = lambda num_classes: ResNet18(num_classes, skip=True)
#   NETS["plain18"] = lambda num_classes: ResNet18(num_classes, skip=False)
NETS = {
    "small_cnn": SmallCNN,
}


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class Classifier(nn.Module):
    """Bọc mạng thật: chuẩn hóa rồi chạy mạng. Mạng thật lấy qua model.net."""

    def __init__(self, net, mean, std):
        super().__init__()
        self.norm, self.net = Normalize(mean, std), net

    def forward(self, x):  # x: (B,3,64,64) trong [0,1]
        return self.net(self.norm(x))


def build_model(name: str, num_classes: int = 43, mean=None, std=None):
    if name not in NETS:
        raise ValueError(f"Không có mô hình '{name}'. Hiện có: {sorted(NETS)}")
    if mean is None:
        stats = json.loads((ROOT / "data/splits/norm_stats.json").read_text())
        mean, std = stats["mean"], stats["std"]
    return Classifier(NETS[name](num_classes), mean, std)
