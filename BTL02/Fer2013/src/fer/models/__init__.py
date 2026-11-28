"""Đăng ký mô hình.

    from fer.models import build_model
    model = build_model(cfg["model"], mean, std)   # x (B,1,48,48) trong [0,1] -> logits (B,7)

cfg["model"]["name"] chọn preset trong PRESETS; các khóa còn lại trong cfg["model"]
(activation, batchnorm, drop_conv, ...) ghi đè preset. Thêm mô hình: thêm một dòng vào PRESETS.
"""
from fer.models.cnn import FerCNN
from fer.models.transfer import VGG16Transfer

# Kiến trúc lấy theo báo cáo mẫu (demo_report.pdf, mục 4)
PRESETS = {
    "small": dict(net=FerCNN, blocks=[[64, 64], [128, 128, 256, 256]], fc=[1024]),
    "medium": dict(net=FerCNN, blocks=[[64, 64], [128] * 3, [256] * 3], fc=[1024]),
    "medium01": dict(net=FerCNN, blocks=[[64] * 3, [128] * 4, [256] * 4], fc=[1024]),
    "large": dict(net=FerCNN, blocks=[[64] * 4, [128] * 4, [256] * 4, [512] * 4], fc=[2048]),
    "vgg16": dict(net=VGG16Transfer),
}


def build_model(model_cfg: dict, mean: float = 0.5, std: float = 0.25, num_classes: int = 7):
    cfg = dict(model_cfg)
    name = cfg.pop("name")
    if name not in PRESETS:
        raise ValueError(f"Không có mô hình '{name}'. Hiện có: {sorted(PRESETS)}")
    spec = {**PRESETS[name], **cfg}
    net = spec.pop("net")
    if net is FerCNN:
        spec.update(mean=mean, std=std)
    return net(num_classes=num_classes, **spec)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
