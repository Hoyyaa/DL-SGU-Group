"""Model registry: tạo mô hình từ dict config (khóa ``model``)."""

from typing import Any, Dict

from src import NUM_CLASSES
from src.models.base_model import BaseEmotionModel
from src.models.cnn import CNN_SPECS, VGGStyleCNN, build_custom_cnn
from src.models.pretrained import PretrainedEmotionModel

PRETRAINED = {"vgg16", "resnet50", "efficientnet_b0"}
AVAILABLE_MODELS = sorted(set(CNN_SPECS) | PRETRAINED)


def build_model(cfg: Dict[str, Any]) -> BaseEmotionModel:
    """Ví dụ cfg: ``{"model": "medium_cnn", "activation": "relu", "batch_norm": true}``."""
    name = cfg["model"]
    num_classes = int(cfg.get("num_classes", NUM_CLASSES))
    if name in CNN_SPECS:
        return build_custom_cnn(
            name,
            num_classes=num_classes,
            in_channels=3 if cfg.get("input_mode") == "imagenet" else 1,
            image_size=int(cfg.get("image_size", 48)),
            activation=cfg.get("activation", "relu"),
            batch_norm=bool(cfg.get("batch_norm", False)),
            conv_dropout=float(cfg.get("conv_dropout", 0.2)),
            fc_dropout=float(cfg.get("dropout", 0.5)),
            global_pool=bool(cfg.get("global_pool", False)),
            blocks=cfg.get("blocks"),
            fc_dim=cfg.get("fc_dim"),
        )
    if name in PRETRAINED:
        return PretrainedEmotionModel(
            backbone=name,
            num_classes=num_classes,
            pretrained=bool(cfg.get("pretrained", True)),
            freeze_backbone=bool(cfg.get("freeze_backbone", False)),
            dropout=float(cfg.get("dropout", 0.5)),
        )
    raise ValueError(f"Model '{name}' chưa đăng ký. Có: {AVAILABLE_MODELS}")


__all__ = ["AVAILABLE_MODELS", "BaseEmotionModel", "PretrainedEmotionModel", "VGGStyleCNN", "build_model"]
