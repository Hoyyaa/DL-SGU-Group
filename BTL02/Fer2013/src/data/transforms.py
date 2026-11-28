"""Tiền xử lý & tăng cường dữ liệu cho FER2013.

Hai chế độ đầu vào:
- ``gray``: ảnh 48x48x1, chuẩn hóa theo thống kê FER2013 (cho Custom CNN).
- ``imagenet``: phóng to lên ``image_size`` và lặp thành 3 kênh, chuẩn hóa ImageNet
  (cho VGG16 / ResNet50 / EfficientNet pretrained).

Augmentation CHỈ áp dụng cho tập train; tập PublicTest/PrivateTest luôn xác định.
"""

from typing import Any, Dict, Optional

from torchvision import transforms as T

FER2013_MEAN, FER2013_STD = (0.5077,), (0.2550,)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

DEFAULT_AUG: Dict[str, Any] = {
    "horizontal_flip": 0.5,   # xác suất lật ngang
    "rotation": 10.0,         # độ
    "translate": 0.1,         # tỉ lệ dịch chuyển (giống width/height_shift_range=0.1)
    "scale": 0.1,             # zoom ±10%
    "random_erasing": 0.0,    # xác suất RandomErasing
}


def build_transforms(
    train: bool,
    mode: str = "gray",
    image_size: int = 48,
    augment: bool = True,
    aug: Optional[Dict[str, Any]] = None,
) -> T.Compose:
    """Tạo pipeline biến đổi ảnh (đầu vào là PIL Image mức xám 48x48).

    Args:
        train: True cho tập huấn luyện (có augmentation nếu ``augment``).
        mode: ``"gray"`` hoặc ``"imagenet"``.
        image_size: Kích thước cạnh ảnh đầu ra.
        augment: Bật/tắt augmentation (chỉ có tác dụng khi ``train``).
        aug: Ghi đè tham số trong ``DEFAULT_AUG``.
    """
    if mode not in {"gray", "imagenet"}:
        raise ValueError(f"mode không hợp lệ: {mode}")
    p = {**DEFAULT_AUG, **(aug or {})}
    ops = []
    if image_size != 48:
        ops.append(T.Resize((image_size, image_size), antialias=True))
    if mode == "imagenet":
        ops.append(T.Grayscale(num_output_channels=3))

    if train and augment:
        if p["horizontal_flip"] > 0:
            ops.append(T.RandomHorizontalFlip(p["horizontal_flip"]))
        if p["rotation"] or p["translate"] or p["scale"]:
            ops.append(
                T.RandomAffine(
                    degrees=p["rotation"],
                    translate=(p["translate"], p["translate"]) if p["translate"] else None,
                    scale=(1 - p["scale"], 1 + p["scale"]) if p["scale"] else None,
                )
            )

    ops.append(T.ToTensor())
    mean, std = (FER2013_MEAN, FER2013_STD) if mode == "gray" else (IMAGENET_MEAN, IMAGENET_STD)
    ops.append(T.Normalize(mean, std))

    if train and augment and p["random_erasing"] > 0:
        ops.append(T.RandomErasing(p=p["random_erasing"]))
    return T.Compose(ops)


def denormalize(x, mode: str = "gray"):
    """Đưa tensor đã chuẩn hóa về [0, 1] để hiển thị."""
    import torch

    mean, std = (FER2013_MEAN, FER2013_STD) if mode == "gray" else (IMAGENET_MEAN, IMAGENET_STD)
    mean = torch.tensor(mean, device=x.device).view(-1, 1, 1)
    std = torch.tensor(std, device=x.device).view(-1, 1, 1)
    return (x * std + mean).clamp(0, 1)
