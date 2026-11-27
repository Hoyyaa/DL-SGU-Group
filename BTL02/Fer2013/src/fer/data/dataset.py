"""Dataset, augmentation và DataLoader.

Mỗi batch: x (B,1,48,48) float32 trong [0,1] (chưa chuẩn hóa), y (B,) int64 trong 0..6.
Chuẩn hóa mean/std nằm trong mô hình. Augmentation CHỈ áp dụng cho tập train.
"""
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from fer.utils.config import ROOT


def load_cache(cfg) -> dict:
    path = Path(cfg["data"]["cache"])
    path = path if path.is_absolute() else ROOT / path
    if not path.exists():
        raise SystemExit(f"Không thấy {path}. Chạy `python -m fer.data.prepare` trước.")
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


class FERDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, transform=None):
        self.x, self.y, self.transform = x, y, transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = torch.from_numpy(self.x[i]).float().div_(255.0).unsqueeze(0)  # (1,48,48)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.y[i])


def build_transform(aug: dict):
    """Augmentation giống báo cáo mẫu: lật ngang, xoay 10°, dịch 10%, zoom 10% (+ tùy chọn RandomErasing)."""
    if not aug.get("enable", False):
        return None
    tfs = []
    if aug.get("hflip", True):
        tfs.append(v2.RandomHorizontalFlip())
    z = aug.get("zoom", 0.0)
    tfs.append(v2.RandomAffine(degrees=aug.get("rotation", 0.0),
                               translate=(aug.get("shift", 0.0),) * 2,
                               scale=(1 - z, 1 + z) if z else None))
    if aug.get("erasing", 0.0) > 0:
        tfs.append(v2.RandomErasing(p=aug["erasing"], scale=(0.02, 0.15)))
    return v2.Compose(tfs)


def _subset(idx: np.ndarray, fast: bool, seed: int) -> np.ndarray:
    if not fast:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.permutation(idx)[: max(len(idx) // 10, 1)])


def resolve_workers(value) -> int:
    """'auto' -> 0 trên Windows, 2 nơi khác (Colab/Linux).

    Trên Windows mỗi worker là một tiến trình mới nạp lại toàn bộ DLL CUDA của torch (vài GB bộ nhớ ảo),
    dễ gây "WinError 1455: paging file is too small" hoặc CUDA OOM giả dù VRAM còn trống.
    Dữ liệu 48x48 đã nằm sẵn trong RAM nên 0 worker gần như không chậm hơn.
    """
    if value in (None, "auto"):
        return 0 if os.name == "nt" else 2
    return int(value)


def build_loader(cfg, split: str, fast: bool = False, data: dict | None = None) -> DataLoader:
    """split: 'train' (có augmentation, xáo trộn) | 'val' | 'test' (không augmentation)."""
    data = data if data is not None else load_cache(cfg)
    idx = _subset(np.flatnonzero(data["split"] == split), fast, cfg.get("seed", 42))
    is_train = split == "train"
    ds = FERDataset(data["x"][idx], data["y"][idx], build_transform(cfg["aug"]) if is_train else None)
    t = cfg["train"]
    workers = resolve_workers(t.get("num_workers", "auto"))
    return DataLoader(ds, batch_size=t["batch_size"], shuffle=is_train, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                      generator=torch.Generator().manual_seed(cfg.get("seed", 42)) if is_train else None)


def build_dataloaders(cfg, fast: bool = False):
    """Trả về (train_loader, val_loader, data). Tập test chỉ dùng trong evaluate / visualize."""
    data = load_cache(cfg)
    return build_loader(cfg, "train", fast, data), build_loader(cfg, "val", fast, data), data
