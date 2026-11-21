"""Dataset và DataLoader.

Hợp đồng: mỗi batch x có dạng (B, 3, 64, 64), float32, giá trị trong [0, 1], CHƯA chuẩn hóa.
Việc chuẩn hóa (mean/std) nằm trong mô hình, nên làm tối/mờ trên ảnh [0, 1] luôn đúng.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from gtsrb.data.cache import load_checked_cache
from gtsrb.utils.config import ROOT


class GTSRBDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, transform=None):
        self.x, self.y, self.transform = x, y, transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = torch.from_numpy(self.x[i]).permute(2, 0, 1).float() / 255.0  # HWC uint8 -> CHW [0,1]
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.y[i])


def build_train_transform(aug: dict):
    """Thứ tự: hình học -> độ sáng -> làm mờ (đều trên ảnh [0,1]). Không lật ngang vì làm sai nghĩa biển."""
    ts = []
    if aug.get("geo", True):
        ts.append(v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)))
    p = aug.get("p", 0.5)
    if aug.get("brightness", False):
        lo, hi = aug["brightness_range"]
        ts.append(v2.RandomApply([v2.ColorJitter(brightness=(lo, hi))], p=p))
    if aug.get("blur", False):
        lo, hi = aug["blur_sigma"]
        ts.append(v2.RandomApply([v2.GaussianBlur(kernel_size=11, sigma=(lo, hi))], p=p))
    return v2.Compose(ts) if ts else None


def build_dataloaders(cfg: dict, fast: bool = False):
    """Trả về (train_loader, val_loader). fast=True: chỉ ~10% dữ liệu để chạy thử."""
    d = cfg["data"]
    manifest = pd.read_csv(ROOT / d["manifest"])
    x, y = load_checked_cache(manifest, ROOT / d["cache"], d["img_size"], d.get("use_roi", True))

    idx_train = np.flatnonzero((manifest["split"] == "train").to_numpy())
    idx_val = np.flatnonzero((manifest["split"] == "val").to_numpy())
    if fast:
        rng = np.random.default_rng(0)
        idx_train = rng.choice(idx_train, max(1, len(idx_train) // 10), replace=False)
        idx_val = rng.choice(idx_val, max(1, len(idx_val) // 10), replace=False)

    t = cfg["train"]
    train_ds = GTSRBDataset(x[idx_train], y[idx_train], build_train_transform(cfg["aug"]))
    val_ds = GTSRBDataset(x[idx_val], y[idx_val])  # val: ảnh gốc, không aug
    kw = dict(batch_size=t["batch_size"], num_workers=t["num_workers"], pin_memory=torch.cuda.is_available())
    return DataLoader(train_ds, shuffle=True, **kw), DataLoader(val_ds, shuffle=False, **kw)
