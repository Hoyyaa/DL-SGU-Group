"""Dataset & DataLoader cho FER2013 – hỗ trợ 2 định dạng nguồn:

1. **Thư mục ảnh** (Kaggle ``msambare/fer2013``)::

       data/raw/train/<emotion>/Training_*.jpg
       data/raw/test/<emotion>/PublicTest_*.jpg | PrivateTest_*.jpg

   Nhãn lấy từ tên thư mục con, ``Usage`` lấy từ tiền tố tên file. Vì bản này gộp
   PublicTest + PrivateTest vào ``test/``, ta dựa vào tiền tố để khôi phục ĐÚNG cách chia gốc
   thay vì tự chia ngẫu nhiên (tránh lệch so với mốc 69%/71% của đề).
2. **CSV gốc** ``fer2013.csv`` (cột emotion, pixels, Usage).

Chia tập chuẩn (KHÔNG tự chia lại để tránh rò rỉ dữ liệu):
- ``Training``    – 28.709 ảnh → huấn luyện
- ``PublicTest``  –  3.589 ảnh → validation (chọn checkpoint, early stopping)
- ``PrivateTest`` –  3.589 ảnh → test cuối cùng (chỉ dùng để báo cáo)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src import EMOTION_LABELS, NUM_CLASSES
from src.data.transforms import build_transforms

SPLITS = {
    "train": "Training",
    "val": "PublicTest",
    "public": "PublicTest",
    "test": "PrivateTest",
    "private": "PrivateTest",
}
EXPECTED_COUNTS = {"Training": 28709, "PublicTest": 3589, "PrivateTest": 3589}
USAGES = tuple(EXPECTED_COUNTS)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
LABEL_TO_IDX = {name.lower(): i for i, name in enumerate(EMOTION_LABELS)}

# Cache trong RAM theo đường dẫn: train/val/test cùng nguồn chỉ giải mã 1 lần,
# run_suite chạy nhiều thí nghiệm trong cùng tiến trình cũng không đọc lại.
_SOURCE_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def get_data_path(cfg: Dict[str, Any]) -> str:
    """``data_path`` là key mới; ``csv_path`` giữ để đọc được config/checkpoint cũ."""
    path = cfg.get("data_path") or cfg.get("csv_path")
    if not path:
        raise KeyError("Config thiếu 'data_path' (thư mục ảnh hoặc fer2013.csv)")
    return str(path)


def _list_folder_images(root: Path) -> List[Tuple[Path, int, str]]:
    """Duyệt ``root/<split>/<emotion>/<Usage>_*.jpg`` → (đường dẫn, nhãn, Usage)."""
    items = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS or not p.is_file():
            continue
        label = LABEL_TO_IDX.get(p.parent.name.lower())
        if label is None:
            raise ValueError(f"Thư mục lớp không hợp lệ: {p.parent} (cần một trong {list(LABEL_TO_IDX)})")
        usage = p.name.split("_", 1)[0]
        if usage not in USAGES:
            raise ValueError(f"Tên file không có tiền tố Usage hợp lệ {USAGES}: {p}")
        items.append((p, label, usage))
    if not items:
        raise FileNotFoundError(f"Không thấy ảnh nào trong {root}. Xem data/README.md.")
    return items


def _folder_cache_path(root: Path) -> Path:
    return root.parent / "processed" / f"fer2013_{root.name}.npz"


def _load_folder(root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    items = _list_folder_images(root)
    cache = _folder_cache_path(root)
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        # Chỉ tin cache khi số ảnh khớp – tránh dùng cache cũ sau khi thêm/bớt dữ liệu.
        if len(z["labels"]) == len(items):
            return z["images"], z["labels"], z["usage"]

    images = np.zeros((len(items), 48, 48), dtype=np.uint8)
    for i, (p, _, _) in enumerate(items):
        with Image.open(p) as im:
            im = im.convert("L")
            if im.size != (48, 48):  # bản gốc luôn 48x48; ảnh khác cỡ thì đưa về để không vỡ pipeline
                im = im.resize((48, 48), Image.BILINEAR)
            images[i] = np.asarray(im, dtype=np.uint8)
    labels = np.array([lb for _, lb, _ in items], dtype=np.int64)
    usage = np.array([u for _, _, u in items])
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, images=images, labels=labels, usage=usage)
    except OSError:
        pass  # không ghi được cache (ổ chỉ đọc…) thì vẫn chạy bình thường
    return images, labels, usage


def _load_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    images = np.stack(
        [np.array(p.split(), dtype=np.uint8).reshape(48, 48) for p in df["pixels"]]
    ) if len(df) else np.zeros((0, 48, 48), dtype=np.uint8)
    return images, df["emotion"].to_numpy(dtype=np.int64), df["Usage"].to_numpy(dtype=str)


def load_source(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Đọc toàn bộ nguồn dữ liệu → (images uint8 [N,48,48], labels [N], usage [N])."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không thấy {path}. Xem data/README.md để tải FER2013.")
    key = str(path.resolve())
    if key not in _SOURCE_CACHE:
        _SOURCE_CACHE[key] = _load_folder(path) if path.is_dir() else _load_csv(path)
    return _SOURCE_CACHE[key]


class FER2013Dataset(Dataset):
    """Dataset FER2013: ảnh uint8 48x48 giữ trong RAM (≈80 MB cho cả bộ)."""

    def __init__(
        self,
        data_path: Union[str, Path] = "data/raw",
        split: str = "train",
        transform=None,
        strict: bool = True,
    ) -> None:
        """
        Args:
            data_path: Thư mục ảnh (``data/raw`` chứa train/ test/) hoặc file ``fer2013.csv``.
            split: ``train`` | ``val``/``public`` | ``test``/``private``.
            transform: Biến đổi áp lên PIL Image.
            strict: Kiểm tra số mẫu đúng chuẩn FER2013 (tắt khi test với dữ liệu giả).
        """
        if split.lower() not in SPLITS:
            raise ValueError(f"split phải thuộc {list(SPLITS)}")
        self.data_path = Path(data_path)
        self.usage = SPLITS[split.lower()]
        self.transform = transform

        images, labels, usage = load_source(self.data_path)
        mask = usage == self.usage
        if strict and int(mask.sum()) != EXPECTED_COUNTS[self.usage]:
            raise ValueError(f"{self.usage}: {int(mask.sum())} mẫu, kỳ vọng {EXPECTED_COUNTS[self.usage]}")
        self.images = images[mask]
        self.labels = labels[mask]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.fromarray(self.images[idx])  # uint8 2D → ảnh mức xám "L"
        x = self.transform(img) if self.transform else torch.from_numpy(self.images[idx]).float().unsqueeze(0) / 255.0
        return x, int(self.labels[idx])

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.labels, minlength=NUM_CLASSES)

    def class_weights(self) -> torch.Tensor:
        """Trọng số lớp kiểu 'balanced': w_c = N / (K * N_c) – dùng cho Weighted CE / Focal."""
        counts = self.class_counts().astype(np.float64)
        counts[counts == 0] = 1.0
        return torch.tensor(counts.sum() / (NUM_CLASSES * counts), dtype=torch.float32)


def build_dataloaders(cfg: Dict[str, Any], strict: bool = True) -> Dict[str, DataLoader]:
    """Tạo DataLoader cho 3 tập train / val (PublicTest) / test (PrivateTest) từ config."""
    mode = cfg.get("input_mode", "gray")
    size = int(cfg.get("image_size", 48))
    augment = bool(cfg.get("augment", True))
    aug = cfg.get("aug", {})
    generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    data_path = get_data_path(cfg)

    loaders = {}
    for split in ("train", "val", "test"):
        is_train = split == "train"
        ds = FER2013Dataset(
            data_path,
            split,
            build_transforms(is_train, mode, size, augment, aug),
            strict=strict,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=int(cfg.get("batch_size", 64)),
            shuffle=is_train,
            num_workers=int(cfg.get("num_workers", 0)),
            pin_memory=torch.cuda.is_available(),
            generator=generator if is_train else None,
        )
    return loaders
