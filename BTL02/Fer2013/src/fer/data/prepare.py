"""Đọc FER2013 bản Kaggle (thư mục train/ và test/) thành một file cache .npz.

    python -m fer.data.prepare                       # mặc định đọc data/raw/{train,test}
    python -m fer.data.prepare --src D:/fer2013      # thư mục chứa train/ và test/

Bản Kaggle msambare/fer2013 đã chia sẵn:
    data/raw/train/<emotion>/Training_*.jpg                        28.709 ảnh
    data/raw/test/<emotion>/PublicTest_*.jpg, PrivateTest_*.jpg    3.589 + 3.589 ảnh

Pipeline cần 3 tập: train (học), val (chọn checkpoint, early stopping), test (chỉ để báo cáo).
    train = toàn bộ train/
    val   = ảnh PublicTest_*  trong test/   ┐ đúng cách chia gốc của cuộc thi,
    test  = ảnh PrivateTest_* trong test/   ┘ so sánh được với mốc 69% / 71% của đề
Nếu tên file trong test/ không có tiền tố này (bản tải từ nguồn khác), tự chuyển sang:
    val = 10% train/ (chia đều theo lớp, seed 42), test = toàn bộ test/

Kết quả data/processed/fer48.npz:
    x      (N,48,48) uint8
    y      (N,) int64, 0..6 theo CLASSES
    split  (N,) 'train' | 'val' | 'test'
    files  (N,) đường dẫn ảnh tương đối so với --src (để tra lại ảnh khi phân tích lỗi)
    mean, std  tính trên ảnh train đã chia 255
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from fer.utils.config import ROOT

# Thứ tự nhãn chuẩn của đề. Thư mục Kaggle tên thường, xếp theo ABC nên phải ánh xạ lại.
CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
NAME_TO_IDX = {c.lower(): i for i, c in enumerate(CLASSES)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
EXPECTED = {"train": 28709, "val": 3589, "test": 3589}


def read_folder(folder: Path, root: Path):
    """Đọc folder/<emotion>/*.jpg -> (x, y, đường dẫn tương đối)."""
    if not folder.is_dir():
        raise SystemExit(f"Không thấy {folder}. Cần cấu trúc <src>/train/<emotion>/ và <src>/test/<emotion>/.")
    xs, ys, files = [], [], []
    for cls_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        idx = NAME_TO_IDX.get(cls_dir.name.lower())
        if idx is None:
            raise SystemExit(f"Thư mục lớp lạ: {cls_dir}. Hợp lệ: {sorted(NAME_TO_IDX)}")
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() not in IMAGE_EXTS:
                continue
            img = Image.open(f).convert("L")
            if img.size != (48, 48):
                img = img.resize((48, 48))
            xs.append(np.asarray(img, dtype=np.uint8))
            ys.append(idx)
            files.append(f.relative_to(root).as_posix())
    if not xs:
        raise SystemExit(f"{folder} không có ảnh nào.")
    return np.stack(xs), np.asarray(ys, dtype=np.int64), np.asarray(files)


def stratified_holdout(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Trả về mask True cho `frac` mẫu của mỗi lớp (dùng làm val)."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        mask[idx[: max(1, round(len(idx) * frac))]] = True
    return mask


def assign_splits(test_files: np.ndarray, y_train: np.ndarray, val_frac: float, seed: int):
    """Trả về (split_train, split_test, mô tả cách chia)."""
    names = np.asarray([Path(f).name for f in test_files])
    is_pub = np.char.startswith(names, "PublicTest")
    is_priv = np.char.startswith(names, "PrivateTest")
    split_train = np.full(len(y_train), "train", dtype="U5")
    if (is_pub | is_priv).all() and is_pub.any() and is_priv.any():
        return split_train, np.where(is_pub, "val", "test"), "val = PublicTest, test = PrivateTest (chia gốc)"
    split_train[stratified_holdout(y_train, val_frac, seed)] = "val"
    return split_train, np.full(len(test_files), "test", dtype="U5"), \
        f"không có tiền tố PublicTest/PrivateTest -> val = {val_frac:.0%} train (theo lớp), test = toàn bộ test/"


def build_cache(src: Path, out: Path, val_frac: float = 0.1, seed: int = 42) -> Path:
    x_tr, y_tr, f_tr = read_folder(src / "train", src)
    x_te, y_te, f_te = read_folder(src / "test", src)
    s_tr, s_te, how = assign_splits(f_te, y_tr, val_frac, seed)

    x = np.concatenate([x_tr, x_te])
    y = np.concatenate([y_tr, y_te])
    split = np.concatenate([s_tr, s_te])
    files = np.concatenate([f_tr, f_te])
    train = x[split == "train"].astype(np.float32) / 255.0
    mean, std = float(train.mean()), float(train.std())

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=x, y=y, split=split, files=files, mean=mean, std=std)

    print(f"Đã ghi {out}\n  cách chia: {how}\n  mean={mean:.4f} std={std:.4f}")
    print(f"  {'tập':6s} {'số ảnh':>7s}  " + " ".join(f"{c[:7]:>8s}" for c in CLASSES))
    for s in ("train", "val", "test"):
        counts = np.bincount(y[split == s], minlength=7)
        n = int(counts.sum())
        note = "" if n == EXPECTED[s] else "   (khác bản gốc, kiểm tra lại nếu dùng chia gốc)"
        print(f"  {s:6s} {n:7d}  " + " ".join(f"{c:8d}" for c in counts) + note)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw", help="thư mục chứa train/ và test/")
    ap.add_argument("--out", default="data/processed/fer48.npz")
    ap.add_argument("--val_frac", type=float, default=0.1, help="chỉ dùng khi test/ không có tiền tố PublicTest")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    src, out = Path(a.src), Path(a.out)
    src = src if src.is_absolute() else ROOT / src
    out = out if out.is_absolute() else ROOT / out
    build_cache(src, out, a.val_frac, a.seed)


if __name__ == "__main__":
    main()
