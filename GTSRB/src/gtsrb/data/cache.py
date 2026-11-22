"""Cắt ROI (nếu bật), resize, lưu .npz (uint8), tính norm_stats.json trên tập train.

Mọi tham số (img_size, use_roi, đường dẫn) đọc từ cùng file config với DataLoader.

Chạy:
    python -m gtsrb.data.cache --raw data/raw/GTSRB/Final_Training/Images
    # (tuỳ chọn) cache tập test:
    python -m gtsrb.data.cache --test_dir data/raw/GTSRB/Final_Test/Images \
                               --test_csv data/raw/GTSRB/GT-final_test.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from gtsrb.data.make_split import resolve_train_dir
from gtsrb.utils.config import ROOT, load_config


def load_image(path: Path, size: int, roi=None) -> np.ndarray:
    """Đọc ảnh, cắt theo roi=(x1,y1,x2,y2) nếu có (toạ độ GTSRB gồm cả biên), rồi resize."""
    img = Image.open(path).convert("RGB")
    if roi is not None:
        x1, y1, x2, y2 = (int(v) for v in roi)
        box = (max(x1, 0), max(y1, 0), min(x2 + 1, img.width), min(y2 + 1, img.height))
        img = img.crop(box)
    return np.asarray(img.resize((size, size), Image.BILINEAR))


def _roi_of(row, use_roi: bool):
    if not use_roi:
        return None
    vals = [row.x1, row.y1, row.x2, row.y2]
    if any(pd.isna(v) for v in vals):
        raise ValueError(f"use_roi=true nhưng thiếu ROI cho {row}; đặt data.use_roi: false hoặc bổ sung annotation")
    return vals


def channel_stats(x: np.ndarray, chunk: int = 2048):
    """Mean/std từng kênh trên ảnh [0,1], cộng dồn theo lô để không tạo bản sao float32 của cả tập."""
    s, ss, n = np.zeros(3), np.zeros(3), 0
    for i in range(0, len(x), chunk):
        b = x[i:i + chunk].astype(np.float64) / 255.0
        s += b.sum(axis=(0, 1, 2))
        ss += (b ** 2).sum(axis=(0, 1, 2))
        n += b.shape[0] * b.shape[1] * b.shape[2]
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 0))
    return mean.tolist(), std.tolist()


def build_train_cache(raw_dir: Path, manifest_path: Path, out_path: Path, stats_path: Path, size: int, use_roi: bool):
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Thiếu manifest: {manifest_path}. "
            f"Chạy trước: python -m gtsrb.data.make_split --raw {str(raw_dir)!r} --out {str(manifest_path)!r}"
        )
    raw_dir = resolve_train_dir(raw_dir)
    df = pd.read_csv(manifest_path)
    x = np.stack([load_image(raw_dir / r.path, size, _roi_of(r, use_roi)) for r in df.itertuples()])
    y = df["label"].to_numpy(dtype=np.int64)

    train_mask = (df["split"] == "train").to_numpy()
    mean, std = channel_stats(x[train_mask])  # chỉ train

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # lưu cả path để lúc nạp đối chiếu với manifest; thứ tự hàng = thứ tự manifest
    np.savez(out_path, x=x, y=y, paths=df["path"].to_numpy(dtype=str), size=size, use_roi=use_roi)
    stats_path.write_text(json.dumps({"mean": mean, "std": std}, indent=2))
    print(f"Cache train: {x.shape}, use_roi={use_roi} -> {out_path}\nmean={mean}\nstd={std}")


def build_test_cache(test_dir: Path, csv_path: Path, out_path: Path, size: int, use_roi: bool):
    """Đọc Test.csv (Kaggle: dấu phẩy, cột Path/ClassId/Roi.*; GTSRB gốc: dấu chấm phẩy, cột Filename)."""
    gt = pd.read_csv(csv_path, sep=None, engine="python")  # tự nhận dấu , hoặc ;
    gt = gt.rename(columns={"Path": "Filename", "Roi.X1": "x1", "Roi.Y1": "y1", "Roi.X2": "x2", "Roi.Y2": "y2"})
    files = gt["Filename"].astype(str).str.removeprefix("Test/")  # bỏ tiền tố Test/ nếu có
    x = np.stack([load_image(test_dir / f, size, _roi_of(r, use_roi)) for f, r in zip(files, gt.itertuples())])
    np.savez(out_path, x=x, y=gt["ClassId"].to_numpy(dtype=np.int64),
             paths=files.to_numpy(dtype=str), size=size, use_roi=use_roi)
    print(f"Cache test: {x.shape}, use_roi={use_roi} -> {out_path}")


def load_checked_cache(manifest: pd.DataFrame, cache_path, img_size: int, use_roi: bool):
    """Nạp cache và đối chiếu với manifest. Lệch là báo lỗi, không chạy tiếp với dữ liệu sai."""
    c = np.load(cache_path)
    x, y, paths = c["x"], c["y"], c["paths"]
    if x.shape[1:3] != (img_size, img_size):
        raise ValueError(f"Cache có ảnh {x.shape[1]}x{x.shape[2]} nhưng config yêu cầu img_size={img_size}. Tạo lại cache.")
    if bool(c["use_roi"]) != bool(use_roi):
        raise ValueError(f"Cache tạo với use_roi={bool(c['use_roi'])} nhưng config có use_roi={use_roi}. Tạo lại cache.")
    if len(paths) != len(manifest) or not np.array_equal(paths, manifest["path"].to_numpy(dtype=str)):
        raise ValueError("Thứ tự/đường dẫn trong cache không khớp split_manifest.csv (manifest bị sắp xếp lại hoặc sửa?). Tạo lại cache.")
    if not np.array_equal(y, manifest["label"].to_numpy()):
        raise ValueError("Nhãn trong cache không khớp manifest. Tạo lại cache.")
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--raw", help="thư mục ảnh train, ví dụ data/raw/Train")
    ap.add_argument("--test_dir")
    ap.add_argument("--test_csv")
    a = ap.parse_args()

    d = load_config(a.config)["data"]
    size, use_roi = d["img_size"], d.get("use_roi", True)
    if a.raw:
        build_train_cache(Path(a.raw), ROOT / d["manifest"], ROOT / d["cache"],
                          ROOT / "data/splits/norm_stats.json", size, use_roi)
    if a.test_dir and a.test_csv:
        build_test_cache(Path(a.test_dir), Path(a.test_csv), ROOT / d["cache"].replace(".npz", "_test.npz"), size, use_roi)


if __name__ == "__main__":
    main()