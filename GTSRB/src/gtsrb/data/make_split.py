"""Tạo split_manifest.csv: chia train/val 80/20 theo (class, track), kèm khung ROI.

Mỗi biển thật được chụp thành một "track" ~30 ảnh gần giống nhau. Nếu chia ngẫu nhiên
theo ảnh thì cùng một track nằm ở cả train và val -> val cao giả (rò rỉ dữ liệu).
Vì vậy ta chia theo track.

Chạy:
    python -m gtsrb.data.make_split --raw data/raw/GTSRB/Final_Training/Images
    # Bản dữ liệu có Train.csv và Train/
    python -m gtsrb.data.make_split --raw data/raw/Train
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gtsrb.utils.config import ROOT

EXTS = {".ppm", ".png", ".jpg", ".jpeg"}
ROI_COLS = ["x1", "y1", "x2", "y2"]
GT_ROI = {"Roi.X1": "x1", "Roi.Y1": "y1", "Roi.X2": "x2", "Roi.Y2": "y2"}


def parse_track(stem: str) -> int:
    """'00012_00003' (track_frame) hoặc '00005_00012_00003' (class_track_frame) -> track."""
    parts = stem.split("_")
    return int(parts[0]) if len(parts) == 2 else int(parts[1])


def resolve_train_dir(raw_dir: Path) -> Path:
    """Nhận đường dẫn GTSRB gốc hoặc thư mục Train/ của bản có Train.csv."""
    if raw_dir.is_dir():
        return raw_dir
    original_dir = ROOT / "data/raw/GTSRB/Final_Training/Images"
    local_dir = ROOT / "data/raw/Train"
    if raw_dir.resolve() == original_dir.resolve() and local_dir.is_dir():
        print(f"Không có {raw_dir}; dùng {local_dir}.")
        return local_dir
    raise FileNotFoundError(f"Không tìm thấy thư mục ảnh train: {raw_dir}")


def scan_images(raw_dir: Path) -> pd.DataFrame:
    """Quét raw/<class>/<ảnh> và đọc ROI từ GT theo lớp hoặc Train.csv."""
    raw_dir = resolve_train_dir(raw_dir)
    parts = []
    for class_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        label = int(class_dir.name)
        files = [f for f in sorted(class_dir.iterdir()) if f.suffix.lower() in EXTS]
        df = pd.DataFrame({
            "path": [f.relative_to(raw_dir).as_posix() for f in files],
            "name": [f.name for f in files],
            "label": label,
            "track": [parse_track(f.stem) for f in files],
        })
        gt = class_dir / f"GT-{class_dir.name}.csv"  # annotation gốc, có Roi.X1..Roi.Y2
        if gt.exists():
            roi = pd.read_csv(gt, sep=";").rename(columns={"Filename": "name", **GT_ROI})
            df = df.merge(roi[["name"] + ROI_COLS], on="name", how="left")
        else:
            for c in ROI_COLS:
                df[c] = np.nan
        parts.append(df.drop(columns="name"))
    images = pd.concat(parts, ignore_index=True)
    train_csv = raw_dir.parent / "Train.csv"
    if raw_dir.name == "Train" and train_csv.is_file():
        annotations = pd.read_csv(train_csv, usecols=["Path", "ClassId", *GT_ROI])
        annotations = annotations.rename(columns={"ClassId": "label", **GT_ROI})
        annotations["path"] = annotations.pop("Path").str.removeprefix("Train/")
        images = images.drop(columns=ROI_COLS).merge(
            annotations, on=["path", "label"], how="left", validate="one_to_one"
        )
    return images


def make_manifest(raw_dir, val_frac: float = 0.2, split_seed: int = 0) -> pd.DataFrame:
    if not 0 < val_frac < 1:
        raise ValueError(f"val_frac phải nằm trong (0, 1), nhận được {val_frac}")
    df = scan_images(Path(raw_dir))
    rng = np.random.default_rng(split_seed)  # split_seed cố định, không phụ thuộc seed huấn luyện
    df["split"] = "train"
    for label, g in df.groupby("label"):
        tracks = np.array(sorted(g["track"].unique()))
        if len(tracks) < 2:
            raise ValueError(f"Lớp {label} chỉ có {len(tracks)} track, không chia được train/val")
        rng.shuffle(tracks)
        n_val = min(len(tracks) - 1, max(1, int(round(len(tracks) * val_frac))))  # luôn chừa >= 1 track cho train
        is_val = (df["label"] == label) & df["track"].isin(set(tracks[:n_val]))
        df.loc[is_val, "split"] = "val"
    return df


def check_no_leak(df: pd.DataFrame) -> None:
    """Báo lỗi nếu: rò rỉ track giữa train/val, hoặc có lớp thiếu train hoặc val."""
    keys = df["label"].astype(str) + "_" + df["track"].astype(str)
    both = set(keys[df.split == "train"]) & set(keys[df.split == "val"])
    assert not both, f"Rò rỉ: {len(both)} track nằm ở cả train và val, ví dụ {sorted(both)[:3]}"
    per_class = df.groupby("label")["split"].nunique()
    assert (per_class == 2).all(), f"Lớp thiếu train hoặc val: {per_class[per_class < 2].index.tolist()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="thư mục ảnh train, ví dụ data/raw/Train")
    ap.add_argument("--out", default="data/splits/split_manifest.csv")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--split_seed", type=int, default=0)
    a = ap.parse_args()

    df = make_manifest(a.raw, a.val_frac, a.split_seed)
    check_no_leak(df)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Đã ghi {out}: {len(df)} ảnh, {df.label.nunique()} lớp, có ROI: {df.x1.notna().all()}")
    print(df.split.value_counts().to_string())


if __name__ == "__main__":
    main()
