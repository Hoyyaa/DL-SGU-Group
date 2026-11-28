"""Khám phá dữ liệu (EDA) – mục 1.3 của demo_report: histogram phân bố lớp + ảnh mẫu.

    python -m src.data.explore --data data/raw --out outputs/figures/eda   # --data nhận thư mục ảnh hoặc fer2013.csv
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys
    sys.path[0] = str(Path(__file__).resolve().parents[2])

from src import EMOTION_LABELS
from src.data.dataset import FER2013Dataset


def class_histogram(data_path: Path, out_dir: Path, strict: bool = True) -> pd.DataFrame:
    rows = {}
    for split in ("train", "val", "test"):
        ds = FER2013Dataset(data_path, split, strict=strict)
        rows[ds.usage] = ds.class_counts()
    df = pd.DataFrame(rows, index=EMOTION_LABELS)
    ax = df.plot.bar(figsize=(9, 4), rot=0, title="FER2013 – phân bố lớp theo tập")
    for c in ax.containers:
        ax.bar_label(c, fontsize=6)
    plt.tight_layout(); plt.savefig(out_dir / "class_distribution.png", dpi=150); plt.close()
    df.to_csv(out_dir / "class_distribution.csv")
    return df


def sample_grid(data_path: Path, out_dir: Path, per_class: int = 8, strict: bool = True) -> None:
    ds = FER2013Dataset(data_path, "train", strict=strict)
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(len(EMOTION_LABELS), per_class, figsize=(per_class * 1.1, 8))
    for c, name in enumerate(EMOTION_LABELS):
        idx = np.where(ds.labels == c)[0]
        pick = rng.choice(idx, size=min(per_class, len(idx)), replace=False) if len(idx) else []
        for j in range(per_class):
            ax = axes[c, j]; ax.axis("off")
            if j < len(pick):
                ax.imshow(ds.images[pick[j]], cmap="gray")
        axes[c, 0].set_title(name, fontsize=8, loc="left")
    plt.tight_layout(); plt.savefig(out_dir / "train_samples.png", dpi=150); plt.close()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", "--csv", dest="data", default="data/raw")
    ap.add_argument("--out", default="outputs/figures/eda")
    ap.add_argument("--no-strict", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(class_histogram(Path(a.data), out, not a.no_strict))
    sample_grid(Path(a.data), out, strict=not a.no_strict)
    print(f"Đã lưu hình EDA vào {out}")


if __name__ == "__main__":
    main()
