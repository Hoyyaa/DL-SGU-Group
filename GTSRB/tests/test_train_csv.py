import numpy as np
import pandas as pd
import pytest
from PIL import Image

from gtsrb.data.make_split import check_no_leak, make_manifest, scan_images


def _fake_kaggle(root, n_classes=2, n_tracks=5, n_frames=3):
    """Bản Kaggle: Train/<class>/<class>_<track>_<frame>.png + Train.csv (Path có tiền tố 'Train/')."""
    rows = []
    for c in range(n_classes):
        d = root / "Train" / str(c)
        d.mkdir(parents=True)
        for t in range(n_tracks):
            for f in range(n_frames):
                name = f"{c:05d}_{t:05d}_{f:05d}.png"
                Image.fromarray(np.zeros((40, 40, 3), np.uint8)).save(d / name)
                rows.append(dict(Width=40, Height=40, **{"Roi.X1": 5 + c, "Roi.Y1": 6, "Roi.X2": 30, "Roi.Y2": 31},
                                 ClassId=c, Path=f"Train/{c}/{name}"))
    pd.DataFrame(rows).to_csv(root / "Train.csv", index=False)


def test_train_csv_roi_is_merged_per_image(tmp_path):
    _fake_kaggle(tmp_path)
    df = scan_images(tmp_path / "Train")
    assert len(df) == 2 * 5 * 3
    assert df[["x1", "y1", "x2", "y2"]].notna().all().all()
    assert (df[df.label == 1].x1 == 6).all() and (df[df.label == 0].x1 == 5).all()  # ROI đúng lớp, đúng ảnh


def test_train_csv_manifest_no_leak(tmp_path):
    _fake_kaggle(tmp_path)
    df = make_manifest(tmp_path / "Train")
    check_no_leak(df)
    assert df.path.iloc[0].startswith("0/")   # path tương đối so với Train/


def test_missing_train_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_images(tmp_path / "khong_ton_tai")