import numpy as np
import pandas as pd
from PIL import Image

from gtsrb.data.cache import build_test_cache


def _fake_test(root, n=6):
    """Bản Kaggle: Test/00000.png ... và Test.csv (dấu phẩy, cột Path, ClassId, Roi.*)."""
    (root / "Test").mkdir(parents=True)
    rows = []
    for i in range(n):
        Image.fromarray(np.zeros((40, 40, 3), np.uint8)).save(root / "Test" / f"{i:05d}.png")
        rows.append(dict(Width=40, Height=40, **{"Roi.X1": 5, "Roi.Y1": 5, "Roi.X2": 30, "Roi.Y2": 30},
                         ClassId=i % 3, Path=f"Test/{i:05d}.png"))
    pd.DataFrame(rows).to_csv(root / "Test.csv", index=False)


def test_build_test_cache_kaggle_format(tmp_path):
    _fake_test(tmp_path)
    out = tmp_path / "t.npz"
    build_test_cache(tmp_path / "Test", tmp_path / "Test.csv", out, size=16, use_roi=True)
    c = np.load(out)
    assert c["x"].shape == (6, 16, 16, 3)
    assert c["y"].tolist() == [0, 1, 2, 0, 1, 2]
    assert c["paths"][0] == "00000.png"