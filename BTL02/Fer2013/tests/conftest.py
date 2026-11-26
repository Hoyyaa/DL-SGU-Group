"""Fixture: CSV FER2013 giả (ảnh ngẫu nhiên) để test pipeline không cần dữ liệu thật."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def fake_csv(tmp_path_factory):
    rng = np.random.default_rng(0)
    rows = []
    for usage, n in (("Training", 56), ("PublicTest", 21), ("PrivateTest", 21)):
        for i in range(n):
            pixels = " ".join(map(str, rng.integers(0, 256, 48 * 48)))
            rows.append({"emotion": i % 7, "pixels": pixels, "Usage": usage})
    path = tmp_path_factory.mktemp("data") / "fer2013.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def fake_folder(tmp_path_factory):
    """Thư mục ảnh giả theo cấu trúc Kaggle msambare/fer2013: <split>/<emotion>/<Usage>_<id>.jpg."""
    from PIL import Image

    from src import EMOTION_LABELS

    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("fer_folder") / "raw"
    for split, usage, n in (("train", "Training", 14), ("test", "PublicTest", 7), ("test", "PrivateTest", 7)):
        for i in range(n):
            d = root / split / EMOTION_LABELS[i % 7].lower()
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 256, (48, 48), dtype=np.uint8)).save(d / f"{usage}_{i}.jpg")
    return root
