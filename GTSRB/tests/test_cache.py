import numpy as np
import pandas as pd
import pytest
from PIL import Image

from gtsrb.data.cache import build_train_cache, channel_stats, load_checked_cache, load_image
from gtsrb.data.make_split import check_no_leak, make_manifest


def _fake_raw(root, n_classes=2, n_tracks=5, n_frames=2):
    """Ảnh 40x40: nền đen, ô trắng ở (10..19, 10..19). Có GT-<class>.csv với ROI đúng ô trắng."""
    for c in range(n_classes):
        d = root / f"{c:05d}"
        d.mkdir(parents=True)
        rows = []
        for t in range(n_tracks):
            for f in range(n_frames):
                a = np.zeros((40, 40, 3), np.uint8)
                a[10:20, 10:20] = 255
                name = f"{t:05d}_{f:05d}.ppm"
                Image.fromarray(a).save(d / name)
                rows.append(f"{name};40;40;10;10;19;19;{c}")
        (d / f"GT-{c:05d}.csv").write_text("Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId\n" + "\n".join(rows))


@pytest.fixture
def built(tmp_path):
    raw = tmp_path / "raw"
    _fake_raw(raw)
    df = make_manifest(raw)
    check_no_leak(df)
    mpath = tmp_path / "manifest.csv"
    df.to_csv(mpath, index=False)
    cpath = tmp_path / "cache.npz"
    build_train_cache(raw, mpath, cpath, tmp_path / "stats.json", size=16, use_roi=True)
    return raw, df, cpath


def test_roi_crop_keeps_only_sign(built):
    raw, df, _ = built
    r = df.iloc[0]
    with_roi = load_image(raw / r.path, 16, [r.x1, r.y1, r.x2, r.y2])
    no_roi = load_image(raw / r.path, 16)
    assert with_roi.min() == 255          # chỉ còn vùng biển (trắng)
    assert no_roi.min() == 0              # ảnh đầy đủ còn cả nền đen


def test_checked_cache_ok(built):
    _, df, cpath = built
    x, y = load_checked_cache(df, cpath, img_size=16, use_roi=True)
    assert x.shape == (len(df), 16, 16, 3)


def test_shuffled_manifest_is_rejected(built):
    _, df, cpath = built
    shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
    with pytest.raises(ValueError, match="không khớp"):
        load_checked_cache(shuffled, cpath, img_size=16, use_roi=True)


def test_img_size_mismatch_is_rejected(built):
    _, df, cpath = built
    with pytest.raises(ValueError, match="img_size"):
        load_checked_cache(df, cpath, img_size=64, use_roi=True)


def test_use_roi_mismatch_is_rejected(built):
    _, df, cpath = built
    with pytest.raises(ValueError, match="use_roi"):
        load_checked_cache(df, cpath, img_size=16, use_roi=False)


def test_channel_stats_matches_direct_computation():
    x = np.random.randint(0, 256, (50, 8, 8, 3), dtype=np.uint8)
    mean, std = channel_stats(x, chunk=7)
    f = x.astype(np.float64) / 255
    assert np.allclose(mean, f.mean(axis=(0, 1, 2)))
    assert np.allclose(std, f.std(axis=(0, 1, 2)), atol=1e-9)
