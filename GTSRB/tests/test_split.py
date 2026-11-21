import numpy as np
import pytest
from PIL import Image

from gtsrb.data.make_split import check_no_leak, make_manifest, parse_track


def _fake_raw(root, n_classes=3, n_tracks=10, n_frames=4):
    """Tạo dữ liệu giả đúng định dạng GTSRB: <class>/<track>_<frame>.ppm."""
    for c in range(n_classes):
        d = root / f"{c:05d}"
        d.mkdir(parents=True)
        for t in range(n_tracks):
            for f in range(n_frames):
                Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(d / f"{t:05d}_{f:05d}.ppm")


def test_parse_track():
    assert parse_track("00012_00003") == 12
    assert parse_track("00005_00012_00003") == 12


def test_split_no_leak_and_ratio(tmp_path):
    _fake_raw(tmp_path)
    df = make_manifest(tmp_path, val_frac=0.2, split_seed=0)
    check_no_leak(df)
    assert len(df) == 3 * 10 * 4
    assert (df.split == "val").sum() == 3 * 2 * 4  # 2/10 track mỗi lớp
    assert df.groupby("label")["split"].nunique().eq(2).all()  # lớp nào cũng có cả train và val


def test_split_is_deterministic(tmp_path):
    _fake_raw(tmp_path)
    a = make_manifest(tmp_path, split_seed=0)
    b = make_manifest(tmp_path, split_seed=0)
    assert a.equals(b)


def test_leak_is_detected(tmp_path):
    _fake_raw(tmp_path)
    df = make_manifest(tmp_path)
    df.loc[df.index[0], "split"] = "val" if df.loc[df.index[0], "split"] == "train" else "train"
    with pytest.raises(AssertionError):
        check_no_leak(df)


def test_val_frac_out_of_range(tmp_path):
    _fake_raw(tmp_path)
    for bad in (0, 1, 1.5):
        with pytest.raises(ValueError):
            make_manifest(tmp_path, val_frac=bad)


def test_class_with_one_track_fails(tmp_path):
    _fake_raw(tmp_path, n_tracks=1)
    with pytest.raises(ValueError):
        make_manifest(tmp_path)


def test_high_val_frac_still_keeps_train(tmp_path):
    _fake_raw(tmp_path)
    df = make_manifest(tmp_path, val_frac=0.99)
    check_no_leak(df)  # mỗi lớp vẫn còn >= 1 track train
