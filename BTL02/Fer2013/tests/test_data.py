import pytest
import torch

from src.data import FER2013Dataset, build_dataloaders, build_transforms
from src.train.utils import load_config


def test_dataset_splits(fake_csv):
    ds = FER2013Dataset(fake_csv, "train", build_transforms(True), strict=False)
    x, y = ds[0]
    assert len(ds) == 56 and x.shape == (1, 48, 48) and 0 <= y < 7
    assert ds.class_weights().shape == (7,)
    assert len(FER2013Dataset(fake_csv, "val", strict=False)) == 21


def test_strict_split_rejects_wrong_counts(fake_csv):
    with pytest.raises(ValueError):
        FER2013Dataset(fake_csv, "train", strict=True)


@pytest.mark.parametrize("mode,size,ch", [("gray", 48, 1), ("imagenet", 128, 3)])
def test_transforms_shapes(fake_csv, mode, size, ch):
    ds = FER2013Dataset(fake_csv, "test", build_transforms(False, mode, size), strict=False)
    assert ds[0][0].shape == (ch, size, size)


def test_eval_transform_deterministic(fake_csv):
    ds = FER2013Dataset(fake_csv, "test", build_transforms(False), strict=False)
    assert torch.equal(ds[3][0], ds[3][0])


def test_dataloaders_from_config(fake_csv):
    cfg = load_config(None, [f"data_path={fake_csv}", "batch_size=8", "num_workers=0"])
    loaders = build_dataloaders(cfg, strict=False)
    x, y = next(iter(loaders["train"]))
    assert x.shape == (8, 1, 48, 48) and y.shape == (8,)


def test_folder_dataset_splits_by_filename_prefix(fake_folder):
    """test/ gộp PublicTest + PrivateTest – phải tách đúng theo tiền tố, không lẫn sang nhau."""
    train = FER2013Dataset(fake_folder, "train", build_transforms(True), strict=False)
    val = FER2013Dataset(fake_folder, "val", strict=False)
    test = FER2013Dataset(fake_folder, "test", strict=False)
    assert (len(train), len(val), len(test)) == (14, 7, 7)
    assert sorted(val.labels.tolist()) == list(range(7))  # nhãn lấy từ tên thư mục lớp
    assert train[0][0].shape == (1, 48, 48)
    assert (fake_folder.parent / "processed" / "fer2013_raw.npz").exists()


def test_folder_rejects_unknown_prefix(tmp_path):
    from PIL import Image

    d = tmp_path / "raw" / "test" / "happy"
    d.mkdir(parents=True)
    Image.new("L", (48, 48)).save(d / "img_0.jpg")
    with pytest.raises(ValueError):
        FER2013Dataset(tmp_path / "raw", "test", strict=False)


def test_dataloaders_from_folder_and_legacy_csv_key(fake_folder, fake_csv):
    cfg = load_config(None, [f"data_path={fake_folder}", "batch_size=4", "num_workers=0"])
    x, _ = next(iter(build_dataloaders(cfg, strict=False)["val"]))
    assert x.shape == (4, 1, 48, 48)
    # checkpoint cũ chỉ có csv_path vẫn phải nạp được
    legacy = {"csv_path": str(fake_csv), "batch_size": 4}
    assert len(build_dataloaders(legacy, strict=False)["train"].dataset) == 56
