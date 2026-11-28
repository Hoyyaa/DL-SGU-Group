"""Smoke test toàn pipeline trên dữ liệu giả (không cần tải FER2013, chạy được trên CPU)."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from fer import evaluate, train, visualize
from fer.data.dataset import build_loader, build_transform
from fer.data.prepare import build_cache
from fer.models import PRESETS, build_model
from fer.utils.config import ROOT, load_config, parse_overrides


def make_kaggle_folder(root, prefixed=True, n_train=140, n_test=56, seed=0):
    """Thư mục giả đúng cấu trúc Kaggle: root/{train,test}/<emotion>/<Usage>_<id>.jpg."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    names = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    for split, n in [("train", n_train), ("test", n_test)]:
        for i in range(n):
            emo = names[i % 7]
            usage = "Training" if split == "train" else ("PublicTest" if (i // 7) % 2 == 0 else "PrivateTest")
            fname = f"{usage}_{i}.jpg" if prefixed else f"im{i}.jpg"
            d = root / split / emo
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 256, (48, 48), dtype=np.uint8)).save(d / fname)
    return root


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    return build_cache(make_kaggle_folder(d / "raw"), d / "fer48.npz")


def test_cache_format(cache):
    with np.load(cache) as z:
        assert z["x"].shape == (196, 48, 48) and z["x"].dtype == np.uint8
        assert (z["split"] == "train").sum() == 140
        assert (z["split"] == "val").sum() == 28 and (z["split"] == "test").sum() == 28
        # val = PublicTest, test = PrivateTest, không lẫn nhau
        assert all("PublicTest" in f for f in z["files"][z["split"] == "val"])
        assert all("PrivateTest" in f for f in z["files"][z["split"] == "test"])
        # nhãn theo thứ tự của đề, không theo ABC của thư mục
        neutral = [y for y, f in zip(z["y"], z["files"]) if "/neutral/" in f]
        assert set(neutral) == {6}
        assert 0 < float(z["std"]) < 1


def test_cache_fallback_without_prefix(tmp_path):
    out = build_cache(make_kaggle_folder(tmp_path / "raw", prefixed=False), tmp_path / "c.npz")
    with np.load(out) as z:
        assert (z["split"] == "test").sum() == 56          # toàn bộ test/
        assert (z["split"] == "val").sum() == 7 * 2        # 10% mỗi lớp của 20 ảnh/lớp
        assert np.bincount(z["y"][z["split"] == "val"], minlength=7).tolist() == [2] * 7


@pytest.mark.parametrize("name", [n for n in PRESETS if n != "vgg16"])
def test_presets_shape(name):
    model = build_model({"name": name})
    assert model(torch.rand(2, 1, 48, 48)).shape == (2, 7)
    assert isinstance(model.last_conv(), torch.nn.Conv2d)


def test_vgg16_shape():
    model = build_model({"name": "vgg16", "resize": 64, "pretrained": False})
    assert model(torch.rand(2, 1, 48, 48)).shape == (2, 7)


def test_all_configs_load():
    for f in sorted((ROOT / "configs").glob("e*.yaml")):
        cfg = load_config(f)
        assert cfg["name"] == f.stem, f
        if cfg["model"]["name"] != "vgg16":
            build_model(cfg["model"])


def test_no_aug_on_eval(cache):
    cfg = load_config("configs/base.yaml", overrides={"data": {"cache": str(cache)}, "train": {"num_workers": 0}})
    assert build_loader(cfg, "val").dataset.transform is None
    assert build_loader(cfg, "test").dataset.transform is None
    assert build_transform(cfg["aug"]) is not None


def test_parse_overrides():
    assert parse_overrides(["train.lr=3e-4", "name=x"]) == {"train": {"lr": 3e-4}, "name": "x"}


def test_train_evaluate_visualize(cache, tmp_path):
    run = train.main(["--config", "configs/e1_small.yaml", "--out_dir", str(tmp_path / "runs"), "--set",
                      f"data.cache={cache}", "train.num_workers=0", "train.epochs=2", "train.amp=false"])
    assert (run / "best.pt").exists() and len(pd.read_csv(run / "log.csv")) == 2
    with pytest.raises(SystemExit):  # không ghi đè run đã có
        train.main(["--config", "configs/e1_small.yaml", "--out_dir", str(tmp_path / "runs"),
                    "--set", f"data.cache={cache}"])

    res = tmp_path / "results"
    evaluate.main(["--run", str(run), "--out_dir", str(res), "--data", str(cache)])
    m = json.loads((res / run.name / "metrics.json").read_text(encoding="utf-8"))
    assert 0 <= m["test_acc"] <= 1
    for f in ["curves.png", "confusion_val.png", "confusion_test.png", "true_cases.png", "wrong_cases.png"]:
        assert (res / run.name / f).exists(), f
    assert (res / "summary.csv").exists()

    visualize.main(["--run", str(run), "--out_dir", str(res), "--data", str(cache), "--actmax_steps", "3"])
    for f in ["filters_first_conv.png", "filters_last_conv.png", "actmax_last_conv.png",
              "featuremaps_last_conv.png", "gradcam.png"]:
        assert (res / run.name / "viz" / f).exists(), f


def test_wandb_offline_same_run(cache, tmp_path, monkeypatch):
    """train --wandb (offline) -> evaluate -> visualize cùng ghi vào một run id."""
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_SILENT", "true")
    wb_dir = tmp_path / "wandb"
    run = train.main(["--config", "configs/e1_small.yaml", "--out_dir", str(tmp_path / "runs"), "--wandb", "--set",
                      f"data.cache={cache}", "train.num_workers=0", "train.epochs=1", "train.amp=false",
                      "wandb.mode=offline", f"wandb.dir={wb_dir}"])
    run_id = (run / "wandb_id.txt").read_text().strip()
    evaluate.main(["--run", str(run), "--out_dir", str(tmp_path / "res"), "--data", str(cache)])
    visualize.main(["--run", str(run), "--out_dir", str(tmp_path / "res"), "--data", str(cache), "--actmax_steps", "2"])
    offline = sorted((wb_dir / "wandb").glob(f"offline-run-*-{run_id}"))
    assert len(offline) == 3  # train, evaluate, visualize; `wandb sync` gộp lại theo id
    assert (run / "wandb_id.txt").read_text().strip() == run_id


def test_wandb_off_by_default(cache, tmp_path):
    run = train.main(["--config", "configs/e1_small.yaml", "--out_dir", str(tmp_path / "runs"), "--set",
                      f"data.cache={cache}", "train.num_workers=0", "train.epochs=1", "train.amp=false"])
    assert not (run / "wandb_id.txt").exists()
