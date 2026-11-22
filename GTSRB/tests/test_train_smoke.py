"""Chạy train.py thật trên dữ liệu giả rất nhỏ: kiểm tra file đầu ra, --resume và việc không ghi đè run cũ."""
import csv

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")

from gtsrb.train import main  # noqa: E402


def _fake_data(tmp_path, n=64):
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (n, 64, 64, 3), dtype=np.uint8)
    y = np.arange(n) % 4
    paths = np.array([f"{i}.png" for i in range(n)])
    split = np.where(np.arange(n) % 4 == 0, "val", "train")
    pd.DataFrame({"path": paths, "label": y, "track": np.arange(n), "x1": 0, "y1": 0, "x2": 63, "y2": 63,
                  "split": split}).to_csv(tmp_path / "m.csv", index=False)
    np.savez(tmp_path / "c.npz", x=x, y=y, paths=paths, size=64, use_roi=True)


def _config(tmp_path, epochs=2):
    cfg = {
        "name": "t1",
        "data": {"img_size": 64, "use_roi": True, "manifest": str(tmp_path / "m.csv"), "cache": str(tmp_path / "c.npz")},
        "aug": {"geo": False, "brightness": False, "blur": False},
        "train": {"epochs": epochs, "batch_size": 16, "lr": 1e-3, "weight_decay": 1e-4,
                  "scheduler": "cosine", "amp": False, "num_workers": 0},
        "model": {"name": "small_cnn"},
        "seed": 1,
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def _epochs_in_log(run_dir):
    with open(run_dir / "log.csv") as f:
        return [int(r["epoch"]) for r in csv.DictReader(f)]


def test_run_creates_all_outputs(tmp_path):
    _fake_data(tmp_path)
    run = main(["--config", _config(tmp_path), "--seed", "7", "--out_dir", str(tmp_path / "runs")])
    assert run.name == "t1_s7"
    for name in ["best.pt", "last.pt", "config.yaml", "log.csv", "env.json"]:
        assert (run / name).exists(), name
    assert _epochs_in_log(run) == [1, 2]
    ck = torch.load(run / "best.pt", weights_only=False)
    assert set(ck) == {"model", "cfg", "epoch", "val_macro_f1"}
    assert yaml.safe_load((run / "config.yaml").read_text())["seed"] == 7   # seed lấy từ --seed


def test_existing_run_is_not_overwritten(tmp_path):
    _fake_data(tmp_path)
    args = ["--config", _config(tmp_path), "--seed", "7", "--out_dir", str(tmp_path / "runs")]
    main(args)
    with pytest.raises(SystemExit):
        main(args)


def test_resume_continues_from_last_epoch(tmp_path):
    _fake_data(tmp_path)
    args = ["--config", _config(tmp_path), "--seed", "7", "--out_dir", str(tmp_path / "runs")]
    run = main(args + ["--stop_after", "1"])
    assert _epochs_in_log(run) == [1]
    main(args + ["--resume"])
    assert _epochs_in_log(run) == [1, 2]
