"""Smoke test end-to-end: train 2 epoch trên CSV giả → checkpoint, history, metrics, hình."""

import json
from pathlib import Path

from src.train.trainer import run_experiment
from src.train.utils import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end(fake_csv, tmp_path):
    cfg = load_config(ROOT / "experiments/baseline/exp01_small_cnn.yaml", [
        f"data_path={fake_csv}", "strict_split=false", "epochs=2", "batch_size=16", "num_workers=0",
        f"output_dir={tmp_path / 'runs'}", f"summary_csv={tmp_path / 'summary.csv'}", "loss=weighted_ce",
        "blocks=[[8, 8], [16]]", "fc_dim=32",  # CNN tí hon cho test nhanh, checkpoint nhỏ
    ])
    metrics = run_experiment(cfg)
    run = tmp_path / "runs" / cfg["exp_id"]
    for f in ("best.pt", "last.pt", "history.csv", "metrics.json", "config.yaml",
              "figures/confusion_matrix.png", "figures/last_conv_filters.png",
              "figures/gradcam_per_class.png", "figures/wrong_cases.png", "figures/training_curves.png"):
        assert (run / f).exists(), f
    assert 0 <= metrics["private_acc"] <= 1
    assert json.loads((run / "metrics.json").read_text())["epochs_run"] == 2
    assert (tmp_path / "summary.csv").read_text(encoding="utf-8").count("\n") == 2


def test_wandb_offline_logging(fake_csv, tmp_path, monkeypatch):
    """wandb ở chế độ offline: không cần tài khoản/mạng, kiểm tra run được tạo và đóng đúng."""
    import pytest
    pytest.importorskip("wandb")
    for var in ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR", "WANDB_ARTIFACT_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("WANDB_SILENT", "true")

    cfg = load_config(ROOT / "experiments/baseline/exp01_small_cnn.yaml", [
        f"data_path={fake_csv}", "strict_split=false", "epochs=1", "batch_size=16", "num_workers=0",
        f"output_dir={tmp_path / 'runs'}", f"summary_csv={tmp_path / 'summary.csv'}",
        "blocks=[[8], [16]]", "fc_dim=16",
        "wandb=true", "wandb_mode=offline", f"wandb_dir={tmp_path / 'wandb'}",
    ])
    run_experiment(cfg)
    run_dir = tmp_path / "runs" / cfg["exp_id"]
    run_id = (run_dir / "wandb_run_id.txt").read_text().strip()
    offline = list((tmp_path / "wandb").rglob(f"offline-run-*-{run_id}"))
    assert offline, "không thấy thư mục offline run của wandb"
