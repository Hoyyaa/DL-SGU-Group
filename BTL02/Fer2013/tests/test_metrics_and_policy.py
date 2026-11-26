"""Metric PyTorch đúng như định nghĩa + chính sách 'chỉ PyTorch' + resume giữ trạng thái."""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from src.evaluate.metrics import accuracy, classification_report, confusion_matrix, macro_f1

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {"tensorflow", "keras", "sklearn", "jax", "flax", "mxnet", "paddle"}


def test_confusion_matrix_manual():
    cm = confusion_matrix([0, 0, 1, 2, 2, 2], [0, 1, 1, 2, 0, 2], 3)
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 2]]


def test_metrics_known_values():
    y, p = [0, 0, 1, 1], [0, 1, 1, 1]
    # lớp 0: P=1, R=.5, F1=2/3 ; lớp 1: P=2/3, R=1, F1=.8 ; lớp 2 không có mẫu → F1=0
    assert accuracy(y, p) == pytest.approx(0.75)
    assert macro_f1(y, p, 3) == pytest.approx((2 / 3 + 0.8 + 0) / 3)
    rep = classification_report(y, p, ["a", "b", "c"])
    assert rep.loc["b", "precision"] == pytest.approx(2 / 3)
    assert rep.loc["accuracy", "f1-score"] == pytest.approx(0.75)


def test_metrics_match_sklearn_reference():
    """Chỉ đối chiếu khi máy có sẵn sklearn; dự án KHÔNG phụ thuộc sklearn."""
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 7, 500), rng.integers(0, 7, 500)
    assert macro_f1(y, p, 7) == pytest.approx(skm.f1_score(y, p, average="macro", labels=list(range(7)), zero_division=0))
    assert np.array_equal(confusion_matrix(y, p, 7).numpy(), skm.confusion_matrix(y, p, labels=list(range(7))))


def test_project_uses_only_pytorch():
    """Không file nào trong src/, scripts/, demo/ import framework ML khác ngoài PyTorch."""
    bad = []
    for path in [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("demo/*.py")]:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            bad += [f"{path.relative_to(ROOT)}: {n}" for n in names if n.split(".")[0] in FORBIDDEN]
    assert not bad, bad


def test_resume_restores_state(fake_csv, tmp_path):
    from src.data.dataset import build_dataloaders
    from src.train.trainer import EmotionTrainer
    from src.train.utils import load_config, seed_everything

    cfg = load_config(ROOT / "experiments/baseline/exp01_small_cnn.yaml", [
        f"data_path={fake_csv}", "strict_split=false", "epochs=2", "batch_size=16", "num_workers=0",
        f"output_dir={tmp_path / 'runs'}", f"summary_csv={tmp_path / 's.csv'}", "blocks=[[8], [16]]", "fc_dim=16",
        "evaluate_after_train=false",
    ])
    seed_everything(42)
    loaders = build_dataloaders(cfg, strict=False)
    t1 = EmotionTrainer(cfg, loaders)
    t1.fit()
    t2 = EmotionTrainer({**cfg, "epochs": 3}, loaders)
    t2.resume(t1.run_dir / "last.pt")
    assert t2.start_epoch == 3 and t2.wait == t1.wait and len(t2.history) == 2
    # t1 đã nạp lại best.pt trong _finalize, nên so trọng số của t2 với last.pt
    last = torch.load(t1.run_dir / "last.pt", weights_only=False)
    for k, v in last["model"].items():
        assert torch.equal(v.cpu(), t2.model.state_dict()[k].cpu())
    assert t2.fit()["epochs_run"] == 3
