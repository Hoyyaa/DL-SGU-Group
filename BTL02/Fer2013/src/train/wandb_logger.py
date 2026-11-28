"""Ghi log thí nghiệm lên Weights & Biases (tùy chọn, bật bằng ``wandb: true`` trong config).

Ghi gì:
- config đầy đủ của run, gradient/weight histogram (nếu ``wandb_watch``)
- mỗi epoch: train/val loss & acc, learning rate
- cuối run: public/private acc, macro-F1, best epoch... (run summary)
- toàn bộ hình trong ``figures/`` (confusion matrix, Grad-CAM, filters, fail cases)
  và bảng ``classification_report`` / ``top_confusions``
- ``best.pt`` dưới dạng Artifact (nếu ``wandb_log_model``)

Đăng nhập 1 lần: ``wandb login`` (hoặc đặt biến môi trường ``WANDB_API_KEY`` – KHÔNG commit key).
Không có mạng: ``wandb_mode: offline`` rồi ``wandb sync outputs/wandb/wandb/offline-run-*`` sau.
"""

import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import torch.nn as nn

from src.train.utils import resolve_path


class WandbLogger:
    """Bọc wandb; khi tắt hoặc chưa cài wandb thì mọi phương thức là no-op."""

    def __init__(self, cfg: Dict[str, Any], run_dir: Path) -> None:
        self.cfg, self.run_dir, self.run = cfg, Path(run_dir), None
        if not cfg.get("wandb", False):
            return
        try:
            import wandb
        except ImportError:
            print("[wandb] chưa cài (pip install wandb) – bỏ qua.")
            return
        self._wandb = wandb

        # File tạm/cache của wandb để trong outputs/ (tránh làm đầy ổ hệ thống).
        wb_dir = resolve_path(cfg.get("wandb_dir", "outputs/wandb"))
        wb_dir.mkdir(parents=True, exist_ok=True)
        for var in ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR", "WANDB_ARTIFACT_DIR"):
            os.environ.setdefault(var, str(wb_dir))

        # Lưu run id để --resume tiếp tục ĐÚNG run cũ trên wandb thay vì tạo run mới.
        id_file = self.run_dir / "wandb_run_id.txt"
        run_id = id_file.read_text().strip() if id_file.exists() else uuid.uuid4().hex[:8]
        id_file.write_text(run_id)

        tags = list(cfg.get("wandb_tags") or []) + [str(cfg.get("model", "")), str(cfg.get("input_mode", ""))]
        self.run = wandb.init(
            project=cfg.get("wandb_project", "BTL02_FER2013"),
            entity=cfg.get("wandb_entity") or None,
            name=str(cfg["exp_id"]),
            group=cfg.get("wandb_group") or None,
            tags=[t for t in tags if t],
            notes=cfg.get("notes") or None,
            config=cfg,
            dir=str(wb_dir),
            mode=cfg.get("wandb_mode", "online"),
            id=run_id,
            resume="allow",
            job_type="train",
        )
        self.run.define_metric("epoch")
        self.run.define_metric("*", step_metric="epoch")

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def watch(self, model: nn.Module) -> None:
        if self.enabled and self.cfg.get("wandb_watch", False):
            self.run.watch(model, log="all", log_freq=int(self.cfg.get("wandb_watch_freq", 200)))

    def log_epoch(self, row: Dict[str, float]) -> None:
        if not self.enabled:
            return
        self.run.log({
            "epoch": row["epoch"], "lr": row["lr"],
            "train/loss": row["train_loss"], "train/acc": row["train_acc"],
            "val/loss": row["val_loss"], "val/acc": row["val_acc"],
            "train_val_gap": row["train_acc"] - row["val_acc"],  # chỉ báo overfitting
        })

    def log_summary(self, metrics: Dict[str, Any]) -> None:
        if self.enabled:
            self.run.summary.update(metrics)

    def log_figures(self, fig_dir: Path) -> None:
        """Đẩy mọi PNG/CSV trong figures/ lên wandb (gọi sau EvaluationReport.run_all)."""
        if not self.enabled or not Path(fig_dir).exists():
            return
        import pandas as pd

        payload = {}
        for png in sorted(Path(fig_dir).glob("*.png")):
            payload[f"figures/{png.stem}"] = self._wandb.Image(str(png))
        for csv in sorted(Path(fig_dir).glob("*.csv")):
            df = pd.read_csv(csv)
            payload[f"tables/{csv.stem}"] = self._wandb.Table(dataframe=df.astype(str) if csv.stem == "classification_report" else df)
        if payload:
            self.run.log(payload)

    def log_model(self, path: Path, metrics: Optional[Dict[str, Any]] = None) -> None:
        if not (self.enabled and self.cfg.get("wandb_log_model", False) and Path(path).exists()):
            return
        art = self._wandb.Artifact(f"{self.cfg['exp_id']}-model", type="model", metadata=metrics or {})
        art.add_file(str(path))
        self.run.log_artifact(art, aliases=["best"])

    def finish(self, exit_code: int = 0) -> None:
        if self.enabled:
            self.run.finish(exit_code=exit_code)
            self.run = None
