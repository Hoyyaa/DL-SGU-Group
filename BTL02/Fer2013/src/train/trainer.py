import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    # Chạy trực tiếp (nút Run của VS Code / python src/.../file.py) thì Python không thấy package `src`;
    # thay thư mục chứa file bằng gốc dự án để import giống hệt `python -m`.
    import sys
    sys.path[0] = str(Path(__file__).resolve().parents[2])

from src.data.dataset import build_dataloaders, get_data_path
from src.evaluate.metrics import accuracy, macro_f1
from src.models import build_model
from src.train.losses import build_loss
from src.train.utils import get_device, load_config, resolve_path, seed_everything
from src.train.wandb_logger import WandbLogger

SUMMARY_COLUMNS = [
    "exp_id", "model", "input_mode", "image_size", "activation", "batch_norm", "optimizer", "lr",
    "weight_decay", "scheduler", "loss", "augment", "batch_size", "epochs_run", "best_epoch",
    "train_acc", "best_val_acc", "public_acc", "private_acc", "private_macro_f1",
    "params_M", "train_time_min", "member", "notes",
]


def build_optimizer(cfg: Dict[str, Any], params) -> torch.optim.Optimizer:
    name = cfg.get("optimizer", "adam").lower()
    lr, wd = float(cfg.get("lr", 1e-3)), float(cfg.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name in {"sgd", "sgd_nesterov"}:
        return torch.optim.SGD(params, lr=lr, momentum=float(cfg.get("momentum", 0.9)),
                               nesterov=name == "sgd_nesterov", weight_decay=wd)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Optimizer '{name}' không hỗ trợ")


def build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer):
    name = str(cfg.get("scheduler", "none")).lower()
    p = cfg.get("scheduler_params", {}) or {}
    epochs = int(cfg.get("epochs", 50))
    if name == "none":
        return None
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=p.get("step_size", 10), gamma=p.get("gamma", 0.1))
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=p.get("factor", 0.1), patience=p.get("patience", 3),
            min_lr=float(p.get("min_lr", 1e-6)))
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(p.get("min_lr", 1e-6)))
    if name == "warmup_cosine":
        warm = int(p.get("warmup_epochs", 3))

        def lr_lambda(e: int) -> float:
            if e < warm:
                return (e + 1) / warm
            return 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, epochs - warm)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    raise ValueError(f"Scheduler '{name}' không hỗ trợ")


class EmotionTrainer:
    """Huấn luyện + chọn checkpoint theo val acc (PublicTest) + early stopping."""

    def __init__(self, cfg: Dict[str, Any], loaders: Dict[str, DataLoader],
                 model: Optional[nn.Module] = None, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self.loaders = loaders
        self.device = device or get_device(cfg.get("device", "auto"))
        self.model = (model or build_model(cfg)).to(self.device)

        weights = None
        if "weighted" in cfg.get("loss", "ce"):
            weights = loaders["train"].dataset.class_weights().to(self.device)
        self.criterion = build_loss(cfg.get("loss", "ce"), weights, float(cfg.get("focal_gamma", 2.0)),
                                    float(cfg.get("label_smoothing", 0.0))).to(self.device)
        self.optimizer = build_optimizer(cfg, [p for p in self.model.parameters() if p.requires_grad])
        self.scheduler = build_scheduler(cfg, self.optimizer)
        self.use_amp = bool(cfg.get("amp", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.run_dir = resolve_path(cfg.get("output_dir", "outputs/runs")) / str(cfg["exp_id"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        self.history: list = []
        self.start_epoch, self.best_val_acc, self.best_epoch = 1, -1.0, 0
        self.wait = 0  # số epoch liên tiếp val acc không cải thiện (early stopping)
        self.wandb = WandbLogger(cfg, self.run_dir)
        self.wandb.watch(self.model)

    # ------------------------------------------------------------------ loops
    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        total_loss, correct, n = 0.0, 0, 0
        max_batches = self.cfg.get("max_batches")  # debug nhanh
        clip = self.cfg.get("grad_clip")
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= int(max_batches):
                break
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with torch.set_grad_enabled(train), torch.autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                if clip:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), float(clip))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
        return {"loss": total_loss / max(n, 1), "acc": correct / max(n, 1)}

    @torch.no_grad()
    def predict(self, loader: DataLoader):
        """Trả về (y_true, y_pred, probs) dạng numpy."""
        self.model.eval()
        ys, ps = [], []
        for x, y in loader:
            ps.append(torch.softmax(self.model(x.to(self.device)), 1).float().cpu())
            ys.append(y)
        probs = torch.cat(ps).numpy()
        return torch.cat(ys).numpy(), probs.argmax(1), probs

    # ------------------------------------------------------------ checkpoint
    def save_checkpoint(self, path: Path, epoch: int, full: bool = True) -> None:
        """full=True (last.pt): kèm optimizer/scheduler để resume; False (best.pt): chỉ trọng số."""
        state = {"model": self.model.state_dict(), "epoch": epoch, "best_val_acc": self.best_val_acc,
                 "best_epoch": self.best_epoch, "wait": self.wait, "history": self.history, "cfg": self.cfg}
        if full:
            state["optimizer"] = self.optimizer.state_dict()
            state["scheduler"] = self.scheduler.state_dict() if self.scheduler else None
        torch.save(state, path)

    def resume(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler and ckpt.get("scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val_acc, self.best_epoch = ckpt["best_val_acc"], ckpt["best_epoch"]
        self.history = ckpt.get("history", [])
        self.wait = ckpt.get("wait", 0)
        print(f"[resume] tiếp tục từ epoch {self.start_epoch}")

    # ------------------------------------------------------------------- fit
    def fit(self) -> Dict[str, Any]:
        epochs, patience = int(self.cfg.get("epochs", 50)), int(self.cfg.get("patience", 10))
        t0 = time.time()
        for epoch in range(self.start_epoch, epochs + 1):
            tr = self._run_epoch(self.loaders["train"], train=True)
            va = self._run_epoch(self.loaders["val"], train=False)
            lr = self.optimizer.param_groups[0]["lr"]
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(va["acc"])
            elif self.scheduler:
                self.scheduler.step()

            row = {"epoch": epoch, "lr": lr, "train_loss": tr["loss"], "train_acc": tr["acc"],
                   "val_loss": va["loss"], "val_acc": va["acc"]}
            self.history.append(row)
            self.wandb.log_epoch(row)
            improved = va["acc"] > self.best_val_acc
            if improved:
                self.best_val_acc, self.best_epoch, self.wait = va["acc"], epoch, 0
                self.save_checkpoint(self.run_dir / "best.pt", epoch, full=False)
            else:
                self.wait += 1
            self.save_checkpoint(self.run_dir / "last.pt", epoch)
            self._write_history()
            print(f"[{self.cfg['exp_id']}] ep {epoch:03d} | lr {lr:.2e} | "
                  f"train {tr['loss']:.4f}/{tr['acc']:.4f} | val {va['loss']:.4f}/{va['acc']:.4f}"
                  f"{' *' if improved else ''}")
            if not math.isfinite(tr["loss"]):
                print("Loss = NaN/Inf → dừng."); break
            if self.wait >= patience:
                print(f"Early stopping (patience={patience}) tại epoch {epoch}."); break

        return self._finalize(time.time() - t0)

    def _write_history(self) -> None:
        with open(self.run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self.history[0]))
            w.writeheader(); w.writerows(self.history)

    def _finalize(self, seconds: float) -> Dict[str, Any]:
        ckpt = torch.load(self.run_dir / "best.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        y_pub, p_pub, _ = self.predict(self.loaders["val"])
        y_pri, p_pri, _ = self.predict(self.loaders["test"])
        best_row = next(r for r in self.history if r["epoch"] == self.best_epoch)
        metrics = {
            "exp_id": self.cfg["exp_id"], "best_epoch": self.best_epoch, "epochs_run": len(self.history),
            "train_acc": best_row["train_acc"], "best_val_acc": self.best_val_acc,
            "public_acc": accuracy(y_pub, p_pub), "private_acc": accuracy(y_pri, p_pri),
            "private_macro_f1": macro_f1(y_pri, p_pri, int(self.cfg.get("num_classes", 7))),
            "params_M": sum(p.numel() for p in self.model.parameters()) / 1e6,
            "train_time_min": seconds / 60,
        }
        (self.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        self._append_summary(metrics)
        self._log_mlflow(metrics)
        self.wandb.log_summary(metrics)
        self.wandb.log_model(self.run_dir / "best.pt", metrics)
        print(json.dumps(metrics, indent=2))
        return metrics

    def _append_summary(self, m: Dict[str, Any]) -> None:
        path = resolve_path(self.cfg.get("summary_csv", "outputs/logs/experiments_summary.csv"))
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {k: self.cfg.get(k, "") for k in SUMMARY_COLUMNS}
        row.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k in SUMMARY_COLUMNS})
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            if new:
                w.writeheader()
            w.writerow(row)

    def _log_mlflow(self, m: Dict[str, Any]) -> None:
        if not self.cfg.get("mlflow", False):
            return
        try:
            import mlflow
        except ImportError:
            print("[mlflow] chưa cài – bỏ qua."); return
        mlflow.set_tracking_uri(str(resolve_path(self.cfg.get("mlflow_uri", "mlruns"))))
        mlflow.set_experiment(self.cfg.get("mlflow_experiment", "BTL02_FER2013"))
        with mlflow.start_run(run_name=str(self.cfg["exp_id"])):
            mlflow.log_params({k: v for k, v in self.cfg.items() if not isinstance(v, (dict, list))})
            for r in self.history:
                mlflow.log_metrics({k: v for k, v in r.items() if k != "epoch"}, step=r["epoch"])
            mlflow.log_metrics({k: v for k, v in m.items() if isinstance(v, (int, float))})


def run_experiment(cfg: Dict[str, Any], resume: Optional[Path] = None) -> Dict[str, Any]:
    """Chạy trọn 1 thí nghiệm từ dict config: dữ liệu → train → đánh giá + hình."""
    cfg = dict(cfg)
    cfg["data_path"] = str(resolve_path(get_data_path(cfg)))
    cfg.pop("csv_path", None)  # key cũ, tránh ghi 2 đường dẫn mâu thuẫn vào config.yaml
    seed_everything(int(cfg.get("seed", 42)))
    loaders = build_dataloaders(cfg, strict=bool(cfg.get("strict_split", True)))
    trainer = EmotionTrainer(cfg, loaders)
    try:
        if resume:
            trainer.resume(Path(resume))
        metrics = trainer.fit()

        if cfg.get("evaluate_after_train", True):
            from src.evaluate.report import EvaluationReport
            summary = EvaluationReport(trainer.model, loaders["test"], trainer.run_dir, cfg, trainer.device).run_all()
            trainer.wandb.log_summary({f"eval/{k}": v for k, v in summary.items()})
            trainer.wandb.log_figures(trainer.run_dir / "figures")
    except BaseException:
        trainer.wandb.finish(exit_code=1)  # đánh dấu run lỗi/bị ngắt trên wandb
        raise
    trainer.wandb.finish()
    return metrics


DEFAULT_CONFIG = "experiments/baseline/exp01_small_cnn.yaml"


def main(argv=None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description="Huấn luyện mô hình FER2013")
    # Mặc định EXP01 để bấm nút Run trong IDE vẫn chạy được; thí nghiệm khác thì truyền --config.
    ap.add_argument("--config", default=DEFAULT_CONFIG, help=f"File YAML thí nghiệm (mặc định {DEFAULT_CONFIG})")
    ap.add_argument("--resume", help="Đường dẫn last.pt để train tiếp")
    ap.add_argument("overrides", nargs="*", help="Ghi đè dạng key=value (vd: lr=0.0005 aug.rotation=15)")
    args = ap.parse_args(argv)
    # Terminal của IDE hay mở ở gốc repo chứ không phải FER2013/ → không thấy file thì tìm theo gốc dự án.
    if not Path(args.config).exists():
        args.config = str(resolve_path(args.config))
    print(f"[trainer] config: {args.config}")
    return run_experiment(load_config(args.config, args.overrides), args.resume)


if __name__ == "__main__":
    main()
