"""Huấn luyện một cấu hình với một seed.

    python -m fer.train --config configs/e5_large_bn.yaml --seed 42
    python -m fer.train --config configs/e1_small.yaml --seed 42 --out_dir runs_smoke --fast
    python -m fer.train --config configs/e5_large_bn.yaml --seed 42 --resume       # Colab bị ngắt
    python -m fer.train --config configs/e5_large_bn.yaml --seed 42 --set train.lr=3e-4 name=e5_lr3e-4

Kết quả ở <out_dir>/<name>_s<seed>/: best.pt, last.pt, config.yaml, log.csv, env.json.
Chọn best.pt theo val_acc trên tập val. Tập test không được dùng ở đây.
Tương đương callbacks của báo cáo mẫu: ModelCheckpoint, CSVLogger, EarlyStopping (mẫu tắt ReduceLROnPlateau; ở đây là tùy chọn).
"""
import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score

from fer.data.dataset import build_dataloaders
from fer.models import build_model, count_params
from fer.utils.config import ROOT, load_config, parse_overrides
from fer.utils.seed import set_seed
from fer.utils.wandb_logger import WandbLogger

LOG_COLS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "lr", "sec"]


def env_info(model) -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": git("status", "--porcelain", ".") not in ("", "unknown"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "params": count_params(model),
    }


def amp_supported(device) -> bool:
    """AMP (fp16) chỉ bật trên GPU chạy fp16 đúng.

    GTX 16xx (GTX 1650/1660, Turing không Tensor Core) có lỗi cuDNN: conv fp16 trả NaN ngay batch đầu
    (đã kiểm tra trên GTX 1650: fp32 loss 2.07, fp16 loss NaN). Các card này chạy fp32.
    """
    if device.type != "cuda":
        return False
    name = torch.cuda.get_device_name(device)
    if "GTX 16" in name:
        print(f"[amp] {name}: fp16 bị lỗi NaN trên dòng GTX 16xx, tự tắt AMP (chạy fp32).")
        return False
    return True


def build_optimizer(model, t):
    params = [p for p in model.parameters() if p.requires_grad]
    name = t.get("optimizer", "adam").lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=t["lr"], weight_decay=t.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=t["lr"], weight_decay=t.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(params, lr=t["lr"], momentum=0.9, nesterov=True, weight_decay=t.get("weight_decay", 0.0))
    raise ValueError(f"optimizer '{name}' không hỗ trợ (adam | adamw | sgd)")


def build_scheduler(optimizer, t):
    name = t.get("scheduler", "none")
    if name == "plateau":  # giống ReduceLROnPlateau của báo cáo mẫu, theo dõi val_loss
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=t.get("lr_patience", 3),
                                                          min_lr=1e-6)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])
    if name == "none":
        return None
    raise ValueError(f"scheduler '{name}' không hỗ trợ (plateau | cosine | none)")


def build_criterion(t, train_labels, device):
    weight = None
    if t.get("class_weight", False):  # bù cho lớp Disgust rất ít mẫu
        counts = np.bincount(train_labels, minlength=7).astype(np.float32)
        weight = torch.tensor(counts.sum() / (7 * counts), device=device)
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=t.get("label_smoothing", 0.0))


def train_epoch(model, loader, optimizer, scaler, criterion, device, use_amp):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)
        if not torch.isfinite(loss):  # TerminateOnNaN
            raise SystemExit("Loss = NaN/Inf, dừng huấn luyện. Thử giảm lr.")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return loss_sum / n, correct / n


@torch.no_grad()
def predict(model, loader, device):
    """Trả về (logits, labels) dạng numpy cho cả tập."""
    model.eval()
    outs, labels = [], []
    for x, y in loader:
        outs.append(model(x.to(device)).float().cpu())
        labels.append(y)
    return torch.cat(outs).numpy(), torch.cat(labels).numpy()


def validate(model, loader, criterion, device):
    logits, labels = predict(model, loader, device)
    loss = criterion(torch.from_numpy(logits).to(device), torch.from_numpy(labels).to(device)).item()
    preds = logits.argmax(1)
    return loss, float((preds == labels).mean()), float(f1_score(labels, preds, average="macro", zero_division=0))


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--fast", action="store_true", help="~10%% dữ liệu, tối đa 2 epoch: chỉ để thử pipeline")
    ap.add_argument("--resume", action="store_true", help="tiếp tục từ last.pt")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="ghi đè config, vd train.lr=3e-4")
    ap.add_argument("--wandb", action="store_true", help="ghi log lên Weights & Biases (= --set wandb.enable=true)")
    a = ap.parse_args(argv)

    overrides = parse_overrides(a.set)
    if a.wandb:
        overrides.setdefault("wandb", {})["enable"] = True
    cfg = load_config(a.config, overrides={**overrides, "seed": a.seed})
    t = cfg["train"]
    if a.fast:
        t["epochs"] = min(t["epochs"], 2)
    set_seed(a.seed)

    out_dir = Path(a.out_dir)
    run_dir = (out_dir if out_dir.is_absolute() else ROOT / out_dir) / f"{cfg['name']}_s{a.seed}"
    last_path, best_path, log_path = run_dir / "last.pt", run_dir / "best.pt", run_dir / "log.csv"
    if last_path.exists() and not a.resume:
        raise SystemExit(f"{run_dir} đã có kết quả. Thêm --resume để chạy tiếp, hoặc xóa thư mục để chạy lại.")
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = amp_supported(device) if t.get("amp", False) else False
    train_loader, val_loader, data = build_dataloaders(cfg, fast=a.fast)
    mean, std = float(data["mean"]), float(data["std"])

    model = build_model(cfg["model"], mean, std).to(device)
    criterion = build_criterion(t, train_loader.dataset.y, device)
    optimizer = build_optimizer(model, t)
    scheduler = build_scheduler(optimizer, t)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_acc, best_epoch, bad_epochs = 1, -1.0, 0, 0
    if a.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_acc, bad_epochs = ck["epoch"] + 1, ck["best_acc"], ck["bad_epochs"]
        best_epoch = ck.get("best_epoch", 0)
        if bad_epochs >= t.get("early_stop_patience", 10) or start_epoch > t["epochs"]:
            print(f"{run_dir} đã train xong, bỏ qua.")
            return run_dir
        print(f"Tiếp tục từ epoch {start_epoch}")

    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    if not (run_dir / "env.json").exists():
        (run_dir / "env.json").write_text(json.dumps(env_info(model), indent=2))
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(LOG_COLS)
    print(f"{cfg['name']}: {count_params(model):,} tham số, thiết bị {device}, "
          f"train {len(train_loader.dataset)} / val {len(val_loader.dataset)}")

    wb = WandbLogger(cfg, run_dir, job_type="train")
    wb.summary({"params": count_params(model), "train_size": len(train_loader.dataset)})
    wb.watch(model)
    try:
        best_epoch = _fit(model, train_loader, val_loader, optimizer, scheduler, scaler, criterion, device, use_amp,
                          cfg, run_dir, start_epoch, best_acc, best_epoch, bad_epochs, mean, std, wb)
    except BaseException:  # lỗi hoặc Ctrl+C: đánh dấu run thất bại trên W&B rồi ném lỗi tiếp
        wb.finish(exit_code=1)
        raise
    wb.log_model(best_path, {"best_epoch": best_epoch})
    wb.finish()
    return run_dir


def _fit(model, train_loader, val_loader, optimizer, scheduler, scaler, criterion, device, use_amp,
         cfg, run_dir, start_epoch, best_acc, best_epoch, bad_epochs, mean, std, wb) -> int:
    """Vòng lặp epoch: train, validate, lưu checkpoint, early stopping. Trả về epoch tốt nhất."""
    t = cfg["train"]
    last_path, best_path, log_path = run_dir / "last.pt", run_dir / "best.pt", run_dir / "log.csv"
    patience = t.get("early_stop_patience", 10)
    for epoch in range(start_epoch, t["epochs"] + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler, criterion, device, use_amp)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()
        sec = time.time() - t0

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.5f}", f"{train_acc:.5f}", f"{val_loss:.5f}",
                                    f"{val_acc:.5f}", f"{val_f1:.5f}", f"{lr:.6g}", f"{sec:.1f}"])
        print(f"epoch {epoch}/{t['epochs']}  loss {train_loss:.4f}/{val_loss:.4f}  "
              f"acc {train_acc:.4f}/{val_acc:.4f}  val_f1 {val_f1:.4f}  lr {lr:.2g}  ({sec:.0f}s)")
        wb.log_epoch({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss,
                      "val_acc": val_acc, "val_macro_f1": val_f1, "lr": lr, "sec": sec})

        if val_acc > best_acc:
            best_acc, best_epoch, bad_epochs = val_acc, epoch, 0
            wb.summary({"best_val_acc": val_acc, "best_val_macro_f1": val_f1, "best_epoch": epoch})
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val_acc": val_acc,
                        "mean": mean, "std": std}, best_path)
        else:
            bad_epochs += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict(), "epoch": epoch, "best_acc": best_acc, "best_epoch": best_epoch,
                    "bad_epochs": bad_epochs}, last_path)
        if bad_epochs >= patience:
            print(f"EarlyStopping: {patience} epoch liền không cải thiện val_acc")
            break

    print(f"Xong. best val_acc = {best_acc:.4f}. Kết quả ở {run_dir}")
    return best_epoch


if __name__ == "__main__":
    main()
