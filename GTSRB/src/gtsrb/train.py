"""Huấn luyện một cấu hình với một seed.

    python -m gtsrb.train --config configs/e1_small_cnn.yaml --seed 42 --out_dir runs
    python -m gtsrb.train --config configs/e1_small_cnn.yaml --seed 42 --out_dir runs_smoke --fast
    # Colab ngắt giữa chừng: chạy lại ĐÚNG lệnh cũ, thêm --resume
    python -m gtsrb.train --config configs/e1_small_cnn.yaml --seed 42 --out_dir runs --resume

Kết quả nằm trong <out_dir>/<name>_s<seed>/: best.pt, last.pt, config.yaml, log.csv, env.json.
Chỉ dùng tập train và val. Tập test không được đụng tới ở đây.
"""
import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score

from gtsrb.data.dataset import build_dataloaders
from gtsrb.models import build_model
from gtsrb.utils.config import ROOT, load_config
from gtsrb.utils.seed import set_seed

LOG_COLS = ["epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "lr", "sec"]


def env_info(model) -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": git("status", "--porcelain") not in ("", "unknown"),  # true = có thay đổi chưa commit
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "params": sum(p.numel() for p in model.parameters()),
    }


def train_epoch(model, loader, optimizer, scaler, criterion, device, use_amp):
    model.train()
    loss_sum, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * len(y)
        n += len(y)
    return loss_sum / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_sum, preds, labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * len(y)
        preds.append(out.argmax(1).cpu())
        labels.append(y.cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    acc = float((preds == labels).mean())
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    return loss_sum / len(labels), acc, macro_f1


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--fast", action="store_true", help="~10%% dữ liệu, tối đa 2 epoch: chỉ để thử pipeline")
    ap.add_argument("--resume", action="store_true", help="tiếp tục từ last.pt")
    ap.add_argument("--stop_after", type=int, default=0, help="(dùng để thử --resume) dừng sau N epoch")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, overrides={"seed": a.seed})  # seed luôn lấy từ --seed
    if a.fast:
        cfg["train"]["epochs"] = min(cfg["train"]["epochs"], 2)
    t = cfg["train"]
    set_seed(a.seed)

    run_dir = Path(a.out_dir) / f"{cfg['name']}_s{a.seed}"
    last_path, best_path, log_path = run_dir / "last.pt", run_dir / "best.pt", run_dir / "log.csv"
    if last_path.exists() and not a.resume:
        raise SystemExit(f"{run_dir} đã có kết quả. Thêm --resume để chạy tiếp, hoặc xóa thư mục nếu muốn chạy lại từ đầu.")
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(t.get("amp", False)) and device.type == "cuda"
    train_loader, val_loader = build_dataloaders(cfg, fast=a.fast)

    model = build_model(cfg["model"]["name"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])
                 if t.get("scheduler") == "cosine" else None)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_f1 = 1, -1.0
    if a.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_f1 = ck["epoch"] + 1, ck["best_f1"]
        print(f"Tiếp tục từ epoch {start_epoch}")

    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    if not (run_dir / "env.json").exists():
        (run_dir / "env.json").write_text(json.dumps(env_info(model), indent=2))
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(LOG_COLS)

    for epoch in range(start_epoch, t["epochs"] + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_epoch(model, train_loader, optimizer, scaler, criterion, device, use_amp)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        sec = time.time() - t0

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.5f}", f"{val_loss:.5f}", f"{val_acc:.5f}",
                                    f"{val_f1:.5f}", f"{lr:.6g}", f"{sec:.1f}"])
        print(f"epoch {epoch}/{t['epochs']}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
              f"val_acc {val_acc:.4f}  val_macro_f1 {val_f1:.4f}  ({sec:.0f}s)")

        if val_f1 > best_f1:  # chọn checkpoint theo Macro-F1 trên val (ảnh gốc), không dùng test
            best_f1 = val_f1
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val_macro_f1": val_f1}, best_path)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict(), "epoch": epoch, "best_f1": best_f1}, last_path)

        if a.stop_after and epoch >= a.stop_after:
            print(f"Dừng sau epoch {epoch} theo --stop_after")
            break

    print(f"Xong. best val_macro_f1 = {best_f1:.4f}. Kết quả ở {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
