"""Ghi log lên Weights & Biases (tùy chọn; tắt mặc định).

Bật:  `--wandb` khi train, hoặc `wandb: {enable: true}` trong config.
Đăng nhập một lần: `wandb login` (hoặc biến môi trường WANDB_API_KEY). KHÔNG commit key.
Không có mạng: `--set wandb.mode=offline`, sau đó `wandb sync runs/wandb/wandb/offline-run-*`.

train, evaluate, visualize là 3 lệnh riêng nhưng cùng ghi vào MỘT run trên W&B:
train lưu id vào <run_dir>/wandb_id.txt, hai lệnh sau đọc id đó và nối tiếp (resume="allow").
`train --resume` cũng nối tiếp đúng run cũ thay vì tạo run mới.

Ghi gì:
    train      config, mỗi epoch: train/val loss, acc, val macro-F1, lr, khoảng cách train-val;
               summary: best_val_acc, best_epoch, params; best.pt dạng Artifact nếu log_model
    evaluate   summary val/test acc + macro-F1, bảng F1 từng lớp, confusion matrix, true/wrong cases, curves
    visualize  hình filter conv đầu/cuối, activation maximization, feature map, Grad-CAM
"""
import os
import uuid
from pathlib import Path

from fer.utils.config import ROOT

DEFAULTS = {"enable": False, "project": "BTL02_FER2013", "entity": None, "group": None, "tags": [],
            "mode": "online", "log_model": False, "watch": False, "dir": "runs/wandb"}
ID_FILE = "wandb_id.txt"


class WandbLogger:
    """Bọc wandb. Khi tắt, chưa cài wandb hoặc init lỗi thì mọi phương thức không làm gì."""

    def __init__(self, cfg: dict, run_dir: Path, job_type: str = "train", force: bool | None = None):
        self.cfg = {**DEFAULTS, **(cfg.get("wandb") or {})}
        self.run_dir, self.run = Path(run_dir), None
        id_file = self.run_dir / ID_FILE
        # train: theo config. evaluate/visualize: tự bật nếu lúc train đã bật (có wandb_id.txt)
        enabled = self.cfg["enable"] if force is None else force
        if job_type != "train" and force is None:
            enabled = id_file.exists()
        if not enabled or self.cfg["mode"] == "disabled":
            return
        try:
            import wandb
        except ImportError:
            print("[wandb] chưa cài (pip install wandb), bỏ qua ghi log.")
            return
        self.wandb = wandb

        wb_dir = Path(self.cfg["dir"])
        wb_dir = wb_dir if wb_dir.is_absolute() else ROOT / wb_dir
        wb_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("WANDB_DIR", str(wb_dir))

        run_id = id_file.read_text().strip() if id_file.exists() else uuid.uuid4().hex[:8]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        id_file.write_text(run_id)

        m, t = cfg["model"], cfg["train"]
        tags = list(self.cfg["tags"]) + [m["name"], m.get("activation", ""), t.get("optimizer", ""),
                                         "aug" if cfg["aug"].get("enable") else "no_aug"]
        try:
            self.run = wandb.init(project=self.cfg["project"], entity=self.cfg["entity"], group=self.cfg["group"],
                                  name=self.run_dir.name, id=run_id, resume="allow", job_type=job_type,
                                  tags=[x for x in tags if x], config=cfg, dir=str(wb_dir), mode=self.cfg["mode"])
        except Exception as e:  # mất mạng / chưa login: không làm hỏng cả lần train
            print(f"[wandb] không khởi tạo được ({e}); tiếp tục mà không ghi log.")
            return
        self.run.define_metric("epoch")
        for prefix in ("train/*", "val/*", "lr", "train_val_gap", "sec"):
            self.run.define_metric(prefix, step_metric="epoch")

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def watch(self, model):
        if self.enabled and self.cfg["watch"]:
            self.run.watch(model, log="all", log_freq=200)

    def log_epoch(self, row: dict):
        if self.enabled:
            self.run.log({"epoch": row["epoch"], "lr": row["lr"], "sec": row["sec"],
                          "train/loss": row["train_loss"], "train/acc": row["train_acc"],
                          "val/loss": row["val_loss"], "val/acc": row["val_acc"], "val/macro_f1": row["val_macro_f1"],
                          "train_val_gap": row["train_acc"] - row["val_acc"]})  # lớn = overfit

    def summary(self, metrics: dict):
        if self.enabled:
            self.run.summary.update(metrics)

    def log_images(self, folder: Path, prefix: str):
        """Đẩy mọi PNG trong folder lên, khóa '<prefix>/<tên file>'."""
        if not self.enabled or not Path(folder).exists():
            return
        imgs = {f"{prefix}/{p.stem}": self.wandb.Image(str(p)) for p in sorted(Path(folder).glob("*.png"))}
        if imgs:
            self.run.log(imgs)

    def log_table(self, key: str, columns: list, rows: list):
        if self.enabled:
            self.run.log({key: self.wandb.Table(columns=columns, data=rows)})

    def log_model(self, path: Path, metadata: dict | None = None):
        if self.enabled and self.cfg["log_model"] and Path(path).exists():
            art = self.wandb.Artifact(f"{self.run_dir.name}-model", type="model", metadata=metadata or {})
            art.add_file(str(path))
            self.run.log_artifact(art, aliases=["best"])

    def finish(self, exit_code: int = 0):
        if self.enabled:
            self.run.finish(exit_code=exit_code)
            self.run = None
