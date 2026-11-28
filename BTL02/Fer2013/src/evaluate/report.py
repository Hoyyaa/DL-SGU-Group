"""Đánh giá & trực quan hóa sau huấn luyện (yêu cầu đề: visualize + debug + giải thích).

Sinh vào ``<run_dir>/figures/``:
    training_curves.png        – loss/acc train vs val (chẩn đoán over/underfitting)
    confusion_matrix.png       – số lượng + chuẩn hóa theo hàng
    classification_report.csv  – precision/recall/F1 từng lớp
    last_conv_filters.png      – filter của tầng conv cuối (yêu cầu đề bài)
    first_conv_filters.png     – filter tầng đầu (dễ diễn giải: cạnh, góc)
    feature_maps.png           – activation tầng conv cuối cho 1 ảnh
    gradcam_per_class.png      – Grad-CAM cho mỗi lớp (ca dự đoán đúng)
    true_cases.png / wrong_cases.png – ca đúng tự tin nhất / sai tự tin nhất
    top_confusions.csv         – các cặp lớp hay nhầm

CLI:
    python -m src.evaluate.report --run outputs/runs/EXP01_small_cnn [--split test]
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    # Chạy trực tiếp (nút Run của VS Code / python src/.../file.py) thì Python không thấy package `src`;
    # thay thư mục chứa file bằng gốc dự án để import giống hệt `python -m`.
    import sys
    sys.path[0] = str(Path(__file__).resolve().parents[2])

from src import EMOTION_LABELS
from src.data.transforms import denormalize
from src.evaluate.gradcam import GradCAM
from src.evaluate.metrics import classification_report, confusion_matrix

K = len(EMOTION_LABELS)


class EvaluationReport:
    def __init__(self, model: nn.Module, loader: DataLoader, run_dir: Path,
                 cfg: Dict[str, Any], device: torch.device, split_name: str = "PrivateTest") -> None:
        self.model, self.loader, self.cfg, self.device = model.to(device).eval(), loader, cfg, device
        self.run_dir = Path(run_dir)
        self.fig_dir = self.run_dir / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.split_name = split_name
        self.mode = cfg.get("input_mode", "gray")

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def _collect(self):
        """Chỉ giữ nhãn & xác suất; ảnh lấy lại theo chỉ số khi vẽ (tránh giữ cả tập test 128x128x3 trong RAM).
        Yêu cầu loader không shuffle và transform xác định (đúng với tập val/test)."""
        ys, probs = [], []
        for x, y in self.loader:
            probs.append(torch.softmax(self.model(x.to(self.device)), 1).float().cpu())
            ys.append(y)
        self.y, self.probs = torch.cat(ys).numpy(), torch.cat(probs).numpy()
        self.pred = self.probs.argmax(1)
        self.conf = self.probs.max(1)

    def _images(self, idx) -> torch.Tensor:
        """Lấy batch ảnh (đã transform) theo chỉ số trong dataset."""
        ds = self.loader.dataset
        return torch.stack([ds[int(i)][0] for i in idx]) if len(idx) else torch.empty(0)

    def _show(self, ax, img: torch.Tensor, title: str = "", heat=None) -> None:
        im = denormalize(img, self.mode).permute(1, 2, 0).numpy()
        ax.imshow(im.squeeze(), cmap="gray" if im.shape[-1] == 1 else None)
        if heat is not None:
            ax.imshow(heat, cmap="jet", alpha=0.4)
        ax.set_title(title, fontsize=7); ax.axis("off")

    def _save(self, fig, name: str) -> Path:
        path = self.fig_dir / name
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
        return path

    # ----------------------------------------------------------------- plots
    def training_curves(self) -> None:
        hist = self.run_dir / "history.csv"
        if not hist.exists():
            return
        h = pd.read_csv(hist)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        for ax, key in zip(axes, ("loss", "acc")):
            ax.plot(h["epoch"], h[f"train_{key}"], label="train")
            ax.plot(h["epoch"], h[f"val_{key}"], label="val (PublicTest)")
            ax.set_xlabel("epoch"); ax.set_title(key); ax.legend(); ax.grid(alpha=.3)
        self._save(fig, "training_curves.png")

    def confusion(self) -> Dict[str, Any]:
        cm = confusion_matrix(self.y, self.pred, K).numpy()
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, mat, fmt, title in ((axes[0], cm, "d", "Counts"), (axes[1], cmn, ".2f", "Normalized (recall)")):
            ax.imshow(mat, cmap="Blues")
            for i in range(K):
                for j in range(K):
                    ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", fontsize=7,
                            color="white" if mat[i, j] > mat.max() / 2 else "black")
            ax.set_xticks(range(K), EMOTION_LABELS, rotation=45); ax.set_yticks(range(K), EMOTION_LABELS)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{self.split_name} – {title}")
        self._save(fig, "confusion_matrix.png")

        rep = classification_report(self.y, self.pred, EMOTION_LABELS)
        rep.to_csv(self.fig_dir / "classification_report.csv")
        pairs = [(EMOTION_LABELS[i], EMOTION_LABELS[j], int(cm[i, j]), float(cmn[i, j]))
                 for i in range(K) for j in range(K) if i != j]
        pd.DataFrame(sorted(pairs, key=lambda r: -r[2])[:10],
                     columns=["true", "pred", "count", "rate"]).to_csv(self.fig_dir / "top_confusions.csv", index=False)
        return {"accuracy": float(rep.loc["accuracy", "f1-score"]),
                "macro_f1": float(rep.loc["macro avg", "f1-score"])}

    def conv_filters(self, layer: nn.Conv2d, name: str, max_filters: int = 64) -> None:
        """Mỗi filter (out_c) được hiển thị bằng trung bình theo kênh đầu vào."""
        w = layer.weight.detach().cpu()
        w = w.mean(1) if w.shape[1] != 3 else w.permute(0, 2, 3, 1)
        n = min(max_filters, w.shape[0]); cols = 8; rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.1))
        for i, ax in enumerate(np.array(axes).ravel()):
            ax.axis("off")
            if i < n:
                f = w[i]; f = (f - f.min()) / (f.max() - f.min() + 1e-8)
                ax.imshow(f.numpy(), cmap="viridis" if f.ndim == 2 else None)
        fig.suptitle(f"{name}: {tuple(layer.weight.shape)}", fontsize=9)
        self._save(fig, f"{name}.png")

    def feature_maps(self, idx: int = 0, max_maps: int = 32) -> None:
        store = {}
        h = self.model.get_target_layer().register_forward_hook(lambda m, i, o: store.setdefault("a", o.detach()))
        with torch.no_grad():
            self.model(self._images([idx]).to(self.device))
        h.remove()
        a = store["a"][0].cpu()
        n = min(max_maps, a.shape[0]); cols = 8; rows = int(np.ceil(n / cols)) + 1
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
        axes = np.array(axes).ravel()
        for ax in axes: ax.axis("off")
        self._show(axes[0], self._images([idx])[0], f"true={EMOTION_LABELS[self.y[idx]]}")
        for i in range(n):
            axes[cols + i].imshow(a[i].numpy(), cmap="viridis")
        self._save(fig, "feature_maps.png")

    def gradcam_per_class(self, per_class: int = 4) -> None:
        fig, axes = plt.subplots(K, per_class, figsize=(per_class * 1.6, K * 1.7))
        with GradCAM(self.model, self.model.get_target_layer()) as cam:
            for c in range(K):
                idx = np.where((self.y == c) & (self.pred == c))[0]
                idx = idx[np.argsort(-self.conf[idx])][:per_class]
                imgs = self._images(idx)
                heats = cam(imgs.to(self.device))[0].cpu().numpy() if len(idx) else []
                for j in range(per_class):
                    ax = axes[c, j]; ax.axis("off")
                    if j < len(idx):
                        self._show(ax, imgs[j], f"{EMOTION_LABELS[c]} {self.conf[idx[j]]:.2f}", heats[j])
        self._save(fig, "gradcam_per_class.png")

    def cases(self, correct: bool, n: int = 32) -> None:
        mask = (self.pred == self.y) if correct else (self.pred != self.y)
        idx = np.where(mask)[0]
        idx = idx[np.argsort(-self.conf[idx])][:n]  # sai mà tự tin cao → đáng phân tích nhất
        cols = 8; rows = max(1, int(np.ceil(len(idx) / cols)))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.8))
        for ax in np.array(axes).ravel(): ax.axis("off")
        imgs = self._images(idx)
        for ax, i, img in zip(np.array(axes).ravel(), idx, imgs):
            self._show(ax, img, f"T:{EMOTION_LABELS[self.y[i]]}\nP:{EMOTION_LABELS[self.pred[i]]} {self.conf[i]:.2f}")
        self._save(fig, "true_cases.png" if correct else "wrong_cases.png")

    def run_all(self) -> Dict[str, Any]:
        self._collect()
        self.training_curves()
        summary = self.confusion()
        convs = [m for m in self.model.modules() if isinstance(m, nn.Conv2d)]
        self.conv_filters(convs[0], "first_conv_filters")
        self.conv_filters(convs[-1], "last_conv_filters")
        self.feature_maps()
        self.gradcam_per_class()
        self.cases(True); self.cases(False)
        (self.fig_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[report] {self.split_name}: {summary} → {self.fig_dir}")
        return summary


def main(argv=None) -> None:
    from src.data.dataset import build_dataloaders
    from src.models import build_model
    from src.train.utils import get_device, resolve_path

    ap = argparse.ArgumentParser(description="Đánh giá + trực quan hóa 1 run đã train")
    ap.add_argument("--run", required=True, help="Thư mục outputs/runs/<exp_id>")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--checkpoint", default="best.pt")
    args = ap.parse_args(argv)

    run_dir = resolve_path(args.run)
    ckpt = torch.load(run_dir / args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    cfg["pretrained"] = False  # trọng số lấy từ checkpoint, không cần tải ImageNet
    device = get_device(cfg.get("device", "auto"))
    model = build_model(cfg); model.load_state_dict(ckpt["model"])
    loaders = build_dataloaders(cfg, strict=bool(cfg.get("strict_split", True)))
    EvaluationReport(model, loaders[args.split], run_dir, cfg, device,
                     "PrivateTest" if args.split == "test" else "PublicTest").run_all()


if __name__ == "__main__":
    main()
