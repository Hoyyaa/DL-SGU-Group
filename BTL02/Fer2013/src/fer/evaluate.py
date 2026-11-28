"""Chấm một run trên tập val và test, vẽ hình cho báo cáo.

    python -m fer.evaluate --run runs/e5_large_bn_s42

Ghi vào results/<tên run>/:
    metrics.json          acc, macro_f1, report từng lớp cho val và test
    curves.png            loss / acc / lr theo epoch (từ log.csv)
    confusion_val.png, confusion_test.png       (chuẩn hóa theo hàng)
    true_cases.png, wrong_cases.png               (lấy trên tập test)
    preds_test.npz     logits + nhãn, để phân tích thêm
Sau đó gọi summarize để cập nhật results/summary.csv.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, f1_score  # noqa: E402

from fer.data.dataset import build_loader, load_cache  # noqa: E402
from fer.data.prepare import CLASSES  # noqa: E402
from fer.models import build_model  # noqa: E402
from fer.train import predict  # noqa: E402
from fer.utils.config import ROOT  # noqa: E402
from fer.utils.wandb_logger import WandbLogger  # noqa: E402

# mốc world-best của đề: 69% PublicTest (= val khi chia gốc), 71% PrivateTest (= test)
WORLD_BEST = {"val": 0.69, "test": 0.71}


def load_run(run_dir: Path, device, cache: str | None = None):
    ck = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    cfg = ck["cfg"]
    if cache:
        cfg["data"]["cache"] = cache
    cfg["train"]["num_workers"] = 0
    model = build_model(cfg["model"], ck["mean"], ck["std"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), cfg, ck


def plot_curves(log_csv: Path, out: Path):
    df = pd.read_csv(log_csv)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(df.epoch, df.train_loss, label="train"), ax[0].plot(df.epoch, df.val_loss, label="val")
    ax[1].plot(df.epoch, df.train_acc, label="train"), ax[1].plot(df.epoch, df.val_acc, label="val")
    ax[2].plot(df.epoch, df.lr), ax[2].set_yscale("log")
    best = df.val_acc.idxmax()
    ax[1].axvline(df.epoch[best], ls="--", c="gray")
    ax[1].set_title(f"best epoch {df.epoch[best]}: val_acc {df.val_acc[best]:.4f}")
    for a, t in zip(ax, ["loss", "accuracy", "learning rate"]):
        a.set_xlabel("epoch"), a.set_ylabel(t), a.grid(alpha=0.3)
    ax[0].legend(), ax[1].legend()
    fig.tight_layout(), fig.savefig(out, dpi=120), plt.close(fig)


def plot_confusion(labels, preds, title, out: Path):
    cm = confusion_matrix(labels, preds, labels=range(7))
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(7), CLASSES, rotation=45), ax.set_yticks(range(7), CLASSES)
    ax.set_xlabel("dự đoán"), ax.set_ylabel("nhãn thật"), ax.set_title(title)
    fig.colorbar(im), fig.tight_layout(), fig.savefig(out, dpi=120), plt.close(fig)


def plot_cases(images, labels, preds, probs, idx, title, out: Path, n=32, seed=0):
    idx = np.random.default_rng(seed).permutation(idx)[:n]
    cols = 8
    rows = max(1, int(np.ceil(len(idx) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 2.0))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).ravel(), idx):
        ax.imshow(images[i], cmap="gray")
        ok = labels[i] == preds[i]
        ax.set_title(f"T:{CLASSES[labels[i]]}\nP:{CLASSES[preds[i]]} {probs[i, preds[i]]:.2f}", fontsize=7,
                     color="green" if ok else "red")
    fig.suptitle(title), fig.tight_layout(), fig.savefig(out, dpi=120), plt.close(fig)


def evaluate_run(run_dir: Path, out_root: Path | None = None, cache: str | None = None, wandb: bool | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, ck = load_run(run_dir, device, cache)
    out = (out_root or ROOT / "results") / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    data = load_cache(cfg)

    result = {"run": run_dir.name, "exp": cfg["name"], "model": cfg["model"]["name"], "seed": cfg["seed"],
              "best_epoch": ck["epoch"], "params": sum(p.numel() for p in model.parameters())}
    for split in ("val", "test"):
        logits, labels = predict(model, build_loader(cfg, split, data=data), device)
        preds = logits.argmax(1)
        result[f"{split}_acc"] = float((preds == labels).mean())
        result[f"{split}_macro_f1"] = float(f1_score(labels, preds, average="macro", zero_division=0))
        result[f"{split}_report"] = classification_report(labels, preds, labels=range(7), target_names=CLASSES,
                                                          output_dict=True, zero_division=0)
        plot_confusion(labels, preds, f"{run_dir.name} – {split} acc {result[f'{split}_acc']:.4f}",
                       out / f"confusion_{split}.png")

    # ảnh đúng / sai lấy trên tập test (chưa từng dùng để chọn mô hình)
    probs = torch.softmax(torch.from_numpy(logits), 1).numpy()
    images = data["x"][data["split"] == "test"]
    np.savez_compressed(out / "preds_test.npz", logits=logits, labels=labels)
    plot_cases(images, labels, preds, probs, np.flatnonzero(preds == labels), "True cases (test)",
               out / "true_cases.png")
    plot_cases(images, labels, preds, probs, np.flatnonzero(preds != labels), "Wrong cases (test)",
               out / "wrong_cases.png")
    if (run_dir / "log.csv").exists():
        plot_curves(run_dir / "log.csv", out / "curves.png")

    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log_to_wandb(cfg, run_dir, out, result, wandb)
    print(f"{run_dir.name}: val {result['val_acc']:.4f} (world-best {WORLD_BEST['val']})  "
          f"test {result['test_acc']:.4f} (world-best {WORLD_BEST['test']})  -> {out}")
    return result


def log_to_wandb(cfg, run_dir: Path, out: Path, result: dict, force: bool | None = None):
    """Nối vào run W&B lúc train (nếu có): summary val/test, bảng F1 từng lớp, mọi hình đánh giá."""
    wb = WandbLogger(cfg, run_dir, job_type="evaluate", force=force)
    if not wb.enabled:
        return
    wb.summary({f"{s}/{k}": result[f"{s}_{k}"] for s in ("val", "test") for k in ("acc", "macro_f1")})
    rep = result["test_report"]
    wb.log_table("test/per_class", ["class", "precision", "recall", "f1", "support"],
                 [[c, rep[c]["precision"], rep[c]["recall"], rep[c]["f1-score"], rep[c]["support"]] for c in CLASSES])
    wb.log_images(out, "eval")
    wb.finish()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True, help="một hoặc nhiều thư mục run")
    ap.add_argument("--out_dir", default=None, help="mặc định results/")
    ap.add_argument("--data", default=None, help="ghi đè data.cache (dùng trong test)")
    ap.add_argument("--no_wandb", action="store_true", help="không ghi lên W&B dù lúc train đã bật")
    a = ap.parse_args(argv)
    out_root = Path(a.out_dir) if a.out_dir else None
    for r in a.run:
        evaluate_run(Path(r), out_root, a.data, False if a.no_wandb else None)
    from fer.summarize import summarize
    summarize(out_root or ROOT / "results")


if __name__ == "__main__":
    main()
