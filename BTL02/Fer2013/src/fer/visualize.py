"""Trực quan hóa bên trong mô hình (đề bài bắt buộc: "Visualize the final convolution filters").

    python -m fer.visualize --run runs/e5_large_bn_s42

Ghi vào results/<tên run>/viz/:
    filters_first_conv.png   trọng số 3x3 của conv đầu tiên (bộ lọc cạnh / sáng tối)
    filters_last_conv.png    trọng số 3x3 của conv CUỐI (trung bình theo kênh vào)
    actmax_last_conv.png     ảnh đầu vào làm mỗi bộ lọc conv cuối kích hoạt mạnh nhất (gradient ascent)
    featuremaps_last_conv.png  feature map conv cuối cho 1 ảnh mỗi lớp
    gradcam.png              vùng mô hình nhìn vào (Grad-CAM) cho ảnh đúng và sai từng lớp

Trọng số 3x3 của conv cuối khó đọc bằng mắt vì chúng tác động lên đặc trưng trừu tượng,
nên actmax + feature map + Grad-CAM là phần dùng để giải thích trong báo cáo.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from fer.data.dataset import load_cache  # noqa: E402
from fer.data.prepare import CLASSES  # noqa: E402
from fer.evaluate import load_run  # noqa: E402
from fer.utils.config import ROOT  # noqa: E402
from fer.utils.wandb_logger import WandbLogger  # noqa: E402


def _grid(imgs, titles, out: Path, suptitle: str, cols=8, cmap="gray", size=1.4):
    rows = max(1, int(np.ceil(len(imgs) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size * 1.1))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, im, t in zip(axes, imgs, titles):
        ax.imshow(im, cmap=cmap)
        ax.set_title(t, fontsize=6)
    fig.suptitle(suptitle), fig.tight_layout(), fig.savefig(out, dpi=130), plt.close(fig)


def plot_conv_weights(conv: torch.nn.Conv2d, out: Path, title: str, n=64):
    w = conv.weight.detach().float().cpu()  # (C_out, C_in, 3, 3)
    w = w.mean(1) if w.shape[1] > 1 else w[:, 0]
    imgs = [(k - k.min()) / (k.max() - k.min() + 1e-8) for k in w[:n].numpy()]
    _grid(imgs, [f"#{i}" for i in range(len(imgs))], out, f"{title} ({tuple(conv.weight.shape)})")


def capture(model, layer, x):
    """Chạy forward và lấy output của `layer`."""
    store = {}
    h = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    try:
        model(x)
    finally:
        h.remove()
    return store["a"]


def activation_maximization(model, layer, n_filters=32, steps=60, lr=0.1, size=48, device="cpu"):
    """Tối ưu ảnh đầu vào (mỗi bộ lọc một ảnh) để kích hoạt trung bình của bộ lọc đó lớn nhất."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    x = (0.5 + 0.05 * torch.randn(n_filters, 1, size, size, device=device)).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    idx = torch.arange(n_filters, device=device)
    for _ in range(steps):
        opt.zero_grad()
        act = capture(model, layer, x.clamp(0, 1))  # (K, C, h, w)
        loss = -act[idx, idx % act.shape[1]].mean()
        loss.backward()
        opt.step()
    for p in model.parameters():
        p.requires_grad_(True)
    return x.detach().clamp(0, 1)[:, 0].cpu().numpy()


def grad_cam(model, layer, x, target):
    """Grad-CAM: trọng số mỗi kênh = trung bình gradient của logit lớp `target`, rồi tổng có trọng số + ReLU."""
    store = {}

    def fwd(m, i, o):
        store["a"] = o
        o.register_hook(lambda g: store.__setitem__("g", g))

    h = layer.register_forward_hook(fwd)
    try:
        model.eval()
        logits = model(x)
        model.zero_grad()
        logits[torch.arange(len(x)), target].sum().backward()
    finally:
        h.remove()
    w = store["g"].mean((2, 3), keepdim=True)
    cam = F.relu((w * store["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam / (cam.flatten(1).max(1).values.view(-1, 1, 1) + 1e-8)
    return cam.detach().cpu().numpy(), logits.argmax(1).detach().cpu().numpy()


def visualize_run(run_dir: Path, out_root: Path | None = None, cache: str | None = None, actmax_steps=60,
                  wandb: bool | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_run(run_dir, device, cache)
    out = (out_root or ROOT / "results") / run_dir.name / "viz"
    out.mkdir(parents=True, exist_ok=True)
    data = load_cache(cfg)
    sel = data["split"] == "test"
    xs, ys = data["x"][sel], data["y"][sel]

    plot_conv_weights(model.first_conv(), out / "filters_first_conv.png", "Conv đầu tiên")
    plot_conv_weights(model.last_conv(), out / "filters_last_conv.png", "Conv cuối (trung bình theo kênh vào)")

    layer = model.cam_layer()
    am = activation_maximization(model, layer, steps=actmax_steps, device=device)
    _grid(list(am), [f"filter {i}" for i in range(len(am))], out / "actmax_last_conv.png",
          "Activation maximization – conv cuối", cmap="viridis")

    # một ảnh mỗi lớp -> 16 feature map đầu tiên của conv cuối
    rng = np.random.default_rng(0)
    picks = [rng.choice(np.flatnonzero(ys == c)) for c in range(7) if (ys == c).any()]
    x = torch.from_numpy(xs[picks]).float().div(255).unsqueeze(1).to(device)
    with torch.no_grad():
        fmap = capture(model, layer, x).float().cpu().numpy()
    k = min(16, fmap.shape[1])
    fig, axes = plt.subplots(len(picks), k + 1, figsize=((k + 1) * 0.9, len(picks) * 1.0))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(picks):
        axes[r, 0].imshow(xs[i], cmap="gray"), axes[r, 0].set_title(CLASSES[ys[i]], fontsize=6)
        for c in range(k):
            axes[r, c + 1].imshow(fmap[r, c], cmap="viridis")
    for a in axes.ravel():
        a.axis("off")
    fig.suptitle(f"Feature map conv cuối ({fmap.shape[2]}x{fmap.shape[3]})"), fig.tight_layout()
    fig.savefig(out / "featuremaps_last_conv.png", dpi=130), plt.close(fig)

    # Grad-CAM: mỗi lớp lấy 2 ảnh, so sánh với dự đoán
    picks = [i for c in range(7) for i in rng.permutation(np.flatnonzero(ys == c))[:2]]
    x = torch.from_numpy(xs[picks]).float().div(255).unsqueeze(1).to(device)
    pred = model(x).argmax(1)
    cam, _ = grad_cam(model, layer, x, pred)
    pred = pred.cpu().numpy()
    fig, axes = plt.subplots(2, len(picks), figsize=(len(picks) * 1.2, 2.8))
    axes = np.atleast_2d(axes).reshape(2, -1)
    for j, i in enumerate(picks):
        axes[0, j].imshow(xs[i], cmap="gray")
        axes[0, j].set_title(f"T:{CLASSES[ys[i]][:3]}\nP:{CLASSES[pred[j]][:3]}", fontsize=6,
                             color="green" if ys[i] == pred[j] else "red")
        axes[1, j].imshow(xs[i], cmap="gray"), axes[1, j].imshow(cam[j], cmap="jet", alpha=0.45)
    for a in axes.ravel():
        a.axis("off")
    fig.suptitle("Grad-CAM theo lớp dự đoán"), fig.tight_layout()
    fig.savefig(out / "gradcam.png", dpi=130), plt.close(fig)
    print(f"Đã lưu hình trực quan hóa ở {out}")
    wb = WandbLogger(cfg, run_dir, job_type="visualize", force=wandb)
    wb.log_images(out, "viz")
    wb.finish()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--actmax_steps", type=int, default=60)
    ap.add_argument("--no_wandb", action="store_true", help="không ghi lên W&B dù lúc train đã bật")
    a = ap.parse_args(argv)
    for r in a.run:
        visualize_run(Path(r), Path(a.out_dir) if a.out_dir else None, a.data, a.actmax_steps,
                      False if a.no_wandb else None)


if __name__ == "__main__":
    main()
