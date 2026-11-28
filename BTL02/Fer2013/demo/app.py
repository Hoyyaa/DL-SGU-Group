"""Demo web nhận diện cảm xúc + Grad-CAM.

    streamlit run demo/app.py -- --checkpoint outputs/runs/EXP05_large_cnn_bn/best.pt
"""

import argparse
import sys
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference import EmotionPredictor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default=str(ROOT / "outputs/runs/EXP05_large_cnn_bn/best.pt"))
args, _ = ap.parse_known_args()


@st.cache_resource
def load(path: str) -> EmotionPredictor:
    return EmotionPredictor(path)


st.title("BTL02 – Nhận diện cảm xúc khuôn mặt (FER2013)")
ckpt = st.text_input("Checkpoint", args.checkpoint)
if not Path(ckpt).exists():
    st.warning("Chưa có checkpoint – hãy train trước (xem README).")
    st.stop()
predictor = load(ckpt)

src = st.file_uploader("Ảnh khuôn mặt (đã crop sát mặt)", type=["png", "jpg", "jpeg"]) or st.camera_input("Hoặc chụp webcam")
if src:
    img = Image.open(src).convert("L")
    probs = predictor.predict(img)
    heat = predictor.explain(img)
    face = np.asarray(img.resize(heat.shape[::-1]), dtype=np.float32) / 255.0
    overlay = 0.6 * np.stack([face] * 3, -1) + 0.4 * cm.jet(heat)[..., :3]

    c1, c2 = st.columns(2)
    c1.image(img, caption="Ảnh vào", use_container_width=True)
    c2.image(overlay.clip(0, 1), caption="Grad-CAM", use_container_width=True)
    best = max(probs, key=probs.get)
    st.subheader(f"Dự đoán: {best} ({probs[best]:.1%})")
    st.bar_chart(probs)
