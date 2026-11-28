"""EmotionPredictor – suy luận 1 ảnh khuôn mặt (dùng cho demo/app.py)."""

from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from PIL import Image

from src import EMOTION_LABELS
from src.data.transforms import build_transforms
from src.evaluate.gradcam import GradCAM
from src.models import build_model


class EmotionPredictor:
    def __init__(self, checkpoint: Union[str, Path], device: str = "cpu") -> None:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        self.cfg = {**ckpt["cfg"], "pretrained": False}
        self.device = torch.device(device)
        self.model = build_model(self.cfg)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        self.tf = build_transforms(False, self.cfg.get("input_mode", "gray"), int(self.cfg.get("image_size", 48)))

    def preprocess(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """Ảnh bất kỳ → mức xám 48x48 → tensor theo đúng pipeline lúc train.
        (Giả định ảnh đã được crop sát khuôn mặt như FER2013.)"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("L").resize((48, 48))
        return self.tf(image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image) -> Dict[str, float]:
        probs = torch.softmax(self.model(self.preprocess(image)), 1)[0].cpu().numpy()
        return {name: float(p) for name, p in zip(EMOTION_LABELS, probs)}

    def explain(self, image) -> np.ndarray:
        """Heatmap Grad-CAM [H, W] trong [0, 1] cho lớp dự đoán."""
        with GradCAM(self.model, self.model.get_target_layer()) as cam:
            heat, _ = cam(self.preprocess(image))
        return heat[0].cpu().numpy()
