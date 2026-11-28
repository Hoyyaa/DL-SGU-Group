"""Grad-CAM (Selvaraju et al., 2017): L^c = ReLU(sum_k alpha_k^c A^k), alpha_k^c = GAP(dy^c/dA^k)."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Dùng: ``with GradCAM(model, model.get_target_layer()) as cam: heat = cam(x)``."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        # Hook trên tensor (không dùng full_backward_hook) để tương thích ReLU(inplace=True)
        # đứng sau tầng mục tiêu trong VGG/ResNet torchvision.
        self._handles = [target_layer.register_forward_hook(self._save_activation)]

    def _save_activation(self, _m, _inp, out):
        self.activations = out.detach().clone()
        if out.requires_grad:
            out.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def __call__(self, x: torch.Tensor, class_idx: Optional[torch.Tensor] = None):
        """x: [B, C, H, W]. Trả về (heatmap [B, H, W] trong [0,1], logits)."""
        was_training = self.model.training
        self.model.eval()
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1)
        self.model.zero_grad(set_to_none=True)
        logits.gather(1, class_idx.view(-1, 1)).sum().backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        flat = cam.flatten(1)
        mn, mx = flat.min(1)[0].view(-1, 1, 1), flat.max(1)[0].view(-1, 1, 1)
        cam = (cam - mn) / (mx - mn + 1e-8)
        self.model.train(was_training)
        return cam.detach(), logits.detach()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
