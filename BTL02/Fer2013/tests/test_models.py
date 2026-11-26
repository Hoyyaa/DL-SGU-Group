import pytest
import torch

from src.evaluate.gradcam import GradCAM
from src.models import build_model
from src.models.cnn import CNN_SPECS
from src.train.losses import build_loss


@pytest.mark.parametrize("name", list(CNN_SPECS))
@pytest.mark.parametrize("bn,gp", [(False, False), (True, True)])
def test_custom_cnn_forward_and_gradcam(name, bn, gp):
    model = build_model({"model": name, "batch_norm": bn, "global_pool": gp})
    x = torch.randn(2, 1, 48, 48)
    assert model(x).shape == (2, 7)
    with GradCAM(model, model.get_target_layer()) as cam:
        heat, _ = cam(x)
    assert heat.shape == (2, 48, 48) and heat.min() >= 0 and heat.max() <= 1 + 1e-6


@pytest.mark.parametrize("name", ["vgg16", "resnet50", "efficientnet_b0"])
def test_pretrained_head_and_gradcam(name):
    # pretrained=False: không tải trọng số trong test
    model = build_model({"model": name, "pretrained": False, "input_mode": "imagenet"})
    x = torch.randn(1, 3, 64, 64)
    assert model(x).shape == (1, 7)
    with GradCAM(model, model.get_target_layer()) as cam:
        heat, _ = cam(x)
    assert heat.shape == (1, 64, 64)


def test_freeze_backbone_only_head_trainable():
    model = build_model({"model": "resnet50", "pretrained": False, "freeze_backbone": True})
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("net.fc") for n in trainable)


@pytest.mark.parametrize("loss", ["ce", "weighted_ce", "focal", "weighted_focal"])
def test_losses(loss):
    crit = build_loss(loss, torch.ones(7))
    logits = torch.randn(4, 7, requires_grad=True)
    out = crit(logits, torch.tensor([0, 1, 2, 3]))
    out.backward()
    assert out.ndim == 0 and torch.isfinite(out)
