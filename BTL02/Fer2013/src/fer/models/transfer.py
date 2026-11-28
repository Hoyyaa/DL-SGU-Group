"""VGG16 tinh chỉnh từ ImageNet (thí nghiệm "fail case" của báo cáo mẫu).

Ảnh xám 48x48 được phóng lên `resize`, lặp thành 3 kênh rồi chuẩn hóa theo ImageNet.
Ở 48x48, 5 lần MaxPool làm đặc trưng chỉ còn 1x1 nên mô hình học kém: đó là điểm cần phân tích.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VGG16Transfer(nn.Module):
    def __init__(self, num_classes=7, resize=48, pretrained=True, freeze_features=False, drop_fc=0.5, **_):
        super().__init__()
        base = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        for m in base.features:  # bỏ inplace để hook Grad-CAM không lỗi
            if isinstance(m, nn.ReLU):
                m.inplace = False
        self.features = base.features
        if freeze_features:
            for p in self.features.parameters():
                p.requires_grad = False
        self.resize = resize
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(drop_fc),
                                        nn.Linear(256, num_classes))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):  # x: (B,1,48,48) trong [0,1]
        if self.resize != x.shape[-1]:
            x = F.interpolate(x, size=self.resize, mode="bilinear", align_corners=False)
        x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        return self.classifier(self.pool(self.features(x)))

    def first_conv(self):
        return self.features[0]

    def last_conv(self):
        return [m for m in self.features if isinstance(m, nn.Conv2d)][-1]

    def cam_layer(self):
        mods = list(self.features)
        return mods[mods.index(self.last_conv()) + 1]
