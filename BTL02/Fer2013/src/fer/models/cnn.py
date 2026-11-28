"""CNN kiểu VGG cấu hình được: mỗi block = nhiều Conv3x3 -> MaxPool(2) -> Dropout.

Các preset trong models/__init__.py lấy đúng kiến trúc 4 mô hình của báo cáo mẫu
(Small / Medium / Medium01 / Large), dịch từ Keras sang PyTorch.
"""
import torch
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class Normalize(nn.Module):
    """(x - mean) / std với mean, std của tập train (ảnh xám nên chỉ là 2 số)."""

    def __init__(self, mean: float, std: float):
        super().__init__()
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std", torch.tensor(float(std)))

    def forward(self, x):
        return (x - self.mean) / self.std


class FerCNN(nn.Module):
    def __init__(self, blocks, fc, num_classes=7, in_channels=1, img_size=48, activation="relu",
                 batchnorm=False, drop_conv=0.2, drop_fc=0.5, mean=0.5, std=0.25):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation '{activation}' không hỗ trợ. Có: {sorted(ACTIVATIONS)}")
        act = ACTIVATIONS[activation]
        self._act_type = type(act())
        self.norm = Normalize(mean, std)

        layers, c = [], in_channels
        for chans in blocks:
            for co in chans:
                layers.append(nn.Conv2d(c, co, 3, padding=1, bias=not batchnorm))
                if batchnorm:
                    layers.append(nn.BatchNorm2d(co))
                layers.append(act())
                c = co
            layers += [nn.MaxPool2d(2), nn.Dropout(drop_conv)]
        self.features = nn.Sequential(*layers)

        side = img_size // 2 ** len(blocks)  # 48 -> 24 -> 12 -> 6 -> 3
        head, d = [nn.Flatten()], c * side * side
        for h in fc:
            head += [nn.Linear(d, h), act(), nn.Dropout(drop_fc)]
            d = h
        head.append(nn.Linear(d, num_classes))
        self.classifier = nn.Sequential(*head)
        self._init_weights(activation)

    def _init_weights(self, activation: str):
        """Init mặc định của PyTorch (kaiming_uniform a=√5) làm tín hiệu nhỏ đi ~6 lần mỗi lớp ReLU:
        mạng 8–16 conv không BN có gradient ~1e-8, bị eps của Adam nuốt nên kẹt ở 25% (luôn đoán Happy).
        Kaiming giữ phương sai qua ReLU; sigmoid/tanh dùng Xavier. Bias = 0 như Keras."""
        relu_like = activation in ("relu", "leaky_relu", "elu", "gelu")
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if relu_like:
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):  # x: (B,1,48,48) trong [0,1] -> logits (B,7)
        return self.classifier(self.features(self.norm(x)))

    # --- dùng cho trực quan hóa ---
    def first_conv(self) -> nn.Conv2d:
        return next(m for m in self.features if isinstance(m, nn.Conv2d))

    def last_conv(self) -> nn.Conv2d:
        return [m for m in self.features if isinstance(m, nn.Conv2d)][-1]

    def cam_layer(self) -> nn.Module:
        """Lớp kích hoạt ngay sau conv cuối (sau BN nếu có): dùng cho feature map và Grad-CAM."""
        mods = list(self.features)
        i = mods.index(self.last_conv()) + 1
        while not isinstance(mods[i], self._act_type):
            i += 1
        return mods[i]
