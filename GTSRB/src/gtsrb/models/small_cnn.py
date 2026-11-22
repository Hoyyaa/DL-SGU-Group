import torch.nn as nn


def conv_block(cin, cout):
    """Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2 (giảm một nửa kích thước ảnh)."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """CNN nhỏ làm baseline (E1): 3 khối conv, 99.019 tham số.

    Vào (B,3,64,64) đã chuẩn hóa -> 32x32 -> 16x16 -> 8x8 -> GAP -> Linear(128, 43).
    """

    def __init__(self, num_classes: int = 43):
        super().__init__()
        self.features = nn.Sequential(conv_block(3, 32), conv_block(32, 64), conv_block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.head(self.features(x))
