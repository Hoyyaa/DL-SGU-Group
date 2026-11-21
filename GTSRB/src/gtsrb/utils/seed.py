import random

import numpy as np


def set_seed(seed: int) -> None:
    """Cố định seed để chạy lại ra kết quả gần như giống nhau."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # máy chưa cài torch vẫn dùng được các script dữ liệu
        pass
