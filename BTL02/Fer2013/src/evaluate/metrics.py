"""Chỉ số phân loại viết thuần PyTorch (không dùng scikit-learn).

Quy ước giống sklearn với ``zero_division=0``: lớp không có mẫu dự đoán/thật → precision/recall = 0.
"""

from typing import Dict, Sequence, Union

import numpy as np
import torch

ArrayLike = Union[torch.Tensor, np.ndarray, Sequence[int]]


def _to_long(x: ArrayLike) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x) if not isinstance(x, torch.Tensor) else x).long().flatten().cpu()


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike, num_classes: int) -> torch.Tensor:
    """Ma trận nhầm lẫn [K, K]: hàng = nhãn thật, cột = nhãn dự đoán."""
    t, p = _to_long(y_true), _to_long(y_pred)
    idx = t * num_classes + p
    return torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def per_class_metrics(cm: torch.Tensor) -> Dict[str, torch.Tensor]:
    """precision, recall, f1, support cho từng lớp từ ma trận nhầm lẫn."""
    cm = cm.double()
    tp = cm.diag()
    support = cm.sum(1)
    predicted = cm.sum(0)
    precision = torch.where(predicted > 0, tp / predicted.clamp(min=1), torch.zeros_like(tp))
    recall = torch.where(support > 0, tp / support.clamp(min=1), torch.zeros_like(tp))
    denom = precision + recall
    f1 = torch.where(denom > 0, 2 * precision * recall / denom.clamp(min=1e-12), torch.zeros_like(tp))
    return {"precision": precision, "recall": recall, "f1": f1, "support": support}


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    t, p = _to_long(y_true), _to_long(y_pred)
    return float((t == p).double().mean()) if len(t) else 0.0


def macro_f1(y_true: ArrayLike, y_pred: ArrayLike, num_classes: int) -> float:
    """Trung bình F1 của các lớp (mỗi lớp trọng số bằng nhau – nhạy với lớp hiếm như Disgust)."""
    return float(per_class_metrics(confusion_matrix(y_true, y_pred, num_classes))["f1"].mean())


def classification_report(y_true: ArrayLike, y_pred: ArrayLike, class_names: Sequence[str]):
    """Bảng precision/recall/F1/support theo lớp + accuracy, macro avg, weighted avg (pandas DataFrame)."""
    import pandas as pd

    k = len(class_names)
    cm = confusion_matrix(y_true, y_pred, k)
    m = per_class_metrics(cm)
    rows = {name: [float(m["precision"][i]), float(m["recall"][i]), float(m["f1"][i]), int(m["support"][i])]
            for i, name in enumerate(class_names)}
    total = float(m["support"].sum())
    w = m["support"] / max(total, 1.0)
    rows["macro avg"] = [float(m["precision"].mean()), float(m["recall"].mean()), float(m["f1"].mean()), int(total)]
    rows["weighted avg"] = [float((m["precision"] * w).sum()), float((m["recall"] * w).sum()),
                            float((m["f1"] * w).sum()), int(total)]
    acc = float(cm.diag().sum()) / max(total, 1.0)
    rows["accuracy"] = [acc, acc, acc, int(total)]
    return pd.DataFrame.from_dict(rows, orient="index", columns=["precision", "recall", "f1-score", "support"])
