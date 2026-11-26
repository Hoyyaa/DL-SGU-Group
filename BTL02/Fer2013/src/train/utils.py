"""Tiện ích chung: seed, device, đọc config YAML (kế thừa configs/base.yaml)."""

import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Khóa mọi nguồn ngẫu nhiên để tái lập kết quả."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_value(text: str) -> Any:
    return yaml.safe_load(text)


def _set_nested(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(
    path: Optional[Union[str, Path]] = None,
    overrides: Iterable[str] = (),
    base: Union[str, Path] = BASE_CONFIG,
) -> Dict[str, Any]:
    """Đọc ``configs/base.yaml`` rồi ghi đè bằng file thí nghiệm và ``key=value`` từ CLI.

    Ví dụ: ``load_config("experiments/baseline/exp01_small_cnn.yaml", ["lr=0.0005", "aug.rotation=15"])``
    """
    cfg: Dict[str, Any] = {}
    if base and Path(base).exists():
        cfg = yaml.safe_load(Path(base).read_text(encoding="utf-8")) or {}
    if path:
        cfg = deep_update(cfg, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    for item in overrides:
        key, _, value = item.partition("=")
        _set_nested(cfg, key.strip(), _parse_value(value))
    if "exp_id" not in cfg:
        cfg["exp_id"] = Path(path).stem if path else "debug"
    return cfg


def resolve_path(p: Union[str, Path]) -> Path:
    """Đường dẫn tương đối được hiểu theo thư mục gốc dự án."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
