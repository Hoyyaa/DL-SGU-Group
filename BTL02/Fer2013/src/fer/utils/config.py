from pathlib import Path

import yaml

# gốc dự án: .../src/fer/utils/config.py -> lên 3 cấp
ROOT = Path(__file__).resolve().parents[3]


def _merge(base: dict, new: dict) -> dict:
    """Gộp new vào base; dict lồng nhau được gộp từng khóa."""
    out = dict(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_overrides(items) -> dict:
    """['train.lr=3e-4', 'name=x'] -> {'train': {'lr': 0.0003}, 'name': 'x'} (giá trị đọc bằng yaml)."""
    out: dict = {}
    for item in items or []:
        key, _, val = item.partition("=")
        node = out
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        v = yaml.safe_load(val)
        if isinstance(v, str):  # PyYAML (YAML 1.1) đọc "3e-4" thành chuỗi
            try:
                v = float(v)
            except ValueError:
                pass
        node[leaf] = v
    return out


def load_config(path, overrides: dict | None = None) -> dict:
    """Đọc file yaml. Nếu có `_base_: base.yaml` thì đọc base trước rồi ghi đè."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base_name = cfg.pop("_base_", None)
    if base_name:
        cfg = _merge(load_config(path.parent / base_name), cfg)
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg
