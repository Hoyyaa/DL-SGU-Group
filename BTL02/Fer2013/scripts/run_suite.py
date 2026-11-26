import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train.trainer import run_experiment  # noqa: E402
from src.train.utils import deep_update, load_config, resolve_path  # noqa: E402


def expand(path: Path, overrides: List[str]) -> List[Dict]:
    """Một file YAML → danh sách config. File sweep có ``sweep: true, base, variants``."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not raw.get("sweep"):
        cfg = load_config(path, overrides)
        cfg["wandb_group"] = cfg.get("wandb_group") or path.parent.name  # baseline | sota
        return [cfg]
    base = load_config(None, [])
    base = deep_update(base, raw.get("base", {}))
    cfgs = []
    for exp_id, variant in raw["variants"].items():
        cfg = deep_update(base, variant or {})
        cfg["exp_id"] = exp_id
        cfg["wandb_group"] = cfg.get("wandb_group") or path.stem  # vd: abl_loss → so sánh trong 1 nhóm
        for item in overrides:
            k, _, v = item.partition("=")
            node = cfg
            *parents, leaf = k.split(".")
            for p in parents:
                node = node.setdefault(p, {})
            node[leaf] = yaml.safe_load(v)
        cfgs.append(cfg)
    return cfgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+", help="File/thư mục YAML, theo sau là key=value tùy chọn")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    overrides = [t for t in args.targets if "=" in t]
    paths: List[Path] = []
    for t in (t for t in args.targets if "=" not in t):
        p = resolve_path(t)
        paths += sorted(p.rglob("*.yaml")) if p.is_dir() else [p]

    cfgs = [c for p in paths for c in expand(p, overrides)]
    print(f"{len(cfgs)} thí nghiệm:")
    for c in cfgs:
        done = (resolve_path(c.get("output_dir", "outputs/runs")) / c["exp_id"] / "metrics.json").exists()
        print(f"  - {c['exp_id']:<32} model={c['model']:<16} {'(đã xong)' if done else ''}")
    if args.dry_run:
        return

    failed = []
    for c in cfgs:
        if not args.force and (resolve_path(c.get("output_dir", "outputs/runs")) / c["exp_id"] / "metrics.json").exists():
            continue
        print(f"\n========== {c['exp_id']} ==========")
        try:
            run_experiment(c)
        except Exception:  # một run lỗi không làm dừng cả suite
            traceback.print_exc()
            failed.append(c["exp_id"])
    print("\nHoàn tất." + (f" Lỗi: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
