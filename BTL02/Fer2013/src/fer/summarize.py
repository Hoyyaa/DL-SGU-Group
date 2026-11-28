"""Gom mọi results/*/metrics.json thành bảng so sánh cho báo cáo.

    python -m fer.summarize

Ghi results/summary.csv và results/summary.md (bảng markdown dán thẳng vào báo cáo).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from fer.utils.config import ROOT

COLS = ["exp", "seed", "model", "params", "best_epoch", "val_acc", "test_acc", "val_macro_f1",
        "test_macro_f1", "disgust_f1_test"]


def summarize(results_dir: Path = ROOT / "results") -> pd.DataFrame:
    rows = []
    for f in sorted(Path(results_dir).glob("*/metrics.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        m["disgust_f1_test"] = m["test_report"]["Disgust"]["f1-score"]
        rows.append({c: m.get(c) for c in COLS})
    df = pd.DataFrame(rows, columns=COLS).sort_values("exp")
    df.to_csv(Path(results_dir) / "summary.csv", index=False)
    fmt = df.copy()
    for c in COLS[5:]:
        fmt[c] = fmt[c].map(lambda v: f"{v:.4f}")
    (Path(results_dir) / "summary.md").write_text(fmt.to_markdown(index=False) if _has_tabulate()
                                                  else fmt.to_string(index=False), encoding="utf-8")
    print(f"Tổng hợp {len(df)} run -> {Path(results_dir) / 'summary.csv'}")
    return df


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    summarize(Path(ap.parse_args().results))
