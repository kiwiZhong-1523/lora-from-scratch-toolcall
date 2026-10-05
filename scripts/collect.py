"""Collect every finished run into results/summary.csv and copy its small files
(no weights, no predictions) to results/<run>/ for committing to git.

    python scripts/collect.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

SMALL_FILES = ["config.json", "train_log.jsonl", "val_log.jsonl", "train_summary.json", "eval.json"]
METRICS = ["full_exact", "name_correct", "args_exact", "strict_format", "json_valid", "schema_valid"]
COLUMNS = ["run", "exp", "mode", "targets", "r", "alpha", "lr", "seed", "trainable_params", "trainable_pct",
           "train_minutes", "peak_mem_gb", "adapter_mb", "final_val_loss", *METRICS, "n_test", "eval_minutes"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def row_for(run_dir: Path) -> dict:
    cfg, summ, ev = (read_json(run_dir / f) for f in ("config.json", "train_summary.json", "eval.json"))
    a = cfg.get("args", {})
    trained = "trainable_params" in cfg
    is_lora = a.get("mode") in ("lora", "peft")
    row = {
        "run": run_dir.name,
        "exp": run_dir.name.split("_")[0],
        "mode": a.get("mode", "zeroshot") if trained else "zeroshot",
        "targets": a.get("targets") if is_lora else "",
        "r": a.get("r") if is_lora else "",
        "alpha": a.get("alpha") if is_lora else "",
        "lr": a.get("lr", ""),
        "seed": a.get("seed", ""),
        "trainable_params": cfg.get("trainable_params", ""),
        "trainable_pct": round(100 * cfg["trainable_params"] / cfg["total_params"], 4) if trained else "",
        "train_minutes": summ.get("train_minutes", ""),
        "peak_mem_gb": summ.get("peak_mem_gb", ""),
        "adapter_mb": summ.get("adapter_mb", ""),
        "final_val_loss": summ.get("final_val_loss", ""),
        "n_test": ev["metrics"].get("n"),
        "eval_minutes": round(ev.get("eval_seconds", 0) / 60, 1),
    }
    row.update({m: round(ev["metrics"][m], 4) for m in METRICS})
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = Path(args.out)
    rows = []
    for run_dir in sorted(Path(args.runs).iterdir()):
        if not (run_dir / "eval.json").exists():  # unfinished or smoke-only
            continue
        rows.append(row_for(run_dir))
        dst = out / run_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in SMALL_FILES:
            if (run_dir / f).exists():
                shutil.copy2(run_dir / f, dst / f)

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} runs -> {out / 'summary.csv'}")
    for r in rows:
        print(f"  {r['run']:<24} full_exact={r['full_exact']:.4f}  strict={r['strict_format']:.4f}  "
              f"val_loss={r['final_val_loss']}")


if __name__ == "__main__":
    main()
