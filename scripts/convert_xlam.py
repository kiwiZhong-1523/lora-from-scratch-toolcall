"""xLAM-function-calling-60k -> canonical jsonl + fixed train/val/test splits.

    python scripts/convert_xlam.py

First run: keep single-call examples, drop those longer than --max-len tokens,
shuffle with --seed and cut test / val / train / rest. The chosen ids are written
to data/splits/*.ids (committed to git). Later runs (e.g. on the GPU instance)
rebuild the exact same files from those ids instead of re-splitting.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer  # noqa: E402

from src.data import encode_example, write_jsonl, xlam_to_canonical  # noqa: E402

SPLITS = ["test", "val", "train", "rest"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/xlam_function_calling_60k.json")
    ap.add_argument("--tokenizer", default="models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    split_dir = out_dir / "splits"
    raw = json.loads(Path(args.raw).read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in raw}

    if all((split_dir / f"{s}.ids").exists() for s in SPLITS):
        print("rebuilding from existing data/splits/*.ids")
        ids = {s: [int(x) for x in (split_dir / f"{s}.ids").read_text().split()] for s in SPLITS}
        rows = {s: [xlam_to_canonical(by_id[i]) for i in ids[s]] for s in SPLITS}
    else:
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        single = [c for c in map(xlam_to_canonical, raw) if c is not None]
        kept = [c for c in single if encode_example(tok, c, args.max_len) is not None]
        stats = {
            "raw": len(raw),
            "single_call": len(single),
            f"kept_le_{args.max_len}_tokens": len(kept),
            "dropped_too_long": len(single) - len(kept),
            "seed": args.seed,
        }
        random.Random(args.seed).shuffle(kept)
        a, b, c = args.n_test, args.n_test + args.n_val, args.n_test + args.n_val + args.n_train
        rows = {"test": kept[:a], "val": kept[a:b], "train": kept[b:c], "rest": kept[c:]}
        stats.update({f"n_{s}": len(rows[s]) for s in SPLITS})
        split_dir.mkdir(parents=True, exist_ok=True)
        for s in SPLITS:
            (split_dir / f"{s}.ids").write_text("\n".join(str(r["id"]) for r in rows[s]) + "\n")
        (split_dir / "stats.json").write_text(json.dumps(stats, indent=2))
        print(json.dumps(stats, indent=2))

    for s in SPLITS:
        write_jsonl(str(out_dir / f"{s}.jsonl"), rows[s])
        print(f"wrote {out_dir / f'{s}.jsonl'}: {len(rows[s])}")


if __name__ == "__main__":
    main()
