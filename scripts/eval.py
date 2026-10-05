"""Evaluate a base model (or base + LoRA adapter) on the function-calling test set.

Examples (run from the repo root):
    python scripts/eval.py --run smoke --data data/sample.jsonl --limit 3
    python scripts/eval.py --run e0_0.5b --model models/Qwen2.5-0.5B-Instruct
    python scripts/eval.py --run e0_1.5b --model models/Qwen2.5-1.5B-Instruct
    python scripts/eval.py --run e1_attn_r8_s42_merged --lora runs/e1_attn_r8_s42/adapter.pt

Writes runs/<run>/eval.json (+ eval.preds.jsonl, config.json).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from src.data import load_jsonl  # noqa: E402
from src.evaluation import run_eval  # noqa: E402
from src.lora import load_lora, merge_lora  # noqa: E402


def pick_dtype(name: str, device: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="output goes to <out-root>/<run>/eval.json")
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default="data/test.jsonl", help="jsonl in the canonical schema")
    ap.add_argument("--lora", default=None, help="adapter.pt saved by save_lora")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pick_dtype(args.dtype, device)
    print(f"device={device} dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    if args.lora:
        cfg = load_lora(model, args.lora)
        merge_lora(model)
        print(f"loaded adapter {args.lora}: {cfg}")

    run_dir = Path(args.out_root) / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name() if device == "cuda" else "cpu"
    config = {"args": vars(args), "mode": "eval_only", "dtype": str(dtype), "env": {"gpu": gpu}}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows = load_jsonl(args.data, args.limit)
    run_eval(model, tok, rows, str(run_dir / "eval.json"), args.batch_size, args.max_new_tokens,
             extra={"run": args.run, "model": args.model, "lora": args.lora})


if __name__ == "__main__":
    main()
