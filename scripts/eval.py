"""Evaluate a base model (or base + LoRA adapter) on the function-calling test set.

Examples (run from the repo root):
    python scripts/eval.py --data data/sample.jsonl --limit 3 --out results/smoke.json
    python scripts/eval.py --data data/test.jsonl --out results/zeroshot.json
    python scripts/eval.py --data data/test.jsonl --lora runs/r8/adapter.pt --out results/r8.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from src.lora import load_lora, merge_lora  # noqa: E402
from src.toolcall import build_messages, evaluate_predictions  # noqa: E402


def load_jsonl(path: str, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


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
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", required=True, help="jsonl in the canonical schema")
    ap.add_argument("--lora", default=None, help="adapter.pt saved by save_lora")
    ap.add_argument("--out", default="results/eval.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pick_dtype(args.dtype, device)
    print(f"device={device} dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    if args.lora:
        cfg = load_lora(model, args.lora)
        merge_lora(model)
        print(f"loaded adapter {args.lora}: {cfg}")

    rows = load_jsonl(args.data, args.limit)
    prompts = [
        tok.apply_chat_template(build_messages(r["tools"], r["query"]), tokenize=False, add_generation_prompt=True)
        for r in rows
    ]

    preds: list[str] = []
    start = time.time()
    for i in range(0, len(prompts), args.batch_size):
        batch = tok(prompts[i : i + args.batch_size], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        gen = out[:, batch["input_ids"].shape[1] :]
        preds.extend(tok.batch_decode(gen, skip_special_tokens=True))
        print(f"  {min(i + args.batch_size, len(prompts))}/{len(prompts)}", flush=True)
    seconds = time.time() - start

    metrics = evaluate_predictions(preds, [r["answer"] for r in rows])
    result = {"args": vars(args), "metrics": metrics, "seconds": round(seconds, 1)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    preds_path = out_path.with_suffix(".preds.jsonl")
    with open(preds_path, "w", encoding="utf-8") as f:
        for r, p in zip(rows, preds):
            f.write(json.dumps({"query": r["query"], "gold": r["answer"], "pred": p}, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, indent=2))
    print(f"saved {out_path} and {preds_path}")


if __name__ == "__main__":
    main()
