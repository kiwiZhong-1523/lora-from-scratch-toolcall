"""Greedy generation on the test set + metrics, shared by scripts/eval.py and
scripts/train.py (which evaluates right after training, in the same process)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from src.data import prompt_text
from src.toolcall import evaluate_predictions


@torch.no_grad()
def generate(model, tok, rows: list[dict], batch_size: int = 32, max_new_tokens: int = 256) -> list[str]:
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = next(model.parameters()).device
    prompts = [prompt_text(tok, r) for r in rows]
    # Sort by length so each batch has little padding; restore order at the end.
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    preds: list[str | None] = [None] * len(prompts)
    # A full-FT model is kept in fp32; generate under bf16 autocast for speed.
    use_autocast = device.type == "cuda" and next(model.parameters()).dtype == torch.float32
    was_training = model.training
    model.eval()
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        batch = tok([prompts[j] for j in idx], return_tensors="pt", padding=True).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
            out = model.generate(
                **batch, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id
            )
        texts = tok.batch_decode(out[:, batch["input_ids"].shape[1] :], skip_special_tokens=True)
        for j, t in zip(idx, texts):
            preds[j] = t
        print(f"  eval {min(i + batch_size, len(order))}/{len(order)}", flush=True)
    model.train(was_training)
    return preds  # type: ignore[return-value]


def run_eval(model, tok, rows: list[dict], out_path: str, batch_size: int = 32,
             max_new_tokens: int = 256, extra: dict | None = None) -> dict:
    """Writes <out_path> (metrics) and <out_path>.preds.jsonl (every prediction)."""
    start = time.time()
    preds = generate(model, tok, rows, batch_size, max_new_tokens)
    seconds = time.time() - start
    metrics = evaluate_predictions(preds, [r["answer"] for r in rows])
    result = {**(extra or {}), "metrics": metrics, "eval_seconds": round(seconds, 1)}

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".preds.jsonl"), "w", encoding="utf-8") as f:
        for r, p in zip(rows, preds):
            rec = {"id": r.get("id"), "query": r["query"], "gold": r["answer"], "pred": p}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # Metrics file last: its existence marks the run as complete (see run_all.sh).
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics
