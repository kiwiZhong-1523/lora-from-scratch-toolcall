"""Train one run (LoRA via src/lora.py, LoRA via peft, or full fine-tuning), then evaluate it.

    python scripts/train.py --run e1_attn_r8_s42 --targets attn --r 8
    python scripts/train.py --run e5_peft_attn_r8_s42 --mode peft --targets attn --r 8
    python scripts/train.py --run e3_full_s42 --mode full --lr 1e-5 --weight-decay 0.01
    python scripts/train.py --run smoke --max-steps 3 --n-train 64 --n-val 8 --eval-limit 8   # quick check

Writes runs/<run>/: config.json, train_log.jsonl, val_log.jsonl, train_summary.json,
adapter.pt (LoRA only), eval.json, eval.preds.jsonl. eval.json is written last,
so its presence means the run finished.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from src.data import IGNORE, collate, encode_example, load_jsonl  # noqa: E402
from src.evaluation import run_eval  # noqa: E402
from src.lora import ALL_TARGETS, ATTN_TARGETS, MLP_TARGETS, count_parameters, inject_lora, save_lora  # noqa: E402

TARGETS = {"attn": ATTN_TARGETS, "mlp": MLP_TARGETS, "all": ALL_TARGETS}


@torch.no_grad()
def seeded_init_lora_A(model, seed: int) -> int:
    """Re-draw every LoRA A from one dedicated CPU generator, in module order.

    src/lora.py and peft consume the global RNG differently while building the
    adapters, so their A matrices would differ even with the same seed. With this
    both start from bit-identical adapters (B is zero in both), which turns the
    peft cross-check into a strict comparison instead of "within seed noise"."""
    g = torch.Generator().manual_seed(seed)
    n = 0
    for name, p in model.named_parameters():
        if name.endswith(".lora_A") or name.endswith(".lora_A.default.weight"):
            a = torch.empty(p.shape, dtype=torch.float32)
            torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=g)
            p.copy_(a)
            n += 1
    return n


def peft_adapter_state(model) -> dict[str, torch.Tensor]:
    """peft adapter tensors renamed to src/lora.py keys, so scripts/eval.py and
    load_lora can read a peft-trained adapter too."""
    out = {}
    for k, v in model.state_dict().items():
        if "lora_" not in k:
            continue
        k = k.removeprefix("base_model.model.")
        k = k.replace(".lora_A.default.weight", ".lora_A").replace(".lora_B.default.weight", ".lora_B")
        out[k] = v.detach().float().cpu()
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name, output goes to <out-root>/<run>")
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--n-train", type=int, default=None, help="use only the first N training rows")
    ap.add_argument("--n-val", type=int, default=None, help="use only the first N validation rows")
    ap.add_argument("--max-len", type=int, default=1536)

    ap.add_argument("--mode", default="lora", choices=["lora", "peft", "full"])
    ap.add_argument("--targets", default="attn", choices=list(TARGETS))
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=None, help="default: 2 * r")
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=32, help="effective batch (sequences per optimizer step)")
    ap.add_argument("--micro-batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (full FT only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests)")

    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()
    if args.alpha is None:
        args.alpha = 2.0 * args.r
    if args.batch_size % args.micro_batch_size:
        ap.error("--batch-size must be a multiple of --micro-batch-size")
    return args


def env_info() -> dict:
    def git(*cmd: str) -> str:
        try:
            return subprocess.check_output(["git", *cmd], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": git("status", "--porcelain") != "",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "hostname": platform.node(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def answer_loss_sum(model, batch: dict) -> tuple[torch.Tensor, int]:
    """Sum of token cross-entropies over the answer positions only.

    Calls the decoder and lm_head separately so logits are computed just for the
    ~50 answer tokens instead of the whole sequence: with a 151k vocabulary, full
    logits for 8 x 1000 tokens would cost several GB."""
    hidden = model.get_decoder()(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])[0]
    labels = batch["labels"][:, 1:]  # token t+1 is predicted from position t
    sel = labels != IGNORE
    logits = model.get_output_embeddings()(hidden[:, :-1][sel]).float()
    return F.cross_entropy(logits, labels[sel], reduction="sum"), int(sel.sum())


@torch.no_grad()
def val_loss(model, examples: list[dict], pad_id: int, device, autocast: bool, bs: int = 16) -> float:
    model.eval()
    order = sorted(range(len(examples)), key=lambda i: len(examples[i]["input_ids"]))
    total, n_tok = 0.0, 0
    for i in range(0, len(order), bs):
        batch = collate([examples[j] for j in order[i : i + bs]], pad_id)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
            s, n = answer_loss_sum(model, batch)
        total, n_tok = total + s.item(), n_tok + n
    model.train()
    return total / max(n_tok, 1)


def make_batches(examples: list[dict], args, rng: random.Random) -> list[list[list[int]]]:
    """Per optimizer step: a list of micro-batches (lists of example indices).

    Shuffled every epoch; inside one step the examples are sorted by length so
    each micro-batch has little padding (the step's content is unchanged)."""
    steps = []
    n_epochs = math.ceil(args.epochs)
    for _ in range(n_epochs):
        idx = list(range(len(examples)))
        rng.shuffle(idx)
        for i in range(0, len(idx) - args.batch_size + 1, args.batch_size):
            step = sorted(idx[i : i + args.batch_size], key=lambda j: len(examples[j]["input_ids"]))
            steps.append([step[k : k + args.micro_batch_size] for k in range(0, len(step), args.micro_batch_size)])
    n_steps = int(len(steps) * args.epochs / n_epochs)
    return steps[: min(n_steps, args.max_steps or n_steps)]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.out_root) / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # LoRA: frozen base in bf16, adapters in fp32 (see src/lora.py).
    # Full FT: fp32 master weights + bf16 autocast; pure bf16 weights would round
    # away many of the tiny lr=1e-5 updates.
    is_lora = args.mode in ("lora", "peft")
    load_dtype = torch.bfloat16 if (is_lora and use_bf16) else torch.float32
    autocast = args.mode == "full" and use_bf16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=load_dtype).to(device)
    model.config.use_cache = False
    if args.mode == "lora":
        n_wrapped = inject_lora(model, TARGETS[args.targets], r=args.r, alpha=args.alpha, dropout=args.dropout)
    elif args.mode == "peft":
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                         target_modules=TARGETS[args.targets], bias="none")
        # autocast_adapter_dtype keeps the adapters in fp32 on a bf16 base, like src/lora.py.
        model = get_peft_model(model, cfg, autocast_adapter_dtype=True)
        n_wrapped = sum(1 for n, _ in model.named_modules() if n.endswith(".lora_A.default"))
    else:
        n_wrapped = 0
        if args.grad_ckpt:
            model.gradient_checkpointing_enable()
    if is_lora and seeded_init_lora_A(model, args.seed) != n_wrapped:
        raise RuntimeError("seeded A init did not cover every adapter")
    trainable, total = count_parameters(model)
    print(f"trainable {trainable:,} / {total:,} ({100 * trainable / total:.3f}%), wrapped {n_wrapped} layers")

    t0 = time.time()
    train_rows = load_jsonl(args.train, args.n_train)
    train = [e for e in (encode_example(tok, r, args.max_len) for r in train_rows) if e is not None]
    val = [e for e in (encode_example(tok, r, args.max_len) for r in load_jsonl(args.val, args.n_val)) if e is not None]
    print(f"tokenized {len(train)} train / {len(val)} val in {time.time() - t0:.0f}s")

    steps = make_batches(train, args, random.Random(args.seed))
    n_steps = len(steps)
    warmup = max(1, int(args.warmup_ratio * n_steps))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    def lr_lambda(step: int) -> float:  # linear warmup, cosine decay to 0
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, n_steps - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    config = {
        "args": vars(args),
        "n_train": len(train),
        "n_val": len(val),
        "n_steps": n_steps,
        "warmup_steps": warmup,
        "trainable_params": trainable,
        "total_params": total,
        "lora_layers": n_wrapped,
        "load_dtype": str(load_dtype),
        "autocast_bf16": autocast,
        "env": env_info(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    train_log = open(run_dir / "train_log.jsonl", "w", encoding="utf-8")
    val_log = open(run_dir / "val_log.jsonl", "w", encoding="utf-8")

    def log_val(step: int) -> float:
        v = val_loss(model, val, tok.pad_token_id, device, autocast)
        val_log.write(json.dumps({"step": step, "val_loss": round(v, 5)}) + "\n")
        val_log.flush()
        print(f"step {step:5d}  val_loss {v:.4f}", flush=True)
        return v

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    log_val(0)
    start = time.time()
    win_loss, win_tok, win_seq_tok, win_t = 0.0, 0, 0, time.time()

    for step, micro_batches in enumerate(steps, 1):
        # Normalise by the answer tokens of the whole step, so every token has
        # the same weight no matter how it is split into micro-batches.
        n_ans = sum(sum(l != IGNORE for l in train[j]["labels"][1:]) for mb in micro_batches for j in mb)
        step_loss = 0.0
        for mb in micro_batches:
            batch = collate([train[j] for j in mb], tok.pad_token_id)
            win_seq_tok += int(batch["attention_mask"].sum())
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
                s, _ = answer_loss_sum(model, batch)
            (s / n_ans).backward()
            step_loss += s.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm).item()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        win_loss, win_tok = win_loss + step_loss, win_tok + n_ans

        if step % args.log_every == 0 or step == n_steps:
            dt = time.time() - win_t
            rec = {
                "step": step,
                "loss": round(win_loss / win_tok, 5),
                "lr": sched.get_last_lr()[0],
                "grad_norm": round(grad_norm, 4),
                "tok_per_s": round(win_seq_tok / dt),
                "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else 0,
                "elapsed_min": round((time.time() - start) / 60, 2),
            }
            train_log.write(json.dumps(rec) + "\n")
            train_log.flush()
            eta = (time.time() - start) / step * (n_steps - step) / 60
            print(f"step {step:5d}/{n_steps}  loss {rec['loss']:.4f}  gnorm {rec['grad_norm']:.3f}  "
                  f"{rec['tok_per_s']} tok/s  eta {eta:.1f}min", flush=True)
            win_loss, win_tok, win_seq_tok, win_t = 0.0, 0, 0, time.time()
        if step % args.val_every == 0 and step != n_steps:
            log_val(step)

    final_val = log_val(n_steps)
    train_minutes = (time.time() - start) / 60
    train_log.close()
    val_log.close()

    summary = {
        "train_minutes": round(train_minutes, 2),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else 0,
        "final_val_loss": round(final_val, 5),
    }
    if is_lora:
        adapter = run_dir / "adapter.pt"
        lora_cfg = {"target_modules": TARGETS[args.targets], "r": args.r, "alpha": args.alpha,
                    "dropout": args.dropout}
        if args.mode == "lora":
            save_lora(model, str(adapter), lora_cfg)
        else:
            torch.save({"config": lora_cfg, "state": peft_adapter_state(model)}, adapter)
        summary["adapter_mb"] = round(os.path.getsize(adapter) / 2**20, 2)
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))

    if not args.no_eval:
        # Evaluate the unmerged model: merging into bf16 weights would add rounding.
        model.config.use_cache = True
        test = load_jsonl(args.test, args.eval_limit)
        run_eval(model, tok, test, str(run_dir / "eval.json"), args.eval_batch_size, args.max_new_tokens,
                 extra={"run": args.run, "model": args.model})


if __name__ == "__main__":
    main()
