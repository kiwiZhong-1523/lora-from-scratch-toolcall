#!/usr/bin/env bash
# Experiments 0-5 on a single GPU, one after another.
#
#   nohup bash scripts/run_all.sh > run_all.log 2>&1 &
#
# A run whose runs/<name>/eval.json exists is skipped, so after an interruption
# just launch the same command again. RUNS_DIR can point to persistent storage.
set -u
cd "$(dirname "$0")/.."
RUNS_DIR="${RUNS_DIR:-runs}"
M05=models/Qwen2.5-0.5B-Instruct
M15=models/Qwen2.5-1.5B-Instruct

run() {  # run <name> <script> [args...]
  local name=$1 script=$2; shift 2
  if [ -f "$RUNS_DIR/$name/eval.json" ]; then echo "== skip $name (done)"; return; fi
  echo "== $(date '+%F %T') start $name"
  mkdir -p "$RUNS_DIR/$name"
  if python "$script" --run "$name" --out-root "$RUNS_DIR" "$@" > "$RUNS_DIR/$name/stdout.log" 2>&1; then
    echo "== $(date '+%F %T') done  $name"
  else
    echo "== $(date '+%F %T') FAIL  $name (see $RUNS_DIR/$name/stdout.log)"
  fi
  python scripts/collect.py --runs "$RUNS_DIR" > /dev/null
}

# --- Experiment 0: zero-shot baselines
run e0_0.5b scripts/eval.py --model $M05
run e0_1.5b scripts/eval.py --model $M15

# --- Experiment 1: rank sweep on attention, alpha = 2r. r=8 first: it is the
# reference point of experiments 1 and 2 and tells us the real time per run.
run e1_attn_r8_s42 scripts/train.py --targets attn --r 8 --seed 42
for r in 1 2 4 16 32 64; do
  run e1_attn_r${r}_s42 scripts/train.py --targets attn --r $r --seed 42
done
run e1_attn_r8_s43 scripts/train.py --targets attn --r 8 --seed 43
run e1_attn_r8_s44 scripts/train.py --targets attn --r 8 --seed 44

# --- Experiment 2: where to put LoRA (attn_r8 / attn_r32 reused from exp 1)
run e2_mlp_r8_s42 scripts/train.py --targets mlp --r 8 --seed 42
run e2_all_r8_s42 scripts/train.py --targets all --r 8 --seed 42

# --- Experiment 3: full fine-tuning reference
run e3_full_s42 scripts/train.py --mode full --lr 1e-5 --weight-decay 0.01 \
  --micro-batch-size 4 --grad-ckpt --seed 42

# --- Experiment 4: learning-rate sweep, attn r=8 (lr=2e-4 reused from exp 1)
for lr in 5e-5 1e-4 5e-4 1e-3; do
  run e4_attn_r8_lr${lr}_s42 scripts/train.py --targets attn --r 8 --lr $lr --seed 42
done

# --- Experiment 5: same config trained with peft; compare with e1_attn_r8_s42.
# Same seed + same A init -> curves should overlap far tighter than the seed spread.
run e5_peft_attn_r8_s42 scripts/train.py --mode peft --targets attn --r 8 --seed 42

python scripts/collect.py --runs "$RUNS_DIR"
echo "== $(date '+%F %T') all done"
