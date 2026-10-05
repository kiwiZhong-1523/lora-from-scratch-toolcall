# lora-toolcall

Hand-written LoRA (plain PyTorch, no `peft`) + ablations on a 0.5B model for
function calling, followed by GGUF quantization and edge benchmarks.

Status: Day 0 (LoRA module, tests, eval harness). Results table coming.

## Layout
- `src/lora.py`      from-scratch LoRA: inject / merge / unmerge / save / load
- `src/toolcall.py`  prompt format, output parsing, metrics
- `scripts/eval.py`  evaluate base or LoRA-adapted model
- `tests/`           unit tests (`python -m pytest -q`)
- `data/sample.jsonl` 3-example sample of the canonical data schema
