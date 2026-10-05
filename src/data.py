"""Dataset conversion and tokenisation shared by convert / train / eval scripts.

Training target = the canonical answer text followed by the chat template's
end-of-turn token, so the model also learns *when to stop*. Only those tokens
carry a loss; the prompt tokens get label -100.
"""

from __future__ import annotations

import json

import torch

from src.toolcall import answer_to_text, build_messages

IGNORE = -100


def load_jsonl(path: str, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def xlam_to_canonical(raw: dict) -> dict | None:
    """One xLAM record -> canonical schema, or None if it is not a single call.

    xLAM stores `tools` and `answers` as JSON strings; `parameters` uses its own
    {"arg": {"type": "str", ...}} style instead of JSON Schema. We keep it as is:
    to the model it is just text."""
    answers = json.loads(raw["answers"])
    if len(answers) != 1:
        return None
    return {"id": raw["id"], "tools": json.loads(raw["tools"]), "query": raw["query"], "answer": answers[0]}


def prompt_text(tok, row: dict) -> str:
    return tok.apply_chat_template(
        build_messages(row["tools"], row["query"]), tokenize=False, add_generation_prompt=True
    )


def encode_example(tok, row: dict, max_len: int | None = None) -> dict | None:
    """Token ids + labels (prompt masked). None if longer than `max_len`."""
    prompt = prompt_text(tok, row)
    messages = build_messages(row["tools"], row["query"])
    messages.append({"role": "assistant", "content": answer_to_text(row["answer"])})
    full = tok.apply_chat_template(messages, tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("chat template: prompt is not a prefix of the full conversation")
    # The prompt ends with "<|im_start|>assistant\n", a clean token boundary, so
    # tokenising the two halves separately equals tokenising the whole string.
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(full[len(prompt) :], add_special_tokens=False)["input_ids"]
    ids = p_ids + a_ids
    if max_len and len(ids) > max_len:
        return None
    return {"input_ids": ids, "labels": [IGNORE] * len(p_ids) + a_ids}


def collate(examples: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad a list of encoded examples into a batch."""
    n = max(len(e["input_ids"]) for e in examples)
    ids = torch.full((len(examples), n), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), n), IGNORE, dtype=torch.long)
    mask = torch.zeros((len(examples), n), dtype=torch.long)
    for i, e in enumerate(examples):
        k = len(e["input_ids"])
        ids[i, :k] = torch.tensor(e["input_ids"])
        labels[i, :k] = torch.tensor(e["labels"])
        mask[i, :k] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}
