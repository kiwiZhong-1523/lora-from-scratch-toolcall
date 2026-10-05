"""Prompt format, output parsing and metrics for the function-calling task.

Canonical data schema (one JSON object per line, see data/sample.jsonl):

    {
      "tools":  [ {"name": str, "description": str, "parameters": {...JSON schema...}}, ... ],
      "query":  "user request in natural language",
      "answer": {"name": "tool_name", "arguments": {"arg": value, ...}}
    }

The model must reply with ONLY the JSON object  {"name": ..., "arguments": {...}}.
Keeping one canonical schema means any public dataset can be converted into it
by a small script (scripts/convert_*.py), and train/eval code never changes.
"""

from __future__ import annotations

import json
import re
from typing import Any

SYSTEM_PROMPT = (
    "You are a function-calling assistant. Given the available tools and a user "
    "request, reply with ONLY one JSON object of the form "
    '{"name": <tool name>, "arguments": {<argument name>: <value>, ...}}. '
    "No explanation, no markdown."
)


def build_messages(tools: list[dict], query: str) -> list[dict]:
    """Chat messages for the prompt part (without the assistant answer)."""
    tools_json = json.dumps(tools, ensure_ascii=False)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Tools:\n{tools_json}\n\nRequest: {query}"},
    ]


def answer_to_text(answer: dict) -> str:
    """Target string the model is trained to produce."""
    return json.dumps(answer, ensure_ascii=False, sort_keys=True)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_call(text: str) -> dict | None:
    """Extract the first JSON object from model output; None if there is none.

    Tolerates a markdown fence and trailing chatter, because we want to measure
    the *content* separately from the *strict format* (see `strict_format`)."""
    text = _FENCE.sub("", text.strip())
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def strict_format(text: str) -> bool:
    """True iff the whole output is exactly one JSON object (nothing else)."""
    try:
        return isinstance(json.loads(text.strip()), dict)
    except json.JSONDecodeError:
        return False


def _norm(v: Any) -> Any:
    """Normalise values so 1 == 1.0 and key order does not matter, while keeping
    booleans distinct from numbers (in Python, True == 1 is True!).

    Every value is tagged with its kind, so a bool can never equal a number."""
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, (int, float)):
        return ("num", float(v))
    if isinstance(v, dict):
        return ("dict", tuple(sorted((k, _norm(x)) for k, x in v.items())))
    if isinstance(v, list):
        return ("list", tuple(_norm(x) for x in v))
    return ("other", v)


def score_one(pred_text: str, gold: dict) -> dict[str, bool]:
    pred = parse_call(pred_text)
    json_valid = pred is not None
    schema_valid = (
        json_valid and isinstance(pred.get("name"), str) and isinstance(pred.get("arguments"), dict)
    )
    name_ok = schema_valid and pred["name"] == gold["name"]
    args_ok = schema_valid and _norm(pred["arguments"]) == _norm(gold["arguments"])
    return {
        "strict_format": strict_format(pred_text),
        "json_valid": json_valid,
        "schema_valid": bool(schema_valid),
        "name_correct": bool(name_ok),
        "args_exact": bool(args_ok),
        "full_exact": bool(name_ok and args_ok),
    }


def evaluate_predictions(pred_texts: list[str], golds: list[dict]) -> dict[str, float]:
    assert len(pred_texts) == len(golds)
    rows = [score_one(p, g) for p, g in zip(pred_texts, golds)]
    n = max(len(rows), 1)
    out = {k: sum(r[k] for r in rows) / n for k in rows[0]} if rows else {}
    out["n"] = len(rows)
    return out
