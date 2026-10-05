import json

from src.toolcall import (
    answer_to_text,
    build_messages,
    evaluate_predictions,
    parse_call,
    score_one,
    strict_format,
)

GOLD = {"name": "get_weather", "arguments": {"city": "Tokyo", "days": 3}}


def test_parse_plain_and_fenced_and_chatty():
    plain = json.dumps(GOLD)
    assert parse_call(plain) == GOLD
    assert parse_call(f"```json\n{plain}\n```") == GOLD
    assert parse_call(f"Sure! {plain} Hope that helps.") == GOLD


def test_parse_failures():
    assert parse_call("no json here") is None
    assert parse_call('{"name": "x", "arguments": {') is None
    assert parse_call("[1, 2, 3]") is None


def test_strict_format_rejects_extra_text():
    plain = json.dumps(GOLD)
    assert strict_format(plain)
    assert not strict_format(f"```json\n{plain}\n```")
    assert not strict_format(f"Sure! {plain}")


def test_score_perfect_and_wrong_name_and_wrong_args():
    perfect = score_one(json.dumps(GOLD), GOLD)
    assert all(perfect.values())

    wrong_name = score_one(json.dumps({"name": "get_time", "arguments": GOLD["arguments"]}), GOLD)
    assert wrong_name["json_valid"] and not wrong_name["name_correct"] and not wrong_name["full_exact"]

    wrong_args = score_one(json.dumps({"name": "get_weather", "arguments": {"city": "Osaka", "days": 3}}), GOLD)
    assert wrong_args["name_correct"] and not wrong_args["args_exact"] and not wrong_args["full_exact"]


def test_number_normalisation_and_key_order():
    pred = '{"arguments": {"days": 3.0, "city": "Tokyo"}, "name": "get_weather"}'
    assert score_one(pred, GOLD)["full_exact"]


def test_bool_not_confused_with_number():
    gold = {"name": "f", "arguments": {"flag": True}}
    assert not score_one('{"name": "f", "arguments": {"flag": 1}}', gold)["args_exact"]


def test_schema_invalid_when_arguments_missing():
    s = score_one('{"name": "get_weather"}', GOLD)
    assert s["json_valid"] and not s["schema_valid"] and not s["full_exact"]


def test_evaluate_predictions_aggregates():
    preds = [json.dumps(GOLD), "garbage"]
    res = evaluate_predictions(preds, [GOLD, GOLD])
    assert res["n"] == 2
    assert res["full_exact"] == 0.5
    assert res["json_valid"] == 0.5


def test_prompt_and_target_are_deterministic():
    msgs = build_messages([{"name": "get_weather"}], "weather in Tokyo?")
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    assert "weather in Tokyo?" in msgs[1]["content"]
    assert answer_to_text({"b": 1, "a": 2}) == '{"a": 2, "b": 1}'
