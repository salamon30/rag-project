import json
from types import SimpleNamespace

import pytest

from eval.judge import Judge, _parse

GOOD = {"faithfulness": 5, "faithfulness_reason": "ok", "correctness": 4, "correctness_reason": "ok"}


def test_parse_finds_json_inside_extra_text():
    assert _parse("Sure! " + json.dumps(GOOD) + " Done.")["correctness"] == 4


def test_parse_converts_string_scores():
    assert _parse(json.dumps({**GOOD, "faithfulness": "3"}))["faithfulness"] == 3


@pytest.mark.parametrize("bad", ["no json here", json.dumps({**GOOD, "correctness": 7})])
def test_parse_rejects_bad_output(bad):
    with pytest.raises(ValueError):
        _parse(bad)


def judge(client):
    return Judge(SimpleNamespace(client=client, model="m"))


def test_score_retries_then_succeeds_with_thinking_off(fake_client):
    client = fake_client([(None, "length"), json.dumps(GOOD)])
    assert judge(client).score("q", "ref", "ctx", "ans")["faithfulness"] == 5
    assert all(c["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False for c in client.calls)
    assert client.calls[0]["temperature"] == 0


def test_score_gives_none_after_all_retries_fail(fake_client):
    out = judge(fake_client(["nope", "nope", "nope"])).score("q", "ref", "ctx", "ans")
    assert out["faithfulness"] is None and "judge error" in out["faithfulness_reason"]
