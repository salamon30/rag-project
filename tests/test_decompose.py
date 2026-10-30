import json

import pytest

from rag.decompose import DECOMPOSE_PROMPT, Decomposer, interleave, parse_queries
from rag.retriever import Hit


def test_parse_reads_queries():
    text = 'Sure: {"queries": ["CheXNet dataset size", "MedSAM dataset size"]}'
    assert parse_queries(text, "q", 3) == ["CheXNet dataset size", "MedSAM dataset size"]


@pytest.mark.parametrize("text", [None, "", "no json", '{"queries": []}', '{"queries": "one string"}',
                                  '{"other": ["x"]}', '{"queries": [1, "  "]}', "{broken json"])
def test_parse_falls_back_to_original_question(text):
    assert parse_queries(text, "original?", 3) == ["original?"]


def test_parse_caps_number_of_queries():
    assert parse_queries(json.dumps({"queries": list("abcde")}), "q", 3) == ["a", "b", "c"]


def test_prompt_examples_are_not_from_the_test_sets():
    # examples copied from the eval questions would leak answers into the score
    from eval.run_eval import load_tests
    corpus_terms = ["chexnet", "u-net", "medsam", "med-palm", "medprompt", "medrag", "mirage", "lewis"]
    examples = DECOMPOSE_PROMPT.split("Examples:")[1].split("Question: {question}")[0].lower()
    assert not any(term in examples for term in corpus_terms)
    for t in load_tests("testset") + load_tests("testset_hard"):
        assert t["question"].lower() not in examples


def hits(src, n):
    return [Hit(f"{src}{i}", f"{src}.pdf", i, 1.0) for i in range(n)]


def test_interleave_takes_turns():
    merged = interleave([hits("a", 5), hits("b", 5)], 5)
    assert [h.text for h in merged] == ["a0", "b0", "a1", "b1", "a2"]


def test_interleave_drops_duplicates_and_fills_up():
    shared = Hit("same", "a.pdf", 1, 1.0)
    merged = interleave([[shared, *hits("a", 3)], [shared, *hits("b", 1)]], 5)
    assert [h.text for h in merged] == ["same", "a0", "b0", "a1", "a2"]


def test_interleave_handles_empty_lists():
    assert interleave([], 5) == [] and interleave([[], []], 5) == []


def test_split_uses_llm_without_thinking(fake_client):
    from types import SimpleNamespace
    client = fake_client(['{"queries": ["A size", "B size"]}'])
    d = Decomposer(SimpleNamespace(client=client, model="m"))
    assert d.split("Is A bigger than B?") == ["A size", "B size"]
    call = client.calls[0]
    assert call["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert "Is A bigger than B?" in call["messages"][0]["content"]


def test_split_survives_api_error(fake_client):
    from types import SimpleNamespace
    d = Decomposer(SimpleNamespace(client=fake_client([RuntimeError("503")]), model="m"))
    assert d.split("q?") == ["q?"]
