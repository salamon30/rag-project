import json
import math
from types import SimpleNamespace

from eval.judge import Judge
from eval.rejudge import agreement, as_test, rejudge, replay_context
from rag.retriever import Hit

GOOD = {"faithfulness": 4, "faithfulness_reason": "ok", "correctness": 3, "correctness_reason": "partly"}


class FakeRetriever:
    """Returns fixed hits and records the queries it was asked to search."""

    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def search_many(self, queries, k, question=None):
        self.calls.append((queries, question))
        return self.hits


def hit(source, page):
    return Hit(text=f"text of {source} p{page}", source=source, page=page, score=1.0)


def row(**over):
    base = {"id": "q1", "type": "single", "question": "What is X?", "queries": "What is X?",
            "retrieved": "a.pdf:p1; b.pdf:p2", "answer": "X is 4 [1].", "reference_answer": "X is 4.",
            "faithfulness": 5, "correctness": 5}
    return {**base, **over}


def test_judge_uses_the_given_model_without_the_thinking_switch(fake_client):
    client = fake_client([json.dumps(GOOD)])
    judge = Judge(SimpleNamespace(client=client, model="nvidia/nemotron-x"), model="other/model")
    judge.score("q", "ref", "ctx", "ans")
    assert client.calls[0]["model"] == "other/model"
    assert "extra_body" not in client.calls[0]


def test_replay_uses_saved_sub_queries_and_detects_same_pages():
    retriever = FakeRetriever([hit("a.pdf", 1), hit("b.pdf", 2)])
    context, same = replay_context(retriever, row(queries="first part || second part"), k=5)
    assert retriever.calls[0] == (["first part", "second part"], "What is X?")
    assert same and "text of b.pdf p2" in context


def test_replay_reports_when_pages_changed():
    _, same = replay_context(FakeRetriever([hit("a.pdf", 9)]), row(), k=5)
    assert not same


def test_rejudge_keeps_both_scores_and_drops_faithfulness_if_context_differs(fake_client):
    judge = Judge(SimpleNamespace(client=fake_client([json.dumps(GOOD)] * 2), model="m"))
    same = rejudge([row()], FakeRetriever([hit("a.pdf", 1), hit("b.pdf", 2)]), judge, k=5)[0]
    assert (same["correctness_first"], same["correctness_second"]) == (5, 3)
    assert same["faithfulness_second"] == 4
    differs = rejudge([row()], FakeRetriever([hit("a.pdf", 9)]), judge, k=5)[0]
    assert differs["context_matches"] == 0 and differs["faithfulness_second"] is None


def test_rejudge_applies_refusal_rules_to_the_second_judge(fake_client):
    lenient = json.dumps({**GOOD, "correctness": 5})
    judge = Judge(SimpleNamespace(client=fake_client([lenient]), model="m"))
    refusal = row(answer="I don't know based on the provided documents.", correctness=1)
    out = rejudge([refusal], FakeRetriever([hit("a.pdf", 1), hit("b.pdf", 2)]), judge, k=5)[0]
    assert out["correctness_second"] == 1  # an answerable question that was refused


def test_rejudge_skips_rows_that_were_never_scored(fake_client):
    judge = Judge(SimpleNamespace(client=fake_client([]), model="m"))
    assert rejudge([row(correctness=None)], FakeRetriever([]), judge, k=5) == []


def test_as_test_marks_only_unanswerable_rows_as_unanswerable():
    assert as_test(row(type="unanswerable"))["expected_pages"] == {}
    assert as_test(row())["expected_pages"]


def test_agreement_counts_exact_and_within_one():
    rows = [{"correctness_first": 5, "correctness_second": 5},
            {"correctness_first": 5, "correctness_second": 4},
            {"correctness_first": 5, "correctness_second": 2},
            {"correctness_first": 5, "correctness_second": None}]
    a = agreement(rows)
    assert a["n"] == 3 and a["first"] == 5 and math.isclose(a["second"], 11 / 3)
    assert math.isclose(a["exact"], 1 / 3) and math.isclose(a["within_one"], 2 / 3)


def test_agreement_handles_no_pairs():
    assert agreement([{"correctness_first": None, "correctness_second": 3}])["n"] == 0


def test_rejudge_reuses_earlier_scores_and_only_repeats_failed_ones(fake_client):
    client = fake_client([json.dumps(GOOD)])  # one reply: only the failed row may call the judge
    judge = Judge(SimpleNamespace(client=client, model="m"))
    keep = {"q1": {"id": "q1", "correctness_second": 4}, "q2": {"id": "q2", "correctness_second": None}}
    out = rejudge([row(), row(id="q2")], FakeRetriever([hit("a.pdf", 1), hit("b.pdf", 2)]), judge, k=5, keep=keep)
    assert out[0] == keep["q1"] and out[1]["correctness_second"] == 3 and len(client.calls) == 1
