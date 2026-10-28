import json

import pytest

from eval.run_eval import MRR_DEPTH, _is_refusal, load_tests, mean, retrieval_metrics, summarize
from rag.retriever import Hit


@pytest.mark.parametrize("answer", [
    "I don't know based on the provided documents.",
    "The provided documents do not contain this information.",
    "There is no specific information about dosage in the context.",
])
def test_refusals_are_detected(answer):
    assert _is_refusal(answer)


@pytest.mark.parametrize("answer", [
    "The confidence interval does not include zero [2].",   # regression: was flagged as a refusal
    "U-Net uses skip connections [1].",
    "The context does not mention X, but it says Y [3].",   # cited, so it answers
])
def test_answers_are_not_refusals(answer):
    assert not _is_refusal(answer)


def test_mean_ignores_none():
    assert mean([1, None, 3]) == 2
    assert mean([None]) != mean([None])  # NaN


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    def search(self, question, k):
        return self.results[question][:k]


def test_retrieval_metrics():
    tests = [
        {"id": "q1", "question": "a", "expected_source": "x.pdf", "expected_pages": [3]},
        {"id": "q2", "question": "b", "expected_source": "x.pdf", "expected_pages": [9]},
        {"id": "q3", "question": "c", "expected_source": "x.pdf", "expected_pages": [1]},
    ]
    results = {
        "a": [Hit("", "x.pdf", 3, 1.0)],                                  # rank 1, right page
        "b": [Hit("", "y.pdf", 1, 1.0), Hit("", "x.pdf", 2, 0.9)],        # rank 2, wrong page
        "c": [Hit("", "y.pdf", 1, 1.0)] * (MRR_DEPTH + 1),                # never found
    }
    m, misses = retrieval_metrics(FakeRetriever(results), tests, k=5)
    assert m["hit@1"] == pytest.approx(1 / 3)
    assert m["hit@5"] == pytest.approx(2 / 3)
    assert m["page_hit@5"] == pytest.approx(1 / 3)
    assert m[f"MRR@{MRR_DEPTH}"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert misses == ["q2", "q3"]


def test_summarize_counts_refusals_and_citations():
    rows = [
        {"hit@1": 1, "hit@5": 1, "page_hit@5": 1, "rr": 1.0, "answer": "Yes [1].", "faithfulness": 5, "correctness": 5},
        {"hit@1": 0, "hit@5": 1, "page_hit@5": 0, "rr": 0.5, "answer": "No citation here.",
         "faithfulness": 4, "correctness": 3},
        {"hit@1": 0, "hit@5": 0, "page_hit@5": 0, "rr": 0.0,
         "answer": "I don't know based on the provided documents.", "faithfulness": None, "correctness": None},
    ]
    s = summarize(rows, k=5)
    assert s["refusals"] == 1
    assert s["answers with [n] citations"] == 0.5  # refusals are left out
    assert s["faithfulness (1-5)"] == 4.5
    assert s["unscored (errors)"] == 1


def test_testset_is_well_formed():
    tests = load_tests()
    assert len(tests) >= 20
    assert len({t["id"] for t in tests}) == len(tests)
    for t in tests:
        assert t["question"] and t["reference_answer"] and t["expected_source"].endswith(".pdf")
        assert t["expected_pages"] and all(isinstance(p, int) for p in t["expected_pages"])
    json.dumps(tests)
