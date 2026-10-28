import json
import re
from pathlib import Path

import pytest

from eval.run_eval import (
    MRR_DEPTH,
    _declines,
    _is_refusal,
    apply_refusal_rules,
    by_type,
    load_tests,
    mean,
    rank_metrics,
    retrieval_metrics,
    summarize,
)
from rag.retriever import Hit

ROOT = Path(__file__).resolve().parent.parent


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
        {"id": "q1", "type": "single", "question": "a", "expected_pages": {"x.pdf": [3]}},
        {"id": "q2", "type": "single", "question": "b", "expected_pages": {"x.pdf": [9]}},
        {"id": "q3", "type": "single", "question": "c", "expected_pages": {"x.pdf": [1]}},
        {"id": "q4", "type": "unanswerable", "question": "d", "expected_pages": {}},  # skipped
    ]
    results = {
        "a": [Hit("", "x.pdf", 3, 1.0)],                                  # rank 1, right page
        "b": [Hit("", "y.pdf", 1, 1.0), Hit("", "x.pdf", 2, 0.9)],        # rank 2, wrong page
        "c": [Hit("", "y.pdf", 1, 1.0)] * (MRR_DEPTH + 1),                # never found
        "d": [Hit("", "x.pdf", 3, 1.0)],
    }
    m, misses = retrieval_metrics(FakeRetriever(results), tests, k=5)
    assert m["hit@1"] == pytest.approx(1 / 3)
    assert m["hit@5"] == pytest.approx(2 / 3)
    assert m["page_hit@5"] == pytest.approx(1 / 3)
    assert m[f"MRR@{MRR_DEPTH}"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert misses == ["q2", "q3"]


def test_all_sources_needs_every_paper_in_top_k():
    hits = [Hit("", "a.pdf", 1, 1.0), Hit("", "a.pdf", 2, 0.9), Hit("", "b.pdf", 5, 0.8)]
    expected = {"a.pdf": [1], "b.pdf": [5]}
    assert rank_metrics(hits, expected, k=3)["all_sources@3"] == 1
    assert rank_metrics(hits, expected, k=2)["all_sources@2"] == 0   # b.pdf fell out of the top 2
    assert rank_metrics(hits, expected, k=2)["hit@2"] == 1           # but one paper is enough for hit@k


def row(typ, answer, said, f=5, c=5, hit=1):
    hit = None if typ == "unanswerable" else hit
    return {"type": typ, "hit@1": hit, "hit@5": hit, "all_sources@5": hit, "page_hit@5": hit,
            "rr": None if hit is None else float(hit), "answer": answer, "said_dont_know": said,
            "faithfulness": f, "correctness": c}


def test_summarize_separates_answerable_and_unanswerable():
    rows = [
        row("single", "Yes [1].", 0),
        row("multi_hop", "No citation here.", 0, f=4, c=3, hit=0),
        row("single", "I don't know based on the provided documents.", 1, f=None, c=None),
        row("unanswerable", "The documents do not state the cost.", 1),
        row("unanswerable", "It cost $5,000 [1].", 0, c=1),
    ]
    s = summarize(rows, k=5)
    assert s["questions"] == 5
    assert s["hit@5"] == pytest.approx(2 / 3)                 # unanswerable rows are left out
    assert s["all_sources@5"] == pytest.approx(2 / 3)
    assert s["refusals"] == 1                                  # false refusals on answerable only
    assert s["refusal recall (unanswerable)"] == 0.5
    assert s["answers with [n] citations"] == 0.5
    assert s["unscored (errors)"] == 1
    t = by_type(rows, k=5)
    assert t["unanswerable"]["n"] == 2 and t["unanswerable"]["dont_know"] == 1


def test_original_testset_has_no_new_metrics():
    s = summarize([row("single", "Yes [1].", 0)], k=5)
    assert "all_sources@5" not in s and "refusal recall (unanswerable)" not in s


@pytest.mark.parametrize("answer", [
    "The documents do not state how much it cost.",
    "The paper reports 20 A100 GPUs [2], but the monetary cost is not reported.",
    "I don't know based on the provided documents.",
])
def test_declines_accepts_cited_partial_answers(answer):
    assert _declines(answer)


def test_declines_rejects_made_up_answer():
    assert not _declines("Training MedSAM cost about $40,000 [1].")


def judged():
    return {"faithfulness": 5, "correctness": 5, "faithfulness_reason": "", "correctness_reason": ""}


def test_refusal_rules_unanswerable():
    t = {"type": "unanswerable"}
    j = judged()
    assert apply_refusal_rules(t, "It cost $40,000 [1].", j) is False and j["correctness"] == 1
    j = judged()
    assert apply_refusal_rules(t, "The cost is not reported [2].", j) is True and j["correctness"] == 5


def test_refusal_rules_answerable():
    j = judged()
    assert apply_refusal_rules({"type": "single"}, "I don't know based on the provided documents.", j)
    assert j["correctness"] == 1
    j = judged()
    assert not apply_refusal_rules({"type": "false_premise"}, "It uses DenseNet, not ResNet [1].", j)
    assert j["correctness"] == 5


def corpus_files():
    script = (ROOT / "scripts" / "download_data.sh").read_text()
    block = script.split("<<'LIST'\n")[1].split("\nLIST")[0]
    return {line.split()[1] + ".pdf" for line in block.splitlines() if line.strip()}


TYPES = {"single", "unanswerable", "false_premise", "multi_hop", "paraphrase"}


@pytest.mark.parametrize("name", ["testset", "testset_hard"])
def test_testsets_are_well_formed(name):
    tests = load_tests(name)
    files = corpus_files()
    assert len(tests) >= 20
    assert len({t["id"] for t in tests}) == len(tests)
    for t in tests:
        assert t["type"] in TYPES, t["id"]
        assert t["question"] and t["reference_answer"], t["id"]
        if t["type"] == "unanswerable":
            assert t["expected_pages"] == {}, t["id"]
        else:
            assert t["expected_pages"], t["id"]
            for src, pages in t["expected_pages"].items():
                assert src in files, f"{t['id']}: {src} is not in the corpus"
                assert pages and all(isinstance(p, int) and p >= 1 for p in pages), t["id"]
        if t["type"] == "multi_hop":
            assert len(t["expected_pages"]) >= 2, t["id"]
    json.dumps(tests)


def test_hard_set_covers_every_new_type():
    types = {t["type"] for t in load_tests("testset_hard")}
    assert types == TYPES - {"single"}
    assert re.fullmatch(r"h\d\d", load_tests("testset_hard")[0]["id"])
