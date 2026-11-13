import json
from types import SimpleNamespace

from rag.followup import answer_with_followup, is_refusal, merge, parse_query
from rag.retriever import Hit

REFUSAL = "I don't know based on the provided documents."


class FakeGenerator:
    """Returns queued answers; its chat client returns the queued follow-up reply."""

    def __init__(self, answers, client=None):
        self.answers, self.client, self.model, self.seen = list(answers), client, "m", []

    def answer(self, question, hits):
        self.seen.append(list(hits))
        return self.answers.pop(0)


class FakeRetriever:
    def __init__(self, hits):
        self.hits, self.queries = hits, []

    def search(self, query, k):
        self.queries.append((query, k))
        return self.hits

    def add_neighbours(self, hits):
        return hits


def hit(source, page=1):
    return Hit(f"text of {source} p{page}", source, page, 1.0)


def test_is_refusal_only_matches_a_refusal_at_the_start():
    assert is_refusal(REFUSAL) and is_refusal("  I don’t know based on the provided documents.")
    assert not is_refusal("The batch size is 16 [1]. I don't know the learning rate.")


def test_parse_query_reads_json_and_treats_everything_else_as_no_query():
    assert parse_query('Sure: {"query": " Med-PaLM adaptation "}') == "Med-PaLM adaptation"
    assert parse_query('{"query": null}') is None and parse_query("no json") is None
    assert parse_query('{"query": ""}') is None and parse_query(None) is None


def test_no_follow_up_when_the_first_answer_is_not_a_refusal():
    gen, retriever = FakeGenerator(["Adam [1]."]), FakeRetriever([])
    assert answer_with_followup("q", [hit("a.pdf")], retriever, gen, 3) == ("Adam [1].", [hit("a.pdf")], None)
    assert retriever.queries == []


def test_refusal_triggers_one_search_and_a_second_answer(fake_client):
    client = fake_client([json.dumps({"query": "Med-PaLM adaptation"})])
    gen = FakeGenerator([REFUSAL, "Instruction prompt tuning [2]."], client)
    retriever = FakeRetriever([hit("b.pdf", 3)])
    answer, hits, query = answer_with_followup("q", [hit("a.pdf")], retriever, gen, 3)
    assert answer == "Instruction prompt tuning [2]." and query == "Med-PaLM adaptation"
    assert [h.source for h in hits] == ["a.pdf", "b.pdf"] and retriever.queries == [("Med-PaLM adaptation", 3)]
    assert gen.seen[1] == hits  # the second answer read the old and the new passages


def test_no_lead_keeps_the_refusal(fake_client):
    gen = FakeGenerator([REFUSAL], fake_client([json.dumps({"query": None})]))
    retriever = FakeRetriever([hit("b.pdf")])
    assert answer_with_followup("q", [hit("a.pdf")], retriever, gen, 3) == (REFUSAL, [hit("a.pdf")], None)


def test_a_failing_follow_up_call_keeps_the_refusal(fake_client):
    gen = FakeGenerator([REFUSAL], fake_client([RuntimeError("boom")]))
    assert answer_with_followup("q", [hit("a.pdf")], FakeRetriever([]), gen, 3)[0] == REFUSAL


def test_follow_up_can_be_switched_off():
    gen = FakeGenerator([REFUSAL], SimpleNamespace())
    assert answer_with_followup("q", [hit("a.pdf")], FakeRetriever([]), gen, 3, enabled=False)[2] is None


def test_merge_skips_passages_that_are_already_there():
    assert [h.source for h in merge([hit("a.pdf")], [hit("a.pdf"), hit("b.pdf")])] == ["a.pdf", "b.pdf"]
