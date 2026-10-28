import numpy as np

from ingest import config
from rag.retriever import Retriever, _enabled


class FakeEmbedder:
    def __init__(self):
        self.seen = []

    def encode(self, texts, normalize_embeddings=True):
        self.seen += texts
        return np.zeros((len(texts), 3))


class FakeCollection:
    """Returns n chunks ordered by vector similarity: c1 is closest."""

    def __init__(self, total=10):
        self.total = total

    def query(self, query_embeddings, n_results):
        n = min(n_results, self.total)
        return {"documents": [[f"c{i}" for i in range(1, n + 1)]],
                "metadatas": [[{"source": f"s{i}.pdf", "page": i} for i in range(1, n + 1)]],
                "distances": [[0.1 * i for i in range(1, n + 1)]]}


class FakeReranker:
    """Prefers chunks deeper in the vector ranking, to prove the order really changes."""

    def __init__(self):
        self.pairs = None

    def predict(self, pairs, show_progress_bar=False):
        self.pairs = pairs
        return [float(text[1:]) for _, text in pairs]


def make(reranker=None, total=10):
    r = Retriever.__new__(Retriever)  # skip __init__: no model download, no DB
    r.model, r.collection, r.reranker = FakeEmbedder(), FakeCollection(total), reranker
    r.max_per_source = 0
    return r


def test_query_gets_bge_prefix():
    r = make()
    r.vector_search("what is RAG?", 3)
    assert r.model.seen == [config.QUERY_PREFIX + "what is RAG?"]


def test_vector_search_scores_and_ranks():
    hits = make().vector_search("q", 3)
    assert [h.vector_rank for h in hits] == [1, 2, 3]
    assert hits[0].score == hits[0].vector_score == 0.9
    assert hits[0].page == 1 and hits[0].source == "s1.pdf"


def test_without_reranker_returns_vector_top_k():
    hits = make().search("q", k=3)
    assert [h.text for h in hits] == ["c1", "c2", "c3"]


def test_reranker_reorders_candidates_and_keeps_vector_rank():
    rr = FakeReranker()
    hits = make(rr).search("q", k=3, candidates=8)
    assert len(rr.pairs) == 8 and rr.pairs[0] == ("q", "c1")
    assert [h.text for h in hits] == ["c8", "c7", "c6"]
    assert hits[0].vector_rank == 8 and hits[0].score == 8.0
    assert hits[0].vector_score is not None


def test_candidate_pool_is_at_least_k():
    rr = FakeReranker()
    make(rr, total=20).search("q", k=12, candidates=5)
    assert len(rr.pairs) == 12


def test_rerank_can_be_disabled_by_name():
    assert not _enabled("none") and not _enabled("NONE") and not _enabled("") and not _enabled(None)
    assert _enabled("cross-encoder/ms-marco-MiniLM-L-6-v2")


def test_diversify_caps_chunks_per_paper():
    from rag.retriever import Hit, diversify
    ranked = [Hit(f"a{i}", "a.pdf", i, 1.0) for i in range(5)] + [Hit("b0", "b.pdf", 1, 0.5)]
    assert [h.text for h in diversify(ranked, 3, 0)] == ["a0", "a1", "a2"]          # no cap
    assert [h.text for h in diversify(ranked, 3, 2)] == ["a0", "a1", "b0"]          # b.pdf gets a slot


def test_diversify_fills_up_when_only_one_paper():
    from rag.retriever import Hit, diversify
    ranked = [Hit(f"a{i}", "a.pdf", i, 1.0) for i in range(5)]
    assert [h.text for h in diversify(ranked, 4, 2)] == ["a0", "a1", "a2", "a3"]    # still k hits, order kept


def test_search_applies_cap_after_reranking():
    r = make(FakeReranker())
    r.max_per_source = 1
    hits = r.search("q", k=3, candidates=8)
    assert len({h.source for h in hits}) == 3
