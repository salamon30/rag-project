import numpy as np

from ingest import config
from rag.retriever import Hit, Retriever, _enabled, continuation, next_chunk_ids


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


def test_search_many_with_one_query_is_plain_search():
    r = make()
    assert [h.text for h in r.search_many(["q"], k=3)] == [h.text for h in r.search("q", k=3)]


def test_search_many_runs_one_search_per_query():
    r = make()
    r.search_many(["first", "second"], k=4)
    assert r.model.seen[:2] == [config.QUERY_PREFIX + "first", config.QUERY_PREFIX + "second"]


class TwoPaperCollection(FakeCollection):
    """Query 'about a' finds paper a, anything else finds paper b."""

    def __init__(self):
        super().__init__()
        self.next_source = "b"

    def query(self, query_embeddings, n_results):
        res = super().query(query_embeddings, n_results)
        res["metadatas"] = [[{"source": f"{self.next_source}.pdf", "page": i} for i in range(1, n_results + 1)]]
        return res


def test_search_many_drops_split_when_all_parts_hit_one_paper():
    r = make()  # FakeCollection: every query's top hit is s1.pdf
    r.model.seen.clear()
    r.search_many(["aspect one", "aspect two"], k=4, question="original")
    assert r.model.seen[-1] == config.QUERY_PREFIX + "original"


def test_search_many_keeps_split_across_papers():
    r = make()
    col = TwoPaperCollection()
    r.collection = col
    sources = iter(["a", "b"])
    orig_search = r.search

    def search(q, k, candidates=50):
        col.next_source = next(sources)
        return orig_search(q, k, candidates)

    r.search = search
    hits = r.search_many(["about a", "about b"], k=4, question="original", multi_k=4)
    assert [h.source for h in hits] == ["a.pdf", "b.pdf", "a.pdf", "b.pdf"]
    assert r.last_split


def test_kept_split_reads_more_passages_than_a_single_search():
    r = make()
    col = TwoPaperCollection()
    r.collection = col
    sources = iter(["a", "b"])
    orig_search = r.search

    def search(q, k, candidates=50):
        col.next_source = next(sources)
        return orig_search(q, k, candidates)

    r.search = search
    hits = r.search_many(["about a", "about b"], k=4, question="original", multi_k=6)
    assert len(hits) == 6 and r.context_size(4, multi_k=6) == 6
    r.search = orig_search
    r.search_many(["one question"], k=4)
    assert r.context_size(4, multi_k=6) == 4


def test_next_chunk_ids_point_to_same_page_then_next_page():
    assert next_chunk_ids("a.pdf::p3::c1") == ["a.pdf::p3::c2", "a.pdf::p4::c0"]
    assert next_chunk_ids(None) == [] and next_chunk_ids("broken") == []


def test_continuation_removes_the_overlap():
    assert continuation("We train with Adam", "with Adam and a batch of 16.", 100) == " and a batch of 16."
    assert continuation("no overlap here", "Fresh text.", 5) == "Fresh"


class NeighbourCollection:
    def __init__(self, docs):
        self.docs = docs

    def get(self, ids):
        found = [i for i in ids if i in self.docs]
        return {"ids": found, "documents": [self.docs[i] for i in found]}


def test_add_neighbours_appends_the_following_text():
    r = make()
    r.collection = NeighbourCollection({"a.pdf::p2::c1": "We train with a batch of 16."})
    hits = [Hit("Setup. We train", "a.pdf", 2, 1.0, chunk_id="a.pdf::p2::c0"),
            Hit("Last chunk of the paper", "a.pdf", 9, 1.0, chunk_id="a.pdf::p9::c0"),
            Hit("No id", "b.pdf", 1, 1.0)]
    out = r.add_neighbours(hits, limit=100)
    assert out[0].text == "Setup. We train with a batch of 16. […]"
    assert out[1].text == "Last chunk of the paper" and out[2].text == "No id"


def test_add_neighbours_can_be_switched_off():
    r = make()
    r.collection = NeighbourCollection({"a.pdf::p2::c1": "more"})
    assert r.add_neighbours([Hit("text", "a.pdf", 2, 1.0, chunk_id="a.pdf::p2::c0")], limit=0)[0].text == "text"


def test_last_searches_records_what_really_ran():
    r = make()
    r.search_many(["aspect one", "aspect two"], k=3, question="original")
    assert r.last_searches == ["original"]  # split dropped: one paper
