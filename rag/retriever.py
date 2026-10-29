"""Stage B, step 1: fetch the most relevant chunks for a question.

Stage 1 (bi-encoder): question and chunks are embedded separately and compared
by cosine similarity. Fast enough to scan the whole collection.
Stage 2 (cross-encoder reranker, optional): reads each (question, chunk) pair
together and scores how well the chunk answers the question. More accurate but
slower, so it only re-scores the stage-1 candidates.
"""
from collections import Counter
from dataclasses import dataclass

from sentence_transformers import CrossEncoder, SentenceTransformer

from ingest import config
from ingest.store import get_collection

from .decompose import interleave


@dataclass
class Hit:
    text: str
    source: str
    page: int
    score: float                       # final ranking score (rerank score if reranked)
    vector_score: float | None = None  # cosine similarity from stage 1
    vector_rank: int | None = None     # 1-based rank before reranking


def _enabled(name) -> bool:
    return bool(name) and name.lower() != "none"


def diversify(ranked: list[Hit], k: int, max_per_source: int) -> list[Hit]:
    """Take the top k in order, but at most max_per_source chunks from one paper.

    Skipped chunks are used to fill up only if there are not enough other papers,
    so the result still has k hits. max_per_source=0 means no cap."""
    if max_per_source <= 0:
        return ranked[:k]
    picked, skipped, per_source = [], [], Counter()
    for h in ranked:
        if per_source[h.source] < max_per_source:
            picked.append(h)
            per_source[h.source] += 1
        else:
            skipped.append(h)
        if len(picked) == k:
            return picked
    return picked + skipped[: k - len(picked)]


class Retriever:
    def __init__(self, model_name: str = config.EMBED_MODEL, rerank_model: str | None = config.RERANK_MODEL,
                 max_per_source: int = config.MAX_PER_SOURCE):
        self.max_per_source = max_per_source
        self.model = SentenceTransformer(model_name)
        self.collection = get_collection(config.CHROMA_DIR, config.COLLECTION)
        if self.collection.count() == 0:
            raise RuntimeError("Vector store is empty. Run `python -m ingest.run` first.")
        self.rerank_model = rerank_model if _enabled(rerank_model) else None
        self.reranker = CrossEncoder(self.rerank_model, max_length=512) if self.rerank_model else None

    def vector_search(self, question: str, n: int) -> list[Hit]:
        q = self.model.encode([config.QUERY_PREFIX + question], normalize_embeddings=True).tolist()
        res = self.collection.query(query_embeddings=q, n_results=n)
        return [
            Hit(text=doc, source=md["source"], page=md["page"], score=1 - dist,
                vector_score=1 - dist, vector_rank=i)
            for i, (doc, md, dist) in enumerate(
                zip(res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True), start=1)
        ]

    def search(self, question: str, k: int = config.TOP_K,
               candidates: int = config.RERANK_CANDIDATES) -> list[Hit]:
        if not self.reranker:
            pool = k if not self.max_per_source else max(candidates, k)
            return diversify(self.vector_search(question, pool), k, self.max_per_source)
        hits = self.vector_search(question, max(candidates, k))
        scores = self.reranker.predict([(question, h.text) for h in hits], show_progress_bar=False)
        for h, s in zip(hits, scores, strict=True):
            h.score = float(s)
        return diversify(sorted(hits, key=lambda h: h.score, reverse=True), k, self.max_per_source)

    def search_many(self, queries: list[str], k: int = config.TOP_K,
                    candidates: int = config.RERANK_CANDIDATES, question: str | None = None) -> list[Hit]:
        """Search once per sub-query and interleave, so each item gets a share of the top k.

        A split is kept only if the sub-queries' top hits come from at least two papers.
        Otherwise it was a split into aspects of one paper, which only takes slots away
        from the best passages, so the original question is searched instead."""
        if len(queries) == 1:
            return self.search(queries[0], k, candidates)
        results = [self.search(q, k, candidates) for q in queries]
        if len({r[0].source for r in results if r}) < 2:
            return self.search(question or queries[0], k, candidates)
        return interleave(results, k)
