"""Stage B, step 1: fetch the most relevant chunks for a question.

Stage 1 (bi-encoder): question and chunks are embedded separately and compared
by cosine similarity. Fast enough to scan the whole collection.
Stage 2 (cross-encoder reranker, optional): reads each (question, chunk) pair
together and scores how well the chunk answers the question. More accurate but
slower, so it only re-scores the stage-1 candidates.
"""
from dataclasses import dataclass

from sentence_transformers import CrossEncoder, SentenceTransformer

from ingest import config
from ingest.store import get_collection


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


class Retriever:
    def __init__(self, model_name: str = config.EMBED_MODEL, rerank_model: str | None = config.RERANK_MODEL):
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
                zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), start=1)
        ]

    def search(self, question: str, k: int = config.TOP_K,
               candidates: int = config.RERANK_CANDIDATES) -> list[Hit]:
        if not self.reranker:
            return self.vector_search(question, k)
        hits = self.vector_search(question, max(candidates, k))
        scores = self.reranker.predict([(question, h.text) for h in hits], show_progress_bar=False)
        for h, s in zip(hits, scores):
            h.score = float(s)
        return sorted(hits, key=lambda h: h.score, reverse=True)[:k]
