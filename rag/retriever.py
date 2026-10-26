"""Stage B, step 1: embed the question and fetch the top-k most similar chunks."""
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from ingest import config
from ingest.store import get_collection


@dataclass
class Hit:
    text: str
    source: str
    page: int
    score: float  # cosine similarity, higher is better


class Retriever:
    def __init__(self, model_name: str = config.EMBED_MODEL):
        self.model = SentenceTransformer(model_name)
        self.collection = get_collection(config.CHROMA_DIR, config.COLLECTION)
        if self.collection.count() == 0:
            raise RuntimeError("Vector store is empty. Run `python -m ingest.run` first.")

    def search(self, question: str, k: int = config.TOP_K) -> list[Hit]:
        q = self.model.encode([config.QUERY_PREFIX + question], normalize_embeddings=True).tolist()
        res = self.collection.query(query_embeddings=q, n_results=k)
        return [
            Hit(text=doc, source=md["source"], page=md["page"], score=1 - dist)
            for doc, md, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]
