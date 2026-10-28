import numpy as np

from ingest.chunker import Chunk
from ingest.loader import load_pdfs
from ingest.store import embed_and_store, get_collection


def test_corrupt_pdf_is_skipped(tmp_path, capsys):
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")
    assert load_pdfs(tmp_path) == []
    assert "skipping broken.pdf" in capsys.readouterr().out


class FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        return np.array([[len(t), 1.0, 0.0] for t in texts], dtype=float)


def test_store_upserts_with_metadata(tmp_path):
    col = get_collection(tmp_path, "test")
    chunks = [Chunk(f"a.pdf::p{i}::c0", f"text {i}", "a.pdf", i, 0) for i in range(1, 4)]
    embed_and_store(chunks, FakeEmbedder(), col, batch_size=2)
    embed_and_store(chunks, FakeEmbedder(), col)  # same IDs again: upsert, no duplicates
    assert col.count() == 3
    got = col.get(ids=["a.pdf::p2::c0"])
    assert got["metadatas"][0] == {"source": "a.pdf", "page": 2, "chunk_index": 0}


def test_reset_drops_collection(tmp_path):
    col = get_collection(tmp_path, "test")
    embed_and_store([Chunk("x", "t", "a.pdf", 1, 0)], FakeEmbedder(), col)
    assert get_collection(tmp_path, "test", reset=True).count() == 0
