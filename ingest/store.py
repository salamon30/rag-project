"""Steps 3-4: embed chunks and upsert them into a persistent ChromaDB collection."""
import chromadb

from .chunker import Chunk


def get_collection(path, name, reset=False):
    client = chromadb.PersistentClient(path=str(path))
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


def embed_and_store(chunks: list[Chunk], model, collection, batch_size: int = 64) -> None:
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors = model.encode([c.text for c in batch], normalize_embeddings=True,
                               show_progress_bar=False).tolist()
        collection.upsert(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            embeddings=vectors,
            metadatas=[{"source": c.source, "page": c.page, "chunk_index": c.chunk_index}
                       for c in batch],
        )
        print(f"  embedded {min(i + batch_size, len(chunks))}/{len(chunks)}", end="\r")
    print()
