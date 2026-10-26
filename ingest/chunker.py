"""Step 2: split each page into ~N-token windows with overlap.

Tokens are counted with the embedding model's own tokenizer, so chunk size
matches what the model actually sees. Chunking is per page, which keeps the
page number exact for citations.
"""
from dataclasses import dataclass

from .loader import Page


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int
    chunk_index: int


def chunk_pages(pages: list[Page], tokenizer, size: int, overlap: int) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    step = size - overlap
    chunks: list[Chunk] = []
    for pg in pages:
        ids = tokenizer.encode(pg.text, add_special_tokens=False)
        for n, start in enumerate(range(0, max(len(ids), 1), step)):
            window = ids[start : start + size]
            text = tokenizer.decode(window).strip()
            if text:
                chunks.append(Chunk(
                    id=f"{pg.source}::p{pg.page}::c{n}",
                    text=text, source=pg.source, page=pg.page, chunk_index=n,
                ))
            if start + size >= len(ids):
                break
    return chunks
