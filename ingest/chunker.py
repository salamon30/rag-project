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
    """Token windows define the boundaries, but the text is sliced from the original
    page via character offsets. Decoding the tokens instead would return the
    tokenizer's normalised form (bge is uncased: lowercased, '##' word pieces, [UNK]),
    which is what the LLM, the reranker and the citations would then see."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    step = size - overlap
    chunks: list[Chunk] = []
    for pg in pages:
        offsets = tokenizer(pg.text, add_special_tokens=False,
                            return_offsets_mapping=True)["offset_mapping"]
        n_tok = len(offsets)
        for n, start in enumerate(range(0, max(n_tok, 1), step)):
            window = offsets[start : start + size]
            if not window:
                break
            text = pg.text[window[0][0] : window[-1][1]].strip()
            if text:
                chunks.append(Chunk(
                    id=f"{pg.source}::p{pg.page}::c{n}",
                    text=text, source=pg.source, page=pg.page, chunk_index=n,
                ))
            if start + size >= n_tok:
                break
    return chunks
