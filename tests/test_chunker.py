import pytest

from ingest.chunker import chunk_pages
from ingest.loader import Page


def words(n):
    return " ".join(f"Word{i}" for i in range(n))


def test_short_page_is_one_chunk(tokenizer):
    chunks = chunk_pages([Page("a.pdf", 3, "Hello world")], tokenizer, size=10, overlap=2)
    assert len(chunks) == 1
    assert chunks[0].id == "a.pdf::p3::c0"
    assert chunks[0].page == 3


def test_windows_overlap_and_cover_every_token(tokenizer):
    text = words(25)
    chunks = chunk_pages([Page("a.pdf", 1, text)], tokenizer, size=10, overlap=3)
    toks = [c.text.split() for c in chunks]
    assert all(len(t) <= 10 for t in toks)
    for prev, nxt in zip(toks, toks[1:], strict=False):
        assert prev[-3:] == nxt[:3]          # 3-token overlap
    covered = {w for t in toks for w in t}
    assert covered == set(text.split())      # nothing is dropped
    assert toks[-1][-1] == "Word24"


def test_text_keeps_original_casing_and_spacing(tokenizer):
    # regression: decoding tokens used to lowercase text and add '##' pieces
    text = "MedSAM uses   ViT-B\nencoders."
    [chunk] = chunk_pages([Page("a.pdf", 1, text)], tokenizer, size=50, overlap=5)
    assert chunk.text == text


def test_ids_are_unique_across_pages(tokenizer):
    pages = [Page("a.pdf", 1, words(30)), Page("a.pdf", 2, words(30)), Page("b.pdf", 1, words(30))]
    chunks = chunk_pages(pages, tokenizer, size=10, overlap=2)
    assert len({c.id for c in chunks}) == len(chunks)


def test_empty_page_gives_no_chunks(tokenizer):
    assert chunk_pages([Page("a.pdf", 1, "   ")], tokenizer, size=10, overlap=2) == []


def test_overlap_must_be_smaller_than_size(tokenizer):
    with pytest.raises(ValueError):
        chunk_pages([Page("a.pdf", 1, "x")], tokenizer, size=5, overlap=5)
