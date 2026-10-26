"""Stage A entry point: python -m ingest.run [--reset]"""
import argparse
from collections import Counter

from sentence_transformers import SentenceTransformer

from . import config
from .chunker import chunk_pages
from .loader import load_pdfs
from .store import embed_and_store, get_collection


def main():
    ap = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB")
    ap.add_argument("--reset", action="store_true", help="drop and rebuild the collection")
    ap.add_argument("--data", default=str(config.DATA_DIR))
    ap.add_argument("--chunk-tokens", type=int, default=config.CHUNK_TOKENS)
    ap.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    args = ap.parse_args()

    from pathlib import Path
    data_dir = Path(args.data)

    print(f"[1/4] Loading PDFs from {data_dir}")
    pages = load_pdfs(data_dir)
    if not pages:
        print("  no extractable text found. Put PDFs in data/ and retry.")
        return
    print(f"  {len({p.source for p in pages})} PDFs, {len(pages)} pages with text")

    print(f"[2/4] Chunking (~{args.chunk_tokens} tokens, {args.overlap} overlap)")
    model = SentenceTransformer(config.EMBED_MODEL)
    if args.chunk_tokens > model.max_seq_length:
        print(f"  note: {config.EMBED_MODEL} embeds at most {model.max_seq_length} tokens; "
              f"text beyond that in each chunk is not reflected in its vector")
    chunks = chunk_pages(pages, model.tokenizer, args.chunk_tokens, args.overlap)
    print(f"  {len(chunks)} chunks")

    print(f"[3/4] Embedding with {config.EMBED_MODEL}")
    collection = get_collection(config.CHROMA_DIR, config.COLLECTION, reset=args.reset)
    print(f"[4/4] Storing in ChromaDB at {config.CHROMA_DIR}")
    embed_and_store(chunks, model, collection)

    per_file = Counter(c.source for c in chunks)
    print("\nChunks per file:")
    for src, n in per_file.most_common():
        print(f"  {n:5d}  {src}")
    print(f"\nTotal chunks this run: {len(chunks)}")
    print(f"Collection '{config.COLLECTION}' now holds {collection.count()} chunks")


if __name__ == "__main__":
    main()
