# RAG Project

Retrieval-Augmented Generation over a local PDF collection.

## Stages
- **A. Ingestion** (`ingest/`): PDF loading, ~500-token chunking with overlap, `all-MiniLM-L6-v2` embeddings, local ChromaDB. ✅
- **B. Retrieval + generation** (`rag/`): planned
- **C. Evaluation** (`eval/`): planned
- **UI** (`app/`): planned

## Setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ingest
Put PDFs in `data/`, then:
```bash
python -m ingest.run            # incremental
python -m ingest.run --reset    # rebuild the collection from scratch
```
Chunks are stored in `chroma_db/` (collection `documents`) with `source` and `page` metadata.
