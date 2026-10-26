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

## Data
The corpus is 10 open-access arXiv papers on medical AI and RAG:
```bash
./scripts/download_data.sh
```

## Ingest
Put PDFs in `data/` (or use the script above), then:
```bash
python -m ingest.run            # incremental
python -m ingest.run --reset    # rebuild the collection from scratch
```
Chunks are stored in `chroma_db/` (collection `documents`) with `source` and `page` metadata.
