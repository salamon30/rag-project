# RAG Project

Retrieval-Augmented Generation over a local PDF collection.

## Stages
- **A. Ingestion** (`ingest/`): PDF loading, ~500-token chunking with overlap, `BAAI/bge-small-en-v1.5` embeddings (512-token window), local ChromaDB. ✅
- **B. Retrieval + generation** (`rag/`): top-k cosine retrieval, grounded answers with `[n]` citations via NVIDIA NIM (`nemotron-3.5-lightning-30b-a3b`); refuses when the context lacks the answer. ✅
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

## Ask
Copy `.env.example` to `.env` and set `NVIDIA_API_KEY`, then:
```bash
python -m rag.ask "How did Med-PaLM 2 perform on MedQA?"
python -m rag.ask "What is MedSAM?" --k 8 --show-context
```
