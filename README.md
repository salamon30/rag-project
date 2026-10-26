# RAG Project

Retrieval-Augmented Generation over a local PDF collection.

## Stages
- **A. Ingestion** (`ingest/`): PDF loading, ~500-token chunking with overlap, `BAAI/bge-small-en-v1.5` embeddings (512-token window), local ChromaDB. ✅
- **B. Retrieval + generation** (`rag/`): top-k cosine retrieval, grounded answers with `[n]` citations via NVIDIA NIM (`nemotron-3.5-lightning-30b-a3b`); refuses when the context lacks the answer. ✅
- **C. Evaluation** (`eval/`): 20-question test set grounded in the papers; retrieval hit@k / MRR and LLM-as-judge faithfulness + correctness. ✅
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

## Evaluate
```bash
python -m eval.run_eval          # writes eval/report.md and eval/results.csv
```
- **hit@k**: the expected source file is among the top-k retrieved chunks. **page_hit@k** also requires an expected page.
- **MRR@10**: mean of 1/rank of the first chunk from the expected file.
- **Faithfulness / correctness (1-5)**: LLM judge checks the answer against the retrieved context and against a reference answer. Refusals on these answerable questions are scored correctness 1 by rule.

Lessons from building it: with the reasoning model's thinking off, answers dropped citations and misread context, so the generator keeps thinking on (with a no-thinking fallback when the token budget runs out). The judge runs with thinking off, because otherwise it spent its whole budget reasoning and returned no score.
