# Medical AI Paper QA: an evaluated RAG pipeline

Ask questions about medical-AI research papers and get answers that cite the exact passage and page they came from. The system says *"I don't know"* instead of guessing when the retrieved text does not contain the answer.

Built end to end in Python: PDF ingestion, vector search, grounded generation, a measured evaluation suite and a web UI.

![Streamlit UI answering a question with cited sources](docs/screenshot.png)

## Highlights

- **Grounded answers with citations.** Every claim is tied to a numbered passage `[n]`, shown with file, page and similarity score.
- **Measured, not assumed.** A 20-question test set built from the papers scores retrieval (hit@k, MRR) and answers (LLM-as-judge faithfulness and correctness).
- **Refuses instead of hallucinating.** When retrieval misses, the model declines. Faithfulness stays at 4.8 / 5 even on misses.
- **Debugged with data.** Several design choices below were changed because the evaluation exposed a problem.

## Results

20 questions over 10 papers, k = 5 ([full report](eval/report.md), [per-question CSV](eval/results.csv)).

| Metric | Score |
|---|---|
| Retrieval hit@1 | 0.70 |
| Retrieval hit@5 | 0.95 |
| MRR@10 | 0.81 |
| Answers with citations | 94% |
| Faithfulness (1-5, LLM judge) | 4.80 |
| Correctness (1-5, LLM judge) | 4.00 |

**What the misses show.** All four refusals are retrieval misses, not generation errors. The main pattern: the RAG *survey* paper quotes the original RAG paper, so its chunks outrank the original for questions about it. The one unfaithful answer came from the same confusion. A reranker or hybrid BM25 search is the next experiment, and this table is the baseline to beat.

## Architecture

```mermaid
flowchart LR
    subgraph Ingestion
        A[PDFs] --> B[Page text<br/>pypdf]
        B --> C[~500-token chunks<br/>75 overlap, per page]
        C --> D[Embeddings<br/>bge-small-en-v1.5]
        D --> E[(ChromaDB<br/>cosine)]
    end
    subgraph Query
        Q[Question] --> QE[Query embedding]
        QE --> E
        E --> K[Top-k chunks<br/>+ source, page]
        K --> L[LLM<br/>NVIDIA NIM]
        L --> R[Answer with citations<br/>or refusal]
    end
    subgraph Evaluation
        T[Test set<br/>20 Q + references] --> M[hit@k, MRR]
        R --> J[LLM judge<br/>faithfulness, correctness]
    end
```

| Component | Choice |
|---|---|
| PDF parsing | `pypdf`, page by page so citations keep exact page numbers |
| Chunking | ~500 tokens, 75 overlap, counted with the embedding model's own tokenizer |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-dim, 512-token window, runs locally) |
| Vector store | ChromaDB, persistent, cosine distance, deterministic chunk IDs for idempotent re-ingestion |
| LLM | `nvidia/nemotron-3.5-lightning-30b-a3b` via NVIDIA NIM's OpenAI-compatible API |
| UI | Streamlit, with shareable `?q=` question links |

## Engineering decisions and lessons

These came from measuring, not from the first draft.

1. **Embedding window vs. chunk size.** The first embedding model, `all-MiniLM-L6-v2`, reads only 256 tokens, so half of every 500-token chunk was invisible to search. Switching to `bge-small-en-v1.5` with a 512-token window fixed this without shrinking chunks.
2. **Reasoning on for the generator.** With the model's hidden reasoning turned off, answers dropped citations entirely and misread passages with confidence. Reasoning stays on, with a no-reasoning fallback if the token budget runs out.
3. **Reasoning off for the judge.** With reasoning on, the judge spent its whole budget thinking and returned no score for 7 of 20 questions. Scoring is a short rubric task, so the judge runs without it.
4. **Do not trust the judge blindly.** The judge rated an *"I don't know"* answer as fully correct. Since every test question is answerable, refusals are now scored correctness 1 by rule. A refusal detector that was too broad also flagged a correct answer containing "does not include zero"; it now requires a refusal phrase and no citations.
5. **Errors must not lose the run.** A single failed API call used to abort the whole evaluation. Each question is now isolated and failures are reported as unscored.

## Quickstart

Requires Python 3.11 and an [NVIDIA NIM API key](https://build.nvidia.com/).

```bash
git clone https://github.com/salamon30/rag-project.git && cd rag-project
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # then set NVIDIA_API_KEY

./scripts/download_data.sh         # 10 open-access arXiv papers into data/
python -m ingest.run --reset       # build the vector store
```

Then use any of the three entry points:

```bash
streamlit run app/streamlit_app.py                         # web UI at localhost:8501
python -m rag.ask "What is the MIRAGE benchmark?"          # command line
python -m eval.run_eval                                    # rerun the evaluation (~15 min)
```

## Project structure

```
ingest/    load PDFs, chunk, embed, store (Stage A)
rag/       retriever, generator with citation rules, pipeline, CLI (Stage B)
eval/      test set, metrics, LLM judge, report.md + results.csv (Stage C)
app/       Streamlit UI
scripts/   corpus download
```

## Corpus

Ten open-access arXiv papers on medical AI and RAG: the original RAG paper, a RAG survey, MedRAG, Med-PaLM and Med-PaLM 2, GPT-4 on medical exams, Medprompt, MedSAM, CheXNet and U-Net. The PDFs are not redistributed here; `scripts/download_data.sh` fetches them from arXiv.

## Limitations and next steps

- **Same model as generator and judge.** This can inflate answer scores through self-preference. A different judge model would be more reliable.
- **Small test set.** 20 questions give a useful signal but wide error bars; answers also vary slightly between runs.
- **Retrieval is the bottleneck.** Next: a cross-encoder reranker and hybrid BM25 + dense search, measured against the baseline above.
- **Per-page chunking** keeps citations exact but splits passages that cross page boundaries.
