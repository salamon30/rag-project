# Medical AI Paper QA: an evaluated RAG pipeline

[![CI](https://github.com/salamon30/rag-project/actions/workflows/ci.yml/badge.svg)](https://github.com/salamon30/rag-project/actions/workflows/ci.yml)

Ask questions about medical-AI research papers and get answers that cite the exact passage and page they came from. The system says *"I don't know"* instead of guessing when the retrieved text does not contain the answer.

Built end to end in Python: PDF ingestion, two-stage retrieval (vector search + cross-encoder reranker), grounded generation, a measured evaluation suite and a web UI.

![Streamlit UI answering a question with cited sources](docs/screenshot.png)

## Highlights

- **Grounded answers with citations.** Every claim is tied to a numbered passage `[n]`, shown with file, page and similarity score.
- **Measured, not assumed.** A 20-question test set built from the papers scores retrieval (hit@k, MRR) and answers (LLM-as-judge faithfulness and correctness).
- **Refuses instead of hallucinating.** When retrieval misses, the model declines rather than inventing an answer.
- **Two-stage retrieval.** A cross-encoder reranker re-scores the vector-search candidates. It eliminated every refusal in the test set.
- **Debugged with data.** Several design choices below were changed because the evaluation exposed a problem.

## Results

20 questions over 10 papers, k = 5. Both runs use the same corpus, prompts and LLM; the only difference is the reranker.

| Metric | Vector search only | + Reranker |
|---|---|---|
| Retrieval hit@1 | 0.70 | **0.80** |
| Retrieval hit@5 | 0.90 | **1.00** |
| Right page in top 5 (page_hit@5) | 0.45 | **0.90** |
| MRR@10 | 0.81 | **0.88** |
| Refusals ("I don't know") | 5 | **0** |
| Answers with citations | 100% | 100% |
| Faithfulness (1-5, LLM judge) | 5.00 | 5.00 |
| Correctness (1-5, LLM judge) | 3.95 | **5.00** |

Reports: [with reranker](eval/report.md) · [without reranker](eval/report_no_rerank.md) · [reranker comparison](eval/retrieval_comparison.md)

**What changed.** Vector search usually found the right *paper* but often not the *page* with the answer, so the model correctly refused. In the hardest case the answer passage sat at rank 28. The cross-encoder reads the question and each candidate passage together and moves the answering passage to the top. The model's answers were already faithful; the reranker gave them the right evidence.

**Reading the scores honestly.** Correctness of 5.0 means this test set no longer separates good from great; a harder, larger set is the next step. Manual spot checks agree with the judge, except one answer that omitted half of the reference and would deserve a 4.

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
        E --> C50[Top 50 candidates]
        C50 --> RR[Cross-encoder reranker<br/>MiniLM-L-6]
        RR --> K[Top 5 chunks<br/>+ source, page]
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
| Chunking | ~500 tokens, 75 overlap; boundaries from the embedding model's tokenizer, text sliced from the original page |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-dim, 512-token window, runs locally) |
| Vector store | ChromaDB, persistent, cosine distance, deterministic chunk IDs for idempotent re-ingestion |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores the top 50 vector hits; chosen over `bge-reranker-base` for equal hit rates at ~6x lower latency |
| LLM | `nvidia/nemotron-3.5-lightning-30b-a3b` via NVIDIA NIM's OpenAI-compatible API |
| UI | Streamlit, with shareable `?q=` question links |

## Engineering decisions and lessons

These came from measuring, not from the first draft.

1. **Embedding window vs. chunk size.** The first embedding model, `all-MiniLM-L6-v2`, reads only 256 tokens, so half of every 500-token chunk was invisible to search. Switching to `bge-small-en-v1.5` with a 512-token window fixed this without shrinking chunks.
2. **Reasoning on for the generator.** With the model's hidden reasoning turned off, answers dropped citations entirely and misread passages with confidence. Reasoning stays on, with a no-reasoning fallback if the token budget runs out.
3. **Reasoning off for the judge.** With reasoning on, the judge spent its whole budget thinking and returned no score for 7 of 20 questions. Scoring is a short rubric task, so the judge runs without it.
4. **Do not trust the judge blindly.** The judge rated an *"I don't know"* answer as fully correct. Since every test question is answerable, refusals are now scored correctness 1 by rule. A refusal detector that was too broad also flagged a correct answer containing "does not include zero"; it now requires a refusal phrase and no citations.
5. **Keep the original text, not the tokenizer's version.** Chunks were first built by decoding token windows. Because the embedding tokenizer is uncased, stored text came out lowercased with `##` word pieces and `[UNK]`, and that is what the LLM and citations saw. Chunks are now sliced from the original page using the tokenizer's character offsets.
6. **Rerankers: measure the trade-off.** A retrieval-only evaluation mode (no LLM calls, under a minute) compared two cross-encoders. Both doubled page-level hits; the smaller one was ~6x faster, so it is the default.
7. **Errors must not lose the run.** A single failed API call used to abort the whole evaluation. Each question is now isolated and failures are reported as unscored.

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
python -m eval.run_eval                                    # full evaluation (~15 min)
python -m eval.run_eval --retrieval-only cross-encoder/ms-marco-MiniLM-L-6-v2   # compare retrievers, no LLM
```

### Tests

```bash
pip install -r requirements-dev.txt
pytest -q          # 36 unit tests, about 5 s
ruff check .
```

The tests use small fakes for the tokenizer, embedder, vector store, reranker and LLM client, so they need no API key, no model downloads and no PDFs. They cover chunk boundaries and overlap, the reranker reordering, the thinking fallback in the generator, judge output parsing, the refusal detector and the retrieval metrics. Several are regression tests for bugs found during development. GitHub Actions runs lint and tests on every push and pull request.

## Project structure

```
ingest/    load PDFs, chunk, embed, store (Stage A)
rag/       two-stage retriever, generator with citation rules, pipeline, CLI (Stage B)
eval/      test set, metrics, LLM judge, report.md + results.csv (Stage C)
app/       Streamlit UI
scripts/   corpus download
tests/     offline unit tests (pytest)
```

## Corpus

Ten open-access arXiv papers on medical AI and RAG: the original RAG paper, a RAG survey, MedRAG, Med-PaLM and Med-PaLM 2, GPT-4 on medical exams, Medprompt, MedSAM, CheXNet and U-Net. The PDFs are not redistributed here; `scripts/download_data.sh` fetches them from arXiv.

## Limitations and next steps

- **Same model as generator and judge.** This can inflate answer scores through self-preference. A different judge model would be more reliable.
- **Test set is now too easy.** With the reranker every question is answered correctly, so 20 questions no longer show headroom. A larger set with multi-hop and unanswerable questions is the next evaluation step.
- **Latency.** Reranking adds about 0.6 s per question on a laptop CPU. Hybrid BM25 + dense search could reduce the candidate set further.
- **Per-page chunking** keeps citations exact but splits passages that cross page boundaries.
