# Medical AI Paper QA: an evaluated RAG pipeline

[![CI](https://github.com/salamon30/rag-project/actions/workflows/ci.yml/badge.svg)](https://github.com/salamon30/rag-project/actions/workflows/ci.yml)

> **At a glance**
>
> - **What:** ask questions about a set of PDFs and get answers that cite the exact passage and page, or "I don't know" when the documents do not say.
> - **Results:** right page in the top 5 rose from 0.45 to 0.90 with a reranker; answer correctness 5.0 / 4.8 / 4.5 (1-5) on the original, hard and held-out test sets; 72 offline unit tests in CI.
> - **Stack:** Python 3.11 · pypdf · bge-small embeddings · ChromaDB · cross-encoder reranker · NVIDIA NIM (Nemotron) · Streamlit · pytest + GitHub Actions.
> - **Run:** `pip install -r requirements.txt`, set `NVIDIA_API_KEY`, then `./scripts/download_data.sh`, `python -m ingest.run --reset` and `streamlit run app/streamlit_app.py` ([full steps](#quickstart)).

Ask questions about medical-AI research papers and get answers that cite the exact passage and page they came from. The system says *"I don't know"* instead of guessing when the retrieved text does not contain the answer.

Built end to end in Python: PDF ingestion, two-stage retrieval (vector search + cross-encoder reranker), grounded generation, a measured evaluation suite and a web UI.

![Streamlit UI answering a question with cited sources](docs/screenshot.png)

## Highlights

- **Grounded answers with citations.** Every claim is tied to a numbered passage `[n]`, shown with file, page and relevance score.
- **Measured, not assumed.** Three test sets (60 questions, including unanswerable, false-premise, multi-paper and a held-out set never tuned on) score retrieval (hit@k, MRR) and answers (LLM-as-judge faithfulness and correctness).
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

**Reading the scores honestly.** Correctness of 5.0 means this test set no longer separates good from great, so a second, harder set was built (below). Manual spot checks agree with the judge, except one answer that omitted half of the reference and would deserve a 4.

### Hard test set

[20 more questions](eval/testset_hard.jsonl) of four kinds the first set never asked, each checked against the PDF text:

- **Unanswerable** (6): the topic is in the corpus but the fact is not, e.g. *"How much did it cost to train MedSAM?"* The system must say it does not know.
- **False premise** (4): *"Why did CheXNet use ResNet-50?"* (it uses DenseNet-121). The system must correct the question.
- **Multi-paper** (6): the answer needs two papers, e.g. comparing two models' MedQA scores.
- **Paraphrase** (4): the question avoids the paper's keywords.

| Question type | First prompt | + false-premise rule | + query decomposition (current) |
|---|---|---|---|
| Unanswerable (declined) | 5.0 (6 / 6) | 5.0 (6 / 6) | 5.0 (6 / 6) ✅ |
| False premise | 2.0 | **5.0** | 4.0 ¹ |
| Multi-paper | 3.0 | 3.7 | **5.0** |
| Paraphrase | 5.0 | 5.0 | 5.0 |
| **All 20** | 3.8 | 4.6 | **4.8** |

Correctness 1-5. Reports: [current](eval/report_hard.md) · [without decomposition](eval/report_hard_no_decompose.md). The original 20 questions score 5.0 in every column.

¹ One false-premise answer (h08) flipped to a refusal although it received exactly the same five passages as the run before: run-to-run variation of the LLM, not an effect of decomposition.

**What the hard set found.**
1. **The prompt was too strict.** The model refused false-premise questions because the passages did not explain *why ResNet-50 was chosen*. One added rule ("if the passages contradict the question's assumption, correct it with citations") fixed all four without making it answer any unanswerable question.
2. **The reranker can crowd out a second paper.** For multi-paper questions it sometimes filled all five slots from one paper, and the model then correctly said it lacked the other half. A fixed cap on chunks per paper fixed that but pushed the right page out for single-paper questions (page_hit@5 0.90 → 0.70), so it stays off ([diversity_comparison.md](eval/diversity_comparison.md)).
3. **Query decomposition fixed it per question.** An LLM call splits a comparison into one search per paper, and the results are interleaved. The first prompt split 35 of 40 questions, including single-paper ones ("U-Net challenge" + "U-Net image types"), and page hits on the original set fell to 0.65. Rather than tuning the prompt further, the retriever now checks each split: it is kept only if the sub-queries' best hits come from different papers. Multi-paper retrieval (all papers in top 5) rose from 0.79 to 0.93 and multi-paper correctness from 3.7 to 5.0; the original set's answers are unchanged.

### Held-out test set

Every change above was tuned while looking at the first two sets, so their scores could be optimistic. [20 new questions](eval/testset_heldout.jsonl) were written afterwards and run once, with no changes to the code or prompts. Each answer was checked against the PDF text, with page numbers.

| Question type | Hard set (tuned on) | Held-out set |
|---|---|---|
| Single paper (10) | – | 4.8 |
| Paraphrase (3) | 5.0 | 5.0 |
| Unanswerable, declined (2) | 5.0 | 5.0 ✅ |
| False premise (2) | 4.0 | 5.0 |
| Multi-paper (3) | 5.0 | **2.3** |
| **All 20** | 4.8 | **4.5** |

Correctness 1-5. Retrieval on the held-out set: hit@5 1.00, page_hit@5 0.94, MRR@10 0.94. [Full report](eval/report_heldout.md).

**What the held-out set found.**
1. **Most of the gains generalise.** Single-paper, paraphrase, unanswerable and false-premise questions score 4.8-5.0 on questions the system was never tuned on. The false-premise prompt rule holds up.
2. **Multi-paper questions do not.** The perfect 5.0 on the hard set was partly fitted to those six questions. Each held-out failure has a different cause:
    - **A chained question (x16).** *"Which model did the GPT-4 study cite but not test, and how was it adapted?"* The second search depends on the answer to the first, and decomposition only splits side-by-side comparisons. No Med-PaLM page was retrieved, so the model declined.
    - **A chunk boundary (x14).** The CheXNet passage in the top 5 ends at "We train", one sentence before the batch size and learning rate. The overlapping next chunk has them but ranked lower, so the comparison came out half empty.
    - **A misread number (x15).** The model said MedRAG uses 4 snippets by default; the paper says 32, on a page that was not retrieved.
3. **The judge missed a wrong number.** Re-scored after its reference was corrected, x15 got 5/5 four times in a row, each time accepting "4 snippets" against a reference of 32. It is scored by hand (3/5) and marked as manual in the report.
4. **The test set had a bug too.** The x15 reference only gave RAG's training setting (5 or 10 documents). The paper's appendix also reports 15 and 50 documents at test time, which the model had correctly cited. The reference was fixed and the saved answer re-scored without regenerating it.

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
        Q[Question] --> QD[Query decomposition<br/>one search per paper]
        QD --> QE[Query embedding]
        QE --> E
        E --> C50[Top 50 candidates]
        C50 --> RR[Cross-encoder reranker<br/>MiniLM-L-6]
        RR --> K[Top 5 chunks<br/>+ source, page]
        K --> L[LLM<br/>NVIDIA NIM]
        L --> R[Answer with citations<br/>or refusal]
    end
    subgraph Evaluation
        T[2 test sets<br/>40 Q + references] --> M[hit@k, all_sources@k, MRR]
        R --> J[LLM judge<br/>faithfulness, correctness]
    end
```

| Component | Choice |
|---|---|
| PDF parsing | `pypdf`, page by page so citations keep exact page numbers |
| Chunking | ~500 tokens, 75 overlap; boundaries from the embedding model's tokenizer, text sliced from the original page |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-dim, 512-token window, runs locally) |
| Vector store | ChromaDB, persistent, cosine distance, deterministic chunk IDs for idempotent re-ingestion |
| Query decomposition | One short LLM call splits multi-paper questions into one search per paper; a split is kept only if the searches reach different papers |
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
8. **Let the LLM propose and the code check.** Prompt tuning alone swung query decomposition between splitting everything and splitting nothing. An eager prompt plus a cheap retrieval check (do the sub-queries reach different papers?) gave both recall and precision.
9. **Check the input before blaming the change.** Two answers changed after decomposition was switched on, but both received identical passages in both runs. One was LLM variation; the other was a correct decline that the scoring rule did not recognise. The rule was fixed and saved answers are re-scored without new LLM calls (`--rescore`).

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
pytest -q          # 72 unit tests, about 5 s
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
- **Multi-paper questions are the weak spot.** They score 2.3 on the held-out set. Chained questions, where one search depends on another's answer, would need iterative retrieval: search, read, then search again. Adding the neighbouring chunk to each hit could fix boundary cuts like x14.
- **Small held-out set.** 20 questions, 3 of them multi-paper, so one answer moves a type's average a lot.
- **The judge is lenient on partial answers and can miss wrong numbers.** It gave 5/5 to a multi-paper answer that covered only one of the two papers (h15), and 5/5 to an answer with a wrong snippet count (x15). Answer scores are spot-checked by hand, and refusals are scored by rule rather than by the judge.
- **Run-to-run variation.** LLM answers and decomposition can differ between identical runs; single-question changes should be checked against the retrieved passages before they are attributed to a code change.
- **Decomposition cost.** One extra LLM call and up to three searches per question (about 1.5 s more on a laptop). `DECOMPOSE=0` turns it off.
- **Latency.** Reranking adds about 0.6 s per question on a laptop CPU. Hybrid BM25 + dense search could reduce the candidate set further.
- **Per-page chunking** keeps citations exact but splits passages that cross page boundaries.

## License

MIT, see [LICENSE](LICENSE).
