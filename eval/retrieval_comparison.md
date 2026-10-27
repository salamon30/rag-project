# Retrieval comparison: vector search vs. cross-encoder rerankers

Run with `python -m eval.run_eval --retrieval-only cross-encoder/ms-marco-MiniLM-L-6-v2 BAAI/bge-reranker-base`.
Same 20-question test set, k = 5, rerankers re-score the top 50 vector candidates. No LLM calls involved.
Latency is per question on a MacBook (CPU/MPS), including the vector search.

| Retriever | hit@1 | hit@5 | page_hit@5 | MRR@10 | s / question | page misses |
|---|---|---|---|---|---|---|
| Vector search only (bge-small) | 0.70 | 0.90 | 0.45 | 0.81 | 0.18 | q01 q02 q04 q08 q11 q12 q14 q15 q16 q18 q20 |
| + ms-marco-MiniLM-L-6-v2 reranker | 0.80 | **1.00** | **0.90** | 0.88 | 0.74 | q11 q14 |
| + bge-reranker-base reranker | 0.80 | **1.00** | **0.90** | **0.90** | 4.25 | q01 q14 |

**Choice:** MiniLM is the default. It matches bge-reranker-base on hit@5 and page_hit@5 at about 6x lower latency, which matters for an interactive UI.

**Why page_hit doubles.** Vector search finds the right *paper* but often the wrong *page*: a passage about the topic, not the one with the answer. The cross-encoder reads question and passage together and promotes the passage that actually answers. For q01 the answer passage (RAG paper, p.3) sat at vector rank 28, outside any reasonable top-k without reranking.

**Note on the baseline.** These numbers use the corrected chunk text (see README). With the earlier, tokenizer-normalised chunk text, vector search scored hit@5 0.95 instead of 0.90; page_hit@5 and MRR were essentially unchanged.
