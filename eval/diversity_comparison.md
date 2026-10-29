# Per-paper cap on the reranked top 5 (retrieval only, no LLM)

Idea: the reranker can fill all five slots from one paper, so questions that need two
papers lose the second one. A cap of N chunks per paper makes room for other papers.

| Reranker + cap | hard set: all_sources@5 | hard set: page_hit@5 | original set: page_hit@5 |
|---|---|---|---|
| no cap (default) | 0.786 | 0.714 | **0.900** |
| max 3 per paper | 0.857 | 0.786 | 0.800 |
| max 2 per paper | **0.929** | 0.714 | 0.700 |

**Decision: keep the cap off.** It helps multi-paper questions but pushes the right page out
of the top 5 for single-paper questions, where the answer is often the 4th or 5th chunk of the
same paper. One fixed rule cannot serve both question types. A per-question approach, such as
splitting a comparison question into one sub-query per paper, is the better next step.

Reproduce: `python -m eval.run_eval --testset testset_hard --retrieval-only cross-encoder/ms-marco-MiniLM-L-6-v2 --max-per-source 0 3 2`
