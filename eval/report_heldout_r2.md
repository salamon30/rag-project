# RAG evaluation report

Date: 2026-11-13  
Embeddings: `BAAI/bge-small-en-v1.5` · Generator and judge: `nvidia/nemotron-3.5-lightning-30b-a3b` (generator thinking on with no-thinking fallback, judge thinking off) · k = 5  
Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` over top 50 vector candidates · query decomposition on

## Summary

| Metric | Value |
|---|---|
| questions | 20 |
| hit@1 | 0.889 |
| hit@5 | 1.000 |
| all_sources@5 | 0.944 |
| page_hit@5 | 0.944 |
| MRR@10 | 0.944 |
| answers with [n] citations | 0.944 |
| refusals | 0 |
| refusal recall (unanswerable) | 1.000 |
| faithfulness (1-5) | 4.850 |
| correctness (1-5) | 5.000 |
| unscored (errors) | 0 |

**Metric definitions.** hit@k: expected source file is among the top-k chunks. page_hit@k: a chunk from the expected file *and* an expected page is in the top-k. MRR@10: mean of 1/rank of the first chunk from the expected file (0 if not in top 10). Faithfulness: answer is supported by the retrieved context. Correctness: answer matches the reference. Both are scored 1-5 by an LLM judge.

**Caveat.** The judge is the same model as the generator, which can bias scores upward (self-preference). Treat answer scores as indicative, not absolute.

## By question type

| type | n | page_hit@5 | correctness (1-5) | said don't know |
|---|---|---|---|---|
| single | 10 | 1.000 | 5.000 | 0 |
| paraphrase | 3 | 1.000 | 5.000 | 0 |
| multi_hop | 3 | 0.667 | 5.000 | 0 |
| unanswerable | 2 | – | 5.000 | 2 |
| false_premise | 2 | 1.000 | 5.000 | 0 |

Unanswerable questions should all be declined; every other type should be answered.

## Per-question results

| id | type | question | first rank | hit@5 | page hit | F | C |
|---|---|---|---|---|---|---|---|
| x01 | single | Which optimizer, mini-batch size and initial learning rate were used to train CheXNet? | 1 | 1 | 1 | 5 | 5 |
| x02 | single | How does U-Net's training make the network learn to separate touching cells of the same class? | 1 | 1 | 1 | 5 | 5 |
| x03 | single | Which components of the RAG model are updated during fine-tuning, and why is the document encoder left unchanged? | 1 | 1 | 1 | 5 | 5 |
| x04 | single | In the human evaluation of Jeopardy question generation, how often was RAG judged more factual than BART? | 1 | 1 | 1 | 5 | 5 |
| x05 | single | What share of Flan-PaLM and Med-PaLM long-form answers did clinicians rate as aligned with scientific consensus and as potentially harmful? | 1 | 1 | 1 | 5 | 5 |
| x06 | single | Which adversarial question datasets were introduced with Med-PaLM 2, and how many questions does each contain? | 1 | 1 | 1 | 4 | 5 |
| x07 | single | How accurate was text-only GPT-4 on USMLE questions that referred to images or graphs it could not see? | 1 | 1 | 1 | 5 | 5 |
| x08 | single | According to the Medprompt ablation study, which component added the most MedQA accuracy? | 1 | 1 | 1 | 5 | 5 |
| x09 | single | How does the position of the ground-truth snippet in the retrieved context affect accuracy in the MedRAG experiments? | 1 | 1 | 1 | 5 | 5 |
| x10 | single | Which quality scores and abilities does the RAG survey list for evaluating RAG systems? | 1 | 1 | 1 | 5 | 5 |
| x11 | paraphrase | The 2015 network that segments microscopy images from very few labelled examples: how long did it take to train, and on what hardware? | 1 | 1 | 1 | 5 | 5 |
| x12 | paraphrase | How did the study of GPT-4 on medical exams check whether the exam questions had leaked into the model's training data, and what did it find? | 2 | 1 | 1 | 5 | 5 |
| x13 | paraphrase | What is the name of the indexing approach where short sentences are used to search but the neighbouring sentences are handed to the LLM as context? | 1 | 1 | 1 | 5 | 5 |
| x14 | multi_hop | Compare the optimizer, learning rate and batch size used to train CheXNet with those used to train MedSAM. | 1 | 1 | 1 | 5 | 5 |
| x15 | multi_hop | How many retrieved passages does the original RAG model use per query, compared with the default number of snippets in MedRAG? | 1 | 1 | 1 | 3 | 5 |
| x16 | multi_hop | Which medical model did the GPT-4 exam study cite but could not test itself, and how was that model adapted to medicine? | 1 | 1 | 0 | 5 | 5 |
| x17 | unanswerable | How much did it cost, in US dollars, to run Medprompt over the MedQA test set? | – | – | – | 5 | 5 |
| x18 | unanswerable | What is the average latency per question, in seconds, of the MedRAG pipeline? | – | – | – | 5 | 5 |
| x19 | false_premise | Why did Lewis et al. fine-tune the document encoder together with the generator in RAG? | 2 | 1 | 1 | 5 | 5 |
| x20 | false_premise | Why was CheXNet trained on lateral-view chest X-rays? | 1 | 1 | 1 | 5 | 5 |

## Example failures

### x15: How many retrieved passages does the original RAG model use per query, compared with the default number of snippets in MedRAG?

- **Type:** multi_hop · **Expected:** rag_lewis_2020.pdf:p8 17; medrag_benchmark_2024.pdf:p4
- **Retrieved:** rag_lewis_2020.pdf:p4; medrag_benchmark_2024.pdf:p7; rag_lewis_2020.pdf:p17; medrag_benchmark_2024.pdf:p7; rag_lewis_2020.pdf:p2; medrag_benchmark_2024.pdf:p8; rag_lewis_2020.pdf:p3; medrag_benchmark_2024.pdf:p7
- **Answer:** The original RAG model uses 15 retrieved documents for RAG-Token models and 50 for RAG-Sequence models per query (or 10 for Open-MSMarco and Jeopardy question generation), as stated in [3]. In contrast, MedRAG employs a variable number of snippets k, with the study exploring k ∈ {1, 2, 4, …, 64} across different tasks, but no single default number of snippets is explicitly stated in the passages; the text only examines scaling behavior across these k values without designating one as the default setting.
- **Reference:** RAG is trained with either 5 or 10 retrieved documents (no significant difference between them); for open-domain QA test results it uses 15 documents for RAG-Token and 50 for RAG-Sequence. MedRAG retrieves 32 snippets by default.
- **Faithfulness 3:** The answer claims MedRAG has no single default number of snippets, but the context explicitly states MedRAG retrieves 32 snippets by default.
- **Correctness 5:** The answer accurately reports the original RAG model's retrieval numbers (15 for RAG-Token, 50 for RAG-Sequence) and correctly notes MedRAG's variable k values without claiming a specific default.
