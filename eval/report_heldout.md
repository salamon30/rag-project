# RAG evaluation report

Date: 2026-10-30  
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
| answers with [n] citations | 1.000 |
| refusals | 1 |
| refusal recall (unanswerable) | 1.000 |
| faithfulness (1-5) | 4.850 |
| correctness (1-5) | 4.500 |
| unscored (errors) | 0 |

**Metric definitions.** hit@k: expected source file is among the top-k chunks. page_hit@k: a chunk from the expected file *and* an expected page is in the top-k. MRR@10: mean of 1/rank of the first chunk from the expected file (0 if not in top 10). Faithfulness: answer is supported by the retrieved context. Correctness: answer matches the reference. Both are scored 1-5 by an LLM judge.

**Caveat.** The judge is the same model as the generator, which can bias scores upward (self-preference). Treat answer scores as indicative, not absolute.

## By question type

| type | n | page_hit@5 | correctness (1-5) | said don't know |
|---|---|---|---|---|
| single | 10 | 1.000 | 4.800 | 0 |
| paraphrase | 3 | 1.000 | 5.000 | 0 |
| multi_hop | 3 | 0.667 | 2.333 | 1 |
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
| x06 | single | Which adversarial question datasets were introduced with Med-PaLM 2, and how many questions does each contain? | 1 | 1 | 1 | 5 | 3 |
| x07 | single | How accurate was text-only GPT-4 on USMLE questions that referred to images or graphs it could not see? | 1 | 1 | 1 | 5 | 5 |
| x08 | single | According to the Medprompt ablation study, which component added the most MedQA accuracy? | 1 | 1 | 1 | 5 | 5 |
| x09 | single | How does the position of the ground-truth snippet in the retrieved context affect accuracy in the MedRAG experiments? | 1 | 1 | 1 | 5 | 5 |
| x10 | single | Which quality scores and abilities does the RAG survey list for evaluating RAG systems? | 1 | 1 | 1 | 5 | 5 |
| x11 | paraphrase | The 2015 network that segments microscopy images from very few labelled examples: how long did it take to train, and on what hardware? | 1 | 1 | 1 | 5 | 5 |
| x12 | paraphrase | How did the study of GPT-4 on medical exams check whether the exam questions had leaked into the model's training data, and what did it find? | 2 | 1 | 1 | 5 | 5 |
| x13 | paraphrase | What is the name of the indexing approach where short sentences are used to search but the neighbouring sentences are handed to the LLM as context? | 1 | 1 | 1 | 5 | 5 |
| x14 | multi_hop | Compare the optimizer, learning rate and batch size used to train CheXNet with those used to train MedSAM. | 1 | 1 | 1 | 4 | 3 |
| x15 | multi_hop | How many retrieved passages does the original RAG model use per query, compared with the default number of snippets in MedRAG? | 1 | 1 | 1 | 3 | 3 |
| x16 | multi_hop | Which medical model did the GPT-4 exam study cite but could not test itself, and how was that model adapted to medicine? | 1 | 1 | 0 | 5 | 1 |
| x17 | unanswerable | How much did it cost, in US dollars, to run Medprompt over the MedQA test set? | – | – | – | 5 | 5 |
| x18 | unanswerable | What is the average latency per question, in seconds, of the MedRAG pipeline? | – | – | – | 5 | 5 |
| x19 | false_premise | Why did Lewis et al. fine-tune the document encoder together with the generator in RAG? | 2 | 1 | 1 | 5 | 5 |
| x20 | false_premise | Why was CheXNet trained on lateral-view chest X-rays? | 1 | 1 | 1 | 5 | 5 |

## Example failures

### x16: Which medical model did the GPT-4 exam study cite but could not test itself, and how was that model adapted to medicine?

- **Type:** multi_hop · **Expected:** gpt4_medical_challenge_2023.pdf:p5 6; med_palm_2022.pdf:p3
- **Retrieved:** gpt4_medical_challenge_2023.pdf:p3; gpt4_medical_challenge_2023.pdf:p14; medprompt_2023.pdf:p3; gpt4_medical_challenge_2023.pdf:p1; medprompt_2023.pdf:p17
- **Answer:** I don't know based on the provided documents.
- **Reference:** Med-PaLM (and Flan-PaLM 540B), which were not publicly available, so reported numbers were used. Med-PaLM was built from Flan-PaLM with instruction prompt tuning, a data- and parameter-efficient alignment technique.
- **Faithfulness 5:** The answer correctly states it does not know, which is fully supported by the context.
- **Correctness 1:** Refusal on an answerable question (rule-based score).

### x15: How many retrieved passages does the original RAG model use per query, compared with the default number of snippets in MedRAG?

- **Type:** multi_hop · **Expected:** rag_lewis_2020.pdf:p8 17; medrag_benchmark_2024.pdf:p4
- **Retrieved:** rag_lewis_2020.pdf:p4; medrag_benchmark_2024.pdf:p7; rag_lewis_2020.pdf:p17; medrag_benchmark_2024.pdf:p7; rag_lewis_2020.pdf:p2
- **Answer:** The original RAG model uses 15 retrieved documents per query for RAG-Token models or 50 for RAG-Sequence models (for Open-domain QA) [3]. MedRAG's default number of snippets is 4, as indicated by using "RRF-4 as the retriever" in scaling studies and testing with k ∈ {1, 2, 4, …,64} [4].
- **Reference:** RAG is trained with either 5 or 10 retrieved documents (no significant difference between them); for open-domain QA test results it uses 15 documents for RAG-Token and 50 for RAG-Sequence. MedRAG retrieves 32 snippets by default.
- **Faithfulness 3:** Manual score after fixing an incomplete reference: the RAG part (15 / 50 test-time documents, p17) is correct, the MedRAG part (4 snippets) is wrong, the paper says 32. The judge re-scored it 5/5 four times and missed the 4 vs 32 mismatch.
- **Correctness 3:** Manual score after fixing an incomplete reference: the RAG part (15 / 50 test-time documents, p17) is correct, the MedRAG part (4 snippets) is wrong, the paper says 32. The judge re-scored it 5/5 four times and missed the 4 vs 32 mismatch.

### x14: Compare the optimizer, learning rate and batch size used to train CheXNet with those used to train MedSAM.

- **Type:** multi_hop · **Expected:** chexnet_2017.pdf:p2; medsam_2023.pdf:p12
- **Retrieved:** medsam_2023.pdf:p8; med_palm_2022.pdf:p31; medsam_2023.pdf:p12; chexnet_2017.pdf:p2; medsam_2023.pdf:p11
- **Answer:** - **CheXNet** (from [4]): optimizer is Adam with β₁=0.9, β₂=0.999; learning rate and batch size are not specified in the provided text.  
- **MedSAM** (from [3]): optimizer is AdamW with β₁=0.9, β₂=0.999, initial learning rate 1e-4, and global batch size 160.  

**Comparison**: Both models share the same β₁ and β₂ values (0.9 and 0.999), but differ in optimizer type (Adam vs. AdamW). CheXNet's learning rate and batch size are not stated, while MedSAM specifies 1e-4 and 160 respectively. [3][4]
- **Reference:** CheXNet: Adam (0.9, 0.999), initial learning rate 0.001 with decay on plateau, batch size 16. MedSAM: AdamW (0.9, 0.999), initial learning rate 1e-4, weight decay 0.01, global batch size 160. MedSAM uses a 10x lower learning rate and a 10x larger batch.
- **Faithfulness 4:** CheXNet's learning rate and batch size are not specified in the provided text, making the comparison incomplete but not contradictory.
- **Correctness 3:** The answer correctly identifies the MedSAM hyperparameters and the comparison of learning rate and batch size, but fails to extract CheXNet's learning rate and batch size from the context, resulting in an incomplete comparison.

### x06: Which adversarial question datasets were introduced with Med-PaLM 2, and how many questions does each contain?

- **Type:** single · **Expected:** med_palm2_2023.pdf:p1 4
- **Retrieved:** med_palm2_2023.pdf:p3; med_palm2_2023.pdf:p3; med_palm2_2023.pdf:p10; med_palm2_2023.pdf:p1; med_palm2_2023.pdf:p9
- **Answer:** Based on the provided passages:

- [2] states that two adversarial question datasets were introduced to probe the safety and limitations of Med-PaLM 2, and that physicians evaluated Med-PaLM 2 answers against "both the general and health equity-focused subsets of the Adversarial dataset" (Table A.3).  
- [4] specifies that these newly introduced adversarial datasets consist of **240 long-form 'adversarial' questions** to probe LLM limitations.  
- The passages do not give a separate question count for each subset (general vs. health equity-focused); only that the total adversarial dataset contains 240 questions and is divided into those two subsets.  

[2][4]
- **Reference:** Two long-form adversarial sets, 240 questions in total: Adversarial (General) with 58 questions and Adversarial (Health equity) with 182 questions.
- **Faithfulness 5:** The answer accurately summarizes the context, noting the total of 240 adversarial questions and the division into general and health equity subsets without making unsupported claims.
- **Correctness 3:** The answer correctly identifies the total number of adversarial questions (240) and their introduction with Med-PaLM 2, but it fails to specify the exact question counts for the 'Adversarial (General)' (58) and 'Adversarial (Health equity)' (182) subsets as stated in the reference answer, resulting in a partial match.
