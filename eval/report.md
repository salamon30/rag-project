# RAG evaluation report

Date: 2026-10-27  
Embeddings: `BAAI/bge-small-en-v1.5` · Generator and judge: `nvidia/nemotron-3.5-lightning-30b-a3b` (generator thinking on with no-thinking fallback, judge thinking off) · k = 5  
Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` over top 50 vector candidates

## Summary

| Metric | Value |
|---|---|
| questions | 20 |
| hit@1 | 0.800 |
| hit@5 | 1.000 |
| page_hit@5 | 0.900 |
| MRR@10 | 0.877 |
| answers with [n] citations | 1.000 |
| refusals | 0 |
| faithfulness (1-5) | 5.000 |
| correctness (1-5) | 5.000 |
| unscored (errors) | 0 |

**Metric definitions.** hit@k: expected source file is among the top-k chunks. page_hit@k: a chunk from the expected file *and* an expected page is in the top-k. MRR@10: mean of 1/rank of the first chunk from the expected file (0 if not in top 10). Faithfulness: answer is supported by the retrieved context. Correctness: answer matches the reference. Both are scored 1-5 by an LLM judge.

**Caveat.** The judge is the same model as the generator, which can bias scores upward (self-preference). Treat answer scores as indicative, not absolute.

## Per-question results

| id | question | first rank | hit@5 | page hit | F | C |
|---|---|---|---|---|---|---|
| q01 | What are the two RAG model variants proposed by Lewis et al., and how do they differ? | 3 | 1 | 1 | 5 | 5 |
| q02 | How was Wikipedia split to build the document index in the original RAG paper, and how many documents resulted? | 2 | 1 | 1 | 5 | 5 |
| q03 | Which pre-trained model is used as the generator in the original RAG architecture? | 1 | 1 | 1 | 5 | 5 |
| q04 | How can a retrieval-augmented model's world knowledge be updated without retraining it? | 1 | 1 | 1 | 5 | 5 |
| q05 | What neural network architecture does CheXNet use? | 1 | 1 | 1 | 5 | 5 |
| q06 | How large is the ChestX-ray14 dataset used to train CheXNet? | 1 | 1 | 1 | 5 | 5 |
| q07 | How did CheXNet's F1 score for pneumonia detection compare to the radiologists? | 1 | 1 | 1 | 5 | 5 |
| q08 | How many convolutional layers does the U-Net network have in total? | 1 | 1 | 1 | 5 | 5 |
| q09 | Which data augmentation technique was key for training a biomedical segmentation network with very few annotated images? | 1 | 1 | 1 | 5 | 5 |
| q10 | Which challenge did U-Net win in 2015, and on which kinds of images? | 1 | 1 | 1 | 5 | 5 |
| q11 | What is MultiMedQA and which new dataset was introduced with it? | 5 | 1 | 0 | 5 | 5 |
| q12 | What accuracy did Flan-PaLM reach on MedQA, and how did it compare to the prior state of the art? | 2 | 1 | 1 | 5 | 5 |
| q13 | What score did Med-PaLM 2 achieve on MedQA, and how much did it improve over Med-PaLM? | 1 | 1 | 1 | 5 | 5 |
| q14 | What novel prompting strategy was introduced with Med-PaLM 2? | 1 | 1 | 0 | 5 | 5 |
| q15 | How did GPT-4 perform on the USMLE relative to the passing score, without specialized prompting? | 1 | 1 | 1 | 5 | 5 |
| q16 | What three prompting techniques are combined in Medprompt, and what MedQA accuracy does it reach with GPT-4? | 1 | 1 | 1 | 5 | 5 |
| q17 | What is the MIRAGE benchmark and how many questions does it contain? | 1 | 1 | 1 | 5 | 5 |
| q18 | By how much does the MedRAG toolkit improve LLM accuracy over chain-of-thought prompting? | 1 | 1 | 1 | 5 | 5 |
| q19 | Into which three paradigms does the RAG survey categorize the evolution of RAG? | 1 | 1 | 1 | 5 | 5 |
| q20 | How large and diverse is the dataset used to develop MedSAM? | 1 | 1 | 1 | 5 | 5 |

## No hard failures; weakest cases

### q11: What is MultiMedQA and which new dataset was introduced with it?

- **Expected:** `med_palm_2022.pdf` p.1 2
- **Retrieved:** med_palm2_2023.pdf:p3; med_palm2_2023.pdf:p1; gpt4_medical_challenge_2023.pdf:p3; med_palm2_2023.pdf:p5; med_palm_2022.pdf:p8
- **Answer:** MultiMedQA is a diverse benchmark for medical question-answering spanning medical exams, consumer health, and medical research [1]. The new dataset introduced alongside it is HealthSearchQA, a consumer medical question-answering dataset curated from seed medical conditions and their associated symptoms [5][4]. Additionally, two adversarial long-form datasets (MultiMedQA 140 and 1066) were introduced for evaluation [1].
- **Reference:** A benchmark combining six existing medical QA datasets (MedQA, MedMCQA, PubMedQA, LiveQA, MedicationQA, MMLU clinical topics) plus the new HealthSearchQA dataset of commonly searched health questions.
- **Faithfulness 5:** All claims in the answer are directly supported by the provided context.
- **Correctness 5:** The answer accurately identifies MultiMedQA as a benchmark and HealthSearchQA as the new dataset introduced alongside it, matching the reference answer's key facts.

### q01: What are the two RAG model variants proposed by Lewis et al., and how do they differ?

- **Expected:** `rag_lewis_2020.pdf` p.3
- **Retrieved:** medrag_benchmark_2024.pdf:p2; medrag_benchmark_2024.pdf:p2; rag_lewis_2020.pdf:p3; rag_survey_gao_2023.pdf:p17; rag_survey_gao_2023.pdf:p12
- **Answer:** The two RAG model variants proposed by Lewis et al. are **RAG-Sequence** and **RAG-Token**, as described in [3].  
- **RAG-Sequence** uses the same retrieved document to generate the entire sequence, treating the document as a single latent variable marginalized via a top-K approximation.  
- **RAG-Token** allows a different latent document to be selected for each target token, enabling the generator to draw content from multiple documents when producing an answer.  

The key difference is that RAG-Sequence relies on a single document throughout, while RAG-Token can switch documents per token, allowing integration of information from several retrieved passages. [3]
- **Reference:** RAG-Sequence uses the same retrieved document to generate the whole output sequence; RAG-Token can use a different retrieved document for each target token.
- **Faithfulness 5:** The answer accurately summarizes the two variants and their difference as described in the context.
- **Correctness 5:** The answer correctly identifies RAG-Sequence and RAG-Token and accurately describes their key difference regarding document usage per token.

### q02: How was Wikipedia split to build the document index in the original RAG paper, and how many documents resulted?

- **Expected:** `rag_lewis_2020.pdf` p.4
- **Retrieved:** rag_survey_gao_2023.pdf:p3; rag_lewis_2020.pdf:p4; rag_lewis_2020.pdf:p8; rag_lewis_2020.pdf:p7; rag_survey_gao_2023.pdf:p8
- **Answer:** According to [2], Wikipedia was split into disjoint 100-word chunks, resulting in a total of 21M documents to build the document index.
- **Reference:** The December 2018 Wikipedia dump was split into disjoint 100-word chunks, giving about 21 million documents, indexed with FAISS.
- **Faithfulness 5:** The answer accurately reflects the context without any unsupported claims.
- **Correctness 5:** The answer exactly matches the reference answer regarding the chunking method and document count.
