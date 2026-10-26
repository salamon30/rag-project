# RAG evaluation report

Date: 2026-10-26  
Embeddings: `BAAI/bge-small-en-v1.5` · Generator and judge: `nvidia/nemotron-3.5-lightning-30b-a3b` (generator thinking on with no-thinking fallback, judge thinking off) · k = 5

## Summary

| Metric | Value |
|---|---|
| questions | 20 |
| hit@1 | 0.700 |
| hit@5 | 0.950 |
| page_hit@5 | 0.450 |
| MRR@10 | 0.810 |
| answers with [n] citations | 0.938 |
| refusals | 4 |
| faithfulness (1-5) | 4.800 |
| correctness (1-5) | 4.000 |
| unscored (errors) | 0 |

**Metric definitions.** hit@k: expected source file is among the top-k chunks. page_hit@k: a chunk from the expected file *and* an expected page is in the top-k. MRR@10: mean of 1/rank of the first chunk from the expected file (0 if not in top 10). Faithfulness: answer is supported by the retrieved context. Correctness: answer matches the reference. Both are scored 1-5 by an LLM judge.

**Caveat.** The judge is the same model as the generator, which can bias scores upward (self-preference). Treat answer scores as indicative, not absolute.

## Per-question results

| id | question | first rank | hit@5 | page hit | F | C |
|---|---|---|---|---|---|---|
| q01 | What are the two RAG model variants proposed by Lewis et al., and how do they differ? | 4 | 1 | 0 | 1 | 1 |
| q02 | How was Wikipedia split to build the document index in the original RAG paper, and how many documents resulted? | 2 | 1 | 0 | 5 | 1 |
| q03 | Which pre-trained model is used as the generator in the original RAG architecture? | 2 | 1 | 1 | 5 | 1 |
| q04 | How can a retrieval-augmented model's world knowledge be updated without retraining it? | 3 | 1 | 0 | 5 | 5 |
| q05 | What neural network architecture does CheXNet use? | 1 | 1 | 1 | 5 | 5 |
| q06 | How large is the ChestX-ray14 dataset used to train CheXNet? | 1 | 1 | 1 | 5 | 5 |
| q07 | How did CheXNet's F1 score for pneumonia detection compare to the radiologists? | 1 | 1 | 1 | 5 | 5 |
| q08 | How many convolutional layers does the U-Net network have in total? | 1 | 1 | 0 | 5 | 1 |
| q09 | Which data augmentation technique was key for training a biomedical segmentation network with very few annotated images? | 1 | 1 | 1 | 5 | 5 |
| q10 | Which challenge did U-Net win in 2015, and on which kinds of images? | 1 | 1 | 1 | 5 | 5 |
| q11 | What is MultiMedQA and which new dataset was introduced with it? | 1 | 1 | 0 | 5 | 5 |
| q12 | What accuracy did Flan-PaLM reach on MedQA, and how did it compare to the prior state of the art? | 1 | 1 | 0 | 5 | 5 |
| q13 | What score did Med-PaLM 2 achieve on MedQA, and how much did it improve over Med-PaLM? | 1 | 1 | 1 | 5 | 5 |
| q14 | What novel prompting strategy was introduced with Med-PaLM 2? | 2 | 1 | 0 | 5 | 5 |
| q15 | How did GPT-4 perform on the USMLE relative to the passing score, without specialized prompting? | 1 | 1 | 0 | 5 | 5 |
| q16 | What three prompting techniques are combined in Medprompt, and what MedQA accuracy does it reach with GPT-4? | 1 | 1 | 0 | 5 | 5 |
| q17 | What is the MIRAGE benchmark and how many questions does it contain? | 1 | 1 | 1 | 5 | 5 |
| q18 | By how much does the MedRAG toolkit improve LLM accuracy over chain-of-thought prompting? | 9 | 0 | 0 | 5 | 1 |
| q19 | Into which three paradigms does the RAG survey categorize the evolution of RAG? | 1 | 1 | 1 | 5 | 5 |
| q20 | How large and diverse is the dataset used to develop MedSAM? | 1 | 1 | 0 | 5 | 5 |

## Example failures

### q18: By how much does the MedRAG toolkit improve LLM accuracy over chain-of-thought prompting?

- **Expected:** `medrag_benchmark_2024.pdf` p.1 2
- **Retrieved:** gpt4_medical_challenge_2023.pdf:p2; gpt4_medical_challenge_2023.pdf:p13; med_palm_2022.pdf:p3; gpt4_medical_challenge_2023.pdf:p12; med_palm_2022.pdf:p4
- **Answer:** I don't know based on the provided documents.
- **Reference:** By up to 18% over chain-of-thought prompting across six LLMs, raising GPT-3.5 and Mixtral to GPT-4-level performance.
- **Faithfulness 5:** The answer correctly states it does not know, which is fully supported.
- **Correctness 1:** Refusal on an answerable question (rule-based score).

### q01: What are the two RAG model variants proposed by Lewis et al., and how do they differ?

- **Expected:** `rag_lewis_2020.pdf` p.3
- **Retrieved:** rag_survey_gao_2023.pdf:p12; rag_survey_gao_2023.pdf:p4; rag_survey_gao_2023.pdf:p17; rag_lewis_2020.pdf:p7; rag_survey_gao_2023.pdf:p5
- **Answer:** Based on the provided documents, the two RAG model variants proposed by Lewis et al. are **naive RAG** and **advanced RAG**. 

According to [2], modular RAG "builds upon the foundational principles of advanced and naive rag, illustrating a progression and refinement within the rag family." [4] further clarifies the distinction: the original RAG work compared "rag's dense retriever to a word overlap-based BM25 retriever," noting that "for fever, BM25 performs best, perhaps since fever claims are heavily entity-centric and thus well-suited for word overlap-based retrieval," while "differentiable retrieval improves results on all other tasks, especially for open-domain QA, where it is crucial." 

Thus, the variants differ in their retrieval mechanisms: naive RAG relies on traditional word-overlap methods like BM25, while advanced RAG employs learned/dense retrievers.
- **Reference:** RAG-Sequence uses the same retrieved document to generate the whole output sequence; RAG-Token can use a different retrieved document for each target token.
- **Faithfulness 1:** The answer claims the two variants are naive RAG and advanced RAG, but the reference answer explicitly identifies RAG-Sequence and RAG-Token as the variants proposed by Lewis et al., making the answer's identification incorrect.
- **Correctness 1:** The answer incorrectly identifies the variants as naive and advanced RAG, whereas the reference answer specifies RAG-Sequence and RAG-Token as the two variants proposed by Lewis et al.

### q02: How was Wikipedia split to build the document index in the original RAG paper, and how many documents resulted?

- **Expected:** `rag_lewis_2020.pdf` p.4
- **Retrieved:** rag_survey_gao_2023.pdf:p8; rag_lewis_2020.pdf:p8; rag_lewis_2020.pdf:p7; rag_survey_gao_2023.pdf:p3; rag_survey_gao_2023.pdf:p14
- **Answer:** I don't know based on the provided documents.
- **Reference:** The December 2018 Wikipedia dump was split into disjoint 100-word chunks, giving about 21 million documents, indexed with FAISS.
- **Faithfulness 5:** The answer correctly states it does not know, which is fully supported by the context.
- **Correctness 1:** Refusal on an answerable question (rule-based score).
