# RAG evaluation report

Date: 2026-10-28  
Embeddings: `BAAI/bge-small-en-v1.5` · Generator and judge: `nvidia/nemotron-3.5-lightning-30b-a3b` (generator thinking on with no-thinking fallback, judge thinking off) · k = 5  
Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` over top 50 vector candidates

## Summary

| Metric | Value |
|---|---|
| questions | 20 |
| hit@1 | 0.857 |
| hit@5 | 1.000 |
| all_sources@5 | 0.786 |
| page_hit@5 | 0.714 |
| MRR@10 | 0.907 |
| answers with [n] citations | 1.000 |
| refusals | 6 |
| refusal recall (unanswerable) | 1.000 |
| faithfulness (1-5) | 5.000 |
| correctness (1-5) | 3.800 |
| unscored (errors) | 0 |

**Metric definitions.** hit@k: expected source file is among the top-k chunks. page_hit@k: a chunk from the expected file *and* an expected page is in the top-k. MRR@10: mean of 1/rank of the first chunk from the expected file (0 if not in top 10). Faithfulness: answer is supported by the retrieved context. Correctness: answer matches the reference. Both are scored 1-5 by an LLM judge.

**Caveat.** The judge is the same model as the generator, which can bias scores upward (self-preference). Treat answer scores as indicative, not absolute.

## By question type

| type | n | page_hit@5 | correctness (1-5) | said don't know |
|---|---|---|---|---|
| unanswerable | 6 | – | 5.000 | 6 |
| false_premise | 4 | 0.750 | 2.000 | 3 |
| multi_hop | 6 | 0.667 | 3.000 | 3 |
| paraphrase | 4 | 0.750 | 5.000 | 0 |

Unanswerable questions should all be declined; every other type should be answered.

## Per-question results

| id | type | question | first rank | hit@5 | page hit | F | C |
|---|---|---|---|---|---|---|---|
| h01 | unanswerable | How much money, in US dollars, did it cost to train MedSAM? | – | – | – | 5 | 5 |
| h02 | unanswerable | What AUROC did CheXNet achieve when it was validated on chest X-rays from an external hospital? | – | – | – | 5 | 5 |
| h03 | unanswerable | What was the age and sex distribution of the patients in CheXNet's radiologist-annotated test set? | – | – | – | 5 | 5 |
| h04 | unanswerable | Has Med-PaLM 2 received FDA clearance for clinical use? | – | – | – | 5 | 5 |
| h05 | unanswerable | How many parameters does the PaLM 2 base model behind Med-PaLM 2 have? | – | – | – | 5 | 5 |
| h06 | unanswerable | What is the inference latency, in milliseconds per query, of the RAG-Token model? | – | – | – | 5 | 5 |
| h07 | false_premise | Why did the CheXNet authors choose ResNet-50 as their backbone? | 1 | 1 | 1 | 5 | 1 |
| h08 | false_premise | Why did Lewis et al. use GPT-3 as the generator in their RAG models? | 5 | 1 | 1 | 5 | 1 |
| h09 | false_premise | Which medical dataset was GPT-4 fine-tuned on to build Medprompt? | 1 | 1 | 0 | 5 | 5 |
| h10 | false_premise | Why did U-Net require thousands of annotated training images to work well? | 1 | 1 | 1 | 5 | 1 |
| h11 | multi_hop | Compare the network architectures of CheXNet and U-Net. | 1 | 1 | 1 | 5 | 5 |
| h12 | multi_hop | Which training dataset is larger: the one used for CheXNet or the one used for MedSAM? | 1 | 1 | 0 | 5 | 1 |
| h13 | multi_hop | How much higher is Medprompt's MedQA accuracy than the MedQA accuracy Flan-PaLM reached in the original Med-PaLM paper? | 2 | 1 | 0 | 5 | 1 |
| h14 | multi_hop | Which retriever did the original RAG paper use, and which retrievers does the MedRAG benchmark compare? | 1 | 1 | 1 | 5 | 5 |
| h15 | multi_hop | What knowledge source does the original RAG model retrieve from, and which corpora does MedRAG use? | 1 | 1 | 1 | 5 | 1 |
| h16 | multi_hop | How do Med-PaLM and Medprompt differ in how they adapt a general LLM to medicine? | 1 | 1 | 1 | 5 | 5 |
| h17 | paraphrase | A 2015 network that learns to outline cells from only a handful of labelled microscope pictures won which competition? | 1 | 1 | 1 | 5 | 5 |
| h18 | paraphrase | In the study that compared a pneumonia-detecting algorithm with doctors, how many doctors labelled the test images? | 1 | 1 | 1 | 5 | 5 |
| h19 | paraphrase | In the 2020 paper that pairs a sequence-to-sequence model with a searchable Wikipedia index, what do its two kinds of memory refer to? | 1 | 1 | 1 | 5 | 5 |
| h20 | paraphrase | How many evaluation tasks were used to check the general-purpose medical image segmentation model, split into seen and unseen data? | 1 | 1 | 0 | 5 | 5 |

## Example failures

### h08: Why did Lewis et al. use GPT-3 as the generator in their RAG models?

- **Type:** false_premise · **Expected:** rag_lewis_2020.pdf:p2 3
- **Retrieved:** medrag_benchmark_2024.pdf:p2; medrag_benchmark_2024.pdf:p2; rag_survey_gao_2023.pdf:p17; medrag_benchmark_2024.pdf:p1; rag_lewis_2020.pdf:p3
- **Answer:** I don't know based on the provided documents.
- **Reference:** The premise is wrong: the RAG generator is BART-large, a pre-trained seq2seq transformer with 400M parameters, not GPT-3.
- **Faithfulness 5:** The answer correctly states it does not know, which is fully supported by the context.
- **Correctness 1:** Refusal on an answerable question (rule-based score).

### h13: How much higher is Medprompt's MedQA accuracy than the MedQA accuracy Flan-PaLM reached in the original Med-PaLM paper?

- **Type:** multi_hop · **Expected:** medprompt_2023.pdf:p1 2 10; med_palm_2022.pdf:p1 3 13
- **Retrieved:** med_palm2_2023.pdf:p8; med_palm_2022.pdf:p32; med_palm_2022.pdf:p23; med_palm2_2023.pdf:p2; med_palm2_2023.pdf:p8
- **Answer:** I don't know based on the provided documents.
- **Reference:** Medprompt reaches 90.2% versus Flan-PaLM's 67.6%, about 22.6 percentage points higher.
- **Faithfulness 5:** The answer correctly states it does not know, which is fully supported and consistent with the evaluation guidelines.
- **Correctness 1:** Refusal on an answerable question (rule-based score).

### h07: Why did the CheXNet authors choose ResNet-50 as their backbone?

- **Type:** false_premise · **Expected:** chexnet_2017.pdf:p1 2
- **Retrieved:** chexnet_2017.pdf:p3; chexnet_2017.pdf:p3; chexnet_2017.pdf:p2; chexnet_2017.pdf:p2; chexnet_2017.pdf:p4
- **Answer:** I don't know based on the provided documents.
- **Reference:** The premise is wrong: CheXNet does not use ResNet-50. It is a 121-layer Dense Convolutional Network (DenseNet).
- **Faithfulness 5:** The answer correctly states it does not know, which is fully supported by the context.
- **Correctness 1:** Refusal on an answerable question (rule-based score).

### h10: Why did U-Net require thousands of annotated training images to work well?

- **Type:** false_premise · **Expected:** unet_2015.pdf:p1 3
- **Retrieved:** unet_2015.pdf:p1; unet_2015.pdf:p8; unet_2015.pdf:p7; unet_2015.pdf:p6; medsam_2023.pdf:p13
- **Answer:** I don't know based on the provided documents.
- **Reference:** The premise is wrong: U-Net is designed to train from very few annotated images, relying on strong data augmentation (especially elastic deformations).
- **Faithfulness 5:** The answer correctly states it does not know, which is consistent with the evaluation guidelines.
- **Correctness 1:** Refusal on an answerable question (rule-based score).

### h12: Which training dataset is larger: the one used for CheXNet or the one used for MedSAM?

- **Type:** multi_hop · **Expected:** chexnet_2017.pdf:p1 2; medsam_2023.pdf:p1 2
- **Retrieved:** medsam_2023.pdf:p8; medsam_2023.pdf:p9; medsam_2023.pdf:p9; medsam_2023.pdf:p3; medsam_2023.pdf:p11
- **Answer:** I don't know based on the provided documents.
- **Reference:** MedSAM's: 1,570,263 image-mask pairs across 10 modalities, versus ChestX-ray14's 112,120 frontal chest X-rays used for CheXNet.
- **Faithfulness 5:** The answer correctly states it does not know, which is fully supported by the context.
- **Correctness 1:** Refusal on an answerable question (rule-based score).
