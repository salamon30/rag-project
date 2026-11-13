# RAG evaluation report

Date: 2026-11-13  
Embeddings: `BAAI/bge-small-en-v1.5` · Generator and judge: `nvidia/nemotron-3.5-lightning-30b-a3b` (generator thinking on with no-thinking fallback, judge thinking off) · k = 5  
Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` over top 50 vector candidates · query decomposition on

## Summary

| Metric | Value |
|---|---|
| questions | 20 |
| hit@1 | 0.857 |
| hit@5 | 1.000 |
| all_sources@5 | 0.929 |
| page_hit@5 | 0.857 |
| MRR@10 | 0.907 |
| answers with [n] citations | 0.929 |
| refusals | 0 |
| refusal recall (unanswerable) | 1.000 |
| faithfulness (1-5) | 5.000 |
| correctness (1-5) | 5.000 |
| unscored (errors) | 0 |

**Metric definitions.** hit@k: expected source file is among the top-k chunks. page_hit@k: a chunk from the expected file *and* an expected page is in the top-k. MRR@10: mean of 1/rank of the first chunk from the expected file (0 if not in top 10). Faithfulness: answer is supported by the retrieved context. Correctness: answer matches the reference. Both are scored 1-5 by an LLM judge.

**Caveat.** The judge is the same model as the generator, which can bias scores upward (self-preference). Treat answer scores as indicative, not absolute.

## By question type

| type | n | page_hit@5 | correctness (1-5) | said don't know |
|---|---|---|---|---|
| unanswerable | 6 | – | 5.000 | 6 |
| false_premise | 4 | 0.750 | 5.000 | 0 |
| multi_hop | 6 | 1.000 | 5.000 | 0 |
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
| h07 | false_premise | Why did the CheXNet authors choose ResNet-50 as their backbone? | 1 | 1 | 1 | 5 | 5 |
| h08 | false_premise | Why did Lewis et al. use GPT-3 as the generator in their RAG models? | 5 | 1 | 1 | 5 | 5 |
| h09 | false_premise | Which medical dataset was GPT-4 fine-tuned on to build Medprompt? | 1 | 1 | 0 | 5 | 5 |
| h10 | false_premise | Why did U-Net require thousands of annotated training images to work well? | 1 | 1 | 1 | 5 | 5 |
| h11 | multi_hop | Compare the network architectures of CheXNet and U-Net. | 1 | 1 | 1 | 5 | 5 |
| h12 | multi_hop | Which training dataset is larger: the one used for CheXNet or the one used for MedSAM? | 1 | 1 | 1 | 5 | 5 |
| h13 | multi_hop | How much higher is Medprompt's MedQA accuracy than the MedQA accuracy Flan-PaLM reached in the original Med-PaLM paper? | 1 | 1 | 1 | 5 | 5 |
| h14 | multi_hop | Which retriever did the original RAG paper use, and which retrievers does the MedRAG benchmark compare? | 1 | 1 | 1 | 5 | 5 |
| h15 | multi_hop | What knowledge source does the original RAG model retrieve from, and which corpora does MedRAG use? | 2 | 1 | 1 | 5 | 5 |
| h16 | multi_hop | How do Med-PaLM and Medprompt differ in how they adapt a general LLM to medicine? | 1 | 1 | 1 | 5 | 5 |
| h17 | paraphrase | A 2015 network that learns to outline cells from only a handful of labelled microscope pictures won which competition? | 1 | 1 | 1 | 5 | 5 |
| h18 | paraphrase | In the study that compared a pneumonia-detecting algorithm with doctors, how many doctors labelled the test images? | 1 | 1 | 1 | 5 | 5 |
| h19 | paraphrase | In the 2020 paper that pairs a sequence-to-sequence model with a searchable Wikipedia index, what do its two kinds of memory refer to? | 1 | 1 | 1 | 5 | 5 |
| h20 | paraphrase | How many evaluation tasks were used to check the general-purpose medical image segmentation model, split into seen and unseen data? | 1 | 1 | 0 | 5 | 5 |

## No hard failures; weakest cases

### h01: How much money, in US dollars, did it cost to train MedSAM?

- **Type:** unanswerable · **Expected:** nothing (unanswerable)
- **Retrieved:** medsam_2023.pdf:p8; medsam_2023.pdf:p9; medsam_2023.pdf:p13; medsam_2023.pdf:p9; medsam_2023.pdf:p7
- **Answer:** I don't know based on the provided documents.
- **Reference:** Not in the documents. The MedSAM paper reports hardware (20 A100 GPUs) and annotation time savings, but no monetary training cost. The system should say it does not know.
- **Faithfulness 5:** The answer correctly states it does not know, which is fully supported by the absence of training cost information in the context.
- **Correctness 5:** Correctly declined an unanswerable question (rule-based score).

### h02: What AUROC did CheXNet achieve when it was validated on chest X-rays from an external hospital?

- **Type:** unanswerable · **Expected:** nothing (unanswerable)
- **Retrieved:** chexnet_2017.pdf:p5; chexnet_2017.pdf:p3; chexnet_2017.pdf:p1; chexnet_2017.pdf:p2; chexnet_2017.pdf:p2
- **Answer:** I don't know based on the provided documents.
- **Reference:** Not in the documents. CheXNet is evaluated only on ChestX-ray14 and a radiologist-annotated test set; no external-hospital validation is reported. The system should say it does not know.
- **Faithfulness 5:** The answer correctly states it does not know, which is consistent with the context.
- **Correctness 5:** Correctly declined an unanswerable question (rule-based score).

### h03: What was the age and sex distribution of the patients in CheXNet's radiologist-annotated test set?

- **Type:** unanswerable · **Expected:** nothing (unanswerable)
- **Retrieved:** chexnet_2017.pdf:p3; chexnet_2017.pdf:p3; chexnet_2017.pdf:p4; chexnet_2017.pdf:p2; chexnet_2017.pdf:p5
- **Answer:** I don't know based on the provided documents.
- **Reference:** Not in the documents. The CheXNet paper does not report patient demographics for the test set. The system should say it does not know.
- **Faithfulness 5:** The answer correctly states it does not know and makes no unsupported claims.
- **Correctness 5:** Correctly declined an unanswerable question (rule-based score).
