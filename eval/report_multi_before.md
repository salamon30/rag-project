# RAG evaluation report

Date: 2026-11-13  
Embeddings: `BAAI/bge-small-en-v1.5` · Generator and judge: `nvidia/nemotron-3.5-lightning-30b-a3b` (generator thinking on with no-thinking fallback, judge thinking off) · k = 5  
Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` over top 50 vector candidates · query decomposition on

## Summary

| Metric | Value |
|---|---|
| questions | 10 |
| hit@1 | 1.000 |
| hit@5 | 1.000 |
| all_sources@5 | 0.800 |
| page_hit@5 | 0.800 |
| MRR@10 | 1.000 |
| answers with [n] citations | 0.625 |
| refusals | 2 |
| faithfulness (1-5) | 5.000 |
| correctness (1-5) | 3.800 |
| unscored (errors) | 0 |

**Metric definitions.** hit@k: expected source file is among the top-k chunks. page_hit@k: a chunk from the expected file *and* an expected page is in the top-k. MRR@10: mean of 1/rank of the first chunk from the expected file (0 if not in top 10). Faithfulness: answer is supported by the retrieved context. Correctness: answer matches the reference. Both are scored 1-5 by an LLM judge.

**Caveat.** The judge is the same model as the generator, which can bias scores upward (self-preference). Treat answer scores as indicative, not absolute.

## Per-question results

| id | type | question | first rank | hit@5 | page hit | F | C |
|---|---|---|---|---|---|---|---|
| y01 | multi_hop | By how many percentage points does GPT-4 with Medprompt beat the best MedQA score that the Med-PaLM 2 paper reports for Med-PaLM 2? | 1 | 1 | 1 | 5 | 5 |
| y02 | multi_hop | How does the amount of training data U-Net had for the EM segmentation challenge compare with the size of the dataset MedSAM was developed on? | 1 | 1 | 1 | 5 | 5 |
| y03 | multi_hop | Compare the optimizer and batch size used to train U-Net with those used to train CheXNet. | 1 | 1 | 1 | 5 | 1 |
| y04 | multi_hop | MedSAM is compared against specialist segmentation models. One of those architectures has its original paper in this collection. How many training images did that original paper have for the EM segmentation challenge? | 1 | 1 | 0 | 5 | 1 |
| y05 | multi_hop | Which general-purpose model is Medprompt applied to, and what average score did the earlier exam study report for that model, zero-shot, on the USMLE Self-Assessment? | 1 | 1 | 1 | 5 | 5 |
| y06 | multi_hop | How many datasets make up the MIRAGE benchmark and how many make up MultiMedQA? | 1 | 1 | 0 | 5 | 2 |
| y07 | multi_hop | By how many percentage points does Med-PaLM 2's best MedQA score exceed the MedQA accuracy that Flan-PaLM reached in the first Med-PaLM paper? | 1 | 1 | 1 | 5 | 5 |
| y08 | multi_hop | Both CheXNet and Med-PaLM 2 were compared with human experts. What was the result of each comparison? | 1 | 1 | 1 | 5 | 4 |
| y09 | multi_hop | Which generator model does the original RAG paper use, and which LLMs does the MedRAG benchmark evaluate? | 1 | 1 | 1 | 5 | 5 |
| y10 | multi_hop | The Med-PaLM 2 paper says it improves on an earlier model by over 19% on MedQA. What technique did that earlier model's own paper introduce to align a large language model to the medical domain? | 1 | 1 | 1 | 5 | 5 |

## Example failures

### y03: Compare the optimizer and batch size used to train U-Net with those used to train CheXNet.

- **Type:** multi_hop · **Expected:** unet_2015.pdf:p4; chexnet_2017.pdf:p2
- **Retrieved:** unet_2015.pdf:p7; unet_2015.pdf:p1; chexnet_2017.pdf:p2; medsam_2023.pdf:p12; chexnet_2017.pdf:p7
- **Answer:** I don't know based on the provided documents.
- **Reference:** U-Net: stochastic gradient descent (Caffe) with a high momentum of 0.99 and a batch of a single image, favouring large input tiles. CheXNet: Adam with standard parameters (beta1 0.9, beta2 0.999) and mini-batches of size 16.
- **Faithfulness 5:** The answer states it does not know, which is consistent with the context and does not make unsupported claims.
- **Correctness 1:** Refusal on an answerable question (rule-based score).

### y04: MedSAM is compared against specialist segmentation models. One of those architectures has its original paper in this collection. How many training images did that original paper have for the EM segmentation challenge?

- **Type:** multi_hop · **Expected:** medsam_2023.pdf:p5; unet_2015.pdf:p6
- **Retrieved:** medsam_2023.pdf:p3; medsam_2023.pdf:p6; unet_2015.pdf:p8; medsam_2023.pdf:p9; medsam_2023.pdf:p12
- **Answer:** I don't know based on the provided documents.
- **Reference:** The architecture is U-Net (MedSAM is compared with specialist U-Net and DeepLabV3+ models). The U-Net paper had 30 training images (512x512 pixels) for the EM segmentation challenge.
- **Faithfulness 5:** The answer states it does not know, which is a valid response that does not contradict the context.
- **Correctness 1:** Refusal on an answerable question (rule-based score).

### y06: How many datasets make up the MIRAGE benchmark and how many make up MultiMedQA?

- **Type:** multi_hop · **Expected:** medrag_benchmark_2024.pdf:p1; med_palm_2022.pdf:p1 2
- **Retrieved:** medrag_benchmark_2024.pdf:p5; med_palm2_2023.pdf:p3; medrag_benchmark_2024.pdf:p6; gpt4_medical_challenge_2023.pdf:p3; medrag_benchmark_2024.pdf:p14
- **Answer:** MIRAGE benchmark consists of 4 datasets: MMLU-Med, MedQA-US, MedMCQA, and PubMedQA* [5]. MultiMedQA comprises 4 datasets (MedQA, PubMedQA, MedMCQA, and MMLU) that constitute a large part of the benchmark [4].
- **Reference:** MIRAGE contains 7,663 questions from five medical QA datasets. MultiMedQA comprises seven datasets: six existing ones (MedQA, MedMCQA, PubMedQA, LiveQA, MedicationQA, MMLU clinical topics) plus the new HealthSearchQA.
- **Faithfulness 5:** All claims in the answer are fully supported by the provided context.
- **Correctness 2:** The answer incorrectly states there are 4 datasets in MIRAGE and 4 datasets in MultiMedQA, contradicting the reference answer which specifies 5 and 7 datasets respectively.
