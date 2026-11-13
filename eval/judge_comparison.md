# Judge comparison

Date: 2026-11-13  
First judge (same model as the generator): `nvidia/nemotron-3.5-lightning-30b-a3b`  
Second judge (independent): `minimaxai/minimax-m3.1-flash`

The saved answers were scored again by a model from a different vendor. Nothing was regenerated: both judges saw the same question, passages, answer and reference. Refusals are scored by the same code rules for both.

## Correctness (1-5)

| Test set | Answers | First judge | Second judge | Same score | Within 1 point |
|---|---|---|---|---|---|
| Original | 20 | 5.000 | 4.550 | 60% | 95% |
| Hard | 20 | 4.800 | 4.350 | 75% | 90% |
| Held-out | 20 | 4.500 | 4.350 | 85% | 100% |

## Faithfulness (1-5)

| Test set | Answers | First judge | Second judge | Same score | Within 1 point |
|---|---|---|---|---|---|
| Original | 16 | 5.000 | 5.000 | 100% | 100% |
| Hard | 18 | 5.000 | 4.500 | 72% | 89% |
| Held-out | 20 | 4.850 | 4.900 | 95% | 100% |

## Where the judges disagree on correctness

| Set | Id | Type | First | Second | Second judge's reason |
|---|---|---|---|---|---|
| Original | q01 | single | 5 | 3 | The answer correctly names the two variants but misses the reference's key distinction that RAG-Sequence uses one retrieved document for the whole sequence while RAG-Token can use a different document per token, instead describing differences in retrieval-count effects. |
| Original | q02 | single | 5 | 4 | The answer correctly states the 100-word chunking and the 21M document count, but omits the reference's details about the December 2018 Wikipedia dump and the FAISS index. |
| Original | q06 | single | 5 | 4 | The answer correctly gives the image and patient counts but omits the reference's detail that images are labeled with up to 14 thoracic diseases. |
| Original | q10 | single | 5 | 4 | The answer matches the reference's key facts (challenge, image modalities, large margin) but omits the specific dataset names PhC-U373 and DIC-HeLa. |
| Original | q11 | single | 5 | 4 | The answer correctly identifies MultiMedQA as a diverse medical QA benchmark and HealthSearchQA as the newly introduced dataset, but omits the reference's enumeration of the six existing datasets (MedQA, MedMCQA, PubMedQA, LiveQA, MedicationQA, MMLU clinical topics) that make up the benchmark. |
| Original | q13 | single | 5 | 4 | The answer correctly gives the 86.5% score and the over-19% improvement, but omits the reference's additional fact that the result set a new state of the art. |
| Original | q16 | single | 5 | 4 | The three techniques match the reference exactly, but the answer only says it surpasses 90% rather than giving the specific 90.2% MedQA accuracy. |
| Original | q18 | single | 5 | 4 | The answer correctly states the up-to-18% improvement figure but omits the reference's additional detail that this raises GPT-3.5 and Mixtral to GPT-4-level performance across six LLMs. |
| Hard | h09 | false_premise | 5 | 4 | The answer correctly rejects the premise and identifies the prompting-only methodology, but omits the reference's mention of dynamic few-shot selection and choice-shuffle ensembling as key components. |
| Hard | h12 | multi_hop | 5 | 3 | The answer correctly identifies MedSAM's dataset as larger but gives incorrect figures, omitting the reference's 1,570,263 image-mask pairs and 112,120 ChestX-ray14 images and instead asserting a false ~1,470-image training set. |
| Hard | h13 | multi_hop | 5 | 1 | The reference answer states Medprompt reached 90.2% versus Flan-PaLM's 67.6%, a difference of about 22.6 percentage points, not 13.5. |
| Hard | h15 | multi_hop | 5 | 4 | The answer correctly identifies Wikipedia Dump and the four MedRAG corpora, but omits the reference's specifics that the dump was from December 2018 and split into about 21M 100-word passages. |
| Hard | h16 | multi_hop | 5 | 4 | The answer correctly captures the core distinction—Med-PaLM uses parameter-efficient instruction prompt tuning with exemplars while Medprompt uses prompting only—but omits the reference's specific Medprompt components of dynamic few-shot prompting and choice-shuffle ensembling. |
| Held-out | x03 | single | 5 | 4 | The answer correctly identifies the fine-tuned components and the index-rebuilding cost rationale, but omits the reference's additional point that this step was not necessary for strong performance. |
| Held-out | x05 | single | 5 | 4 | The harmful percentages match the reference exactly and the consensus figures are given, but the answer's primary attribution of 92.9% to Med-PaLM (rather than 92.6% for Med-PaLM with 92.9% being the clinician figure) is a minor misattribution relative to the reference. |
| Held-out | x12 | paraphrase | 5 | 4 | The answer correctly identifies MELD as the blackbox memorization test and reports the key findings (no evidence of memorization for USMLE, evidence for SQuAD 2.0), but omits the reference's description of how MELD works (feeding the first part of an item and comparing the continuation to the real remainder via Levenshtein distance) and the note that MELD has high precision but unknown recall. |
