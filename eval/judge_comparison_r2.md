# Judge comparison

Date: 2026-11-13  
First judge (same model as the generator): `nvidia/nemotron-3.5-lightning-30b-a3b`  
Second judge (independent): `minimaxai/minimax-m3.1-flash`

The saved answers were scored again by a model from a different vendor. Nothing was regenerated: both judges saw the same question, passages, answer and reference. Refusals are scored by the same code rules for both.

## Correctness (1-5)

| Test set | Answers | First judge | Second judge | Same score | Within 1 point |
|---|---|---|---|---|---|
| _multi | 10 | 4.000 | 3.600 | 50% | 80% |
| _r2 | 20 | 4.950 | 4.750 | 80% | 100% |
| _hard_r2 | 20 | 5.000 | 4.750 | 80% | 95% |
| _heldout_r2 | 20 | 5.000 | 4.350 | 50% | 85% |

## Faithfulness (1-5)

| Test set | Answers | First judge | Second judge | Same score | Within 1 point |
|---|---|---|---|---|---|
| _multi | 10 | 4.500 | 4.400 | 60% | 80% |
| _r2 | 20 | 5.000 | 4.900 | 90% | 100% |
| _hard_r2 | 20 | 5.000 | 4.900 | 90% | 100% |
| _heldout_r2 | 20 | 4.850 | 4.900 | 85% | 90% |

## Where the judges disagree on correctness

| Set | Id | Type | First | Second | Second judge's reason |
|---|---|---|---|---|---|
| _multi | y02 | multi_hop | 5 | 4 | The answer correctly states the two dataset sizes and the magnitude of the difference (consistent with the reference's 'more than four orders of magnitude'), but omits the minor details of the 512x512 pixel resolution and the 10 imaging modalities. |
| _multi | y06 | multi_hop | 1 | 3 | The MIRAGE figure of five datasets is correct, but MultiMedQA comprises seven datasets (six existing plus HealthSearchQA), not nine. |
| _multi | y07 | multi_hop | 5 | 2 | The reference answer gives a difference of about 18.9 percentage points, whereas the answer states 19.3, which is numerically incorrect despite being close. |
| _multi | y08 | multi_hop | 5 | 4 | The answer matches the reference's key facts for both comparisons but omits the detail that the Med-PaLM 2 pairwise ranking involved 1066 consumer medical questions. |
| _multi | y10 | multi_hop | 5 | 4 | The answer correctly identifies instruction prompt-tuning as the technique used to align Med-PaLM to the medical domain, but omits the reference's characterization of it as a parameter-efficient approach using a few exemplars applied to Flan-PaLM. |
| _r2 | q11 | single | 5 | 4 | The answer correctly identifies MultiMedQA and the new HealthSearchQA dataset, but omits the reference's enumeration of the six existing datasets (MedQA, MedMCQA, PubMedQA, LiveQA, MedicationQA, and MMLU clinical topics) that make up the benchmark. |
| _r2 | q13 | single | 5 | 4 | The answer correctly states the 86.5% score and the over-19% improvement, but omits the reference's additional fact that this set a new state of the art. |
| _r2 | q17 | single | 5 | 4 | The answer correctly identifies MIRAGE as a medical RAG benchmark with 7,663 questions from five medical QA datasets, but omits the full acronym expansion 'Medical Information Retrieval-Augmented Generation Evaluation' given in the reference. |
| _r2 | q18 | single | 4 | 3 | The answer captures the GPT-3.5/Mixtral improvement to roughly GPT-4-level performance but omits the reference's key figure of up to 18% improvement across six LLMs, substituting a different average estimate. |
| _hard_r2 | h08 | false_premise | 5 | 3 | The answer correctly flags the premise as unsupported but omits the reference's key corrective fact that the RAG generator is BART-large, a 400M-parameter pre-trained seq2seq transformer. |
| _hard_r2 | h09 | false_premise | 5 | 4 | The answer correctly rejects the false premise and identifies prompting-only steering with in-context learning and self-generated chain-of-thought, but omits the reference's mention of choice-shuffle ensembling. |
| _hard_r2 | h15 | multi_hop | 5 | 4 | The answer correctly identifies Wikipedia Dump as the original RAG source and all four MedRAG corpora, but omits the reference's specifics that the dump was from December 2018 and split into about 21M 100-word passages. |
| _hard_r2 | h16 | multi_hop | 5 | 4 | The answer correctly captures the core distinction (Med-PaLM uses parameter-efficient instruction prompt tuning of Flan-PaLM with exemplars, while Medprompt does no fine-tuning and relies on prompting), but omits the reference's mention of choice-shuffle ensembling and the dynamic few-shot component. |
| _heldout_r2 | x04 | single | 5 | 4 | The answer correctly states the key figure of 42.7% but omits the reference's additional details about the 452 pairs, BART's 7.1% rate, and the 17% 'both factual' category. |
| _heldout_r2 | x05 | single | 5 | 4 | The answer matches the reference on all model figures but omits the clinician-generated answers' 92.9% scientific-consensus rate. |
| _heldout_r2 | x06 | single | 5 | 3 | The answer correctly identifies the two datasets (general and health equity) and the 240-question total but omits the key per-dataset counts from the reference (58 for Adversarial General and 182 for Adversarial Health equity). |
| _heldout_r2 | x07 | single | 5 | 4 | The answer matches the reference's key facts (about 70-80% accuracy, 69.75% on Self Assessment media and 79.59% on Sample Exam media with 5-shot prompting) but omits the reference's comparison to text-question accuracy (89.51% and 87.77%). |
| _heldout_r2 | x08 | single | 5 | 4 | The answer correctly identifies the GPT-4 self-generated chain-of-thought steps as the largest contributor (+3.4%) but omits the reference's additional detail that kNN few-shot selection and choice-shuffle ensembling followed at +2.2% each. |
| _heldout_r2 | x09 | single | 5 | 3 | The answer states the reference's key fact (accuracy is highest at the start/end and lowest in the middle), but the fabricated numeric values and their monotonic increase misrepresent the actual decreasing-then-increasing pattern. |
| _heldout_r2 | x12 | paraphrase | 5 | 4 | The answer correctly identifies MELD and the key findings (no evidence of memorization for official USMLE material, evidence for SQuAD 2.0) but omits the specific MELD mechanism of comparing the model's continuation of a partial item to the real remainder via Levenshtein distance, and the note that MELD has high precision but unknown recall. |
| _heldout_r2 | x14 | multi_hop | 5 | 4 | The answer correctly states the optimizer, learning rate, and batch size for both models, but omits the reference's explicit comparison that MedSAM uses a 10x lower learning rate and 10x larger batch, as well as the weight decay and beta parameters. |
| _heldout_r2 | x15 | multi_hop | 5 | 3 | The answer correctly gives the 15 (RAG-Token) and 50 (RAG-Sequence) test-time document counts, but it omits the reference's key facts that RAG was trained with either 5 or 10 retrieved documents and that MedRAG retrieves 32 snippets by default, instead claiming no default is stated. |
| _heldout_r2 | x16 | multi_hop | 5 | 4 | The answer correctly identifies Med-PaLM and its prompt-tuning adaptation from Flan-PaLM 540B, but omits that these models were not publicly available so reported numbers were used. |
