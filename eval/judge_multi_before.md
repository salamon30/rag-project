# Judge comparison

Date: 2026-11-13  
First judge (same model as the generator): `nvidia/nemotron-3.5-lightning-30b-a3b`  
Second judge (independent): `minimaxai/minimax-m3.1-flash`

The saved answers were scored again by a model from a different vendor. Nothing was regenerated: both judges saw the same question, passages, answer and reference. Refusals are scored by the same code rules for both.

## Correctness (1-5)

| Test set | Answers | First judge | Second judge | Same score | Within 1 point |
|---|---|---|---|---|---|
| _multi_before | 10 | 3.800 | 3.400 | 70% | 90% |

## Faithfulness (1-5)

| Test set | Answers | First judge | Second judge | Same score | Within 1 point |
|---|---|---|---|---|---|
| _multi_before | 10 | 5.000 | 4.600 | 80% | 90% |

## Where the judges disagree on correctness

| Set | Id | Type | First | Second | Second judge's reason |
|---|---|---|---|---|---|
| _multi_before | y02 | multi_hop | 5 | 4 | The answer matches the reference's key comparison (30 images vs. over a million pairs, a difference of more than four orders of magnitude) but omits the 512x512 pixel size, the exact 1,570,263 count, and the 10 imaging modalities. |
| _multi_before | y08 | multi_hop | 4 | 3 | The CheXNet comparison matches the reference, but the Med-PaLM 2 comparison is wrong, citing PubMedQA accuracy and label noise instead of the reference's key facts that physicians preferred Med-PaLM 2's answers on eight of nine clinical-utility axes in a pairwise ranking of 1066 consumer medical questions. |
| _multi_before | y09 | multi_hop | 5 | 3 | The MedRAG LLM list is correct, but the answer misses the key fact that the original RAG paper's generator is BART-large (400M parameters), instead describing RAG-Sequence/RAG-Token as the generator models and mentioning BART only as a baseline. |
