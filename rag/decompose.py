"""Query decomposition: split a question that needs several papers into one search per item.

Vector search turns the whole question into one vector. For a comparison such as
"Is A's dataset larger than B's?" that vector lands near whichever paper talks most
about dataset size, and the reranker can then fill every slot from that one paper.
Searching once per item and interleaving the results gives each item room in the top k.

Design: the LLM proposes, the retriever checks. The prompt splits eagerly, which also
splits single-paper questions into aspects ("U-Net challenge", "U-Net image types").
Retriever.search_many keeps a split only if the sub-queries' best hits come from
different papers; otherwise it searches the original question as before.
"""
import json
import re

DECOMPOSE_PROMPT = """You prepare search queries for a search engine over research papers.

If the question asks about two or more different papers, models, methods or datasets
(for example a comparison), write one short, self-contained search query per item.
Otherwise return the question unchanged as the only query. Use at most {max_queries} queries.

Examples:
Question: How do BERT and GPT-2 differ in their pre-training objective?
{{"queries": ["BERT pre-training objective", "GPT-2 pre-training objective"]}}
Question: What optimizer was used to train ResNet?
{{"queries": ["What optimizer was used to train ResNet?"]}}

Question: {question}
Reply with only the JSON object."""


def parse_queries(text: str, question: str, max_queries: int) -> list[str]:
    """Read the LLM's JSON. Anything unusable falls back to the original question."""
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        queries = json.loads(m.group(0))["queries"] if m else []
    except (json.JSONDecodeError, KeyError, TypeError):
        queries = []
    if not isinstance(queries, list):
        queries = []
    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    return queries[:max_queries] or [question]


class Decomposer:
    def __init__(self, generator, max_queries: int = 3):
        self.gen = generator  # reuse the generator's client and model
        self.max_queries = max_queries

    def split(self, question: str) -> list[str]:
        try:
            resp = self.gen.client.chat.completions.create(
                model=self.gen.model, temperature=0, max_tokens=300,
                messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(
                    question=question, max_queries=self.max_queries)}],
                # a short structured task: hidden reasoning only costs time here
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            text = resp.choices[0].message.content
        except Exception as e:  # decomposition is an optimisation, never a reason to fail
            print(f"  decomposition failed, using the original question: {e}")
            return [question]
        return parse_queries(text, question, self.max_queries)


def interleave(result_lists: list[list], k: int) -> list:
    """Round-robin merge: first hit of each list, then the second of each, ...

    The same chunk found by two sub-queries is kept once."""
    merged, seen = [], set()
    for rank in range(max((len(r) for r in result_lists), default=0)):
        for results in result_lists:
            if rank < len(results):
                h = results[rank]
                key = (h.source, h.page, h.text)
                if key not in seen:
                    seen.add(key)
                    merged.append(h)
                    if len(merged) == k:
                        return merged
    return merged
