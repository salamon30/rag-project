"""LLM-as-a-judge: scores an answer for faithfulness (vs context) and correctness (vs reference)."""
import json
import re

from rag.generator import Generator

JUDGE_PROMPT = """You are a strict evaluator of a retrieval-augmented QA system.

Score the ANSWER on two 1-5 scales.

FAITHFULNESS: is every claim in the ANSWER supported by the CONTEXT?
5 = fully supported; 4 = minor unsupported detail; 3 = some unsupported claims;
2 = mostly unsupported; 1 = contradicts or ignores the context.
An answer that correctly says it does not know scores 5 for faithfulness.

CORRECTNESS: does the ANSWER match the REFERENCE ANSWER?
5 = same key facts; 4 = mostly correct, small omission; 3 = partially correct;
2 = mostly wrong or missing key facts; 1 = wrong or no answer.

QUESTION: {question}

REFERENCE ANSWER: {reference}

CONTEXT:
{context}

ANSWER: {answer}

Reply with ONLY a JSON object, no other text:
{{"faithfulness": <1-5>, "faithfulness_reason": "<one sentence>", "correctness": <1-5>, "correctness_reason": "<one sentence>"}}"""


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in judge output: {text[:200]!r}")
    d = json.loads(m.group(0))
    for k in ("faithfulness", "correctness"):
        d[k] = int(d[k])
        if not 1 <= d[k] <= 5:
            raise ValueError(f"{k} out of range: {d[k]}")
    return d


class Judge:
    def __init__(self, generator: Generator):
        self.gen = generator  # reuse the same client and model

    def score(self, question, reference, context, answer, retries=2) -> dict:
        prompt = JUDGE_PROMPT.format(question=question, reference=reference,
                                     context=context, answer=answer)
        last = None
        for _ in range(retries + 1):
            try:
                resp = self.gen.client.chat.completions.create(
                    model=self.gen.model, temperature=0, max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}],
                    # Scoring is a short rubric task. With thinking on, the model spent its
                    # whole token budget on hidden reasoning and returned no JSON.
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                choice = resp.choices[0]
                if not choice.message.content:
                    raise ValueError(f"empty judge output (finish_reason={choice.finish_reason})")
                return _parse(choice.message.content)
            except Exception as e:
                last = e
        return {"faithfulness": None, "correctness": None,
                "faithfulness_reason": f"judge error: {last}", "correctness_reason": ""}
