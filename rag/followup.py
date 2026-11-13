"""Follow-up search for chained questions.

"Which model did study A cite but not test, and how was that model adapted?" cannot be
answered with one search: the second half is about a model that only the first passages
name. Query decomposition does not help, because the question never names it.

So when the first answer is a refusal, the LLM reads the question and the passages and may
propose one more search. It costs nothing for questions that are answered the first time.
"""
import json
import re

FOLLOWUP_PROMPT = """A search over research papers returned the passages below, but they do not
answer the whole question.

If the passages name something (a model, method, paper or dataset) that the unanswered part
of the question is about, write ONE short search query that would find the missing information.
If the passages give no such lead, use null.

Question: {question}

Passages:
{passages}

Reply with only a JSON object: {{"query": "<search query>"}} or {{"query": null}}"""

REFUSAL_START = "i don't know"


def is_refusal(answer: str) -> bool:
    return answer.strip().lower().replace("’", "'").startswith(REFUSAL_START)


def parse_query(text: str) -> str | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        query = json.loads(m.group(0)).get("query") if m else None
    except (json.JSONDecodeError, AttributeError):
        return None
    return query.strip() if isinstance(query, str) and query.strip() else None


def propose_query(generator, question: str, hits, chars: int = 700) -> str | None:
    """Ask the LLM for one follow-up search. Any failure means no follow-up."""
    passages = "\n\n".join(f"[{i}] ({h.source}, page {h.page}) {h.text[:chars]}"
                           for i, h in enumerate(hits, start=1))
    try:
        resp = generator.client.chat.completions.create(
            model=generator.model, temperature=0, max_tokens=200, timeout=120,
            messages=[{"role": "user", "content": FOLLOWUP_PROMPT.format(question=question, passages=passages)}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return parse_query(resp.choices[0].message.content)
    except Exception as e:  # an optimisation, never a reason to fail
        print(f"  follow-up search failed, keeping the first answer: {e}")
        return None


def merge(hits, new_hits):
    """Keep the first passages and add the new ones that are not already there."""
    seen = {(h.source, h.page, h.text[:200]) for h in hits}
    return hits + [h for h in new_hits if (h.source, h.page, h.text[:200]) not in seen]


def answer_with_followup(question, hits, retriever, generator, followup_k: int, enabled: bool = True):
    """Answer; on a refusal, try one follow-up search and answer again.

    Returns (answer, hits the answer was written from, follow-up query or None)."""
    answer = generator.answer(question, hits)
    if not enabled or not is_refusal(answer):
        return answer, hits, None
    query = propose_query(generator, question, hits)
    if not query:
        return answer, hits, None
    more = retriever.add_neighbours(retriever.search(query, followup_k))
    hits = merge(hits, more)
    return generator.answer(question, hits), hits, query
