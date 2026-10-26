"""Stage B, step 2: answer the question using only the retrieved chunks, with citations."""
import os

from openai import OpenAI

from ingest import config
from .retriever import Hit

SYSTEM_PROMPT = """You answer questions about a collection of research papers.
Rules:
- Use ONLY the numbered context passages. Do not use outside knowledge.
- Cite every claim with the passage number in square brackets, e.g. [2] or [1][3].
- If the passages do not contain the answer, reply exactly: "I don't know based on the provided documents."
- Be concise."""


def build_context(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {h.source}, page {h.page})\n{h.text}" for i, h in enumerate(hits, start=1)
    )


class Generator:
    def __init__(self, model: str = config.LLM_MODEL):
        key = os.getenv("NVIDIA_API_KEY")
        if not key:
            raise RuntimeError("NVIDIA_API_KEY is not set (put it in .env or your shell).")
        self.client = OpenAI(base_url=config.LLM_BASE_URL, api_key=key)
        self.model = model

    def answer(self, question: str, hits: list[Hit]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            max_tokens=3000,  # reasoning models spend part of this on hidden thinking
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{build_context(hits)}\n\nQuestion: {question}"},
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            raise RuntimeError(f"{self.model} returned no answer (finish_reason="
                               f"{resp.choices[0].finish_reason}); try a larger max_tokens.")
        return content
