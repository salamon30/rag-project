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

    def _call(self, messages, thinking: bool):
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            max_tokens=config.LLM_MAX_TOKENS if thinking else 1000,
            messages=messages,
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        )
        return resp.choices[0]

    def answer(self, question: str, hits: list[Hit]) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{build_context(hits)}\n\nQuestion: {question}"},
        ]
        choice = self._call(messages, config.LLM_THINKING)
        if not choice.message.content and config.LLM_THINKING:
            # reasoning used the whole budget; retry once without hidden thinking
            choice = self._call(messages, thinking=False)
        content = (choice.message.content or "").strip()
        if not content:
            raise RuntimeError(f"{self.model} returned no answer (finish_reason={choice.finish_reason})")
        return content
