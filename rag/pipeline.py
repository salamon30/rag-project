"""Retrieve-then-generate pipeline used by the CLI, eval and UI."""
from dataclasses import dataclass

from ingest import config

from .generator import Generator
from .retriever import Hit, Retriever


@dataclass
class RagResult:
    question: str
    answer: str
    hits: list[Hit]


class RagPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()

    def ask(self, question: str, k: int = config.TOP_K) -> RagResult:
        hits = self.retriever.search(question, k)
        return RagResult(question, self.generator.answer(question, hits), hits)
