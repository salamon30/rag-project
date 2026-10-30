"""Retrieve-then-generate pipeline used by the CLI, eval and UI."""
from dataclasses import dataclass, field

from ingest import config

from .decompose import Decomposer
from .generator import Generator
from .retriever import Hit, Retriever


@dataclass
class RagResult:
    question: str
    answer: str
    hits: list[Hit]
    queries: list[str] = field(default_factory=list)  # the searches that were run (several if split)


class RagPipeline:
    def __init__(self, decompose: bool = config.DECOMPOSE):
        self.retriever = Retriever()
        self.generator = Generator()
        self.decomposer = Decomposer(self.generator) if decompose else None

    def ask(self, question: str, k: int = config.TOP_K) -> RagResult:
        queries = self.decomposer.split(question) if self.decomposer else [question]
        hits = self.retriever.search_many(queries, k, question=question)
        # the generator always sees the original question, only the search is split
        return RagResult(question, self.generator.answer(question, hits), hits, self.retriever.last_searches)
