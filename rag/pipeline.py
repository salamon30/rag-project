"""Retrieve-then-generate pipeline used by the CLI, eval and UI."""
from dataclasses import dataclass, field

from ingest import config

from .decompose import Decomposer
from .followup import answer_with_followup
from .generator import Generator
from .retriever import Hit, Retriever


@dataclass
class RagResult:
    question: str
    answer: str
    hits: list[Hit]
    queries: list[str] = field(default_factory=list)  # the searches that were run (several if split)
    followup: str | None = None  # extra search made after a first refusal (chained questions)


class RagPipeline:
    def __init__(self, decompose: bool = config.DECOMPOSE, followup: bool = config.FOLLOWUP):
        self.followup = followup
        self.retriever = Retriever()
        self.generator = Generator()
        self.decomposer = Decomposer(self.generator) if decompose else None

    def ask(self, question: str, k: int = config.TOP_K) -> RagResult:
        queries = self.decomposer.split(question) if self.decomposer else [question]
        hits = self.retriever.search_many(queries, k, question=question)
        hits = hits[: self.retriever.context_size(k)]  # a split question reads more passages
        searches = self.retriever.last_searches
        # the generator always sees the original question, only the search is split
        answer, hits, followup = answer_with_followup(
            question, hits, self.retriever, self.generator, config.FOLLOWUP_K, self.followup)
        return RagResult(question, answer, hits, searches, followup)
