import pytest

from ingest import config
from rag.generator import Generator, build_context
from rag.retriever import Hit

HITS = [Hit("Alpha text", "a.pdf", 2, 0.9), Hit("Beta text", "b.pdf", 7, 0.8)]


def make(client):
    g = Generator.__new__(Generator)  # skip __init__: no API key needed
    g.client, g.model = client, "test-model"
    return g


def test_context_is_numbered_with_source_and_page():
    ctx = build_context(HITS)
    assert ctx.startswith("[1] (source: a.pdf, page 2)\nAlpha text")
    assert "[2] (source: b.pdf, page 7)\nBeta text" in ctx


def test_answer_sends_system_prompt_and_context(fake_client, monkeypatch):
    monkeypatch.setattr(config, "LLM_THINKING", True)
    client = fake_client(["  It is alpha [1].  "])
    assert make(client).answer("What?", HITS) == "It is alpha [1]."
    call = client.calls[0]
    assert call["messages"][0]["role"] == "system"
    assert "Alpha text" in call["messages"][1]["content"]
    assert call["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_empty_thinking_answer_retries_without_thinking(fake_client, monkeypatch):
    monkeypatch.setattr(config, "LLM_THINKING", True)
    client = fake_client([(None, "length"), "Answer [1]."])
    assert make(client).answer("What?", HITS) == "Answer [1]."
    assert [c["extra_body"]["chat_template_kwargs"]["enable_thinking"] for c in client.calls] == [True, False]


def test_empty_answer_after_fallback_raises(fake_client, monkeypatch):
    monkeypatch.setattr(config, "LLM_THINKING", True)
    with pytest.raises(RuntimeError, match="no answer"):
        make(fake_client([None, None])).answer("What?", HITS)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        Generator()
