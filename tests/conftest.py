"""Shared fakes so tests run offline: no model downloads, no API key, no PDFs."""
import re
from types import SimpleNamespace

import pytest


class FakeTokenizer:
    """Whitespace tokenizer with the same call signature and offset output as a HF tokenizer."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        return {"offset_mapping": [m.span() for m in re.finditer(r"\S+", text)]}


class FakeChatClient:
    """Stands in for openai.OpenAI. Returns queued replies and records every request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        content, finish = reply if isinstance(reply, tuple) else (reply, "stop")
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason=finish)])


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def fake_client():
    return FakeChatClient
