"""Unit tests for the shared OpenAI-shaped request mapping."""

from __future__ import annotations

from typing import cast

from wmh.providers import _openai_common
from wmh.providers.base import Message


def test_to_messages_prepends_system_when_present() -> None:
    out = _openai_common.to_messages(
        "sys", [Message(role="user", content="a"), Message(role="assistant", content="b")]
    )
    assert out == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]


def test_to_messages_omits_empty_system() -> None:
    out = _openai_common.to_messages("", [Message(role="user", content="a")])
    assert out == [{"role": "user", "content": "a"}]


def test_complete_handles_missing_usage() -> None:
    class _Choice:
        def __init__(self) -> None:
            self.message = type("M", (), {"content": "hi"})()

    class _Resp:
        choices = [_Choice()]
        usage = None

    class _Chat:
        def create(self, **kwargs: object) -> _Resp:
            return _Resp()

    chat = cast("_openai_common._ChatCompletions", _Chat())
    completion = _openai_common.complete(chat, "m", "", [Message(role="user", content="x")], 8)
    assert completion.text == "hi"
    assert completion.usage.input_tokens == 0
    assert completion.usage.output_tokens == 0
