"""Tests for trace-adapter registration and the OTel GenAI file parser."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from wmh.core.types import ActionKind
from wmh.ingest import get_adapter
from wmh.ingest.adapter import VendorPull
from wmh.ingest.otel_genai import VENDOR_ENDPOINT_ENV, OtelGenAIAdapter

_TESTDATA = Path(__file__).parent / "testdata"


def test_default_otel_adapter_is_registered_on_import() -> None:
    # DESIGN/README claim the OTel adapter ships registered; importing wmh.ingest must suffice.
    assert get_adapter("otel-genai").name == "otel-genai"


def test_from_file_parses_otlp_json_into_one_trace() -> None:
    traces = OtelGenAIAdapter().from_file(str(_TESTDATA / "sample_otlp.json"))

    assert len(traces) == 1
    trace = traces[0]
    assert trace.trace_id == "0af7651916cd43dd8448eb211c80319c"
    assert trace.source.endswith("sample_otlp.json")

    # 3 spans (llm+tool_call, execute_tool, final llm) -> 2 steps: paired tool call + final message.
    assert len(trace.steps) == 2

    call_step = trace.steps[0]
    assert call_step.action.kind == ActionKind.TOOL_CALL
    assert call_step.action.name == "get_weather"
    assert call_step.action.arguments == {"city": "Paris"}
    assert call_step.observation.content == "18C and sunny"
    assert call_step.observation.is_error is False
    # The originating prompt is carried onto every step's `task`.
    assert call_step.task == "What is the weather in Paris?"
    # Both the LLM span and the tool span are recorded as provenance.
    assert call_step.raw_span_ids == ["b7ad6b7169203331", "c8be7c8270314442"]

    final_step = trace.steps[1]
    assert final_step.action.kind == ActionKind.MESSAGE
    assert final_step.action.content == "It is 18C and sunny in Paris."
    # No following tool span -> empty observation.
    assert final_step.observation.content == ""


def test_from_file_parses_jsonl_with_multiple_traces() -> None:
    traces = OtelGenAIAdapter().from_file(str(_TESTDATA / "sample_spans.jsonl"))

    assert [t.trace_id for t in traces] == [
        "aaaa0000aaaa0000aaaa0000aaaa0000",
        "bbbb1111bbbb1111bbbb1111bbbb1111",
    ]

    # Trace 1: paired tool call whose execution errored.
    first = traces[0]
    assert len(first.steps) == 1
    assert first.steps[0].action.name == "rm"
    assert first.steps[0].action.arguments == {"path": "/tmp/x"}
    assert first.steps[0].observation.content == "permission denied"
    assert first.steps[0].observation.is_error is True

    # Trace 2: a lone execute_tool span with no preceding LLM span becomes a self-contained step.
    second = traces[1]
    assert len(second.steps) == 1
    assert second.steps[0].action.kind == ActionKind.TOOL_CALL
    assert second.steps[0].action.name == "search"
    assert second.steps[0].action.arguments == {"q": "otel"}
    assert second.steps[0].observation.content == "3 results"


def test_from_file_skips_corrupt_jsonl_lines(tmp_path: Path) -> None:
    good = (
        '{"traceId": "cccc", "spanId": "01", "name": "chat", '
        '"attributes": [{"key": "gen_ai.completion", "value": {"stringValue": "hi"}}]}'
    )
    path = tmp_path / "partial.jsonl"
    # A truncated middle line (crashed exporter) must not abort the whole ingest.
    path.write_text(f"{good}\n{{truncated\n{good}\n", encoding="utf-8")

    traces = OtelGenAIAdapter().from_file(str(path))

    assert len(traces) == 1
    assert traces[0].trace_id == "cccc"
    assert len(traces[0].steps) == 2  # both valid lines parsed; the corrupt one skipped


def test_from_vendor_without_endpoint_raises_friendly_error() -> None:
    saved = os.environ.pop(VENDOR_ENDPOINT_ENV, None)
    try:
        with pytest.raises(ValueError, match=VENDOR_ENDPOINT_ENV):
            OtelGenAIAdapter().from_vendor(VendorPull(project="demo"))
    finally:
        if saved is not None:
            os.environ[VENDOR_ENDPOINT_ENV] = saved
