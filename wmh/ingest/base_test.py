"""Tests for BaseTraceAdapter + the shared normalizer's OpenInference handling.

Proves a provider adapter that only sets `name` + (optionally) `spans_from_payload` gets correct
file/JSONL loading and span->Trace normalization for free — including OpenInference-vocabulary spans
(`openinference.span.kind`, `tool.name`, `output.value`), which providers like Phoenix/Langfuse use.
"""

from __future__ import annotations

import json
from pathlib import Path

from wmh.core.types import ActionKind
from wmh.ingest.adapter import VendorPull
from wmh.ingest.base import BaseTraceAdapter


def _oi_span(span_id: str, kind: str, attrs: dict, *, start: int, name: str = "") -> dict:
    """An OpenInference-style OTLP span with a FLAT attribute map (provider export shape)."""
    return {
        "traceId": "oitrace0000000000000000000000000",
        "spanId": span_id,
        "name": name,
        "startTimeUnixNano": start,
        "attributes": {"openinference.span.kind": kind, **attrs},
    }


def _otlp(spans: list[dict]) -> dict:
    return {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}


class _DefaultAdapter(BaseTraceAdapter):
    name = "test-default"


def test_base_from_file_normalizes_openinference_spans(tmp_path: Path) -> None:
    # LLM span issues a tool call (OpenInference: tool.name + input.value); TOOL span has output.
    spans = [
        _oi_span(
            "a1",
            "LLM",
            {"tool.name": "get_user", "input.value": '{"id": "u1"}', "input": "look up u1"},
            start=1,
        ),
        _oi_span("t1", "TOOL", {"tool.name": "get_user", "output.value": "found u1"}, start=2),
    ]
    path = tmp_path / "oi.json"
    path.write_text(json.dumps(_otlp(spans)), encoding="utf-8")

    traces = _DefaultAdapter().from_file(str(path))

    assert len(traces) == 1
    assert traces[0].source.startswith("test-default:")
    step = traces[0].steps[0]
    assert step.action.kind == ActionKind.TOOL_CALL
    assert step.action.name == "get_user"
    assert step.action.arguments == {"id": "u1"}
    assert step.observation.content == "found u1"


def test_base_from_file_handles_jsonl_and_skips_corrupt_lines(tmp_path: Path) -> None:
    good = json.dumps(_otlp([_oi_span("a1", "LLM", {"llm.model_name": "gpt"}, start=1)]))
    path = tmp_path / "spans.jsonl"
    path.write_text(f"{good}\n{{truncated\n{good}\n", encoding="utf-8")

    traces = _DefaultAdapter().from_file(str(path))
    # Two valid lines -> two payloads -> same trace id grouped into one trace; corrupt line skipped.
    assert len(traces) == 1


def test_base_vendor_pull_unsupported_is_friendly() -> None:
    try:
        _DefaultAdapter().from_vendor(VendorPull())
    except ValueError as exc:
        assert "does not support live vendor pulls" in str(exc)
        assert "test-default" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_subclass_can_override_spans_from_payload(tmp_path: Path) -> None:
    """A provider whose export is NOT OTLP maps its own shape via spans_from_payload."""
    from wmh.ingest.normalize import SpanRecord

    class CustomAdapter(BaseTraceAdapter):
        name = "custom"

        def spans_from_payload(self, payload):  # noqa: ANN001, ANN202 - test stub
            # payload is {"events": [{"call": "...", "result": "..."}]}
            spans: list[SpanRecord] = []
            events = payload.get("events", []) if isinstance(payload, dict) else []
            for i, ev in enumerate(events):
                spans.append(
                    SpanRecord(
                        trace_id="c" * 32,
                        span_id=f"a{i}",
                        start_nano=i * 2,
                        attributes={
                            "gen_ai.operation.name": "chat",
                            "gen_ai.tool.name": ev["call"],
                            "gen_ai.tool.call.arguments": "{}",
                        },
                    )
                )
                spans.append(
                    SpanRecord(
                        trace_id="c" * 32,
                        span_id=f"t{i}",
                        start_nano=i * 2 + 1,
                        attributes={
                            "gen_ai.operation.name": "execute_tool",
                            "gen_ai.tool.message": ev["result"],
                        },
                    )
                )
            return spans

    path = tmp_path / "custom.json"
    path.write_text(json.dumps({"events": [{"call": "ping", "result": "pong"}]}), encoding="utf-8")

    traces = CustomAdapter().from_file(str(path))
    assert len(traces) == 1
    assert traces[0].steps[0].action.name == "ping"
    assert traces[0].steps[0].observation.content == "pong"
