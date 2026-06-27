"""Tests for trace-source transports feeding the OTel adapter."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic import JsonValue

from wmh.ingest.otel_genai import OtelGenAIAdapter
from wmh.ingest.trace_source import (
    BRAINTRUST_API_KEY_ENV,
    BRAINTRUST_PROJECT_ENV,
    PHOENIX_API_KEY_ENV,
    PHOENIX_BASE_URL_ENV,
    TraceSourceConfig,
    TraceSourceKind,
    load_traces,
    trace_source_kind,
)


def _span(
    trace_id: str,
    span_id: str,
    name: str,
    attrs: list[dict[str, JsonValue]],
    *,
    start: int = 1,
) -> dict[str, JsonValue]:
    return {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "startTimeUnixNano": start,
        "attributes": attrs,
    }


def _attr(key: str, value: JsonValue) -> dict[str, JsonValue]:
    return {"key": key, "value": {"stringValue": value}}


def _otel_payload() -> list[JsonValue]:
    return [
        _span(
            "trace-1",
            "s1",
            "chat",
            [
                _attr("gen_ai.operation.name", "chat"),
                _attr("gen_ai.prompt", "find docs"),
                _attr("gen_ai.tool.name", "search"),
                _attr("gen_ai.tool.call.arguments", '{"q": "otel"}'),
            ],
            start=1,
        ),
        _span(
            "trace-1",
            "s2",
            "execute_tool search",
            [
                _attr("gen_ai.operation.name", "execute_tool"),
                _attr("gen_ai.tool.name", "search"),
                _attr("gen_ai.tool.message", "3 docs"),
            ],
            start=2,
        ),
    ]


def _response(endpoint: str, payload: JsonValue) -> httpx.Response:
    return httpx.Response(200, json=payload, request=httpx.Request("GET", endpoint))


def test_file_source_loads_exported_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps(span) for span in _otel_payload()), encoding="utf-8")

    traces = load_traces(
        TraceSourceConfig(kind=TraceSourceKind.FILE, path=str(path)),
        OtelGenAIAdapter(),
    )

    assert len(traces) == 1
    assert traces[0].steps[0].action.name == "search"
    assert traces[0].steps[0].observation.content == "3 docs"


def test_generic_otlp_source_fetches_endpoint(monkeypatch) -> None:  # noqa: ANN001
    seen: dict[str, JsonValue] = {}

    def fake_get(
        endpoint: str,
        *,
        headers: dict[str, str],
        params: dict[str, str],
        timeout: float,
    ) -> httpx.Response:
        seen["endpoint"] = endpoint
        seen["headers"] = headers
        seen["params"] = params
        seen["timeout"] = timeout
        return _response(endpoint, _otel_payload())

    import wmh.ingest.trace_source as module

    monkeypatch.setattr(module.httpx, "get", fake_get)
    traces = load_traces(
        TraceSourceConfig(
            kind=TraceSourceKind.OTLP,
            endpoint="https://otel.example/query",
            api_key="secret",
            project="proj",
            since="2026-06-01T00:00:00Z",
            limit=1,
        ),
        OtelGenAIAdapter(),
    )

    assert seen["endpoint"] == "https://otel.example/query"
    assert seen["headers"] == {"Authorization": "Bearer secret"}
    assert seen["params"] == {
        "project": "proj",
        "since": "2026-06-01T00:00:00Z",
        "limit": "1",
    }
    assert traces[0].source == "otlp:https://otel.example/query"


def test_braintrust_source_posts_btql_and_normalizes_rows(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv(BRAINTRUST_API_KEY_ENV, "bt-key")
    monkeypatch.setenv(BRAINTRUST_PROJECT_ENV, "customer-support")
    seen: dict[str, JsonValue] = {}

    def fake_post(
        endpoint: str,
        *,
        headers: dict[str, str],
        json: dict[str, JsonValue],
        timeout: float,
    ) -> httpx.Response:
        seen["endpoint"] = endpoint
        seen["headers"] = headers
        seen["body"] = json
        seen["timeout"] = timeout
        return _response(endpoint, {"rows": _vendor_rows()})

    import wmh.ingest.trace_source as module

    monkeypatch.setattr(module.httpx, "post", fake_post)
    traces = load_traces(
        TraceSourceConfig(kind=TraceSourceKind.BRAINTRUST, limit=4),
        OtelGenAIAdapter(),
    )

    assert seen["endpoint"] == "https://api.braintrust.dev/btql"
    assert seen["headers"] == {"Authorization": "Bearer bt-key"}
    body = seen["body"]
    assert isinstance(body, dict)
    assert "project_logs('customer-support'" in str(body["query"])
    assert traces[0].source == "braintrust:customer-support"
    assert traces[0].steps[0].action.name == "search"
    assert traces[0].steps[0].action.arguments == {"q": "otel"}
    assert traces[0].steps[0].observation.content == "3 docs"


def test_phoenix_source_fetches_spans_endpoint_and_normalizes_rows(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv(PHOENIX_BASE_URL_ENV, "https://phoenix.example")
    monkeypatch.setenv(PHOENIX_API_KEY_ENV, "px-key")
    seen: dict[str, JsonValue] = {}

    def fake_get(
        endpoint: str,
        *,
        headers: dict[str, str],
        params: dict[str, str],
        timeout: float,
    ) -> httpx.Response:
        seen["endpoint"] = endpoint
        seen["headers"] = headers
        seen["params"] = params
        seen["timeout"] = timeout
        return _response(endpoint, {"data": _phoenix_otlp_rows(), "next_cursor": None})

    import wmh.ingest.trace_source as module

    monkeypatch.setattr(module.httpx, "get", fake_get)
    traces = load_traces(
        TraceSourceConfig(
            kind=TraceSourceKind.PHOENIX,
            project="42",
            since="2026-06-01T00:00:00Z",
            until="2026-06-02T00:00:00Z",
            limit=4,
        ),
        OtelGenAIAdapter(),
    )

    assert seen["endpoint"] == "https://phoenix.example/v1/projects/42/spans/otlpv1"
    assert seen["headers"] == {"Authorization": "Bearer px-key"}
    assert seen["params"] == {
        "start_time": "2026-06-01T00:00:00Z",
        "end_time": "2026-06-02T00:00:00Z",
        "limit": "4",
    }
    assert traces[0].source == "phoenix:42"
    assert traces[0].steps[0].observation.content == "3 docs"


def test_trace_source_aliases() -> None:
    assert trace_source_kind("arize") is TraceSourceKind.PHOENIX
    assert trace_source_kind("generic-otlp") is TraceSourceKind.OTLP
    with pytest.raises(ValueError, match="unknown trace source"):
        trace_source_kind("unknown")


def _vendor_rows() -> list[JsonValue]:
    return [
        {
            "trace_id": "vendor-trace",
            "span_id": "a",
            "name": "chat",
            "start_time": "2026-06-01T00:00:00Z",
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.prompt": "find docs",
                "gen_ai.tool.name": "search",
                "gen_ai.tool.call.arguments": '{"q": "otel"}',
            },
        },
        {
            "trace_id": "vendor-trace",
            "span_id": "b",
            "name": "execute_tool search",
            "start_time": "2026-06-01T00:00:01Z",
            "attributes": {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "search",
                "gen_ai.tool.message": "3 docs",
            },
        },
    ]


def _phoenix_otlp_rows() -> list[JsonValue]:
    return [
        {
            "trace_id": "phoenix-trace",
            "span_id": "a",
            "name": "chat",
            "start_time_unix_nano": 1,
            "attributes": [
                {"key": "gen_ai.operation.name", "value": {"string_value": "chat"}},
                {"key": "gen_ai.prompt", "value": {"string_value": "find docs"}},
                {"key": "gen_ai.tool.name", "value": {"string_value": "search"}},
                {
                    "key": "gen_ai.tool.call.arguments",
                    "value": {"string_value": '{"q": "otel"}'},
                },
            ],
            "status": {"code": 1},
        },
        {
            "trace_id": "phoenix-trace",
            "span_id": "b",
            "name": "execute_tool search",
            "start_time_unix_nano": 2,
            "attributes": [
                {"key": "gen_ai.operation.name", "value": {"string_value": "execute_tool"}},
                {"key": "gen_ai.tool.name", "value": {"string_value": "search"}},
                {"key": "gen_ai.tool.message", "value": {"string_value": "3 docs"}},
            ],
            "status": {"code": 1},
        },
    ]
