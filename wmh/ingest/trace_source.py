"""Trace source transports for OTel ingestion.

The harness has two separate extension points:

- `TraceSource`: fetch/load raw telemetry from a place (file, Braintrust, Phoenix, generic OTLP).
- `TraceAdapter`: normalize that payload into `Trace` objects.

Keeping transport here lets the OTel GenAI adapter remain a pure schema parser while vendor query
semantics live behind small, testable source implementations.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, cast, runtime_checkable
from urllib.parse import quote, urljoin

import httpx
from pydantic import BaseModel, Field, JsonValue

from wmh.core.types import JsonObject, Trace
from wmh.ingest.adapter import TraceAdapter

OTLP_ENDPOINT_ENV = "WMH_OTLP_QUERY_ENDPOINT"
OTLP_API_KEY_ENV = "WMH_OTLP_API_KEY"

BRAINTRUST_API_URL_ENV = "BRAINTRUST_API_URL"
BRAINTRUST_API_KEY_ENV = "BRAINTRUST_API_KEY"
BRAINTRUST_PROJECT_ENV = "BRAINTRUST_PROJECT"
BRAINTRUST_DEFAULT_API_URL = "https://api.braintrust.dev"

PHOENIX_BASE_URL_ENV = "PHOENIX_BASE_URL"
PHOENIX_API_KEY_ENV = "PHOENIX_API_KEY"
PHOENIX_PROJECT_ENV = "PHOENIX_PROJECT_ID"
PHOENIX_DEFAULT_BASE_URL = "http://localhost:6006"


class TraceSourceKind(StrEnum):
    """Built-in trace source transports."""

    FILE = "file"
    OTLP = "otlp"
    BRAINTRUST = "braintrust"
    PHOENIX = "phoenix"


class TraceSourceConfig(BaseModel):
    """Resolved inputs for loading raw telemetry before adapter normalization."""

    kind: TraceSourceKind
    path: str | None = None
    endpoint: str | None = None
    api_key: str | None = None
    project: str | None = None
    since: str | None = None
    until: str | None = None
    limit: int | None = Field(default=None, gt=0)
    query: str | None = None


@runtime_checkable
class TraceSource(Protocol):
    """Fetch raw telemetry and normalize it with the requested adapter."""

    name: str

    def load(self, config: TraceSourceConfig, adapter: TraceAdapter) -> list[Trace]:
        """Return normalized traces."""
        ...


_SOURCE_ALIASES = {
    "arize": TraceSourceKind.PHOENIX.value,
    "arize-phoenix": TraceSourceKind.PHOENIX.value,
    "arize_phoenix": TraceSourceKind.PHOENIX.value,
    "generic": TraceSourceKind.OTLP.value,
    "generic-otlp": TraceSourceKind.OTLP.value,
    "generic_otlp": TraceSourceKind.OTLP.value,
    "http": TraceSourceKind.OTLP.value,
    "otlp-http": TraceSourceKind.OTLP.value,
    "otlp_http": TraceSourceKind.OTLP.value,
}
_SOURCES: dict[str, TraceSource] = {}


def trace_source_kind(value: str) -> TraceSourceKind:
    """Resolve user-facing source names and aliases to a built-in source kind."""
    normalized = value.strip().lower()
    normalized = _SOURCE_ALIASES.get(normalized, normalized)
    try:
        return TraceSourceKind(normalized)
    except ValueError:
        choices = ", ".join(k.value for k in TraceSourceKind)
        aliases = ", ".join(sorted(_SOURCE_ALIASES))
        raise ValueError(
            f"unknown trace source {value!r}; choose one of: {choices} ({aliases})"
        ) from None


def register_trace_source(source: TraceSource) -> None:
    _SOURCES[source.name] = source


def get_trace_source(kind: TraceSourceKind | str) -> TraceSource:
    key = trace_source_kind(kind).value if isinstance(kind, str) else kind.value
    if key not in _SOURCES:
        raise ValueError(f"no trace source registered for {key!r}; have {list(_SOURCES)}")
    return _SOURCES[key]


def load_traces(config: TraceSourceConfig, adapter: TraceAdapter) -> list[Trace]:
    """Fetch telemetry through a registered source, then normalize with `adapter`."""
    return get_trace_source(config.kind).load(config, adapter)


def _required(value: str | None, message: str) -> str:
    if value:
        return value
    raise ValueError(message)


def _auth_headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _json_response(response: httpx.Response) -> JsonValue:
    response.raise_for_status()
    return cast(JsonValue, response.json())


def _get_json(
    endpoint: str, *, api_key: str | None, params: dict[str, str] | None = None
) -> JsonValue:
    response = httpx.get(
        endpoint,
        headers=_auth_headers(api_key),
        params=params or {},
        timeout=30.0,
    )
    return _json_response(response)


def _post_json(endpoint: str, *, api_key: str | None, body: JsonObject) -> JsonValue:
    response = httpx.post(endpoint, headers=_auth_headers(api_key), json=body, timeout=30.0)
    return _json_response(response)


class FileTraceSource:
    name = TraceSourceKind.FILE.value

    def load(self, config: TraceSourceConfig, adapter: TraceAdapter) -> list[Trace]:
        path = _required(config.path, "file trace source requires a path")
        return adapter.from_file(path)


class GenericOtlpTraceSource:
    name = TraceSourceKind.OTLP.value

    def load(self, config: TraceSourceConfig, adapter: TraceAdapter) -> list[Trace]:
        endpoint = config.endpoint or os.environ.get(OTLP_ENDPOINT_ENV)
        endpoint = _required(
            endpoint,
            f"set ${OTLP_ENDPOINT_ENV} or pass --trace-endpoint for the generic OTLP source",
        )
        api_key = config.api_key or os.environ.get(OTLP_API_KEY_ENV)
        params = _query_params(config)
        payload = _get_json(endpoint, api_key=api_key, params=params)
        return _limit_traces(adapter.from_payload(payload, source=f"otlp:{endpoint}"), config.limit)


class BraintrustTraceSource:
    """Fetch Braintrust logs via BTQL and normalize span rows into OTLP-like spans."""

    name = TraceSourceKind.BRAINTRUST.value

    def load(self, config: TraceSourceConfig, adapter: TraceAdapter) -> list[Trace]:
        base_url = config.endpoint or os.environ.get(BRAINTRUST_API_URL_ENV)
        if base_url is None:
            base_url = BRAINTRUST_DEFAULT_API_URL
        endpoint = (
            base_url if base_url.endswith("/btql") else urljoin(f"{base_url.rstrip('/')}/", "btql")
        )
        api_key = config.api_key or os.environ.get(BRAINTRUST_API_KEY_ENV)
        project = config.project or os.environ.get(BRAINTRUST_PROJECT_ENV)
        query = config.query or _braintrust_default_query(project, config)
        payload = _post_json(endpoint, api_key=api_key, body={"query": query})
        normalized = _payload_or_vendor_spans(payload, _vendor_record_to_span)
        label = project or endpoint
        traces = adapter.from_payload(normalized, source=f"braintrust:{label}")
        return _limit_traces(traces, config.limit)


class PhoenixTraceSource:
    """Fetch Arize Phoenix spans from its REST API and normalize them into OTLP-like spans."""

    name = TraceSourceKind.PHOENIX.value

    def load(self, config: TraceSourceConfig, adapter: TraceAdapter) -> list[Trace]:
        project = config.project or os.environ.get(PHOENIX_PROJECT_ENV)
        endpoint = config.endpoint
        if endpoint is None:
            endpoint = _phoenix_spans_endpoint(
                _required(
                    project,
                    f"set ${PHOENIX_PROJECT_ENV}, pass --trace-project, "
                    "or pass --trace-endpoint",
                )
            )
        api_key = config.api_key or os.environ.get(PHOENIX_API_KEY_ENV)
        params = _phoenix_params(config, include_project=config.endpoint is not None)
        payloads: list[JsonValue] = []
        next_cursor: str | None = None
        fetched_records = 0
        while True:
            page_params = dict(params)
            if next_cursor is not None:
                page_params["cursor"] = next_cursor
            page = _get_json(endpoint, api_key=api_key, params=page_params)
            payloads.append(page)
            records = _extract_records(page)
            if records is not None:
                fetched_records += len(records)
            if (
                config.limit is not None
                and records is not None
                and fetched_records >= config.limit
            ):
                break
            next_cursor = _next_cursor(page)
            if next_cursor is None:
                break
        normalized = _payloads_or_vendor_spans(payloads, _vendor_record_to_span)
        label = project or endpoint
        traces = adapter.from_payload(normalized, source=f"phoenix:{label}")
        return _limit_traces(traces, config.limit)


def _query_params(config: TraceSourceConfig) -> dict[str, str]:
    params: dict[str, str] = {}
    if config.project is not None:
        params["project"] = config.project
    if config.since is not None:
        params["since"] = config.since
    if config.until is not None:
        params["until"] = config.until
    if config.limit is not None:
        params["limit"] = str(config.limit)
    return params


def _braintrust_default_query(project: str | None, config: TraceSourceConfig) -> str:
    project = _required(
        project,
        f"set ${BRAINTRUST_PROJECT_ENV}, pass --trace-project, or pass --trace-query",
    )
    since = _required(
        config.since,
        "Braintrust default queries require --trace-since; pass --trace-query for custom SQL",
    )
    query = f"select * from project_logs('{_btql_quote(project)}', shape => 'traces')"
    clauses = [f"created >= '{_btql_quote(since)}'"]
    if config.until is not None:
        clauses.append(f"created <= '{_btql_quote(config.until)}'")
    if clauses:
        query = f"{query} where {' and '.join(clauses)}"
    if config.limit is not None:
        query = f"{query} limit {config.limit}"
    return query


def _btql_quote(value: str) -> str:
    return value.replace("'", "''")


def _phoenix_spans_endpoint(project: str) -> str:
    base_url = os.environ.get(PHOENIX_BASE_URL_ENV, PHOENIX_DEFAULT_BASE_URL)
    escaped_project = quote(project, safe="")
    return urljoin(f"{base_url.rstrip('/')}/", f"v1/projects/{escaped_project}/spans/otlpv1")


def _phoenix_params(config: TraceSourceConfig, *, include_project: bool) -> dict[str, str]:
    params: dict[str, str] = {}
    project = config.project or os.environ.get(PHOENIX_PROJECT_ENV)
    if include_project and project is not None:
        params["project_id"] = project
    if config.since is not None:
        params["start_time"] = config.since
    if config.until is not None:
        params["end_time"] = config.until
    if config.limit is not None:
        params["limit"] = str(config.limit)
    return params


def _payloads_or_vendor_spans(
    payloads: list[JsonValue], converter: VendorSpanConverter
) -> JsonValue:
    if len(payloads) == 1:
        return _payload_or_vendor_spans(payloads[0], converter)
    spans: list[JsonValue] = []
    for payload in payloads:
        converted = _payload_or_vendor_spans(payload, converter)
        if isinstance(converted, list):
            spans.extend(converted)
        else:
            spans.append(converted)
    return spans


class VendorSpanConverter(Protocol):
    def __call__(self, record: JsonObject) -> JsonValue | None:
        ...


def _payload_or_vendor_spans(payload: JsonValue, converter: VendorSpanConverter) -> JsonValue:
    if _looks_like_otlp(payload):
        return payload
    records = _extract_records(payload)
    if records is None:
        return payload
    spans: list[JsonValue] = []
    for record in records:
        obj = _as_object(record)
        if obj is None:
            continue
        if _looks_like_otlp(obj):
            spans.append(obj)
            continue
        span = converter(obj)
        if span is not None:
            spans.append(span)
    return spans


def _looks_like_otlp(payload: JsonValue) -> bool:
    if isinstance(payload, dict):
        return "resourceSpans" in payload or "traceId" in payload
    if isinstance(payload, list):
        return all(_looks_like_otlp(item) for item in payload if isinstance(item, dict))
    return False


def _extract_records(payload: JsonValue) -> list[JsonValue] | None:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return None
    for key in ("spans", "data", "rows", "records", "results", "items", "traces"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return None


def _next_cursor(payload: JsonValue) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("next_cursor", "nextCursor", "cursor", "next"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _vendor_record_to_span(record: JsonObject) -> JsonValue | None:
    attrs = _record_attributes(record)
    trace_id = _first_str(record, "trace_id", "traceId", "root_span_id") or _first_str(
        attrs, "trace_id", "traceId", "root_span_id"
    )
    if trace_id is None:
        return None
    span_id = (
        _first_str(record, "span_id", "spanId", "id")
        or _first_str(attrs, "span_id", "spanId", "id")
        or trace_id
    )
    span: JsonObject = {
        "traceId": trace_id,
        "spanId": span_id,
        "parentSpanId": _parent_span_id(record, attrs),
        "name": _first_str(record, "name", "span_name")
        or _first_str(attrs, "name", "span_name")
        or "",
        "startTimeUnixNano": _time_nanos(
            _first_value(
                record,
                "startTimeUnixNano",
                "start_time_unix_nano",
                "start_time",
            )
            or _nested_value(record, "metrics", "start")
            or _nested_value(record, "metrics", "start_time")
        ),
        "endTimeUnixNano": _time_nanos(
            _first_value(
                record,
                "endTimeUnixNano",
                "end_time_unix_nano",
                "end_time",
            )
            or _nested_value(record, "metrics", "end")
            or _nested_value(record, "metrics", "end_time")
        ),
        "attributes": _record_attributes_to_otlp(record, attrs),
        "status": _status(record),
    }
    return span


def _record_attributes(record: JsonObject) -> JsonObject:
    for key in ("attributes", "span_attributes", "spanAttributes"):
        value = record.get(key)
        obj = _as_object(value)
        if obj is not None:
            return obj
    return {}


def _record_attributes_to_otlp(record: JsonObject, attrs: JsonObject) -> list[JsonObject]:
    raw = record.get("attributes")
    if isinstance(raw, list):
        normalized: list[JsonObject] = []
        for item in raw:
            obj = _as_object(item)
            if obj is None:
                continue
            key = obj.get("key")
            value = obj.get("value")
            if isinstance(key, str):
                normalized.append({"key": key, "value": _normalize_any_value(value)})
        return normalized
    return _attributes_to_otlp(attrs)


def _as_object(value: JsonValue) -> JsonObject | None:
    return value if isinstance(value, dict) else None


def _first_value(record: JsonObject, *keys: str) -> JsonValue | None:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def _nested_value(record: JsonObject, parent: str, key: str) -> JsonValue | None:
    obj = _as_object(record.get(parent))
    if obj is None:
        return None
    return obj.get(key)


def _first_str(record: JsonObject, *keys: str) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _parent_span_id(record: JsonObject, attrs: JsonObject) -> str:
    parent = _first_str(record, "parent_span_id", "parentSpanId", "parent_id")
    if parent is not None:
        return parent
    parent = _first_str(attrs, "parent_span_id", "parentSpanId", "parent_id")
    if parent is not None:
        return parent
    for source in (record, attrs):
        span_parents = source.get("span_parents")
        if isinstance(span_parents, list):
            for candidate in reversed(span_parents):
                if isinstance(candidate, str) and candidate:
                    return candidate
    return ""


def _time_nanos(value: JsonValue | None) -> int:
    if value is None or isinstance(value, bool):
        return 0
    if isinstance(value, int | float):
        numeric = float(value)
        absolute = abs(numeric)
        if absolute >= 1e17:
            return int(numeric)
        if absolute >= 1e12:
            return int(numeric * 1_000_000)
        if absolute >= 1e9:
            return int(numeric * 1_000_000_000)
        return int(numeric)
    if isinstance(value, str):
        try:
            return _time_nanos(float(value))
        except ValueError:
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return 0
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return int(parsed.timestamp() * 1_000_000_000)
    return 0


def _attributes_to_otlp(attrs: JsonObject) -> list[JsonObject]:
    out: list[JsonObject] = []
    for key, value in attrs.items():
        if value is None:
            continue
        out.append({"key": key, "value": _any_value(value)})
    return out


def _any_value(value: JsonValue) -> JsonObject:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": value}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, list):
        return {"arrayValue": {"values": [_any_value(item) for item in value]}}
    if isinstance(value, dict):
        return {
            "kvlistValue": {
                "values": [
                    {"key": key, "value": _any_value(nested)}
                    for key, nested in value.items()
                    if nested is not None
                ]
            }
        }
    return {"stringValue": ""}


def _normalize_any_value(value: JsonValue | None) -> JsonObject:
    if not isinstance(value, dict):
        return _any_value(value)
    mapping = {
        "string_value": "stringValue",
        "int_value": "intValue",
        "double_value": "doubleValue",
        "bool_value": "boolValue",
        "array_value": "arrayValue",
        "kvlist_value": "kvlistValue",
    }
    out: JsonObject = {}
    for key, nested in value.items():
        mapped = mapping.get(key, key)
        if nested is None:
            continue
        if mapped == "arrayValue" and isinstance(nested, dict):
            values = nested.get("values")
            if isinstance(values, list):
                out[mapped] = {"values": [_normalize_any_value(item) for item in values]}
                continue
        if mapped == "kvlistValue" and isinstance(nested, dict):
            values = nested.get("values")
            if isinstance(values, list):
                out[mapped] = {"values": values}
                continue
        out[mapped] = nested
    return out or {"stringValue": ""}


def _status(record: JsonObject) -> JsonObject:
    existing = _as_object(record.get("status"))
    if existing is not None:
        code = existing.get("code")
        if code in {"ERROR", "STATUS_CODE_ERROR"}:
            return {"code": "STATUS_CODE_ERROR"}
        if code == 2:
            return {"code": 2}
        return existing
    status = _first_str(record, "status_code", "statusCode", "status")
    if status is not None and status.lower() in {"error", "errored", "failed"}:
        return {"code": "STATUS_CODE_ERROR"}
    error = record.get("error")
    if error not in (None, False, ""):
        return {"code": "STATUS_CODE_ERROR"}
    return {"code": "STATUS_CODE_OK"}


def _limit_traces(traces: list[Trace], limit: int | None) -> list[Trace]:
    return traces[:limit] if limit is not None else traces


register_trace_source(FileTraceSource())
register_trace_source(GenericOtlpTraceSource())
register_trace_source(BraintrustTraceSource())
register_trace_source(PhoenixTraceSource())
