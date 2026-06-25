"""Official OpenTelemetry GenAI semantic-convention adapter.

Maps `gen_ai.*` spans into our normalized schema:
  - LLM / agent spans  -> Action (tool_call from `gen_ai.tool.*`, else message from `gen_ai.prompt`
    / `gen_ai.completion`)
  - tool / execution spans -> Observation (tool output attribute / span status)
  - one trace_id worth of spans, ordered by start time -> one `Trace`

This is the reference adapter; it also documents the generic shape other adapters target.

Span classification follows the OTel GenAI semantic conventions (`gen_ai.operation.name`):
  - LLM spans:  `chat`, `text_completion`, `invoke_agent`, `generate_content`
  - tool spans: `execute_tool`
When `gen_ai.operation.name` is absent we fall back to the conventional span-name prefix
(`execute_tool ...`) and to attribute-presence heuristics.

Action/Observation pairing: each LLM Action is paired with the next tool Observation in start-time
order. An LLM message with no following tool span (e.g. the final answer) becomes a Step with an
empty Observation; a tool span with no preceding LLM span becomes a self-contained tool_call Step
from its own `gen_ai.tool.*` attributes.

Supported file formats:
  - OTLP-JSON: a single object with `resourceSpans` -> `scopeSpans` -> `spans`
  - JSON array of the above, or of bare span objects
  - JSONL: one OTLP-JSON object or one bare span object per line
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
from pydantic import BaseModel, Field, JsonValue

from wmh.core.types import Action, ActionKind, JsonObject, Observation, Step, Trace
from wmh.ingest.adapter import VendorPull, register_adapter

# `gen_ai.operation.name` values, per the OTel GenAI semantic conventions.
_LLM_OPS = frozenset({"chat", "text_completion", "invoke_agent", "generate_content"})
_TOOL_OPS = frozenset({"execute_tool"})

# Attribute keys carrying a tool call's serialized arguments, in priority order.
_TOOL_ARG_KEYS = (
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.arguments",
    "gen_ai.tool.input",
    "gen_ai.request.arguments",
)
# Attribute keys carrying a tool execution's output, in priority order.
_TOOL_OUTPUT_KEYS = (
    "gen_ai.tool.message",
    "gen_ai.tool.output",
    "gen_ai.tool.call.result",
    "gen_ai.tool.result",
    "gen_ai.completion",
    "output",
)

# Env vars the (placeholder) vendor pull reads. The real query semantics are vendor-specific; see
# the TODO in `from_vendor`.
VENDOR_ENDPOINT_ENV = "WMH_OTLP_QUERY_ENDPOINT"
VENDOR_API_KEY_ENV = "WMH_OTLP_API_KEY"


class _ParsedSpan(BaseModel):
    """A flattened OTLP span with its attributes already decoded to plain JSON values."""

    trace_id: str
    span_id: str = ""
    parent_span_id: str = ""
    name: str = ""
    start_nano: int = 0
    end_nano: int = 0
    attributes: JsonObject = Field(default_factory=dict)
    status_error: bool = False


# --- OTLP AnyValue / attribute decoding -------------------------------------------------------


def _any_value(value: JsonValue) -> JsonValue:
    """Decode an OTLP `AnyValue` (`{"stringValue": ...}` etc.) to a plain JSON value."""
    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return _to_int(value["intValue"])
    if "doubleValue" in value:
        return value["doubleValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "arrayValue" in value:
        arr = value["arrayValue"]
        values = arr.get("values") if isinstance(arr, dict) else None
        return [_any_value(v) for v in values] if isinstance(values, list) else []
    if "kvlistValue" in value:
        kv = value["kvlistValue"]
        values = kv.get("values") if isinstance(kv, dict) else None
        return _attrs_to_dict(values) if isinstance(values, list) else {}
    return value


def _attrs_to_dict(attrs: JsonValue) -> JsonObject:
    """Turn an OTLP attribute list (`[{"key": ..., "value": <AnyValue>}, ...]`) into a dict."""
    out: JsonObject = {}
    if not isinstance(attrs, list):
        return out
    for attr in attrs:
        if isinstance(attr, dict):
            key = attr.get("key")
            if isinstance(key, str):
                out[key] = _any_value(attr.get("value"))
    return out


def _to_int(value: JsonValue) -> int:
    if isinstance(value, bool):  # bool is an int subclass; treat as non-numeric here.
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _as_text(value: JsonValue) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _as_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


# --- span collection --------------------------------------------------------------------------


def _parse_span(raw: JsonValue) -> _ParsedSpan | None:
    if not isinstance(raw, dict):
        return None
    trace_id = raw.get("traceId")
    if not isinstance(trace_id, str) or not trace_id:
        return None
    status = raw.get("status")
    status_error = False
    if isinstance(status, dict):
        code = status.get("code")
        status_error = code in (2, "STATUS_CODE_ERROR")
    return _ParsedSpan(
        trace_id=trace_id,
        span_id=_as_str(raw.get("spanId")),
        parent_span_id=_as_str(raw.get("parentSpanId")),
        name=_as_str(raw.get("name")),
        start_nano=_to_int(raw.get("startTimeUnixNano")),
        end_nano=_to_int(raw.get("endTimeUnixNano")),
        attributes=_attrs_to_dict(raw.get("attributes")),
        status_error=status_error,
    )


def _collect_spans(obj: JsonValue) -> list[_ParsedSpan]:
    """Walk an OTLP-JSON payload, a list of payloads/spans, or a bare span into `_ParsedSpan`s."""
    spans: list[_ParsedSpan] = []
    if isinstance(obj, list):
        for item in obj:
            spans.extend(_collect_spans(item))
        return spans
    if not isinstance(obj, dict):
        return spans
    if "resourceSpans" in obj:
        resource_spans = obj["resourceSpans"]
        if isinstance(resource_spans, list):
            for resource_span in resource_spans:
                spans.extend(_spans_in_resource(resource_span))
        return spans
    parsed = _parse_span(obj)
    if parsed is not None:
        spans.append(parsed)
    return spans


def _spans_in_resource(resource_span: JsonValue) -> list[_ParsedSpan]:
    spans: list[_ParsedSpan] = []
    if not isinstance(resource_span, dict):
        return spans
    scope_spans = resource_span.get("scopeSpans")
    if not isinstance(scope_spans, list):
        return spans
    for scope_span in scope_spans:
        if not isinstance(scope_span, dict):
            continue
        raw_spans = scope_span.get("spans")
        if not isinstance(raw_spans, list):
            continue
        for raw in raw_spans:
            parsed = _parse_span(raw)
            if parsed is not None:
                spans.append(parsed)
    return spans


# --- classification + mapping -----------------------------------------------------------------


def _operation(span: _ParsedSpan) -> str:
    op = span.attributes.get("gen_ai.operation.name")
    return op if isinstance(op, str) else ""


def _is_tool_span(span: _ParsedSpan) -> bool:
    op = _operation(span)
    if op in _TOOL_OPS:
        return True
    if op in _LLM_OPS:
        return False
    return span.name.startswith("execute_tool")


def _is_llm_span(span: _ParsedSpan) -> bool:
    op = _operation(span)
    if op in _LLM_OPS:
        return True
    if op in _TOOL_OPS:
        return False
    attrs = span.attributes
    return any(
        attrs.get(key) is not None
        for key in ("gen_ai.request.model", "gen_ai.completion", "gen_ai.prompt")
    )


def _tool_args(attrs: JsonObject) -> JsonObject:
    for key in _TOOL_ARG_KEYS:
        raw = attrs.get(key)
        if raw is None:
            continue
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed: JsonValue = json.loads(raw)
            except json.JSONDecodeError:
                return {"value": raw}
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        return {"value": raw}
    return {}


def _action_from_llm_span(span: _ParsedSpan) -> Action:
    attrs = span.attributes
    tool_name = attrs.get("gen_ai.tool.name")
    if isinstance(tool_name, str) and tool_name:
        return Action(kind=ActionKind.TOOL_CALL, name=tool_name, arguments=_tool_args(attrs))
    completion = attrs.get("gen_ai.completion")
    content = attrs.get("gen_ai.prompt") if completion is None else completion
    return Action(kind=ActionKind.MESSAGE, content=_as_text(content))


def _tool_call_action_from_tool_span(span: _ParsedSpan) -> Action:
    name = span.attributes.get("gen_ai.tool.name")
    return Action(
        kind=ActionKind.TOOL_CALL,
        name=name if isinstance(name, str) and name else None,
        arguments=_tool_args(span.attributes),
    )


def _observation_from_tool_span(span: _ParsedSpan) -> Observation:
    content = ""
    for key in _TOOL_OUTPUT_KEYS:
        value = span.attributes.get(key)
        if value is not None:
            content = _as_text(value)
            break
    return Observation(content=content, is_error=span.status_error)


def _trace_task(spans: list[_ParsedSpan]) -> str | None:
    for span in spans:
        prompt = span.attributes.get("gen_ai.prompt")
        if prompt is not None:
            return _as_text(prompt)
    return None


def _build_steps(spans: list[_ParsedSpan]) -> list[Step]:
    """Pair ordered Action spans with their following Observation spans into Steps."""
    task = _trace_task(spans)
    steps: list[Step] = []
    pending: Action | None = None
    pending_ids: list[str] = []

    def flush(action: Action, observation: Observation, span_ids: list[str]) -> None:
        steps.append(Step(action=action, observation=observation, task=task, raw_span_ids=span_ids))

    for span in spans:
        if _is_tool_span(span):
            observation = _observation_from_tool_span(span)
            if pending is None:
                action = _tool_call_action_from_tool_span(span)
                flush(action, observation, [span.span_id])
            else:
                if pending.kind == ActionKind.TOOL_CALL and not pending.arguments:
                    pending.arguments = _tool_args(span.attributes)
                if pending.kind == ActionKind.TOOL_CALL and pending.name is None:
                    pending.name = _tool_call_action_from_tool_span(span).name
                flush(pending, observation, [*pending_ids, span.span_id])
            pending, pending_ids = None, []
        elif _is_llm_span(span):
            if pending is not None:
                flush(pending, Observation(content=""), pending_ids)
            pending, pending_ids = _action_from_llm_span(span), [span.span_id]
        # Non-GenAI spans are ignored.

    if pending is not None:
        flush(pending, Observation(content=""), pending_ids)
    return steps


def _spans_to_traces(spans: list[_ParsedSpan], source: str) -> list[Trace]:
    by_trace: dict[str, list[_ParsedSpan]] = {}
    for span in spans:
        by_trace.setdefault(span.trace_id, []).append(span)
    traces: list[Trace] = []
    for group in by_trace.values():
        group.sort(key=lambda s: (s.start_nano, s.span_id))
        traces.append(Trace(trace_id=group[0].trace_id, steps=_build_steps(group), source=source))
    # Deterministic ordering: by each trace's earliest span start.
    traces.sort(key=lambda t: _earliest_start(by_trace[t.trace_id]))
    return traces


def _earliest_start(spans: list[_ParsedSpan]) -> int:
    return min((s.start_nano for s in spans), default=0)


class OtelGenAIAdapter:
    name = "otel-genai"

    def from_file(self, path: str) -> list[Trace]:
        text = Path(path).read_text(encoding="utf-8")
        spans: list[_ParsedSpan] = []
        try:
            payload: JsonValue = json.loads(text)
        except json.JSONDecodeError:
            # Not a single JSON document; treat as JSONL (one object/span per line).
            for line in text.splitlines():
                stripped = line.strip()
                if stripped:
                    spans.extend(_collect_spans(json.loads(stripped)))
        else:
            spans = _collect_spans(payload)
        return _spans_to_traces(spans, source=f"file:{path}")

    def from_vendor(self, pull: VendorPull) -> list[Trace]:
        """Pull OTLP-JSON spans from an OTLP-compatible query backend.

        OTLP itself is push-only, so "pulling" traces is inherently vendor-specific (Grafana Tempo,
        Jaeger, Honeycomb, ... each expose their own query API and auth). This implementation does a
        best-effort generic OTLP-JSON fetch and reuses the same span->Step mapping as `from_file`.

        TODO: implement the vendor-specific query protocol and auth. Today we:
          - read the query URL from ``$WMH_OTLP_QUERY_ENDPOINT``;
          - send ``pull.api_key`` (or ``$WMH_OTLP_API_KEY``) as a Bearer token;
          - pass project/since/limit as query params (real backends name these differently).
        """
        endpoint = os.environ.get(VENDOR_ENDPOINT_ENV)
        if not endpoint:
            raise ValueError(
                f"set ${VENDOR_ENDPOINT_ENV} to an OTLP-compatible query URL to pull traces "
                "(vendor-specific query support is not yet implemented; use from_file meanwhile)"
            )
        api_key = pull.api_key or os.environ.get(VENDOR_API_KEY_ENV)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # TODO: these param names are placeholders; map to the target backend's query schema.
        params: dict[str, str] = {}
        if pull.project is not None:
            params["project"] = pull.project
        if pull.since is not None:
            params["since"] = pull.since
        if pull.limit is not None:
            params["limit"] = str(pull.limit)
        response = httpx.get(endpoint, headers=headers, params=params, timeout=30.0)
        response.raise_for_status()
        spans = _collect_spans(response.json())
        return _spans_to_traces(spans, source=f"vendor:{pull.project or endpoint}")


register_adapter(OtelGenAIAdapter())
