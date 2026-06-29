"""Official OpenTelemetry GenAI semantic-convention adapter.

Reads OTLP-JSON spans (file export or generic OTLP query backend) and normalizes them into `Trace`s
via the shared `wmh.ingest.normalize` core. The mapping itself (LLM/agent span -> Action, tool span
-> Observation, `wmh.*` enrichments -> state/metadata) lives in `normalize` and is shared with every
other span-based adapter (Phoenix, Langfuse, LangSmith, ...). This module is just the GenAI-semconv
*transport*: it loads the bytes and hands the decoded spans to `spans_to_traces`.

Supported file formats:
  - OTLP-JSON: a single object with `resourceSpans` -> `scopeSpans` -> `spans`
  - JSON array of the above, or of bare span objects
  - JSONL: one OTLP-JSON object or one bare span object per line

Optional `wmh.*` enrichments are honored by the shared normalizer (see `wmh.ingest.normalize`):
`wmh.state.structured`/`wmh.state.scratchpad` on an action span -> `Step.state_before`, and
`wmh.trace.metadata` on any span -> `Trace.metadata`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
from pydantic import JsonValue

from wmh.core.types import Trace
from wmh.ingest.adapter import VendorPull, register_adapter
from wmh.ingest.normalize import SpanRecord, collect_spans, spans_to_traces

# Env vars the (placeholder) vendor pull reads. The real query semantics are vendor-specific; see
# the TODO in `from_vendor`.
VENDOR_ENDPOINT_ENV = "WMH_OTLP_QUERY_ENDPOINT"
VENDOR_API_KEY_ENV = "WMH_OTLP_API_KEY"


def _spans_from_text(text: str) -> list[SpanRecord]:
    """Collect spans from a whole-document JSON payload, or per-line JSONL on decode failure."""
    spans: list[SpanRecord] = []
    try:
        payload: JsonValue = json.loads(text)
    except json.JSONDecodeError:
        # Not a single JSON document; treat as JSONL (one object/span per line). A single corrupt
        # line (truncated by a crashed exporter, say) is skipped rather than aborting the whole
        # ingest and losing every valid trace.
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                spans.extend(collect_spans(json.loads(stripped)))
            except json.JSONDecodeError:
                continue
        return spans
    return collect_spans(payload)


class OtelGenAIAdapter:
    name = "otel-genai"

    def from_file(self, path: str) -> list[Trace]:
        text = Path(path).read_text(encoding="utf-8")
        return spans_to_traces(_spans_from_text(text), source=f"file:{path}")

    def from_vendor(self, pull: VendorPull) -> list[Trace]:
        """Pull OTLP-JSON spans from an OTLP-compatible query backend.

        OTLP itself is push-only, so "pulling" traces is inherently vendor-specific (Grafana Tempo,
        Jaeger, Honeycomb, ... each expose their own query API and auth). This implementation does a
        best-effort generic OTLP-JSON fetch and reuses the same span->Trace mapping as `from_file`.

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
        spans = collect_spans(response.json())
        traces = spans_to_traces(spans, source=f"vendor:{pull.project or endpoint}")
        # The `limit` query param is a placeholder a real backend may ignore; enforce it locally
        # so the `VendorPull.limit` contract holds regardless of what the backend honors.
        if pull.limit is not None:
            traces = traces[: pull.limit]
        return traces


register_adapter(OtelGenAIAdapter())
