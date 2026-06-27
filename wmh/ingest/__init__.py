"""Trace ingestion: trace sources + schema adapters -> normalized `Trace` objects.

A `TraceSource` loads raw telemetry from file / provider query APIs. A `TraceAdapter` turns that
payload into the generic `Trace` schema. One concrete adapter ships (official OTel GenAI semconv);
other schemas can register behind the same protocol.
"""

# Import for the registration side effect so `get_adapter("otel-genai")` works on package import.
from wmh.ingest import otel_genai as otel_genai  # noqa: F401
from wmh.ingest import trace_source as trace_source  # noqa: F401
from wmh.ingest.adapter import TraceAdapter, get_adapter, register_adapter
from wmh.ingest.trace_source import (
    TraceSource,
    TraceSourceConfig,
    TraceSourceKind,
    get_trace_source,
    load_traces,
    register_trace_source,
    trace_source_kind,
)

__all__ = [
    "TraceAdapter",
    "TraceSource",
    "TraceSourceConfig",
    "TraceSourceKind",
    "get_adapter",
    "get_trace_source",
    "load_traces",
    "register_adapter",
    "register_trace_source",
    "trace_source_kind",
]
