"""Trace ingestion: vendor SDK pulls and file uploads -> normalized `Trace` objects.

A `TraceAdapter` turns one source's spans into the generic `Trace` schema. One concrete adapter
ships (official OTel GenAI semconv); others register behind the same protocol.
"""

# Import for the registration side effect so `get_adapter("otel-genai")` works on package import.
from wmh.ingest import otel_genai as otel_genai  # noqa: F401
from wmh.ingest.adapter import TraceAdapter, VendorPull, get_adapter, register_adapter
from wmh.ingest.quality import drop_degenerate_traces

__all__ = [
    "TraceAdapter",
    "VendorPull",
    "drop_degenerate_traces",
    "get_adapter",
    "register_adapter",
]
