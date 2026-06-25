"""Trace ingestion: vendor SDK pulls and file uploads -> normalized `Trace` objects.

A `TraceAdapter` turns one source's spans into the generic `Trace` schema. One concrete adapter
ships (official OTel GenAI semconv); others register behind the same protocol.
"""

from wmh.ingest.base import TraceAdapter, get_adapter, register_adapter

__all__ = ["TraceAdapter", "get_adapter", "register_adapter"]
