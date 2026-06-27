"""TraceAdapter protocol + a small registry.

Sources differ in two ways: *transport* (file, Braintrust, Phoenix, ... query APIs) and *schema*
(which OTel semantic convention the spans follow). A `TraceAdapter` owns only the schema mapping:
raw payloads -> normalized traces. Transport lives in `trace_source.py`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from wmh.core.types import Trace


@runtime_checkable
class TraceAdapter(Protocol):
    """Turns one source's raw telemetry into normalized `Trace` objects."""

    name: str

    def from_file(self, path: str) -> list[Trace]:
        """Read traces from an exported file (OTLP-JSON / vendor JSONL)."""
        ...

    def from_payload(self, payload: JsonValue, *, source: str) -> list[Trace]:
        """Normalize an already-fetched trace payload."""
        ...


_ADAPTERS: dict[str, TraceAdapter] = {}


def register_adapter(adapter: TraceAdapter) -> None:
    _ADAPTERS[adapter.name] = adapter


def get_adapter(name: str) -> TraceAdapter:
    if name not in _ADAPTERS:
        raise ValueError(f"no trace adapter registered for {name!r}; have {list(_ADAPTERS)}")
    return _ADAPTERS[name]
