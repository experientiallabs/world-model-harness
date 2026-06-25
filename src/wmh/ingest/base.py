"""TraceAdapter protocol + a small registry.

Sources differ in two ways: *transport* (file vs. vendor SDK) and *schema* (which OTel semantic
convention the spans follow). An adapter owns both: it pulls/reads raw spans and normalizes them.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from wmh.types import Trace


@runtime_checkable
class TraceAdapter(Protocol):
    """Turns one source's raw telemetry into normalized `Trace` objects."""

    name: str

    def from_file(self, path: str) -> list[Trace]:
        """Read traces from an exported file (OTLP-JSON / vendor JSONL)."""
        ...

    def from_vendor(self, **options: Any) -> list[Trace]:
        """Pull traces via a vendor SDK/API (creds + filters in options)."""
        ...


_ADAPTERS: dict[str, TraceAdapter] = {}


def register_adapter(adapter: TraceAdapter) -> None:
    _ADAPTERS[adapter.name] = adapter


def get_adapter(name: str) -> TraceAdapter:
    if name not in _ADAPTERS:
        raise ValueError(f"no trace adapter registered for {name!r}; have {list(_ADAPTERS)}")
    return _ADAPTERS[name]
