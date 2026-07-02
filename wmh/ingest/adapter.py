"""TraceAdapter protocol + a small registry.

Sources differ in two ways: *transport* (file vs. vendor SDK) and *schema* (which OTel semantic
convention the spans follow). An adapter owns both: it pulls/reads raw spans and normalizes them.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from wmh.core.types import Trace


class VendorPull(BaseModel):
    """Parameters for pulling traces from an observability vendor's API."""

    api_key: str | None = None  # falls back to the vendor's env var when None
    project: str | None = None  # vendor project / workspace to pull from
    since: str | None = None  # ISO-8601 lower bound on trace start time
    limit: int | None = None  # max traces to pull


@runtime_checkable
class TraceAdapter(Protocol):
    """Turns one source's raw telemetry into normalized `Trace` objects."""

    name: str

    def from_file(self, path: str) -> list[Trace]:
        """Read traces from an exported file (OTLP-JSON / vendor JSONL)."""
        ...

    def from_vendor(self, pull: VendorPull) -> list[Trace]:
        """Pull traces via a vendor SDK/API."""
        ...


_ADAPTERS: dict[str, TraceAdapter] = {}


def register_adapter(adapter: TraceAdapter) -> None:
    """Register a trace adapter by name."""
    _ADAPTERS[adapter.name] = adapter


def get_adapter(name: str) -> TraceAdapter:
    """Return the registered adapter for a source name."""
    if name not in _ADAPTERS:
        raise ValueError(f"no trace adapter registered for {name!r}; have {list(_ADAPTERS)}")
    return _ADAPTERS[name]
