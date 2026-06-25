"""Tests for trace-adapter registration."""

from __future__ import annotations

from wmh.ingest import get_adapter


def test_default_otel_adapter_is_registered_on_import() -> None:
    # DESIGN/README claim the OTel adapter ships registered; importing wmh.ingest must suffice.
    assert get_adapter("otel-genai").name == "otel-genai"
