"""Official OpenTelemetry GenAI semantic-convention adapter.

Maps `gen_ai.*` spans into our normalized schema:
  - LLM / agent spans  -> Action (message or tool_call from gen_ai.tool.* / gen_ai.prompt)
  - tool execution spans -> Observation (gen_ai.tool output / span result)
  - one trace_id worth of spans, ordered by start time -> one `Trace`

This is the reference adapter; it also documents the generic shape other adapters target.
"""

from __future__ import annotations

from wmh.core.types import Trace
from wmh.ingest.adapter import VendorPull, register_adapter


class OtelGenAIAdapter:
    name = "otel-genai"

    def from_file(self, path: str) -> list[Trace]:
        # TODO: parse OTLP-JSON; group spans by trace_id; map gen_ai.* -> Step list.
        raise NotImplementedError

    def from_vendor(self, pull: VendorPull) -> list[Trace]:
        # TODO: pull from an OTLP-compatible backend; reuse the same span->Step mapping.
        raise NotImplementedError


register_adapter(OtelGenAIAdapter())
