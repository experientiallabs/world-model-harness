"""The build pipeline behind `wmh build`.

ingest -> normalize -> split(train/test) -> embed/index -> GEPA optimize -> write `.wmh/` artifact.
Ingestion is part of the build: there is no separate ingest step.
"""

from __future__ import annotations

from wmh.config import HarnessConfig
from wmh.core.types import Trace
from wmh.ingest import VendorPull


def ingest(
    config: HarnessConfig, *, file: str | None = None, vendor: VendorPull | None = None
) -> list[Trace]:
    """Load + normalize traces from a file upload or a vendor SDK pull into `Trace` objects."""
    # adapter = get_adapter(config.trace_adapter)
    # return adapter.from_file(file) if file is not None else adapter.from_vendor(vendor)
    raise NotImplementedError


def split_traces(traces: list[Trace], train_split: float) -> tuple[list[Trace], list[Trace]]:
    """Deterministic train/held-out split for GEPA (held-out is never seen during evolution)."""
    # TODO: stable split (e.g. by hashing trace_id) so rebuilds are reproducible.
    raise NotImplementedError


def build(
    config: HarnessConfig, *, file: str | None = None, vendor: VendorPull | None = None
) -> None:
    """Ingest traces and run the full build, creating + persisting the artifact under `.wmh/`."""
    # traces = ingest(config, file=file, vendor=vendor)
    # train, test = split_traces(traces, config.train_split)
    # retriever.index(train); optimizer.optimize(train, test, BASE_ENV_PROMPT, config.gepa_budget)
    # persist prompts, frontier, metrics, and the index.
    raise NotImplementedError
