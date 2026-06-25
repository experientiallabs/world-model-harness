"""The build pipeline behind `wmh build`.

ingest -> normalize -> split(train/test) -> embed/index -> GEPA optimize -> write `.wmh/` artifact.
"""

from __future__ import annotations

from wmh.config import HarnessConfig
from wmh.types import Trace


def split_traces(traces: list[Trace], train_split: float) -> tuple[list[Trace], list[Trace]]:
    """Deterministic train/held-out split for GEPA (held-out is never seen during evolution)."""
    # TODO: stable split (e.g. by hashing trace_id) so rebuilds are reproducible.
    raise NotImplementedError


def build(config: HarnessConfig, traces: list[Trace]) -> None:
    """Run the full build and persist the artifact under `.wmh/`."""
    # train, test = split_traces(traces, config.train_split)
    # retriever.index(train); optimizer.optimize(train, test, BASE_ENV_PROMPT, config.gepa_budget)
    # persist prompts, frontier, metrics, and the index.
    raise NotImplementedError
