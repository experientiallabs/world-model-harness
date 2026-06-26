"""The build pipeline behind `wmh build`.

ingest -> normalize -> split(train/test) -> embed/index -> GEPA optimize -> write `.wmh/` artifact.
Ingestion is part of the build: there is no separate ingest step.
"""

from __future__ import annotations

import hashlib
import json

from wmh.config import ArtifactPaths, HarnessConfig, save_config
from wmh.core.types import Trace
from wmh.engine.prompts import BASE_ENV_PROMPT
from wmh.engine.reporting import BuildReporter, NullReporter
from wmh.ingest import VendorPull, get_adapter
from wmh.optimize import GEPAOptimizer, LLMJudge, OptimizeResult
from wmh.providers import get_provider
from wmh.providers.base import Embedder, Provider
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder


def _count_steps(traces: list[Trace]) -> int:
    return sum(len(trace.steps) for trace in traces)


def ingest(
    config: HarnessConfig, *, file: str | None = None, vendor: VendorPull | None = None
) -> list[Trace]:
    """Load + normalize traces from a file upload or a vendor SDK pull into `Trace` objects."""
    adapter = get_adapter(config.trace_adapter)
    if file is not None:
        return adapter.from_file(file)
    if vendor is not None:
        return adapter.from_vendor(vendor)
    raise ValueError("ingest needs either a file path or a vendor pull")


def split_traces(traces: list[Trace], train_split: float) -> tuple[list[Trace], list[Trace]]:
    """Deterministic train/held-out split for GEPA (held-out is never seen during evolution).

    Assignment is by a stable hash of `trace_id`, so the same corpus always splits the same way
    regardless of order and rebuilds are reproducible. `train_split` is the target train fraction.
    """
    train: list[Trace] = []
    test: list[Trace] = []
    for trace in traces:
        digest = hashlib.blake2b(trace.trace_id.encode("utf-8"), digest_size=8).digest()
        # Map the hash to [0, 1); below the threshold -> train.
        fraction = int.from_bytes(digest, "big") / 2**64
        (train if fraction < train_split else test).append(trace)
    return train, test


def build(
    config: HarnessConfig,
    *,
    file: str | None = None,
    vendor: VendorPull | None = None,
    root: str = ".wmh",
    serve_provider: Provider | None = None,
    embedder: Embedder | None = None,
    reporter: BuildReporter | None = None,
) -> OptimizeResult:
    """Ingest traces and run the full build, creating + persisting the artifact under `root`.

    `serve_provider` / `embedder` are injectable for testing; in production they are constructed
    from `config` (serve provider via the registry, embedder = offline HashingEmbedder sized to
    `config.embed_dim`). `reporter` receives progress events (defaults to a no-op). Returns the
    GEPA OptimizeResult (also persisted).
    """
    report = reporter or NullReporter()
    paths = ArtifactPaths(root)
    traces = ingest(config, file=file, vendor=vendor)
    if not traces:
        raise ValueError("no traces ingested; nothing to build")
    report.ingest_done(len(traces), _count_steps(traces))

    train, test = split_traces(traces, config.train_split)
    report.split_done(len(train), len(test))

    provider = serve_provider or get_provider(config.serve_provider_config())
    embed = embedder or HashingEmbedder(dim=config.embed_dim)

    # Serving index over the full corpus: at serve time we retrieve from everything we have seen.
    retriever = EmbeddingRetriever(embed)
    retriever.index(traces)
    report.index_done(_count_steps(traces))

    # GEPA evolves the env prompt under serving conditions: it retrieves demos the same way the
    # world model will, but from a SEPARATE retriever it re-indexes over train-only (so held-out
    # steps never retrieve themselves). The embedder is stateless, so it's safe to share.
    report.optimize_start(config.gepa_budget)
    budget = config.gepa_budget

    def _on_rollout(done: int, score: float | None) -> None:
        report.rollout(done, budget, score)

    optimizer = GEPAOptimizer(
        provider,
        LLMJudge(provider),
        retriever=EmbeddingRetriever(embed),
        on_rollout=_on_rollout,
    )
    result = optimizer.optimize(train, test or train, BASE_ENV_PROMPT, config.gepa_budget)
    report.optimize_done(
        result.metrics.held_out_accuracy, len(result.frontier), result.metrics.rollouts_used
    )

    _persist(paths, config, retriever, result)
    return result


def _persist(
    paths: ArtifactPaths,
    config: HarnessConfig,
    retriever: EmbeddingRetriever,
    result: OptimizeResult,
) -> None:
    """Write config, prompts, frontier, metrics, and the retrieval index under `.wmh/`."""
    save_config(config, paths.root)
    paths.base_prompt.parent.mkdir(parents=True, exist_ok=True)
    paths.base_prompt.write_text(BASE_ENV_PROMPT, encoding="utf-8")
    paths.optimized_prompt.write_text(result.prompt, encoding="utf-8")
    paths.frontier.write_text(json.dumps(result.frontier, indent=2), encoding="utf-8")
    paths.metrics.write_text(result.metrics.model_dump_json(indent=2), encoding="utf-8")
    retriever.save(paths.index)
