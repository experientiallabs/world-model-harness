"""Reconstruction-fidelity evaluation over trace files (the `wmh eval` backend).

`replay` (in `wmh.engine.replay`) scores one corpus of held-out steps. This orchestration layer is
what `wmh eval` calls: it loads one or more OTel trace files, splits each into train/holdout,
replays the holdout through a world-model prompt with leak-free RAG, and aggregates a per-file +
overall scorecard. Keeping it here (not in the CLI) keeps the command thin and the logic testable.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from wmh.engine.build import split_traces
from wmh.engine.replay import ReplayReport, replay
from wmh.ingest import get_adapter
from wmh.optimize.judge import Judge
from wmh.providers.base import Embedder, Provider
from wmh.retrieval import EmbeddingRetriever


class EvalReport(BaseModel):
    """Per-file fidelity reports plus the step-weighted overall mean."""

    per_file: dict[str, ReplayReport] = Field(default_factory=dict)
    overall_fidelity: float = 0.0
    total_steps: int = 0


def evaluate_files(
    files: list[Path],
    prompt: str,
    provider: Provider,
    judge: Judge,
    *,
    embedder: Embedder | None = None,
    train_split: float = 0.7,
    top_k: int = 5,
    adapter_name: str = "otel-genai",
) -> EvalReport:
    """Replay-score each trace file's held-out split. `embedder=None` -> zero-shot (no retrieval).

    Each file is split deterministically; tiny corpora with no held-out trace fall back to scoring
    every trace. RAG, when enabled, retrieves from that file's own train split only (leak-free).
    """
    adapter = get_adapter(adapter_name)
    per_file: dict[str, ReplayReport] = {}
    for path in files:
        traces = adapter.from_file(str(path))
        if not traces:
            continue
        train, holdout = split_traces(traces, train_split)
        if not holdout:  # tiny corpus: evaluate on everything
            train, holdout = traces, traces
        retriever = EmbeddingRetriever(embedder) if embedder is not None else None
        # Clean display name: "tau2-bench.otel.jsonl" -> "tau2-bench".
        name = path.name.removesuffix(".jsonl").removesuffix(".otel")
        per_file[name] = replay(
            prompt,
            holdout,
            provider,
            judge,
            retriever=retriever,
            train=train if embedder is not None else None,
            top_k=top_k,
        )

    total_steps = sum(r.n_steps for r in per_file.values())
    overall = (
        sum(r.mean_score * r.n_steps for r in per_file.values()) / total_steps
        if total_steps
        else 0.0
    )
    return EvalReport(per_file=per_file, overall_fidelity=overall, total_steps=total_steps)
