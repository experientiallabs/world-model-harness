"""Convert a frozen baseline cache of REAL database-exploration runs into the wmh trace corpus.

The cache holds REAL runs (real ``sqlite3``/``python3`` commands, real recorded outputs); this
re-emits them on the wmh OTel GenAI wire format with provenance in the trace metadata.

One transformation is applied to the command/observation text: the reference harness's submission
sentinel token (``SIB_SUBMIT``) is renamed to the neutral ``SUBMIT``. That token is the reference
harness's own submission protocol keyword, not environment content — no query result, schema, or
number is altered — and renaming it drops the source project's identifier from the corpus. The
conversion then asserts no source-project reference survives. Zero-transition trajectories (the
agent submitted without running a command) produce no spans and are skipped explicitly.

Usage:
    uv run python environment-capture/continual-learning/convert_cache.py \
        --cache <path-to-baseline-cache-train-dir> --out traces.otel.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from environment_capture import (
    Trajectory,
    load_baseline_cache,
    trajectory_to_spans,
    write_spans_jsonl,
)

_BENCHMARK = "continual-learning"
# The reference harness's submission sentinel -> a neutral, source-agnostic keyword.
_SENTINEL = "SIB_SUBMIT"
_NEUTRAL_SENTINEL = "SUBMIT"
# Source-project identifiers that must never appear in the committed corpus.
_FORBIDDEN = ("SIB_SUBMIT", "sib_bash", "self-improvement", "self_improvement", "/Users/")


def _sanitize(text: str) -> str:
    return text.replace(_SENTINEL, _NEUTRAL_SENTINEL)


def _sanitize_trajectory(trajectory: Trajectory) -> Trajectory:
    steps = [
        dataclasses.replace(
            step,
            action=dataclasses.replace(
                step.action,
                arguments={
                    key: _sanitize(value) if isinstance(value, str) else value
                    for key, value in step.action.arguments.items()
                },
            ),
            output=_sanitize(step.output),
        )
        for step in trajectory.steps
    ]
    final_answer = _sanitize(trajectory.final_answer)
    return dataclasses.replace(trajectory, steps=steps, final_answer=final_answer)


def _assert_clean(trajectory: Trajectory) -> None:
    blobs = [trajectory.final_answer, trajectory.task.prompt]
    for step in trajectory.steps:
        blobs.append(step.output)
        blobs.extend(str(v) for v in step.action.arguments.values())
    for blob in blobs:
        for token in _FORBIDDEN:
            if token in blob:
                raise AssertionError(
                    f"{trajectory.task.task_id}: source-project reference {token!r} survived "
                    f"sanitization in: {blob[:120]!r}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="Baseline-cache dir (manifest/tasks/traces)")
    parser.add_argument("--out", required=True, help="Output OTel GenAI JSONL path")
    args = parser.parse_args()

    trajectories = [_sanitize_trajectory(t) for t in load_baseline_cache(Path(args.cache))]
    for trajectory in trajectories:
        _assert_clean(trajectory)
    kept = [t for t in trajectories if t.steps]
    skipped = len(trajectories) - len(kept)

    out = Path(args.out)
    n_spans = 0
    for index, trajectory in enumerate(kept):
        spans = trajectory_to_spans(trajectory, benchmark=_BENCHMARK)
        n_spans += write_spans_jsonl(spans, out, append=index > 0)

    n_steps = sum(len(t.steps) for t in kept)
    print(
        f"wrote {len(kept)} traces / {n_steps} steps / {n_spans} spans -> {out} "
        f"(skipped {skipped} zero-step trajectories)"
    )


if __name__ == "__main__":
    main()
