"""Convert a DABstep baseline-cache of real runs into the wmh OTel GenAI trace corpus.

The cache holds REAL benchmark runs (real bash commands, real recorded outputs); this script
re-emits them on the wmh wire format with provenance in the trace metadata. Trajectories with zero
environment transitions (the agent submitted without running a command) produce no spans and are
skipped explicitly, not silently. The shared ``load_baseline_cache`` normalizes the recording
harness's ``*_SUBMIT`` sentinel to the neutral ``SUBMIT`` (apparatus protocol, not environment
content), so nothing benchmark-specific is done to the loaded text here.

Usage:
    uv run python environment-capture/dabstep/convert_cache.py \
        --cache <path-to-baseline-cache-train-dir> --out traces.otel.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from environment_capture import load_baseline_cache, trajectory_to_spans, write_spans_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="Baseline-cache dir (manifest/tasks/traces)")
    parser.add_argument("--out", required=True, help="Output OTel GenAI JSONL path")
    parser.add_argument("--benchmark", default="dabstep")
    args = parser.parse_args()

    trajectories = load_baseline_cache(Path(args.cache))
    kept = [t for t in trajectories if t.steps]
    skipped = len(trajectories) - len(kept)

    n_spans = 0
    out = Path(args.out)
    for index, trajectory in enumerate(kept):
        spans = trajectory_to_spans(trajectory, benchmark=args.benchmark)
        n_spans += write_spans_jsonl(spans, out, append=index > 0)

    n_steps = sum(len(t.steps) for t in kept)
    print(
        f"wrote {len(kept)} traces / {n_steps} steps / {n_spans} spans -> {out} "
        f"(skipped {skipped} zero-step trajectories)"
    )


if __name__ == "__main__":
    main()
