"""Build the `examples/` OTel trace corpus from self-improvement-bench (SIB) caches.

Converts every SIB benchmark that has cached agent transcripts (excluding swebench) into OTel GenAI
JSONL the harness ingests, writing one file per benchmark under `examples/<benchmark>.otel.jsonl`.
The traces are committed so the harness has a real, offline multi-domain corpus to build and replay
against without needing the SIB repo present.

Usage:
    python scripts/build_examples.py [<sib_repo_root>] [<out_dir>]

Defaults: sib_repo_root=../self-improvement-bench, out_dir=examples/.
All SIB benchmarks share one transcript shape (system/user/assistant with ```sib_bash``` commands),
so a single converter (`scripts.sib_to_otel`) handles them all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.sib_to_otel import convert_dir
from wmh.core.types import JsonObject

# Benchmarks to include. swebench is intentionally excluded (per project decision).
BENCHMARKS = (
    "tau2-bench",
    "terminal-bench",
    "dabstep",
    "echo-bench",
    "bird-sql",
    "gaia",
    "financebench",
    "continual-learning-bench",
)
_DEFAULT_SIB_ROOT = Path("../self-improvement-bench")
_DEFAULT_OUT = Path("examples")


def _trace_dirs(sib_root: Path, benchmark: str) -> list[Path]:
    """All cached transcript dirs for a benchmark (a benchmark may have several baseline runs)."""
    cache = sib_root / "results"
    return sorted(
        d for d in cache.glob(f"**/{benchmark}/**/traces") if d.is_dir() and any(d.glob("*.json"))
    )


def build_examples(sib_root: Path, out_dir: Path) -> dict[str, int]:
    """Convert each benchmark's transcripts to one OTel JSONL file. Returns {benchmark: spans}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for benchmark in BENCHMARKS:
        spans: list[JsonObject] = []
        for traces_dir in _trace_dirs(sib_root, benchmark):
            spans.extend(convert_dir(traces_dir, label=benchmark))
        if not spans:
            continue
        out_path = out_dir / f"{benchmark}.otel.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for span in spans:
                fh.write(json.dumps(span) + "\n")
        counts[benchmark] = len(spans)
    return counts


def main(argv: list[str]) -> int:
    sib_root = Path(argv[1]) if len(argv) > 1 else _DEFAULT_SIB_ROOT
    out_dir = Path(argv[2]) if len(argv) > 2 else _DEFAULT_OUT
    if not sib_root.exists():
        print(f"SIB repo not found at {sib_root}; pass its path as the first argument")
        return 2
    counts = build_examples(sib_root, out_dir)
    total = sum(counts.values())
    for benchmark, n in counts.items():
        print(f"  {benchmark}: {n} spans")
    print(f"wrote {total} spans across {len(counts)} benchmarks -> {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
