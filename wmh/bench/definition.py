"""Benchmark definitions: the committed, versioned `benchmark.toml` ("filesystem as DB").

A benchmark lives in its own directory under `benchmarks/<name>/`:

    benchmarks/
      tau-bench/
        benchmark.toml      <- the definition (this module loads it)
        results/            <- persisted runs (wmh.bench.results)

`benchmark.toml` names the recorded trace files to score and the eval knobs that make a run
reproducible — which turns to sample ("all" or Qwen-AgentWorld's 5-turn "sampled"), how many
world-model rollouts to draw per turn, the seeds to sweep, and which judge model grades. Trace
paths resolve relative to the benchmark directory so a checked-out benchmark is self-contained.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

# How many held-out turns to score per trace. Matches the eval scorer's contract
# (`wmh.engine.replay`): "all" scores every step; "sampled" takes Qwen-AgentWorld's 5-turn protocol
# (first, last, and 3 uniformly-spaced middle turns) for a fixed, cheaper turn budget.
SampleTurns = Literal["all", "sampled"]

# The benchmark.toml file name inside each benchmark directory.
BENCHMARK_FILE = "benchmark.toml"


class JudgeConfig(BaseModel):
    """Which model grades predicted-vs-real observations. Pinned in the definition for repro."""

    provider: str = "bedrock"
    model: str = "us.anthropic.claude-opus-4-8"
    region: str | None = None


class EvalConfig(BaseModel):
    """The reproducibility knobs of a benchmark, passed through to the open-loop scorer.

    `sample_turns` is `"all"` or `"sampled"` (Qwen-AgentWorld's 5-turn protocol). `rollouts` is how
    many world-model samples to draw per scored turn — with `temperature` > 0 each rollout yields a
    different score, so we report mean + std over them. `seeds` is the sweep: one scoring pass per
    seed, so a benchmark can report variance across seeds as well as across rollouts.
    """

    sample_turns: SampleTurns = "all"
    rollouts: int = Field(default=1, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    seeds: list[int] = Field(default_factory=lambda: [0], min_length=1)
    train_split: float = Field(default=0.7, gt=0.0, lt=1.0)
    top_k: int = Field(default=5, ge=0)
    no_rag: bool = False
    embed_dim: int = Field(default=512, ge=1)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)


class BenchmarkDef(BaseModel):
    """A loaded benchmark definition. `dir`/`name` come from the file location, not the TOML."""

    name: str
    version: str = "0"
    description: str = ""
    traces: list[str] = Field(default_factory=list)  # paths relative to the benchmark dir
    eval: EvalConfig = Field(default_factory=EvalConfig)
    dir: Path = Field(exclude=True)  # the benchmark directory; not serialized back to TOML

    def trace_files(self) -> list[Path]:
        """Resolve the configured trace paths against the benchmark directory."""
        return [self._resolve(t) for t in self.traces]

    def _resolve(self, trace: str) -> Path:
        path = Path(trace)
        return path if path.is_absolute() else (self.dir / path)

    def missing_traces(self) -> list[Path]:
        """Configured trace files that don't exist on disk (a definition can reference fixtures
        produced by another chat that haven't landed yet)."""
        return [p for p in self.trace_files() if not p.exists()]


def load_benchmark(path: str | Path) -> BenchmarkDef:
    """Load a benchmark from its directory or its `benchmark.toml` file.

    Accepts either `benchmarks/tau-bench/` or `benchmarks/tau-bench/benchmark.toml`. The benchmark
    name defaults to the directory name unless the TOML overrides `name`.
    """
    toml_path, bench_dir = _locate(path)
    try:
        with toml_path.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{toml_path} is not valid TOML ({exc})") from exc

    data.setdefault("name", bench_dir.name)
    data["dir"] = bench_dir
    try:
        return BenchmarkDef.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"{toml_path} does not match the benchmark schema ({exc})") from exc


def discover_benchmarks(root: str | Path) -> list[BenchmarkDef]:
    """Load every benchmark under `root` (each `<root>/<name>/benchmark.toml`), sorted by name.

    Returns an empty list if `root` doesn't exist. Subdirectories without a `benchmark.toml` are
    skipped, so `results/` and other artifacts never look like benchmarks.
    """
    base = Path(root)
    if not base.exists():
        return []
    defs = [
        load_benchmark(child)
        for child in sorted(base.iterdir())
        if child.is_dir() and (child / BENCHMARK_FILE).exists()
    ]
    return defs


def _locate(path: str | Path) -> tuple[Path, Path]:
    """Resolve `path` to `(benchmark.toml, benchmark_dir)`, accepting either as input."""
    p = Path(path)
    if p.is_dir():
        return p / BENCHMARK_FILE, p
    if p.name == BENCHMARK_FILE:
        return p, p.parent
    raise FileNotFoundError(f"{p} is not a benchmark directory or a {BENCHMARK_FILE} file")
