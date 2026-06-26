"""Tests for benchmark definition loading + discovery."""

from __future__ import annotations

import pytest

from wmh.bench.definition import discover_benchmarks, load_benchmark


def _write_benchmark(tmp_path, name: str, body: str) -> str:  # noqa: ANN001 - pytest path
    bench_dir = tmp_path / name
    bench_dir.mkdir(parents=True)
    (bench_dir / "benchmark.toml").write_text(body, encoding="utf-8")
    return str(bench_dir)


def test_loads_definition_with_defaults(tmp_path) -> None:  # noqa: ANN001
    bench_dir = _write_benchmark(
        tmp_path,
        "tau",
        'version = "2"\ntraces = ["a.jsonl"]\n',
    )
    bench = load_benchmark(bench_dir)
    assert bench.name == "tau"  # defaulted from the directory name
    assert bench.version == "2"
    assert bench.eval.sample_turns == "all"
    assert bench.eval.rollouts == 1
    assert bench.eval.seeds == [0]
    # Trace paths resolve relative to the benchmark dir.
    assert bench.trace_files()[0] == tmp_path / "tau" / "a.jsonl"


def test_loads_full_eval_and_judge_config(tmp_path) -> None:  # noqa: ANN001
    bench_dir = _write_benchmark(
        tmp_path,
        "retail",
        "\n".join(
            [
                'traces = ["t.jsonl"]',
                "[eval]",
                'sample_turns = "sampled"',
                "rollouts = 3",
                "temperature = 0.7",
                "seeds = [1, 2, 3]",
                "[eval.judge]",
                'provider = "bedrock"',
                'model = "us.anthropic.claude-opus-4-8"',
            ]
        ),
    )
    bench = load_benchmark(bench_dir)
    assert bench.eval.sample_turns == "sampled"
    assert bench.eval.rollouts == 3
    assert bench.eval.temperature == 0.7
    assert bench.eval.seeds == [1, 2, 3]
    assert bench.eval.judge.model == "us.anthropic.claude-opus-4-8"


def test_accepts_toml_path_directly(tmp_path) -> None:  # noqa: ANN001
    bench_dir = _write_benchmark(tmp_path, "tau", 'traces = ["a.jsonl"]\n')
    bench = load_benchmark(f"{bench_dir}/benchmark.toml")
    assert bench.name == "tau"


def test_missing_traces_reports_absent_files(tmp_path) -> None:  # noqa: ANN001
    bench_dir = _write_benchmark(tmp_path, "tau", 'traces = ["nope.jsonl"]\n')
    bench = load_benchmark(bench_dir)
    missing = bench.missing_traces()
    assert len(missing) == 1
    assert missing[0].name == "nope.jsonl"


def test_rejects_empty_seeds(tmp_path) -> None:  # noqa: ANN001
    bench_dir = _write_benchmark(tmp_path, "tau", "traces = []\n[eval]\nseeds = []\n")
    with pytest.raises(ValueError, match="schema"):
        load_benchmark(bench_dir)


def test_rejects_invalid_sample_turns(tmp_path) -> None:  # noqa: ANN001
    bench_dir = _write_benchmark(tmp_path, "tau", '[eval]\nsample_turns = "first-five"\n')
    with pytest.raises(ValueError, match="schema"):
        load_benchmark(bench_dir)


def test_rejects_malformed_toml(tmp_path) -> None:  # noqa: ANN001
    bench_dir = _write_benchmark(tmp_path, "tau", "this is = = not toml\n")
    with pytest.raises(ValueError, match="not valid TOML"):
        load_benchmark(bench_dir)


def test_discover_skips_non_benchmark_dirs(tmp_path) -> None:  # noqa: ANN001
    _write_benchmark(tmp_path, "tau", 'traces = ["a.jsonl"]\n')
    _write_benchmark(tmp_path, "retail", 'traces = ["b.jsonl"]\n')
    # A stray directory without a benchmark.toml (e.g. a results/ sibling) is ignored.
    (tmp_path / "results").mkdir()
    found = discover_benchmarks(tmp_path)
    assert [b.name for b in found] == ["retail", "tau"]  # sorted by name


def test_discover_missing_root_is_empty(tmp_path) -> None:  # noqa: ANN001
    assert discover_benchmarks(tmp_path / "nope") == []
