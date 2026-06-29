"""Tests for real-sandbox runner specs/capture used by the side-by-side demo."""

from __future__ import annotations

import sys
from pathlib import Path

from wmh.bench.side_by_side import RealSandboxSpec, real_sandbox_spec, run_real_sandbox


def test_real_sandbox_spec_maps_swe_default_to_simplest_trace() -> None:
    spec = real_sandbox_spec(
        "swe-bench", trace_index=None, train_split=0.7, repo_root=_repo_root()
    )

    assert spec.cwd.name == "swe-bench-capture"
    assert spec.label == "real SWE-bench sandbox replaying mini-SWE-agent commands"
    assert spec.command[-4:] == ["--trace", "-1", "--train-split", "0.7"]


def test_real_sandbox_spec_passes_explicit_trace_and_extra_args() -> None:
    spec = real_sandbox_spec(
        "tau-bench",
        trace_index=2,
        train_split=0.8,
        extra_args=["--x", "y"],
        repo_root=_repo_root(),
    )

    assert spec.cwd.name == "tau2-capture"
    assert spec.command[-6:] == ["--trace", "2", "--train-split", "0.8", "--x", "y"]


def test_run_real_sandbox_captures_output(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    spec = RealSandboxSpec(
        benchmark="fake",
        label="fake real env",
        command=[sys.executable, "-c", "print('REAL OUTPUT')"],
        cwd=tmp_path,
    )

    result = run_real_sandbox(spec)

    assert result.ok
    assert result.returncode == 0
    assert "REAL OUTPUT" in result.stdout
    assert "wall" in result.summary()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
