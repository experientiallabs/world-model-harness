"""Real-sandbox runner helpers for side-by-side benchmark demos."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel


class RealSandboxSpec(BaseModel):
    """How to run the real-environment half of a benchmark scenario demo."""

    benchmark: str
    label: str
    command: list[str]
    cwd: Path

    def display_command(self) -> str:
        return " ".join(self.command)


class RealSandboxResult(BaseModel):
    """Captured output and timing from a real sandbox runner."""

    spec: RealSandboxSpec
    returncode: int | None
    seconds: float
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and self.error is None

    def summary(self) -> str:
        if self.timed_out:
            return f"timed out after {self.seconds:.1f}s"
        if self.error is not None:
            return f"failed before launch: {self.error}"
        status = "ok" if self.returncode == 0 else f"exit {self.returncode}"
        return f"{status}, wall {self.seconds:.1f}s"


class _Runner(NamedTuple):
    """How one benchmark's real sandbox runner is invoked."""

    label: str  # display label
    tool_dir: str  # directory under the repo holding run.sh
    simplest_trace: str | None  # --trace value selecting the simplest scenario (None = its default)
    supports_trace_pool: bool  # runner accepts `--trace-pool` to pin which pool an index refers to
    concurrent_purges_images: bool  # concurrent cold runs delete shared images -> force warm+cache


_RUNNERS: dict[str, _Runner] = {
    "tau-bench": _Runner("real tau2 environment", "tools/tau2-capture", None, False, False),
    "tau2-bench": _Runner("real tau2 environment", "tools/tau2-capture", None, False, False),
    "swe-bench": _Runner(
        "real SWE-bench sandbox replaying mini-SWE-agent commands",
        "tools/swe-bench-capture",
        "-1",
        supports_trace_pool=True,
        concurrent_purges_images=True,
    ),
    "terminal-tasks": _Runner(
        "real terminal sandbox", "tools/terminal-tasks-capture", None, False, False
    ),
}


def runner_info(benchmark: str) -> _Runner:
    """Return the real sandbox runner metadata for `benchmark`, or raise for an unsupported one."""
    try:
        return _RUNNERS[benchmark]
    except KeyError as exc:
        supported = ", ".join(sorted(_RUNNERS))
        raise ValueError(
            f"benchmark {benchmark!r} has no real sandbox runner; supported: {supported}"
        ) from exc


def real_sandbox_spec(
    benchmark: str,
    *,
    trace_index: int | None,
    train_split: float,
    trace_pool: str | None = None,
    extra_args: list[str] | None = None,
    repo_root: Path | None = None,
) -> RealSandboxSpec:
    """Build the matching `tools/<benchmark>-capture/run.sh` invocation.

    A `None` trace means "the simplest held-out scenario" on the world-model side. Most real
    runners use that as their default too; SWE-bench uses `--trace -1` for the same selection.

    `trace_pool` pins which pool the index refers to ("held-out" or "all"), so a caller that
    resolved its index against the full corpus selects the SAME trace on the real side. It is
    forwarded only to runners that accept `--trace-pool`; passing it to one that does not is an
    error, since silently dropping it would let the two sides replay different scenarios.
    """
    root = repo_root or Path(__file__).resolve().parents[2]
    info = runner_info(benchmark)

    cwd = root / info.tool_dir
    runner = cwd / "run.sh"
    if not runner.exists():
        raise FileNotFoundError(f"missing real sandbox runner: {runner}")

    command = [str(runner)]
    if trace_index is not None:
        command.extend(["--trace", str(trace_index)])
    elif info.simplest_trace is not None:
        command.extend(["--trace", info.simplest_trace])
    command.extend(["--train-split", str(train_split)])
    if trace_pool is not None:
        if not info.supports_trace_pool:
            raise ValueError(
                f"benchmark {benchmark!r} runner does not support --trace-pool; cannot pin the "
                f"index to the {trace_pool!r} pool, so the real side may replay a different trace"
            )
        command.extend(["--trace-pool", trace_pool])
    command.extend(extra_args or [])
    return RealSandboxSpec(benchmark=benchmark, label=info.label, command=command, cwd=cwd)


def run_real_sandbox(
    spec: RealSandboxSpec,
    *,
    timeout_seconds: float | None = None,
) -> RealSandboxResult:
    """Run the real sandbox command, capturing stdout/stderr for side-by-side rendering."""
    start = time.monotonic()
    try:
        proc = subprocess.run(
            spec.command,
            cwd=spec.cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return RealSandboxResult(
            spec=spec,
            returncode=None,
            seconds=time.monotonic() - start,
            stdout=_coerce_text(exc.stdout),
            stderr=_coerce_text(exc.stderr),
            timed_out=True,
        )
    except OSError as exc:
        return RealSandboxResult(
            spec=spec,
            returncode=None,
            seconds=time.monotonic() - start,
            error=str(exc),
        )
    return RealSandboxResult(
        spec=spec,
        returncode=proc.returncode,
        seconds=time.monotonic() - start,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def _coerce_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


__all__ = [
    "RealSandboxResult",
    "RealSandboxSpec",
    "real_sandbox_spec",
    "run_real_sandbox",
    "runner_info",
]
