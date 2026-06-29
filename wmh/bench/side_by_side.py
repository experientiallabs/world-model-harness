"""Real-sandbox runner helpers for side-by-side benchmark demos."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

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


_RUNNERS: dict[str, tuple[str, str, str | None]] = {
    # benchmark name -> (display label, tool directory, trace arg for default/simplest scenario)
    "tau-bench": ("real tau2 environment", "tools/tau2-capture", None),
    "tau2-bench": ("real tau2 environment", "tools/tau2-capture", None),
    "swe-bench": (
        "real SWE-bench sandbox replaying mini-SWE-agent commands",
        "tools/swe-bench-capture",
        "-1",
    ),
    "terminal-tasks": ("real terminal sandbox", "tools/terminal-tasks-capture", None),
}


def real_sandbox_spec(
    benchmark: str,
    *,
    trace_index: int | None,
    train_split: float,
    extra_args: list[str] | None = None,
    repo_root: Path | None = None,
) -> RealSandboxSpec:
    """Build the matching `tools/<benchmark>-capture/run.sh` invocation.

    A `None` trace means "the simplest held-out scenario" on the world-model side. Most real
    runners use that as their default too; SWE-bench uses `--trace -1` for the same selection.
    """
    root = repo_root or Path(__file__).resolve().parents[2]
    try:
        label, tool_dir, simplest_trace = _RUNNERS[benchmark]
    except KeyError as exc:
        supported = ", ".join(sorted(_RUNNERS))
        raise ValueError(
            f"benchmark {benchmark!r} has no real sandbox runner; supported: {supported}"
        ) from exc

    cwd = root / tool_dir
    runner = cwd / "run.sh"
    if not runner.exists():
        raise FileNotFoundError(f"missing real sandbox runner: {runner}")

    command = [str(runner)]
    if trace_index is not None:
        command.extend(["--trace", str(trace_index)])
    elif simplest_trace is not None:
        command.extend(["--trace", simplest_trace])
    command.extend(["--train-split", str(train_split)])
    command.extend(extra_args or [])
    return RealSandboxSpec(benchmark=benchmark, label=label, command=command, cwd=cwd)


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
]
