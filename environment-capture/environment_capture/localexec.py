"""A real local environment: bash commands executed in a workspace directory.

Each command runs in a fresh subshell with the workspace as cwd (no state leaks between
commands beyond the filesystem itself — the same discipline the shared-sandbox benchmarks use),
with stdout and stderr folded into one observation string, which is what the agent would see in
a terminal.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from environment_capture.adapter import ExecResult

_TIMEOUT_RETURNCODE = 124  # matches coreutils `timeout`


class LocalBashEnv:
    """CommandEnv backed by real subprocess execution in a workspace directory."""

    def __init__(
        self,
        workspace: Path | None = None,
        *,
        timeout_s: int = 60,
        cleanup: bool = False,
    ) -> None:
        """Execute in `workspace` (a fresh temp dir if None, then cleaned up on close)."""
        self._own_workspace = workspace is None
        self.workspace = workspace or Path(tempfile.mkdtemp(prefix="envcap-"))
        self.timeout_s = timeout_s
        self._cleanup = cleanup or self._own_workspace

    def execute(self, command: str) -> ExecResult:
        try:
            completed = subprocess.run(
                ["bash", "-c", command],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(
                output=f"command timed out after {self.timeout_s}s",
                returncode=_TIMEOUT_RETURNCODE,
            )
        output = completed.stdout
        if completed.stderr:
            output = f"{output}{completed.stderr}" if output else completed.stderr
        return ExecResult(output=output.rstrip("\n"), returncode=completed.returncode)

    def close(self) -> None:
        if self._cleanup:
            shutil.rmtree(self.workspace, ignore_errors=True)
