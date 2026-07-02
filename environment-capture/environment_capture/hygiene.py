"""Corpus hygiene: detect trajectories that escaped the task workspace onto the host.

`LocalBashEnv` executes real commands on the host, and an exploring agent that doesn't
immediately find its data can wander out of the workspace — capturing real host content
(home-directory listings, dotfile names, interpreter paths) into a corpus that gets committed
and redistributed. That is a privacy leak AND wrong environment dynamics: the world model should
learn the benchmark's workspace, not this machine.

Two detectors, used at capture emit time and at conversion (and by integrators to audit a
committed corpus): commands that TARGET host locations, and observations that CARRY host
markers. Flagged trajectories are dropped whole (never silently redacted — the discipline is
whole-trajectory drop, matching the terminal-tasks converter's `--exclude-substr`).
"""

from __future__ import annotations

import getpass
import json
import re
from dataclasses import dataclass
from pathlib import Path

from environment_capture.trajectory import Trajectory

# Commands that target host locations: absolute host roots, the home directory, walking out of
# the workspace, or sweeping the filesystem root. Workspace-relative paths never match.
_CMD_ESCAPE_RE = re.compile(
    r"(?:^|[\s;&|(`])("
    r"/(?:Users|home|root|etc|usr|opt|var|private|tmp)(?:/|\b)"
    r"|~(?:[/\s]|$)"
    r"|\$HOME\b"
    r"|cd\s+\.\."
    r"|(?:find|ls|tree|du)\s+(?:-[^\s]+\s+)*/\s"
    r")"
)

# Host content that shows up in observations when an escape succeeded (or partially succeeded —
# even a failed `cd /root` echoes a host-shaped error that isn't benchmark dynamics).
_OBS_MARKERS = (
    "/Users/",
    "/home/",
    "/root",
    "$HOME",
    "~/",
    ".ssh",
    "id_rsa",
    "id_ecdsa",
    "id_ed25519",
    "anaconda3",
    "site-packages",
    "node_modules",
    "/var/folders/",
    "/private/tmp",
    "Application Support",
)

# Learned at runtime, never committed as literals: `ls -l` ownership columns leak the machine
# username even for a listing taken INSIDE the workspace, and any echoed home path leaks the
# account. Detection therefore keys on the current machine's identity.
_RUNTIME_MARKERS = (getpass.getuser(), str(Path.home()))


def command_targets_host(command: str) -> bool:
    """Whether a command targets host locations outside the task workspace."""
    return _CMD_ESCAPE_RE.search(command) is not None


@dataclass(frozen=True)
class HygieneFinding:
    """One host-escape signal in a trajectory: where it was seen and what matched."""

    field: str  # "command" | "output"
    marker: str
    excerpt: str


def _check_text(field: str, text: str) -> list[HygieneFinding]:
    findings: list[HygieneFinding] = []
    if field == "command":
        match = _CMD_ESCAPE_RE.search(text)
        if match is not None:
            findings.append(
                HygieneFinding(field=field, marker=match.group(1).strip(), excerpt=text[:120])
            )
        return findings
    for marker in _OBS_MARKERS + _RUNTIME_MARKERS:
        index = text.find(marker)
        if index != -1:
            findings.append(
                HygieneFinding(
                    field=field, marker=marker, excerpt=text[max(0, index - 40) : index + 80]
                )
            )
            break
    return findings


def host_escape_findings(trajectory: Trajectory) -> list[HygieneFinding]:
    """Every host-escape signal in a trajectory's commands and observations."""
    findings: list[HygieneFinding] = []
    for step in trajectory.steps:
        for value in step.action.arguments.values():
            if isinstance(value, str):
                findings.extend(_check_text("command", value))
        findings.extend(_check_text("output", step.output))
    return findings


def partition_contained(
    trajectories: list[Trajectory],
) -> tuple[list[Trajectory], list[Trajectory]]:
    """Split trajectories into (workspace-contained, flagged), preserving order."""
    clean: list[Trajectory] = []
    flagged: list[Trajectory] = []
    for trajectory in trajectories:
        (flagged if host_escape_findings(trajectory) else clean).append(trajectory)
    return clean, flagged


def scan_spans_jsonl(path: Path) -> dict[str, list[HygieneFinding]]:
    """Audit a committed OTel GenAI corpus; returns flagged trace ids with their findings."""
    flagged: dict[str, list[HygieneFinding]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        span = json.loads(line)
        trace_id = str(span.get("traceId", ""))
        for attribute in span.get("attributes", []):
            key = attribute.get("key", "")
            value = attribute.get("value", {}).get("stringValue", "")
            findings: list[HygieneFinding] = []
            if key == "gen_ai.tool.call.arguments":
                arguments = json.loads(value)
                for argument in arguments.values():
                    if isinstance(argument, str):
                        findings.extend(_check_text("command", argument))
            elif key == "gen_ai.tool.message":
                findings.extend(_check_text("output", value))
            if findings:
                flagged.setdefault(trace_id, []).extend(findings)
    return flagged
