#!/usr/bin/env python3
"""Run ONE real terminal-tasks scenario in a local shell, streaming full stdout + timing.

The **real-environment** half of the scenario comparison for terminal-tasks. `wmh bench scenario
terminal-tasks --trace N` reconstructs a held-out scenario with the world model (LLM, no shell);
this runs the SAME scenario for real — it executes the exact recorded `bash` commands (curl-to-API
calls) in a local shell, in order, printing the real stdout and the wall-clock time. You compare the
two end times by eye.

Because terminal-tasks commands hit live public APIs, a real re-run reflects *current* data, so the
output may differ from the recorded observation (rates change, releases bump) — that is the honest
real environment. There is no container to boot here; the "startup cost" the world model saves is
whatever the real tools' cold start is (shell + curl + network).

Stdlib-only and self-contained (no `wmh` import), like the rest of `tools/terminal-tasks-capture/`:
it reads the committed `examples/terminal-tasks.otel.jsonl`, picks the SAME held-out trace `--trace
N` the world-model side picks (re-implementing the harness's deterministic blake2b split inline),
and runs that trace's recorded commands with `bash -lc`.

Usage:
    python run_real_scenario.py --trace 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

_DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "examples" / "terminal-tasks.otel.jsonl"


def _attr_map(span: dict[str, Any]) -> dict[str, str]:
    return {a["key"]: a.get("value", {}).get("stringValue", "") for a in span.get("attributes", [])}


def _load_traces(corpus: Path) -> "list[dict[str, Any]]":
    """Group the OTel spans into ordered traces: [{trace_id, task, commands:[...]}]."""
    spans = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line]
    order: list[str] = []
    by_trace: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        tid = span["traceId"]
        if tid not in by_trace:
            by_trace[tid] = []
            order.append(tid)
        by_trace[tid].append(span)

    traces: list[dict[str, Any]] = []
    for tid in order:
        task = ""
        commands: list[str] = []
        for span in by_trace[tid]:
            attrs = _attr_map(span)
            task = task or attrs.get("gen_ai.prompt", "")
            args = attrs.get("gen_ai.tool.call.arguments")
            if args:  # an action span (the observation span has no arguments)
                command = json.loads(args).get("command")
                if isinstance(command, str) and command.strip():
                    commands.append(command)
        traces.append({"trace_id": tid, "task": task, "commands": commands})
    return traces


def _holdout(traces: list[dict[str, Any]], train_split: float) -> list[dict[str, Any]]:
    """The held-out traces, by the SAME deterministic blake2b split the wmh harness uses."""
    held: list[dict[str, Any]] = []
    for trace in traces:
        digest = hashlib.blake2b(trace["trace_id"].encode("utf-8"), digest_size=8).digest()
        fraction = int.from_bytes(digest, "big") / 2**64
        if fraction >= train_split:
            held.append(trace)
    return held or traces  # tiny corpora: no held-out -> fall back to all (matches the wmh side)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(_DEFAULT_CORPUS), help="terminal-tasks OTel corpus.")
    parser.add_argument("--trace", type=int, default=0, help="Which held-out trace to replay.")
    parser.add_argument("--train-split", type=float, default=0.7, help="Train/holdout ratio.")
    parser.add_argument("--exec-timeout", type=int, default=120, help="Per-command timeout (s).")
    args = parser.parse_args()

    traces = _load_traces(Path(args.corpus))
    pool = _holdout(traces, args.train_split)
    if not 0 <= args.trace < len(pool):
        raise SystemExit(f"--trace {args.trace} out of range; {len(pool)} held-out trace(s)")
    trace = pool[args.trace]
    commands = trace["commands"]
    task = (trace["task"] or "").strip().splitlines()[0] if trace["task"] else ""
    print(
        f"REAL shell: trace {trace['trace_id'][:8]} ({len(commands)} commands)"
        + (f" — {task[:80]}" if task else "")
        + " — running the recorded commands for real\n"
    )

    start = time.monotonic()
    for i, command in enumerate(commands):
        print(f"--- step {i} ---\n$ {command}")
        try:
            subprocess.run(["bash", "-lc", command], timeout=args.exec_timeout)
        except subprocess.TimeoutExpired:
            print(f"[timed out after {args.exec_timeout}s]")
        print()

    total = time.monotonic() - start
    print(f"done (REAL shell): {len(commands)} commands in {total:.2f}s total")


if __name__ == "__main__":
    main()
