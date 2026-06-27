#!/usr/bin/env python3
"""Run ONE real SWE-bench scenario against the real Docker sandbox, streaming full stdout + timing.

This is the **real-environment** half of the scenario comparison. `wmh bench scenario swe-bench
--trace N` reconstructs a held-out scenario with the world model (no container, just LLM calls);
this script runs the SAME scenario for real — it boots the instance's SWE-bench Docker image and
`docker exec`s the exact recorded commands in order, printing the real stdout and the wall-clock
time (including container boot). You compare the two end times by eye.

Stdlib-only and self-contained (no `wmh` import), like the rest of `tools/swe-bench-capture/`:
- It reads the committed corpus `examples/swe-bench.otel.jsonl` and picks the SAME held-out trace
  `--trace N` the world-model side picks — re-implementing the harness's deterministic blake2b
  train/holdout split inline so trace selection matches exactly.
- It pulls + runs `docker.io/swebench/sweb.eval.x86_64.<instance>:latest` (the real per-instance
  image), execs each recorded `bash` command, and tears the container down.

Usage:
    python run_real_scenario.py --trace 0
    python run_real_scenario.py --corpus ../../examples/swe-bench.otel.jsonl --trace 0 --train-split 0.7
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

_DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "examples" / "swe-bench.otel.jsonl"


def _attr_map(span: dict[str, Any]) -> dict[str, str]:
    return {a["key"]: a.get("value", {}).get("stringValue", "") for a in span.get("attributes", [])}


def _load_traces(corpus: Path) -> "list[dict[str, Any]]":
    """Group the OTel spans into ordered traces: [{trace_id, instance_id, commands:[...]}]."""
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
        instance_id = ""
        commands: list[str] = []
        for span in by_trace[tid]:
            attrs = _attr_map(span)
            if "wmh.trace.metadata" in attrs:
                meta = json.loads(attrs["wmh.trace.metadata"])
                instance_id = meta.get("instance_id", "")
            args = attrs.get("gen_ai.tool.call.arguments")
            if args:  # an action span (the observation span has no arguments)
                command = json.loads(args).get("command")
                if isinstance(command, str) and command.strip():
                    commands.append(command)
        traces.append({"trace_id": tid, "instance_id": instance_id, "commands": commands})
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


def _swebench_image(instance_id: str) -> str:
    """The official per-instance SWE-bench image name for `instance_id`."""
    compat = instance_id.replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{compat}:latest".lower()


def _run(cmd: list[str], *, timeout: int | None = None, quiet: bool = False) -> int:
    """Run `cmd`, streaming its stdout/stderr live to our terminal; return the exit code.

    `quiet` discards stdout (used for `docker run -d`, whose only output is the noisy container id).
    """
    proc = subprocess.run(cmd, timeout=timeout, stdout=subprocess.DEVNULL if quiet else None)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(_DEFAULT_CORPUS), help="swe-bench OTel JSONL corpus.")
    parser.add_argument("--trace", type=int, default=0, help="Which held-out trace to replay.")
    parser.add_argument("--train-split", type=float, default=0.7, help="Train/holdout ratio.")
    parser.add_argument("--platform", default="linux/amd64", help="Docker platform for the image.")
    parser.add_argument("--exec-timeout", type=int, default=600, help="Per-command timeout (s).")
    args = parser.parse_args()

    traces = _load_traces(Path(args.corpus))
    pool = _holdout(traces, args.train_split)
    if not 0 <= args.trace < len(pool):
        raise SystemExit(f"--trace {args.trace} out of range; {len(pool)} held-out trace(s)")
    trace = pool[args.trace]
    instance_id, commands = trace["instance_id"], trace["commands"]
    if not instance_id:
        raise SystemExit(f"trace {trace['trace_id'][:8]} has no instance_id in metadata")

    image = _swebench_image(instance_id)
    container = f"wmh-real-{uuid.uuid4().hex[:8]}"
    print(
        f"REAL sandbox: {instance_id} ({len(commands)} commands) — "
        f"booting {image} then exec'ing the recorded commands\n"
    )

    start = time.monotonic()
    # Boot: pull (if absent) + start the container. This is the cost the world model skips entirely.
    print(f"$ docker run -d --name {container} {image} sleep 2h")
    rc = _run(
        ["docker", "run", "-d", "--name", container, "--platform", args.platform,
         "-w", "/testbed", "--rm", image, "sleep", "2h"],
        timeout=1800,
        quiet=True,  # docker run -d prints only the container id; suppress that noise
    )
    if rc != 0:
        raise SystemExit(f"failed to boot container (docker run exit {rc})")
    boot_done = time.monotonic()
    print(f"[container booted in {boot_done - start:.2f}s]\n")

    try:
        for i, command in enumerate(commands):
            print(f"--- step {i} ---\n$ {command}")
            _run(
                ["docker", "exec", "-w", "/testbed", container, "bash", "-lc", command],
                timeout=args.exec_timeout,
            )
            print()
    finally:
        subprocess.run(["docker", "rm", "-f", container], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)

    total = time.monotonic() - start
    print(
        f"done (REAL sandbox): boot {boot_done - start:.2f}s + "
        f"{len(commands)} commands in {total:.2f}s total"
    )


if __name__ == "__main__":
    main()
