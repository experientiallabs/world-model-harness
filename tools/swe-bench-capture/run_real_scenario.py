#!/usr/bin/env python3
"""Run ONE real SWE-bench scenario against the real Docker sandbox — BUILT FROM SCRATCH — + timing.

The **real-environment** half of the scenario comparison. `wmh bench scenario swe-bench --trace N`
reconstructs a held-out scenario with the world model (no container, just LLM calls); this runs the
SAME scenario for real, and crucially it **builds the environment from scratch** — base image →
environment image (the real conda/pip dependency install) → instance image (clone repo + checkout +
install) — streaming every `docker build` line and counting the whole standup in the total time,
*then* `docker exec`s the recorded commands. That build is the slow, multi-minute cost the world
model skips entirely; it must be in the stdout and the clock, or the comparison is dishonest.

By default it builds with `--no-cache` so the standup cost is the true cold number every run; pass
`--cache` to reuse Docker layers across runs once you've built once.

Needs the swebench `.venv` from this directory's README (it imports `swebench` to get the official
Dockerfiles + setup scripts) and a running local Docker daemon. It never imports `wmh`; it reads the
committed `examples/swe-bench.otel.jsonl` and re-implements the harness's deterministic blake2b
held-out split inline so `--trace N` selects the SAME scenario the world-model side does.

Usage (from tools/swe-bench-capture/, in the swebench venv):
    .venv/bin/python run_real_scenario.py --trace 0            # cold build (default)
    .venv/bin/python run_real_scenario.py --trace 0 --cache    # reuse cached layers
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
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
                instance_id = json.loads(attrs["wmh.trace.metadata"]).get("instance_id", "")
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


def _docker_build(
    tag: str, dockerfile: str, scripts: dict[str, str], platform: str, *, no_cache: bool
) -> None:
    """Write a build context (Dockerfile + setup scripts) and `docker build` it, streaming output.

    Raises on a non-zero build. The streamed output IS the real dependency-install log; the caller
    times the whole call so the install cost lands in the comparison.
    """
    with tempfile.TemporaryDirectory(prefix="wmh-swe-build-") as ctx:
        ctx_dir = Path(ctx)
        (ctx_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8")
        for name, body in scripts.items():
            (ctx_dir / name).write_text(body, encoding="utf-8")
        cmd = ["docker", "build", "-t", tag, "--platform", platform]
        if no_cache:
            cmd.append("--no-cache")
        cmd.append(str(ctx_dir))
        print(f"$ {' '.join(cmd)}")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            raise SystemExit(f"docker build failed for {tag} (exit {rc})")


def _exists(image: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(_DEFAULT_CORPUS), help="swe-bench OTel JSONL corpus.")
    parser.add_argument("--trace", type=int, default=0, help="Which held-out trace to replay.")
    parser.add_argument("--train-split", type=float, default=0.7, help="Train/holdout ratio.")
    parser.add_argument(
        "--dataset", default="SWE-bench/SWE-bench_Verified", help="HF dataset for the build spec."
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Reuse cached Docker layers (skip already-built images). Default: cold --no-cache.",
    )
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

    # Official SWE-bench build spec: the real base/env/instance Dockerfiles + setup scripts.
    try:
        from swebench.harness.test_spec.test_spec import make_test_spec
        from swebench.harness.utils import load_swebench_dataset
    except ImportError as exc:  # pragma: no cover - depends on the isolated venv
        raise SystemExit(
            "swebench is not importable; run this from tools/swe-bench-capture/ in the .venv "
            "(see this directory's README). Docker must be running."
        ) from exc

    ds = load_swebench_dataset(args.dataset, "test", instance_ids=[instance_id])
    if not ds:
        raise SystemExit(f"instance {instance_id} not found in {args.dataset}")
    spec = make_test_spec(ds[0])
    no_cache = not args.cache

    print(
        f"REAL sandbox: {instance_id} ({len(commands)} commands) — building the environment from "
        f"scratch (base -> env deps -> instance), then exec'ing the recorded commands"
        f"{' [--no-cache]' if no_cache else ' [cached]'}\n"
    )
    start = time.monotonic()

    # 1) base image, 2) env image (the real conda/pip dependency install), 3) instance image
    #    (clone repo + checkout + install). Each streams its build log and counts toward the clock.
    layers = [
        ("base", spec.base_image_key, spec.base_dockerfile, {}),
        (
            "env (dependency install)",
            spec.env_image_key,
            spec.env_dockerfile,
            {"setup_env.sh": spec.setup_env_script},
        ),
        (
            "instance (repo + install)",
            spec.instance_image_key,
            spec.instance_dockerfile,
            {"setup_repo.sh": spec.install_repo_script},
        ),
    ]
    for label, tag, dockerfile, scripts in layers:
        if args.cache and _exists(tag):
            print(f"--- {label}: {tag} already built (cached) ---\n")
            continue
        print(f"--- building {label}: {tag} ---")
        _docker_build(tag, dockerfile, scripts, spec.platform, no_cache=no_cache)
        print()
    build_done = time.monotonic()
    print(f"[environment built from scratch in {build_done - start:.1f}s]\n")

    # Run the recorded scenario in a fresh container off the just-built instance image.
    container = f"wmh-real-{uuid.uuid4().hex[:8]}"
    rc = subprocess.run(
        ["docker", "run", "-d", "--name", container, "--platform", spec.platform,
         "-w", "/testbed", "--rm", spec.instance_image_key, "sleep", "2h"],
        stdout=subprocess.DEVNULL,
    ).returncode
    if rc != 0:
        raise SystemExit(f"failed to start container (docker run exit {rc})")
    try:
        for i, command in enumerate(commands):
            print(f"--- step {i} ---\n$ {command}")
            subprocess.run(
                ["docker", "exec", "-w", "/testbed", container, "bash", "-lc", command],
                timeout=args.exec_timeout,
            )
            print()
    finally:
        subprocess.run(["docker", "rm", "-f", container], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)

    total = time.monotonic() - start
    print(
        f"done (REAL sandbox): build {build_done - start:.1f}s + "
        f"{len(commands)} commands, {total:.1f}s total"
    )


if __name__ == "__main__":
    main()
