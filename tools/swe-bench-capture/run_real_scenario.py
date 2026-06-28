#!/usr/bin/env python3
"""Run ONE real SWE-bench scenario against the real Docker sandbox — BUILT FROM SCRATCH — + timing.

The **real-environment** half of the scenario comparison. `wmh bench scenario swe-bench --trace N`
reconstructs a held-out scenario with the world model (no container, just LLM calls); this runs the
SAME scenario for real, and crucially it **builds the environment from scratch** — base image →
environment image (the real conda/pip dependency install) → instance image (clone repo + checkout +
install) — streaming every `docker build` line and counting the whole standup in the total time,
*then* `docker exec`s the recorded commands. That build is the slow, multi-minute cost the world
model skips entirely; it must be in the stdout and the clock, or the comparison is dishonest.

By default the standup is TRULY COLD: it first purges all local swebench/sweb.* images (so no shared
base layers are reused) and builds with `--no-cache`, so the timed standup is the real from-zero
multi-GB cost every run. Pass `--warm` (optionally with `--cache`) to reuse existing images/layers
for a faster repeat run.

Needs the swebench `.venv` from this directory's README (it imports `swebench` to get the official
Dockerfiles + setup scripts) and a running local Docker daemon. It never imports `wmh`; it reads the
committed `examples/swe-bench.otel.jsonl` and re-implements the harness's deterministic blake2b
held-out split inline so `--trace N` selects the SAME scenario the world-model side does (default 0;
pass -1 for the simplest = fewest commands).

By default the stood-up image(s) are wound down in the background after the run (they are multi-GB
and a cold run re-creates them); pass `--keep-image` to keep them.

Usage (from tools/swe-bench-capture/, in the swebench venv):
    .venv/bin/python run_real_scenario.py                          # trace 0, truly cold, then clean up
    .venv/bin/python run_real_scenario.py --warm --cache           # reuse existing image/layers
    .venv/bin/python run_real_scenario.py --trace 2 --keep-image   # a specific trace, keep the image
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
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
    tag: str, dockerfile: str, scripts: dict[str, str], platform_str: str, *, no_cache: bool
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
        cmd = ["docker", "build", "-t", tag, "--platform", platform_str]
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


def _pull_image(instance_id: str, platform_str: str, *, no_cache: bool) -> str:
    """`docker pull` the official prebuilt per-instance image, streaming progress; return its tag.

    The standup for the `pull` mode: a multi-GB image download (the environment + its installed
    dependencies, prebuilt) — a real, timed cost, just not a from-scratch compile. This is the path
    that works under emulation (the from-scratch `build` mode's `apt`/conda steps fail under qemu on
    non-x86 hosts). Raises on a failed pull.

    Cold by default (`no_cache=True`): a locally-cached image makes `docker pull` a no-op (~0.5s
    "up to date" check), which would silently hide the real download cost the comparison is about —
    so we remove it first, mirroring the build path's `--no-cache`. Pass `--cache` to reuse it.
    """
    compat = instance_id.replace("__", "_1776_")
    image = f"docker.io/swebench/sweb.eval.x86_64.{compat}:latest".lower()
    if no_cache and _exists(image):
        print(f"--- removing cached image for a cold pull (pass --cache to reuse): {image} ---")
        subprocess.run(["docker", "rmi", "-f", image], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    cmd = ["docker", "pull", "--platform", platform_str, image]
    print(f"$ {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise SystemExit(f"docker pull failed for {image} (exit {rc})")
    return image


def _purge_swebench_images() -> int:
    """Remove ALL local swebench/sweb.* images so a pull re-downloads every layer (truly cold).

    `docker rmi <image>` only drops the tag; Docker keeps the underlying layer blobs as long as any
    sibling image references them, so a re-pull of one instance only fetches its ~1 instance-specific
    layer and reuses the ~12 shared base layers (the ~3s "warm-ish" standup). The swebench eval
    images all derive from the same base, so evicting the whole family is what forces the next pull
    to download the full multi-GB image cold. Targeted to the swebench family — it does NOT touch
    unrelated images. Returns the count removed.
    """
    listed = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True, text=True,
    )
    refs = [
        r for r in listed.stdout.splitlines()
        if r.startswith("swebench/") or r.startswith(("sweb.eval", "sweb.env", "sweb.base"))
    ]
    removed = 0
    for ref in refs:
        if subprocess.run(["docker", "rmi", "-f", ref],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            removed += 1
    return removed


def _wind_down(images: list[str]) -> None:
    """Remove the stood-up images in a detached background process; return immediately.

    These instance images are multi-GB and a cold run re-creates them anyway, so we don't leave them
    filling the disk. Detached (`Popen`, no wait) so the cleanup doesn't add to the reported run
    time — `docker rmi` can take a moment to unwind layers. `--keep-image` skips this.
    """
    if not images:
        return
    # `docker rmi -f` each image; `|| true` so a missing/shared image doesn't abort the rest.
    script = " ; ".join(f"docker rmi -f {img} >/dev/null 2>&1 || true" for img in images)
    subprocess.Popen(  # noqa: S602 - fixed command, image tags are derived from the dataset spec
        ["bash", "-c", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(f"[winding down {len(images)} image(s) in the background: {', '.join(images)}]")


def _default_mode() -> str:
    """`build` from scratch on a native x86_64 host, else `pull` the prebuilt image.

    SWE-bench's base/env Dockerfiles run `apt`/conda steps that fail under qemu emulation (the apt
    GPG "invalid signature" error) on Apple-Silicon/arm64 hosts, so building from scratch only works
    natively. Everywhere else, pulling the prebuilt image is the robust real-environment standup.
    """
    return "build" if platform.machine().lower() in ("x86_64", "amd64") else "pull"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(_DEFAULT_CORPUS), help="swe-bench OTel JSONL corpus.")
    parser.add_argument(
        "--trace", type=int, default=0,
        help="Held-out trace to replay (default: 0). Pass -1 for the simplest = fewest commands.",
    )
    parser.add_argument("--train-split", type=float, default=0.7, help="Train/holdout ratio.")
    parser.add_argument(
        "--dataset", default="SWE-bench/SWE-bench_Verified", help="HF dataset for the build spec."
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "build", "pull"),
        default="auto",
        help=(
            "How to stand up the env: 'build' compiles base/env/instance from scratch (native "
            "x86_64/Linux only — apt/conda fail under qemu emulation); 'pull' downloads the prebuilt "
            "multi-GB image (works under emulation). 'auto' (default): build on x86_64, else pull."
        ),
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Reuse cached Docker layers (skip already-built images). Default: cold --no-cache.",
    )
    parser.add_argument("--exec-timeout", type=int, default=600, help="Per-command timeout (s).")
    parser.add_argument(
        "--warm",
        action="store_true",
        help=(
            "Skip the truly-cold purge. By default the standup purges ALL local swebench/sweb.* "
            "images first so it re-downloads every layer (no shared-base reuse) and reports the real "
            "cold multi-GB cost. --warm keeps existing images, for a faster repeat run."
        ),
    )
    parser.add_argument(
        "--keep-image",
        action="store_true",
        help=(
            "Keep the stood-up Docker image(s) after the run. Default: wind them down in the "
            "background (these images are multi-GB; a cold run re-creates them anyway)."
        ),
    )
    args = parser.parse_args()

    traces = _load_traces(Path(args.corpus))
    pool = _holdout(traces, args.train_split)
    if not pool:
        raise SystemExit(f"no traces in {args.corpus}; nothing to run")
    if args.trace == -1:
        # The simplest scenario — fewest recorded commands (matches the wmh side's `min` default).
        trace = min(pool, key=lambda t: len(t["commands"]))
    elif 0 <= args.trace < len(pool):
        trace = pool[args.trace]
    else:
        raise SystemExit(f"--trace {args.trace} out of range; {len(pool)} held-out trace(s)")
    instance_id, commands = trace["instance_id"], trace["commands"]
    if not instance_id:
        raise SystemExit(f"trace {trace['trace_id'][:8]} has no instance_id in metadata")

    mode = _default_mode() if args.mode == "auto" else args.mode

    print(f"=== resolving SWE-bench dataset spec for {instance_id} (first run downloads it) ===")
    # Official SWE-bench build spec: the real base/env/instance Dockerfiles + setup scripts.
    try:
        from swebench.harness.test_spec.test_spec import make_test_spec
        from swebench.harness.utils import load_swebench_dataset
    except ImportError as exc:  # pragma: no cover - depends on the isolated venv
        raise SystemExit(
            "swebench is not importable; run via ./run.sh (it sets up the .venv) or install "
            "swebench in tools/swe-bench-capture/.venv. Docker must be running."
        ) from exc

    ds = load_swebench_dataset(args.dataset, "test", instance_ids=[instance_id])
    if not ds:
        raise SystemExit(f"instance {instance_id} not found in {args.dataset}")
    spec = make_test_spec(ds[0])

    print(
        f"\nREAL sandbox: {instance_id} ({len(commands)} commands) — standing up the env "
        f"[mode={mode}], then exec'ing the recorded commands\n"
    )
    # Truly-cold by default: evict the whole swebench image family BEFORE the clock starts, so shared
    # base layers can't be reused and the timed standup is a full from-zero download/build. The
    # eviction is teardown of prior state, so it is deliberately not counted in the standup time.
    # `--warm` skips it for a faster repeat run.
    cold = not args.warm
    if cold:
        print("=== cold standup: purging all local swebench/sweb.* images (no shared-layer reuse) ===")
        n = _purge_swebench_images()
        print(f"--- purged {n} swebench image(s); the standup below is a true cold download ---\n")
    start = time.monotonic()
    # Cold (the default) never reuses anything; --warm + --cache reuses build layers / the image.
    no_cache = cold or not args.cache
    created: list[str] = []  # images this run stood up (for the wind-down cleanup)

    if mode == "build":
        # base image -> env image (the real conda/pip dependency install) -> instance image (clone
        # repo + checkout + install). Each streams its build log and counts toward the clock.
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
            print(f"=== building {label}: {tag} ===")
            _docker_build(tag, dockerfile, scripts, spec.platform, no_cache=no_cache)
            created.append(tag)
            print()
        run_image = spec.instance_image_key
    else:  # pull
        print("=== pulling the prebuilt instance image (multi-GB download — this is the standup) ===")
        run_image = _pull_image(instance_id, spec.platform, no_cache=no_cache)
        created.append(run_image)
        print()
    build_done = time.monotonic()
    print(f"[environment stood up ({mode}) in {build_done - start:.1f}s]\n")

    # Run the recorded scenario in a fresh container off the stood-up instance image.
    container = f"wmh-real-{uuid.uuid4().hex[:8]}"
    rc = subprocess.run(
        ["docker", "run", "-d", "--name", container, "--platform", spec.platform,
         "-w", "/testbed", "--rm", run_image, "sleep", "2h"],
        stdout=subprocess.DEVNULL,
    ).returncode
    if rc != 0:
        raise SystemExit(f"failed to start container (docker run exit {rc})")
    failures = 0
    try:
        for i, command in enumerate(commands):
            print(f"--- step {i} ---\n$ {command}")
            try:
                proc = subprocess.run(
                    ["docker", "exec", "-w", "/testbed", container, "bash", "-lc", command],
                    timeout=args.exec_timeout,
                )
                if proc.returncode != 0:
                    failures += 1
                    print(f"[exit {proc.returncode}]")
            except subprocess.TimeoutExpired:
                failures += 1
                print(f"[timed out after {args.exec_timeout}s]")
            print()
    finally:
        subprocess.run(["docker", "rm", "-f", container], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)

    total = time.monotonic() - start
    note = "" if failures == 0 else f" ({failures} command(s) errored/timed out)"
    print(
        f"done (REAL sandbox): standup {build_done - start:.1f}s + "
        f"{len(commands)} commands, {total:.1f}s total{note}"
    )

    # Wind-down: drop the multi-GB image(s) this run stood up, in the background (after timing, so it
    # never counts against the clock). `--keep-image` or `--cache` (you want to reuse it) opt out.
    if not args.keep_image and not args.cache:
        _wind_down(created)


if __name__ == "__main__":
    main()
