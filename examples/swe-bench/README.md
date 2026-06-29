# SWE-bench Example

Self-contained SWE-bench example for World Model Harness.

## Files

- `traces.otel.jsonl`: OTel trace corpus.
- `convert_to_wmh.py`: converts mini-swe-agent trajectories into `traces.otel.jsonl`.
- `run.sh`: launches the real SWE-bench scenario helper for comparison or trace inspection.
- `run_real_scenario.py`: task-local real-environment runner.

## Harness Commands

```bash
uv run wmh build --name swe-bench --file examples/swe-bench/traces.otel.jsonl
uv run wmh eval examples/swe-bench/traces.otel.jsonl
uv run wmh examples run swe-bench -- --trace 0
```

## Regenerate Traces

Run mini-swe-agent separately, then convert the produced trajectory directory:

```bash
cd examples/swe-bench
python convert_to_wmh.py <run_dir_or_traj.json> --out traces.otel.jsonl --benchmark swe-bench
```
