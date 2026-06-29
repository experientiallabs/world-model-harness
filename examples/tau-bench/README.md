# Tau-Bench Example

Self-contained tau2-bench example for World Model Harness.

## Files

- `traces.otel.jsonl`: OTel trace corpus.
- `convert_to_wmh.py`: converts tau2 results into `traces.otel.jsonl`.
- `run.sh`: launches the real tau2 scenario helper for comparison or trace inspection.
- `run_real_scenario.py`: task-local real-environment runner.

## Harness Commands

```bash
uv run wmh build --name tau-bench --file examples/tau-bench/traces.otel.jsonl
uv run wmh eval examples/tau-bench/traces.otel.jsonl
uv run wmh examples run tau-bench -- --trace 0
```

## Regenerate Traces

Generate tau2 results separately, then convert them:

```bash
cd examples/tau-bench
python convert_to_wmh.py <results.json> --out traces.otel.jsonl --benchmark tau2-bench
```
