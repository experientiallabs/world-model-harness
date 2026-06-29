# Terminal Tasks Example

Self-contained terminal-task example for World Model Harness.

## Files

- `traces.otel.jsonl`: OTel trace corpus.
- `convert_to_wmh.py`: converts terminal command trajectories into `traces.otel.jsonl`.
- `run.sh`: launches the real terminal scenario helper for comparison or trace inspection.
- `run_real_scenario.py`: task-local real-environment runner.

## Harness Commands

```bash
uv run wmh build --name terminal-tasks --file examples/terminal-tasks/traces.otel.jsonl
uv run wmh eval examples/terminal-tasks/traces.otel.jsonl
uv run wmh examples run terminal-tasks -- --trace 0
```

## Regenerate Traces

Convert terminal-task source trajectories into the local corpus:

```bash
cd examples/terminal-tasks
python convert_to_wmh.py <source_dir> --out traces.otel.jsonl --benchmark terminal-tasks
```
