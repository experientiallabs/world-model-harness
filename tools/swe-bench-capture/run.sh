#!/usr/bin/env bash
# End-to-end REAL swe-bench scenario: set up the venv + deps (timed standup), then build the
# environment from scratch and run the recorded scenario, streaming all stdout. One command.
#
#   tools/swe-bench-capture/run.sh [--trace N] [--cache] [...]
#
# The whole thing — Python venv creation, `swebench` install, the base/env/instance Docker build
# (real conda/pip dependency install), and the recorded commands — runs and prints here, so the
# total wall-clock is the true cost of standing up + running the real environment cold. That is the
# cost the world model side (`wmh bench scenario swe-bench`) skips. Re-runs reuse the venv; pass
# --cache to also reuse Docker layers.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "=== setting up the swebench venv (one-time; counts as standup) ==="
  uv venv --python 3.12 .venv
  uv pip install --python .venv swebench boto3
fi

export AWS_REGION="${AWS_REGION:-us-east-1}" AWS_REGION_NAME="${AWS_REGION_NAME:-us-east-1}"
echo "=== running the real swe-bench scenario (build from scratch + exec) ==="
exec .venv/bin/python run_real_scenario.py "$@"
