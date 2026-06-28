#!/usr/bin/env bash
# End-to-end REAL swe-bench scenario: set up the venv + deps (timed standup), then stand up the
# environment (build from scratch on native x86_64, else pull the prebuilt image) and run the
# recorded scenario, streaming all stdout. One command.
#
#   tools/swe-bench-capture/run.sh [--trace N] [--mode build|pull|auto] [--cache] [...]
#
# The whole thing — Python venv creation, `swebench` install, the Docker standup (a from-scratch
# conda/pip build on x86_64, or a multi-GB prebuilt-image pull under emulation), and the recorded
# commands — runs and prints here, so the total wall-clock is the true cost of standing up + running
# the real environment cold. That is the cost the world model side (`wmh bench scenario swe-bench`)
# skips. Re-runs reuse the venv; pass --cache to also reuse Docker build layers.
set -euo pipefail
cd "$(dirname "$0")"

# Guard on the package actually importing, not just the venv dir existing — a half-finished prior
# setup (venv created, pip install interrupted) must re-run the install, not skip it.
if ! .venv/bin/python -c 'import swebench' >/dev/null 2>&1; then
  echo "=== setting up the swebench venv (one-time; counts as standup) ==="
  uv venv --python 3.12 .venv
  uv pip install --python .venv swebench boto3
fi

export AWS_REGION="${AWS_REGION:-us-east-1}" AWS_REGION_NAME="${AWS_REGION_NAME:-us-east-1}"
echo "=== running the real swe-bench scenario (stand up env + exec) ==="
exec .venv/bin/python run_real_scenario.py "$@"
