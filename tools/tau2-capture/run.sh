#!/usr/bin/env bash
# End-to-end REAL tau2-bench scenario: set up the venv + Sierra's tau2 package + data (timed
# standup), then stand up the real domain environment and replay the recorded tool calls, streaming
# all stdout + the real DB records. One command.
#
#   tools/tau2-capture/run.sh [--trace N] [...]
#
# The whole thing — venv creation, `tau2-bench` install, cloning its data, importing tau2, and
# loading the domain DB — runs and prints here, so the total wall-clock is the true cost of standing
# up Sierra's real environment. That is the cost the world model side (`wmh bench scenario
# tau-bench`) skips. No Docker; re-runs reuse the venv + clone.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "=== setting up the tau2 venv + data (one-time; counts as standup) ==="
  [ -d tau2-bench ] || git clone --depth 1 https://github.com/sierra-research/tau2-bench.git
  uv venv --python 3.13 .venv
  # audioop-lts: backport of the audioop module removed from 3.13 stdlib (tau2 imports it).
  uv pip install --python .venv ./tau2-bench audioop-lts boto3
fi

export TAU2_DATA_DIR="${TAU2_DATA_DIR:-$PWD/tau2-bench/data}"
echo "=== running the real tau2 scenario (stand up env + DB, replay tool calls) ==="
exec .venv/bin/python run_real_scenario.py "$@"
