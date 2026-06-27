#!/usr/bin/env bash
# End-to-end REAL terminal-tasks scenario: build a fresh tools container from scratch (apt install
# curl/python3/jq), then run the recorded curl-to-API commands, streaming all stdout. One command.
#
#   tools/terminal-tasks-capture/run.sh [--trace N] [--cache] [...]
#
# The runner is stdlib-only (no benchmark package to install), so the standup here is the Docker
# image build — the real tool install — which it streams and counts in the total time. That is the
# cost the world model side (`wmh bench scenario terminal-tasks`) skips. Needs a running Docker.
set -euo pipefail
cd "$(dirname "$0")"
exec python3 run_real_scenario.py "$@"
