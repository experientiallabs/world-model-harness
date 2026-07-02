#!/usr/bin/env bash
# Overnight SWE-bench Verified capture: grow the trace corpus with a large slice, resiliently.
#
#   examples/swe-bench/capture_overnight.sh
#
# Runs the real mini-swe-agent harness over a wide slice of SWE-bench Verified, live on Bedrock.
# Resilient by design for an unattended overnight run:
#   * Resumable: mini-swe-agent SKIPS instances that already have output (no --redo-existing), so a
#     crash/restart continues where it left off. Re-running this script just fills gaps.
#   * Account/model chain: try the endflow account on Opus 4.6 first (verified: it invokes 4.6 and
#     litellm's Bedrock route works with 4.6 — unlike 4.7, which throws "missing field type"); if a
#     provider becomes unavailable, fall back to the default account on Opus 4.8 (the model the
#     original swe capture used). Whichever model, the SAME output dir is reused so progress carries
#     over.
#   * A watchdog loop restarts the runner if it exits non-zero (transient Docker/Bedrock hiccups),
#     until the slice is exhausted or MAX_RESTARTS is hit.
#
# SWE-bench images are multi-GB x86_64 running under arm64 emulation (~10-30 min/instance), so a wide
# slice is genuinely an overnight job. Output: runs/overnight/<instance_id>/<instance_id>.traj.json.
set -uo pipefail
cd "$(dirname "$0")"

SLICE="${SLICE:-40:240}"          # disjoint-ish from the committed first-40; ~200 instances
WORKERS="${WORKERS:-3}"
OUT="${OUT:-runs/overnight}"
MAX_RESTARTS="${MAX_RESTARTS:-40}"

# (profile, model) chain: primary endflow/4.6, fallback default/4.8.
CHAIN=(
  "endflow:bedrock/us.anthropic.claude-opus-4-6-v1"
  "default:bedrock/us.anthropic.claude-opus-4-8"
)

export AWS_REGION="${AWS_REGION:-us-east-1}" AWS_REGION_NAME="${AWS_REGION_NAME:-us-east-1}"

# Use the TEXT-BASED (backticks) config, NOT the default tool-calling one: litellm's Bedrock route
# injects a malformed `tool_choice` for Opus 4.6/4.7 ("tool_choice.type: Field required" / "missing
# field type"), so the tool-calling config dies on the first step for anything but 4.8. The backticks
# config emits a fenced ```mswea_bash_command``` block instead — no tool_choice — which every Opus
# works with AND which convert_to_wmh.py's fenced-block path parses directly.
CONFIG="${CONFIG:-swebench_backticks.yaml}"

run_once() {
  local profile="$1" model="$2"
  echo "=== capture attempt: profile=$profile model=$model slice=$SLICE workers=$WORKERS ==="
  AWS_PROFILE="$profile" .venv/bin/python -m minisweagent.run.benchmarks.swebench \
    -c "$CONFIG" \
    --subset verified --split test --slice "$SLICE" \
    --environment-class docker \
    -m "$model" -w "$WORKERS" \
    -o "$OUT"
}

restarts=0
while :; do
  progressed=0
  for entry in "${CHAIN[@]}"; do
    profile="${entry%%:*}"; model="${entry#*:}"
    # Quick reachability probe so we skip a dead account fast instead of burning the restart budget.
    if ! AWS_PROFILE="$profile" .venv/bin/python -c "
import boto3,sys
try:
    boto3.client('bedrock-runtime',region_name='$AWS_REGION').converse(
        modelId='${model#bedrock/}', messages=[{'role':'user','content':[{'text':'hi'}]}],
        inferenceConfig={'maxTokens':2})
except Exception as e:
    print(e); sys.exit(1)
" >/dev/null 2>&1; then
      echo "--- $profile/$model unreachable, trying next in chain ---"
      continue
    fi
    run_once "$profile" "$model" && { echo "=== runner exited cleanly ==="; exit 0; }
    progressed=1
    break  # runner errored mid-run; restart the whole loop (re-probe, resume via skip-existing)
  done
  restarts=$((restarts + 1))
  if [ "$progressed" -eq 0 ]; then
    echo "=== no reachable provider in the chain; giving up ==="; exit 1
  fi
  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
    echo "=== hit MAX_RESTARTS=$MAX_RESTARTS; stopping (resume by re-running) ==="; exit 1
  fi
  echo "=== runner exited non-zero; restart $restarts/$MAX_RESTARTS in 20s (resumes, skips done) ==="
  sleep 20
done
