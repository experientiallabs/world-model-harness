#!/usr/bin/env bash
# Ingest Langfuse traces into a world model, end to end.
#
# Langfuse exports a TRACE as an observation tree (not OTLP spans). Grab one (or a page) via the
# public API or the SDK, then let the `langfuse` adapter normalize it into the OTel-GenAI JSONL that
# `wmh build` consumes.
set -euo pipefail

EXPORT="${1:-langfuse_export.json}"
OUT="${2:-langfuse.otel.jsonl}"
MODEL="${3:-langfuse-demo}"

# 1) Export from Langfuse (pick one). The adapter accepts a single trace object, a JSON array of
#    traces, an API list page ({"data": [...]}), or JSONL (one trace per line).
#
#    Public API — a single trace by id (LANGFUSE_* keys are your project credentials):
#      curl -s -u "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" \
#        "$LANGFUSE_HOST/api/public/traces/$TRACE_ID" > "$EXPORT"
#
#    Public API — a page of recent traces (newest first):
#      curl -s -u "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" \
#        "$LANGFUSE_HOST/api/public/traces?limit=50" > "$EXPORT"
#
#    SDK (Python): langfuse.api.trace.get(trace_id).dict() -> dump to JSON.

# 2) Normalize Langfuse -> OTel-GenAI JSONL (the shape build/eval read).
uv run wmh ingest run --source langfuse --file "$EXPORT" --out "$OUT"

# 3) Build a named world model from the ingested traces.
uv run wmh build --name "$MODEL" --file "$OUT" --no-interactive

echo "Built world model '$MODEL' from $EXPORT (via $OUT)."
