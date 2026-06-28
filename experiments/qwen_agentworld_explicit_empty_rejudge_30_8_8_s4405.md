# Qwen AgentWorld Explicit-Empty Rejudge, 30/8/8, Seed 4405

Date: 2026-06-27

Branch: `codex/separate-rag-optimization-corpus`

Starting commit: `2aefa75`

Purpose: rejudge saved Qwen/Qwen-AgentWorld-35B-A3B open-loop next-observation rows after making
the LLM judge prompt explicit about empty observation strings. The target predictions were not
regenerated. The available Bedrock baseline artifact was also rejudged with the same prompt for a
consistent comparison.

Trace corpus: `/private/tmp/qwen3.7-max-pi-traces.otel.jsonl`

Judge credentials source: `/Users/admin/Documents/experientiallabs/world-models/.env.local`

Split:

- Method: sort trace IDs, then `random.Random(4405).shuffle(sorted_traces)`
- Trace counts: train 30, validation 8, test 8
- Step counts: train 841, validation 223, test 225

Commands:

```bash
UV_NO_SYSTEM_CONFIG=1 UV_CACHE_DIR=/tmp/.uv-cache uv run python scripts/rejudge_saved_report.py \
  --source-report .wmh/models/qwen-agentworld-35b-a3b-30train-8val-8test-s4405/eval_30_8_8_llm_judge.json \
  --output .wmh/models/qwen-agentworld-35b-a3b-30train-8val-8test-s4405/eval_30_8_8_opus48_explicit_empty_judge.json \
  --print-every 25

UV_NO_SYSTEM_CONFIG=1 UV_CACHE_DIR=/tmp/.uv-cache uv run python scripts/rejudge_saved_report.py \
  --source-report .wmh/models/qwen-agentworld-35b-a3b-no-rag-agentworld-swe-30train-8val-8test-s4405/eval_30_8_8_llm_judge.json \
  --output .wmh/models/qwen-agentworld-35b-a3b-no-rag-agentworld-swe-30train-8val-8test-s4405/eval_30_8_8_opus48_explicit_empty_judge.json \
  --print-every 25

UV_NO_SYSTEM_CONFIG=1 UV_CACHE_DIR=/tmp/.uv-cache uv run python scripts/rejudge_saved_report.py \
  --source-report .wmh/models/qwen-agentworld-35b-a3b-rag-agentworld-swe-30train-8val-8test-s4405/eval_30_8_8_opus48_judge.json \
  --output .wmh/models/qwen-agentworld-35b-a3b-rag-agentworld-swe-30train-8val-8test-s4405/eval_30_8_8_opus48_explicit_empty_judge.json \
  --print-every 25

UV_NO_SYSTEM_CONFIG=1 UV_CACHE_DIR=/tmp/.uv-cache uv run python scripts/rejudge_saved_report.py \
  --suite-key base_test \
  --source-report .wmh/models/qwen3-7-max-pi-30train-8val-8test-s4405/eval_30_8_8_llm_judge.json \
  --output .wmh/models/qwen3-7-max-pi-30train-8val-8test-s4405/eval_30_8_8_opus48_explicit_empty_judge.json \
  --print-every 25

UV_NO_SYSTEM_CONFIG=1 UV_CACHE_DIR=/tmp/.uv-cache uv run python scripts/rejudge_saved_report.py \
  --suite-key optimized_test \
  --source-report .wmh/models/qwen3-7-max-pi-30train-8val-8test-s4405/eval_30_8_8_opus48_explicit_empty_judge.json \
  --output .wmh/models/qwen3-7-max-pi-30train-8val-8test-s4405/eval_30_8_8_opus48_explicit_empty_judge.json \
  --print-every 25
```

Results:

| Variant | Source report | Corrected report | Mean | Std | Error flag accuracy | Judge cost |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| BASE_ENV_PROMPT + train-only RAG | `.wmh/models/qwen-agentworld-35b-a3b-30train-8val-8test-s4405/eval_30_8_8_llm_judge.json` | `.wmh/models/qwen-agentworld-35b-a3b-30train-8val-8test-s4405/eval_30_8_8_opus48_explicit_empty_judge.json` | 0.1078 | 0.1521 | 0.9467 | $3.2075 |
| AgentWorld SWE prompt, no RAG | `.wmh/models/qwen-agentworld-35b-a3b-no-rag-agentworld-swe-30train-8val-8test-s4405/eval_30_8_8_llm_judge.json` | `.wmh/models/qwen-agentworld-35b-a3b-no-rag-agentworld-swe-30train-8val-8test-s4405/eval_30_8_8_opus48_explicit_empty_judge.json` | 0.5201 | 0.4177 | 0.8756 | $3.5178 |
| AgentWorld SWE prompt + train-only RAG | `.wmh/models/qwen-agentworld-35b-a3b-rag-agentworld-swe-30train-8val-8test-s4405/eval_30_8_8_opus48_judge.json` | `.wmh/models/qwen-agentworld-35b-a3b-rag-agentworld-swe-30train-8val-8test-s4405/eval_30_8_8_opus48_explicit_empty_judge.json` | 0.5360 | 0.4301 | 0.8311 | $3.7338 |

Baseline comparison:

| Variant | Corrected report | Mean | Std | Error flag accuracy | Judge cost |
| --- | --- | ---: | ---: | ---: | ---: |
| Bedrock BASE_ENV_PROMPT baseline | `.wmh/models/qwen3-7-max-pi-30train-8val-8test-s4405/eval_30_8_8_opus48_explicit_empty_judge.json` | 0.3937 | 0.4463 | 0.5289 | $3.2685 |
| Bedrock optimized prompt baseline | `.wmh/models/qwen3-7-max-pi-30train-8val-8test-s4405/eval_30_8_8_opus48_explicit_empty_judge.json` | 0.5118 | 0.4453 | 0.6933 | $3.2815 |

Notes:

- The BASE_ENV_PROMPT + RAG source produced 224 empty predictions out of 225 rows, so the old
  Opus result was inflated by the judge treating empty predictions too generously.
- The AgentWorld SWE no-RAG and AgentWorld SWE + RAG sources produced zero empty predictions.
- The Bedrock baseline artifact had six empty predictions in `base_test` and six in
  `optimized_test`; both suites were rejudged with the explicit-empty prompt.
- The previous `eval_30_8_8_opus48_judge.json` files should be treated as superseded by the
  `eval_30_8_8_opus48_explicit_empty_judge.json` reports.

## Merged-Main AgentWorld SWE No-RAG Rerun

Date: 2026-06-28

Context: merged latest `origin/main` (`f495791`) into
`codex/separate-rag-optimization-corpus` and regenerated only the
Qwen/Qwen-AgentWorld-35B-A3B AgentWorld SWE prompt, no-RAG test predictions with the updated eval
prompt rendering. No RAG demos were supplied.

Report:

`.wmh/models/qwen-agentworld-35b-a3b-no-rag-agentworld-swe-main-f495791-30train-8val-8test-s4405/eval_30_8_8_opus48_explicit_empty_judge.json`

Serving:

- Machine: `h100-dev-box`, GPUs `0,1`
- vLLM: `0.23.0`
- Model: `Qwen/Qwen-AgentWorld-35B-A3B`
- Max model length: 131072
- Target sampling: temperature 0.6, max tokens 2048, thinking disabled
- Prompt: Qwen AgentWorld SWE system prompt from `/private/tmp/Qwen-AgentWorld/prompts/swe/system_prompt.txt`
- Judge: Bedrock `us.anthropic.claude-opus-4-8`, match judge, `us-west-2`

Result:

| Variant | Mean | Std | Error flag accuracy | n |
| --- | ---: | ---: | ---: | ---: |
| AgentWorld SWE prompt, no RAG, merged-main eval prompt | 0.5259 | 0.4126 | 0.8711 | 225 |

Comparison against prior explicit-empty reports:

| Comparison | Delta mean |
| --- | ---: |
| vs prior AgentWorld SWE no-RAG | +0.0057 |
| vs AgentWorld SWE + train-only RAG | -0.0102 |
| vs Bedrock BASE_ENV_PROMPT baseline | +0.1322 |
| vs Bedrock optimized prompt baseline | +0.0140 |

Operational notes:

- The run completed all 225 held-out steps with the same whole-trace split: train 841, validation
  223, test 225.
- Target transport hit a local tunnel read-timeout around row 79 and a transient remote protocol
  disconnect around row 156. The runner resumed from partial artifacts and skipped already-judged
  rows; no completed row was regenerated during resume.
- `scripts/run_agentworld_benchmark.py` was hardened to close vLLM HTTP connections and retry
  transient target transport/5xx errors.
- The final JSON includes all 225 rows. Its usage block is explicitly scoped to the final resumed
  segment, rows 156-225; earlier segment token/cost usage was not persisted by the ad hoc runner.
