"""Tests for the multi-benchmark examples builder.

Run explicitly: `uv run pytest scripts/ -q` (the project's default testpaths is `wmh/`).
"""

from __future__ import annotations

import json

from scripts.build_examples import build_examples
from scripts.sib_to_otel import _trace_id_for
from wmh.ingest import get_adapter


def _write_transcript(path, request: str, command: str, output: str) -> None:  # noqa: ANN001
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "you are an agent"},
                    {"role": "user", "content": f"Customer request: {request}"},
                    {"role": "assistant", "content": f"ok\n```sib_bash\n{command}\n```"},
                    {
                        "role": "user",
                        "content": f"<returncode>0</returncode><output>{output}</output>",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def test_trace_id_namespaced_by_label_avoids_collisions(tmp_path) -> None:  # noqa: ANN001
    f = tmp_path / "task-1.json"
    f.write_text("{}", encoding="utf-8")
    # Same file stem under two benchmarks must yield different trace ids.
    assert _trace_id_for(f, "tau2-bench") != _trace_id_for(f, "terminal-bench")
    # Deterministic across calls.
    assert _trace_id_for(f, "tau2-bench") == _trace_id_for(f, "tau2-bench")


def test_build_examples_writes_per_benchmark_files(tmp_path) -> None:  # noqa: ANN001
    # Lay out a tiny fake SIB cache: results/<run>/<benchmark>/<split>/traces/*.json
    sib = tmp_path / "sib"
    for benchmark in ("tau2-bench", "bird-sql"):
        traces = sib / "results" / "baseline" / benchmark / "train" / "traces"
        traces.mkdir(parents=True)
        _write_transcript(traces / "task-1.json", "do x", "get_user u1", '{"ok": true}')

    out = tmp_path / "examples"
    counts = build_examples(sib, out)

    assert set(counts) == {"tau2-bench", "bird-sql"}
    # Each benchmark file ingests into a valid trace with a paired observation.
    for benchmark in counts:
        traces = get_adapter("otel-genai").from_file(str(out / f"{benchmark}.otel.jsonl"))
        assert len(traces) == 1
        assert traces[0].steps[0].observation.content == '{"ok": true}'
    # Distinct benchmarks produce distinct trace ids even with the same task stem.
    tau = get_adapter("otel-genai").from_file(str(out / "tau2-bench.otel.jsonl"))[0]
    bird = get_adapter("otel-genai").from_file(str(out / "bird-sql.otel.jsonl"))[0]
    assert tau.trace_id != bird.trace_id
