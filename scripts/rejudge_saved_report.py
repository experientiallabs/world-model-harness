"""Rejudge saved open-loop benchmark rows without regenerating target predictions."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from wmh.core.types import Observation, Step, Trace
from wmh.ingest import get_adapter
from wmh.optimize.judge import LLMJudge
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.providers.bedrock import BedrockProvider
from wmh.tracking.metered import MeteredProvider
from wmh.tracking.pricing import price_for
from wmh.tracking.tracker import Phase, RunTracker

JsonObject = dict[str, JsonValue]

DEFAULT_TRACE_FILE = "/private/tmp/qwen3.7-max-pi-traces.otel.jsonl"
DEFAULT_ENV_FILE = "/Users/admin/Documents/experientiallabs/world-models/.env.local"
DEFAULT_JUDGE_MODEL = "us.anthropic.claude-opus-4-8"
DEFAULT_REGION = "us-west-2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-score saved benchmark rows with the current LLMJudge prompt."
    )
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--suite-key", default=None)
    parser.add_argument("--trace-file", default=DEFAULT_TRACE_FILE, type=Path)
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE, type=Path)
    parser.add_argument("--split-seed", default=4405, type=int)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-region", default=DEFAULT_REGION)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    load_env_file(args.env_file)
    split = recreate_split(args.trace_file, args.split_seed)
    source = read_json_object(args.source_report)
    source_rows = report_rows(source, args.suite_key)
    selected_rows = source_rows[: args.limit] if args.limit is not None else source_rows
    if len(selected_rows) != 225 and args.limit is None:
        raise ValueError(f"expected 225 rows in source report, found {len(selected_rows)}")

    verify_report_split(source, split)
    test_steps = index_test_steps(split["test"])

    judge_tracker = RunTracker(
        run_id=f"judge-explicit-empty-{int(time.time())}",
        kind="judge",
    )
    provider = MeteredProvider(
        BedrockProvider(
            ProviderConfig(
                kind=ProviderKind.BEDROCK,
                model=args.judge_model,
                region=args.judge_region,
            )
        ),
        judge_tracker,
        base_phase=Phase.JUDGE,
    )
    judge = LLMJudge(provider)

    judged_rows: list[JsonObject] = []
    with judge_tracker.timed():
        trace_offsets: dict[str, int] = {}
        for index, row in enumerate(selected_rows, start=1):
            key = row_key(row, trace_offsets)
            step = test_steps[key]
            step_index = int(key.rsplit(":", 1)[1])
            predicted = Observation(
                content=string_field(row, "predicted"),
                is_error=bool_field(row.get("is_error_predicted")),
            )
            actual = step.observation
            judgement = judge.score(predicted=predicted, actual=actual, context=step)
            judged_row = dict(row)
            judged_row["actual"] = actual.content
            judged_row["predicted"] = predicted.content
            judged_row["score"] = judgement.score
            judged_row["critique"] = judgement.critique
            judged_row["dimensions"] = cast(JsonObject, judgement.dimensions)
            judged_row["is_error_actual"] = actual.is_error
            judged_row["is_error_predicted"] = predicted.is_error
            judged_row.setdefault("step_index", step_index)
            judged_rows.append(judged_row)

            if index == 1 or index == len(selected_rows) or index % args.print_every == 0:
                write_report(
                    args.output,
                    source,
                    split,
                    judged_rows,
                    judge_tracker,
                    args,
                    source_report=args.source_report,
                    started_at=started_at,
                    status="running",
                )
                print_progress(index, len(selected_rows), judged_rows)

    write_report(
        args.output,
        source,
        split,
        judged_rows,
        judge_tracker,
        args,
        source_report=args.source_report,
        started_at=started_at,
        status="completed",
    )


def load_env_file(path: Path) -> None:
    """Load a dotenv-style file into os.environ without echoing secret values."""
    if not path.exists():
        raise FileNotFoundError(path)
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = strip_env_value(value.strip())
        if key and key not in os.environ:
            os.environ[key] = value


def strip_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if " #" in value:
        return value.split(" #", 1)[0].rstrip()
    return value


def recreate_split(trace_file: Path, split_seed: int) -> dict[str, list[Trace]]:
    traces = get_adapter("otel-genai").from_file(str(trace_file))
    usable = [trace for trace in traces if trace.steps]
    ordered = sorted(usable, key=lambda trace: trace.trace_id)
    random.Random(split_seed).shuffle(ordered)
    split = {
        "train": ordered[:30],
        "val": ordered[30:38],
        "test": ordered[38:46],
    }
    expected_step_counts = {"train": 841, "val": 223, "test": 225}
    actual_step_counts = step_counts(split)
    if len(usable) != 46:
        raise ValueError(f"expected 46 usable traces, found {len(usable)}")
    if trace_counts(split) != {"train": 30, "val": 8, "test": 8}:
        raise ValueError(f"unexpected trace counts: {trace_counts(split)}")
    if actual_step_counts != expected_step_counts:
        raise ValueError(f"unexpected step counts: {actual_step_counts}")
    return split


def step_counts(split: dict[str, list[Trace]]) -> dict[str, int]:
    return {bucket: sum(len(trace.steps) for trace in traces) for bucket, traces in split.items()}


def trace_counts(split: dict[str, list[Trace]]) -> dict[str, int]:
    return {bucket: len(traces) for bucket, traces in split.items()}


def split_metadata(split: dict[str, list[Trace]], split_seed: int) -> JsonObject:
    method = f"sort traces by trace_id, then random.Random({split_seed}).shuffle(sorted_traces)"
    return {
        "seed": split_seed,
        "method": method,
        "trace_counts": trace_counts(split),
        "step_counts": step_counts(split),
        "trace_ids": {
            bucket: [trace.trace_id for trace in traces] for bucket, traces in split.items()
        },
    }


def index_test_steps(test_traces: list[Trace]) -> dict[str, Step]:
    steps: dict[str, Step] = {}
    for trace in test_traces:
        for index, step in enumerate(trace.steps):
            steps[f"{trace.trace_id}:{index}"] = step
    return steps


def read_json_object(path: Path) -> JsonObject:
    with path.open() as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return cast(JsonObject, value)


def report_rows(report: JsonObject, suite_key: str | None) -> list[JsonObject]:
    rows = result_container(report, suite_key).get("results")
    if not isinstance(rows, list):
        raise ValueError("source report has no results list")
    objects: list[JsonObject] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index} is not an object")
        objects.append(cast(JsonObject, row))
    return objects


def result_container(report: JsonObject, suite_key: str | None = None) -> JsonObject:
    if suite_key is None:
        return report
    suite = report.get(suite_key)
    if not isinstance(suite, dict):
        raise ValueError(f"source report has no suite object {suite_key!r}")
    return cast(JsonObject, suite)


def verify_report_split(report: JsonObject, split: dict[str, list[Trace]]) -> None:
    report_split = report.get("split")
    if not isinstance(report_split, dict):
        return
    report_trace_ids = report_split.get("trace_ids")
    if not isinstance(report_trace_ids, dict):
        return
    expected_ids = {
        bucket: [trace.trace_id for trace in traces] for bucket, traces in split.items()
    }
    if report_trace_ids != expected_ids:
        raise ValueError("source report split trace IDs do not match recreated split")


def row_key(row: JsonObject, trace_offsets: dict[str, int]) -> str:
    trace_id = string_field(row, "trace_id")
    value = row.get("step_index")
    if isinstance(value, int) and not isinstance(value, bool):
        step_index = value
    else:
        step_index = trace_offsets.get(trace_id, 0)
        trace_offsets[trace_id] = step_index + 1
    return f"{trace_id}:{step_index}"


def string_field(row: JsonObject, key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(f"expected string field {key!r}")
    return value


def int_field(row: JsonObject, key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"expected integer field {key!r}")
    return value


def bool_field(value: JsonValue | None) -> bool:
    if not isinstance(value, bool):
        raise ValueError("expected boolean error flag")
    return value


def write_report(
    output: Path,
    source: JsonObject,
    split: dict[str, list[Trace]],
    judged_rows: list[JsonObject],
    judge_tracker: RunTracker,
    args: argparse.Namespace,
    *,
    source_report: Path,
    started_at: str,
    status: str,
) -> None:
    report = dict(source)
    aggregate_block = aggregate(judged_rows)
    report["split"] = split_metadata(split, args.split_seed)
    judge_block = {
        "kind": "match",
        "provider": "bedrock",
        "model": args.judge_model,
        "region": args.judge_region,
        "env_file": str(args.env_file),
        "prompt_contract": "LLMJudge with explicit empty-observation JSON payloads",
    }
    if args.suite_key is None:
        report["results"] = judged_rows
        report["aggregate"] = aggregate_block
        report["judge"] = judge_block
        report["usage"] = usage_block(source, judge_tracker, args.judge_model)
        report["run"] = run_block(source, source_report, output, started_at, status)
    else:
        suite = dict(result_container(report, args.suite_key))
        suite["results"] = judged_rows
        suite.update(aggregate_block)
        report[args.suite_key] = suite
        report["rejudge"] = nested_rejudge_block(
            report,
            args.suite_key,
            aggregate_block,
            judge_block,
            judge_tracker,
            args.judge_model,
            source_report,
            output,
            started_at,
            status,
        )
    atomic_write_json(output, report)


def aggregate(rows: list[JsonObject]) -> JsonObject:
    scores = [float_field(row, "score") for row in rows]
    if not scores:
        return {"mean_score": 0.0, "score_std": 0.0, "error_flag_accuracy": 0.0, "n_steps": 0}
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    error_matches = sum(
        1
        for row in rows
        if bool_field(row.get("is_error_actual")) == bool_field(row.get("is_error_predicted"))
    )
    return {
        "mean_score": mean,
        "score_std": math.sqrt(variance),
        "error_flag_accuracy": error_matches / len(rows),
        "n_steps": len(rows),
    }


def float_field(row: JsonObject, key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"expected numeric field {key!r}")
    return float(value)


def usage_block(source: JsonObject, judge_tracker: RunTracker, judge_model: str) -> JsonObject:
    source_usage = source.get("usage")
    usage: JsonObject = {}
    if isinstance(source_usage, dict):
        if "target" in source_usage:
            usage["target"] = cast(JsonValue, source_usage["target"])
        if "judge" in source_usage:
            usage["source_judge"] = cast(JsonValue, source_usage["judge"])
    usage["judge"] = judge_tracker.record_summary().model_dump(mode="json")
    usage["price_available"] = {judge_model: price_for(judge_model) is not None}
    return usage


def nested_rejudge_block(
    report: JsonObject,
    suite_key: str,
    aggregate_block: JsonObject,
    judge_block: JsonObject,
    judge_tracker: RunTracker,
    judge_model: str,
    source_report: Path,
    output: Path,
    started_at: str,
    status: str,
) -> JsonObject:
    existing = report.get("rejudge")
    rejudge: JsonObject = dict(cast(JsonObject, existing)) if isinstance(existing, dict) else {}
    suites_value = rejudge.get("suites")
    suites: JsonObject = (
        dict(cast(JsonObject, suites_value)) if isinstance(suites_value, dict) else {}
    )
    suites[suite_key] = {
        "aggregate": aggregate_block,
        "judge": judge_block,
        "usage": {
            "judge": judge_tracker.record_summary().model_dump(mode="json"),
            "price_available": {judge_model: price_for(judge_model) is not None},
        },
        "run": run_block(report, source_report, output, started_at, status),
    }
    rejudge["judge"] = judge_block
    rejudge["suites"] = suites
    return rejudge


def run_block(
    source: JsonObject,
    source_report: Path,
    output: Path,
    started_at: str,
    status: str,
) -> JsonObject:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    source_run = source.get("run")
    return {
        "status": status,
        "started_at": started_at,
        "updated_at": now,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "source_report": str(source_report),
        "output_path": str(output),
        "source_run": source_run if isinstance(source_run, dict) else None,
    }


def atomic_write_json(path: Path, value: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def print_progress(done: int, total: int, rows: list[JsonObject]) -> None:
    summary = aggregate(rows)
    print(
        f"rejudged {done}/{total}; "
        f"mean={summary['mean_score']:.4f}; "
        f"std={summary['score_std']:.4f}; "
        f"err_acc={summary['error_flag_accuracy']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
