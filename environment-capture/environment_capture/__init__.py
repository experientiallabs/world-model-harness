"""Run benchmarks for real and record agent-environment transitions as OTel GenAI JSONL."""

from environment_capture.adapter import (
    AgentRun,
    BenchmarkAdapter,
    CaptureAgent,
    CaptureResult,
    CommandEnv,
    ExecResult,
    TaskFailure,
    run_capture,
)
from environment_capture.baseline_cache import load_baseline_cache
from environment_capture.hygiene import (
    HygieneFinding,
    host_escape_findings,
    partition_contained,
    scan_spans_jsonl,
)
from environment_capture.otel import trace_id_for, trajectory_to_spans, write_spans_jsonl
from environment_capture.trajectory import JsonValue, StepRecord, Task, ToolCall, Trajectory

__all__ = [
    "AgentRun",
    "BenchmarkAdapter",
    "CaptureAgent",
    "CaptureResult",
    "CommandEnv",
    "ExecResult",
    "HygieneFinding",
    "JsonValue",
    "StepRecord",
    "Task",
    "TaskFailure",
    "ToolCall",
    "Trajectory",
    "host_escape_findings",
    "load_baseline_cache",
    "partition_contained",
    "run_capture",
    "scan_spans_jsonl",
    "trace_id_for",
    "trajectory_to_spans",
    "write_spans_jsonl",
]
