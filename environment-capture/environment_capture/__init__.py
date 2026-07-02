"""Run benchmarks for real and record agent-environment transitions as OTel GenAI JSONL."""

from environment_capture.adapter import (
    AgentRun,
    BenchmarkAdapter,
    CaptureAgent,
    CommandEnv,
    ExecResult,
    run_capture,
)
from environment_capture.otel import trace_id_for, trajectory_to_spans, write_spans_jsonl
from environment_capture.sib_cache import load_sib_cache
from environment_capture.trajectory import JsonValue, StepRecord, Task, ToolCall, Trajectory

__all__ = [
    "AgentRun",
    "BenchmarkAdapter",
    "CaptureAgent",
    "CommandEnv",
    "ExecResult",
    "JsonValue",
    "StepRecord",
    "Task",
    "ToolCall",
    "Trajectory",
    "load_sib_cache",
    "run_capture",
    "trace_id_for",
    "trajectory_to_spans",
    "write_spans_jsonl",
]
