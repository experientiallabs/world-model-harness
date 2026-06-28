"""Benchmark Qwen-AgentWorld through the world-model-harness open-loop replay path.

This script is intentionally narrow: it implements the exact 30/8/8 whole-trace split requested
for the qwen3.7-max PI trace corpus, builds a train-only HashingEmbedder index, calls an
OpenAI-compatible completion server for the world model, and scores every test step with the
existing harness judges.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shlex
import socket
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, pstdev
from typing import cast

import httpx
from pydantic import BaseModel, Field, JsonValue

from wmh.core.render import render_action
from wmh.core.types import Action, EnvState, JsonObject, Observation, Step, Trace
from wmh.engine.prompts import BASE_ENV_PROMPT
from wmh.optimize.gepa import predict_observation
from wmh.optimize.judge import Judge, LLMJudge, RubricJudge
from wmh.providers import get_provider
from wmh.providers.base import (
    Completion,
    Message,
    Provider,
    ProviderConfig,
    ProviderKind,
    TokenUsage,
    VerifyResult,
)
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder
from wmh.retrieval.leakfree import DemoRetriever
from wmh.tracking.pricing import price_for
from wmh.tracking.tracker import Phase, RunTracker

TRACE_FILE = "/private/tmp/qwen3.7-max-pi-traces.otel.jsonl"
MODEL_NAME = "Qwen/Qwen-AgentWorld-35B-A3B"
DEFAULT_MODEL_DIR = ".wmh/models/qwen-agentworld-35b-a3b-30train-8val-8test-s4405"
DEFAULT_NO_RAG_MODEL_DIR = (
    ".wmh/models/qwen-agentworld-35b-a3b-no-rag-agentworld-swe-30train-8val-8test-s4405"
)
DEFAULT_AGENTWORLD_RAG_MODEL_DIR = (
    ".wmh/models/qwen-agentworld-35b-a3b-rag-agentworld-swe-30train-8val-8test-s4405"
)
BASELINE_PATH = ".wmh/models/qwen3-7-max-pi-30train-8val-8test-s4405/eval_30_8_8_llm_judge.json"
SPLIT_SEED = 4405
TRAIN_TRACES = 30
VAL_TRACES = 8
TEST_TRACES = 8
EXPECTED_COUNTS = {"train": 841, "val": 223, "test": 225}

AGENTWORLD_SWE_PROMPT_URL = (
    "https://github.com/QwenLM/Qwen-AgentWorld/blob/main/prompts/swe/system_prompt.txt"
)
AGENTWORLD_SWE_EVAL_URL = (
    "https://github.com/QwenLM/Qwen-AgentWorld/blob/main/eval/eval.py"
)
AGENTWORLD_SWE_TOOL_DEFINITIONS = """### 1. bash
#### Description
Run a Bash command in the Linux workspace.

#### Arguments
- `command` (string, required): command line to execute.
- `timeout` (number, optional): maximum execution time in seconds.

#### Expected observations
Return the exact stdout/stderr text the shell command would produce. If the command exits non-zero,
include the same terminal output and failure wording the tool would return.

### 2. read
#### Description
Read a UTF-8 text file from the workspace filesystem.

#### Arguments
- `path` (string, required): absolute path to read.

#### Expected observations
Return the exact file contents. If the path is missing or invalid, return the tool's file-not-found
or validation error.

### 3. write
#### Description
Write content to a workspace file, creating or replacing it.

#### Arguments
- `path` (string, required): absolute path to write.
- `content` (string, required): complete file content.

#### Expected observations
On success, return `Successfully wrote N bytes to PATH` with the correct byte count and path. If
arguments are missing or invalid, return the tool validation error.

### 4. edit
#### Description
Apply exact string replacements to a workspace file.

#### Arguments
- `path` (string, required): absolute path to edit.
- `edits` (array, required): replacements with `oldText` and `newText`.

#### Expected observations
On success, return the edit tool's success message. If any `oldText` does not match exactly, return
the edit failure message indicating which edit could not be found."""

AGENTWORLD_NO_DEMONSTRATIONS = "\n\nNo demonstrations are provided for this evaluation."
AGENTWORLD_RESPONSE_TAG = "predicted_observation"


class OpenAICompatibleHTTPProvider:
    """Minimal Provider for vLLM's OpenAI-compatible `/v1/chat/completions` endpoint."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float,
        retries: int,
        chat_template_kwargs: JsonObject | None = None,
    ) -> None:
        self.config = ProviderConfig(kind=ProviderKind.OPENAI, model=model, endpoint=base_url)
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout_seconds)
        self._client = httpx.Client(timeout=self._timeout)
        self._headers = {"Connection": "close"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._retries = retries
        self._chat_template_kwargs = chat_template_kwargs

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        payload: JsonObject = {
            "model": self.config.model,
            "messages": _openai_messages(system, messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self._chat_template_kwargs:
            payload["chat_template_kwargs"] = self._chat_template_kwargs
        response: httpx.Response | None = None
        for attempt in range(self._retries + 1):
            try:
                response = self._client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers,
                    json=payload,
                )
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as exc:
                if attempt >= self._retries or exc.response.status_code < 500:
                    raise
                self._reset_client()
                time.sleep(min(2**attempt, 10))
            except httpx.TransportError:
                if attempt >= self._retries:
                    raise
                self._reset_client()
                time.sleep(min(2**attempt, 10))
        if response is None:
            raise RuntimeError(f"{self.config.model} request failed without a response")
        data = cast("JsonObject", response.json())
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"{self.config.model} returned no choices: {data}")
        first = _as_object(choices[0])
        message = _as_object(first.get("message") if first is not None else None)
        text = _as_str(message.get("content") if message is not None else None)
        return Completion(text=text, usage=_usage_from_response(data))

    def _reset_client(self) -> None:
        self._client.close()
        self._client = httpx.Client(timeout=self._timeout)

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Qwen-AgentWorld benchmark uses HashingEmbedder, not vLLM embed")

    def verify(self) -> VerifyResult:
        try:
            self.complete("", [Message(role="user", content="ping")], max_tokens=1)
        except Exception as exc:  # noqa: BLE001 - verify reports failure, never raises
            return VerifyResult(
                ok=False,
                kind=self.config.kind,
                model=self.config.model,
                detail=str(exc),
            )
        return VerifyResult(ok=True, kind=self.config.kind, model=self.config.model)


class MeteredProvider:
    """Provider wrapper that records token usage by phase."""

    def __init__(
        self, provider: Provider, tracker: RunTracker, phase: Phase
    ) -> None:
        self._provider = provider
        self._tracker = tracker
        self._phase = phase

    @property
    def config(self) -> ProviderConfig:
        return self._provider.config

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        completion = self._provider.complete(
            system,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._tracker.record(self._phase, self.config.model, completion.usage)
        return completion

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._provider.embed(texts)

    def verify(self) -> VerifyResult:
        return self._provider.verify()


class GenericMeteredProvider:
    """Meter any Provider returned by the harness registry."""

    def __init__(self, provider: Provider, tracker: RunTracker, phase: Phase) -> None:
        self._provider = provider
        self._tracker = tracker
        self._phase = phase

    @property
    def config(self) -> ProviderConfig:
        return self._provider.config

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        completion = self._provider.complete(
            system,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._tracker.record(self._phase, self.config.model, completion.usage)
        return completion

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._provider.embed(texts)

    def verify(self) -> VerifyResult:
        return self._provider.verify()


class SplitInfo(BaseModel):
    seed: int
    method: str
    trace_counts: dict[str, int]
    step_counts: dict[str, int]
    trace_ids: dict[str, list[str]]


class RetrievalInfo(BaseModel):
    corpus: str
    embedder: str
    embed_dim: int
    top_k: int
    index_dir: str


class Aggregate(BaseModel):
    mean_score: float = 0.0
    score_std: float = 0.0
    error_flag_accuracy: float = 0.0
    n_steps: int = 0


class BenchmarkStepResult(BaseModel):
    trace_id: str
    step_index: int
    rendered_action: str
    actual: str
    predicted: str
    score: float
    critique: str
    is_error_actual: bool
    is_error_predicted: bool
    task: str | None = None
    dimensions: dict[str, float] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.trace_id}:{self.step_index}"


class BenchmarkReport(BaseModel):
    model_name: str
    serving: JsonObject
    prompt: JsonObject
    split: SplitInfo
    retrieval: RetrievalInfo
    judge: JsonObject
    aggregate: Aggregate
    results: list[BenchmarkStepResult]
    usage: JsonObject
    baseline: JsonObject
    run: JsonObject


def _openai_messages(system: str, messages: list[Message]) -> list[JsonObject]:
    out: list[JsonObject] = []
    if system:
        out.append({"role": "system", "content": system})
    out.extend({"role": message.role, "content": message.content} for message in messages)
    return out


def _as_object(value: JsonValue | None) -> JsonObject | None:
    return value if isinstance(value, dict) else None


def _as_str(value: JsonValue | None) -> str:
    return value if isinstance(value, str) else ""


def _as_int(value: JsonValue | None) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _usage_from_response(data: JsonObject) -> TokenUsage:
    usage = _as_object(data.get("usage"))
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=_as_int(usage.get("prompt_tokens")),
        output_tokens=_as_int(usage.get("completion_tokens")),
    )


def _load_traces(path: str) -> list[Trace]:
    from wmh.ingest import get_adapter

    traces = get_adapter("otel-genai").from_file(path)
    return [trace for trace in traces if trace.steps]


def _load_env_file(path: str | None) -> None:
    if not path:
        return
    env_path = Path(path)
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        value = raw_value.strip()
        os.environ[key.strip()] = shlex.split(value, posix=True)[0] if value else ""


def _normalize_bedrock_model(model: str) -> str:
    return model.removeprefix("bedrock/")


def _load_agentworld_swe_template(path: str) -> tuple[str, str]:
    prompt_path = Path(path)
    template = prompt_path.read_text(encoding="utf-8")
    return template, hashlib.sha256(template.encode("utf-8")).hexdigest()


def _agentworld_state_payload(state: EnvState) -> JsonObject:
    return {
        "structured": state.structured,
        "scratchpad": state.scratchpad,
    }


def _agentworld_demo_payload(step: Step) -> JsonObject:
    return {
        "state_before": _agentworld_state_payload(step.state_before),
        "action": _agentworld_action_payload(step.action),
        "observation": {
            "content": step.observation.content,
            "is_error": step.observation.is_error,
        },
    }


def _agentworld_demonstrations(demos: list[Step]) -> str:
    if not demos:
        return AGENTWORLD_NO_DEMONSTRATIONS
    rendered = []
    for index, demo in enumerate(demos, start=1):
        rendered.append(
            f"## Demonstration {index}\n"
            "The following example is retrieved from the train split only. Match its observation "
            "format when the current action is analogous.\n"
            f"{json.dumps(_agentworld_demo_payload(demo), ensure_ascii=False, indent=2)}"
        )
    return "\n\n# Retrieved Demonstrations\n\n" + "\n\n".join(rendered)


def _build_agentworld_swe_system_prompt(template: str, demos: list[Step]) -> str:
    system = template.replace("{tool_definitions}", AGENTWORLD_SWE_TOOL_DEFINITIONS).replace(
        "{demonstrations}", _agentworld_demonstrations(demos)
    )
    return system


def _agentworld_action_payload(action: Action) -> JsonObject:
    return {
        "name": action.name or "",
        "arguments": json.dumps(action.arguments, ensure_ascii=False),
    }


def _build_agentworld_user_prompt(state: EnvState, action: Action, step_index: int) -> str:
    return (
        "### Current Environment State\n"
        "```json\n"
        f"{json.dumps(_agentworld_state_payload(state), ensure_ascii=False, indent=2)}\n"
        "```\n\n"
        f"### Turn {step_index + 1}\n"
        "**Action:**\n"
        "```json\n"
        f"{json.dumps(_agentworld_action_payload(action), ensure_ascii=False, indent=2)}\n"
        "```"
    )


def _remove_thinking_blocks(text: str, response_tag: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    open_match = re.search(r"<think>", cleaned, flags=re.IGNORECASE)
    if open_match is None:
        return cleaned.strip()
    tag_match = re.search(
        rf"<{re.escape(response_tag)}>",
        cleaned[open_match.end() :],
        flags=re.IGNORECASE,
    )
    end = open_match.end() + tag_match.start() if tag_match else len(cleaned)
    return (cleaned[: open_match.start()] + cleaned[end:]).strip()


def _extract_agentworld_observation(raw: str, response_tag: str = AGENTWORLD_RESPONSE_TAG) -> str:
    cleaned = _remove_thinking_blocks(raw, response_tag)
    start_pattern = rf"<{re.escape(response_tag)}>"
    end_pattern = rf"</{re.escape(response_tag)}>"
    starts = list(re.finditer(start_pattern, cleaned, flags=re.IGNORECASE))
    if starts:
        start = starts[-1].end()
        end = re.search(end_pattern, cleaned[start:], flags=re.IGNORECASE)
        cleaned = cleaned[start : start + end.start()] if end else cleaned[start:]
    marker = "**Environment Observation:**"
    return cleaned.replace(marker, "").strip() if marker in cleaned else cleaned.strip()


def _infer_is_error(content: str) -> bool:
    lowered = content.lower()
    patterns = (
        "validation failed",
        "traceback",
        "syntaxerror",
        "typeerror",
        "referenceerror",
        "error:",
        "exception",
        "no such file or directory",
        "file not found",
        "permission denied",
        "command not found",
        "exited with code",
        "could not find",
        "eaddrinuse",
        "err_unknown",
        "failed",
        "must have required",
        "missing required",
    )
    return any(pattern in lowered for pattern in patterns)


def _predict_agentworld_swe_observation(
    provider: Provider,
    system_template: str,
    demos: list[Step],
    state: EnvState,
    action: Action,
    step_index: int,
    *,
    temperature: float,
    max_tokens: int,
) -> Observation:
    system_prompt = _build_agentworld_swe_system_prompt(system_template, demos)
    user = _build_agentworld_user_prompt(state, action, step_index)
    completion = provider.complete(
        system_prompt,
        [Message(role="user", content=user)],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = _extract_agentworld_observation(completion.text)
    return Observation(content=content, is_error=_infer_is_error(content))


def _split(traces: list[Trace]) -> tuple[list[Trace], list[Trace], list[Trace]]:
    ordered = sorted(traces, key=lambda trace: trace.trace_id)
    random.Random(SPLIT_SEED).shuffle(ordered)
    train = ordered[:TRAIN_TRACES]
    val = ordered[TRAIN_TRACES : TRAIN_TRACES + VAL_TRACES]
    test = ordered[TRAIN_TRACES + VAL_TRACES : TRAIN_TRACES + VAL_TRACES + TEST_TRACES]
    return train, val, test


def _step_count(traces: list[Trace]) -> int:
    return sum(len(trace.steps) for trace in traces)


def _split_info(train: list[Trace], val: list[Trace], test: list[Trace]) -> SplitInfo:
    buckets = {"train": train, "val": val, "test": test}
    return SplitInfo(
        seed=SPLIT_SEED,
        method="sort traces by trace_id, then random.Random(4405).shuffle(sorted_traces)",
        trace_counts={name: len(traces) for name, traces in buckets.items()},
        step_counts={name: _step_count(traces) for name, traces in buckets.items()},
        trace_ids={name: [trace.trace_id for trace in traces] for name, traces in buckets.items()},
    )


def _assert_expected_split(info: SplitInfo) -> None:
    expected_traces = {"train": TRAIN_TRACES, "val": VAL_TRACES, "test": TEST_TRACES}
    if info.trace_counts != expected_traces:
        raise ValueError(f"unexpected split trace counts: {info.trace_counts}")
    if info.step_counts != EXPECTED_COUNTS:
        raise ValueError(f"unexpected split step counts: {info.step_counts}")


def _load_existing(path: Path) -> dict[str, BenchmarkStepResult]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    raw_results = data.get("results")
    if not isinstance(raw_results, list):
        return {}
    loaded: dict[str, BenchmarkStepResult] = {}
    for raw in raw_results:
        if isinstance(raw, dict):
            result = BenchmarkStepResult.model_validate(raw)
            loaded[result.key] = result
    return loaded


def _aggregate(results: list[BenchmarkStepResult]) -> Aggregate:
    if not results:
        return Aggregate()
    scores = [result.score for result in results]
    error_matches = [
        1.0 if result.is_error_predicted == result.is_error_actual else 0.0 for result in results
    ]
    return Aggregate(
        mean_score=fmean(scores),
        score_std=pstdev(scores) if len(scores) > 1 else 0.0,
        error_flag_accuracy=fmean(error_matches),
        n_steps=len(results),
    )


def _tracker_summary(tracker: RunTracker) -> JsonObject:
    record = tracker.record_summary()
    return cast("JsonObject", record.model_dump(mode="json"))


def _price_availability(models: list[str]) -> JsonObject:
    return {model: price_for(model) is not None for model in sorted(set(models))}


def _usage_summary(
    target_tracker: RunTracker,
    judge_tracker: RunTracker,
    *,
    resumed_cached_rows: int = 0,
) -> JsonObject:
    models = [
        event.model for event in [*target_tracker.events, *judge_tracker.events] if event.model
    ]
    summary: JsonObject = {
        "target": _tracker_summary(target_tracker),
        "judge": _tracker_summary(judge_tracker),
        "price_available": _price_availability(models),
    }
    if resumed_cached_rows:
        summary["scope"] = "current_process_only"
        summary["resume_note"] = (
            f"{resumed_cached_rows} rows were loaded from an existing progress report; "
            "token and cost counters cover only uncached rows evaluated by this process."
        )
    return summary


def _baseline_summary(path: Path) -> JsonObject:
    if not path.exists():
        return {"available": False, "path": str(path)}
    data = json.loads(path.read_text(encoding="utf-8"))
    summary: JsonObject = {"available": True, "path": str(path)}
    if isinstance(data, dict):
        aggregate = data.get("aggregate")
        if isinstance(aggregate, dict):
            for key in ("mean_score", "score_std", "error_flag_accuracy", "n_steps"):
                value = aggregate.get(key)
                if isinstance(value, int | float):
                    summary[key] = value
        elif "mean_score" in data:
            for key in ("mean_score", "score_std", "error_flag_accuracy", "n_steps"):
                value = data.get(key)
                if isinstance(value, int | float):
                    summary[key] = value
        else:
            reports = [value for value in data.values() if isinstance(value, dict)]
            if reports:
                first = reports[0]
                for key in ("mean_score", "score_std", "error_flag_accuracy", "n_steps"):
                    value = first.get(key)
                    if isinstance(value, int | float):
                        summary[key] = value
    return summary


def _build_report(
    *,
    args: argparse.Namespace,
    split: SplitInfo,
    results: list[BenchmarkStepResult],
    target_tracker: RunTracker,
    judge_tracker: RunTracker,
    started_at: str,
    status: str,
) -> BenchmarkReport:
    output_path = Path(cast("str", args.output))
    baseline_path = Path(cast("str", args.baseline_path))
    model_name = cast("str", args.model_name)
    judge_model = cast("str", args.judge_model)
    judge_provider = cast("str", args.judge_provider)
    judge_kind = cast("str", args.judge_kind)
    prompt_kind = cast("str", args.prompt_kind)
    no_rag = cast("bool", args.no_rag)
    if prompt_kind == "agentworld-swe":
        prompt_info: JsonObject = {
            "name": "Qwen-AgentWorld SWE system prompt",
            "source": AGENTWORLD_SWE_PROMPT_URL,
            "eval_source": AGENTWORLD_SWE_EVAL_URL,
            "template_file": cast("str | None", args.agentworld_prompt_file),
            "template_sha256": cast("str", args.agentworld_template_sha256),
            "static_no_demo_system_prompt_sha256": cast(
                "str", args.agentworld_prompt_sha256
            ),
            "saved_static_no_demo_system_prompt": cast(
                "str", args.agentworld_prompt_saved_path
            ),
            "user_format": (
                "AgentWorldBench current_prompt action block, with harness state_before rendered "
                "as Current Environment State"
            ),
            "demonstrations": (
                "none"
                if no_rag
                else "per-step train-only leak-free HashingEmbedder top-k demos injected into "
                "the official {demonstrations} placeholder"
            ),
            "response_tag": AGENTWORLD_RESPONSE_TAG,
            "output_contract": (
                "raw environment observation; <predicted_observation> is extracted when present; "
                "is_error is inferred from observation text"
            ),
        }
    else:
        prompt_info = {
            "name": "BASE_ENV_PROMPT",
            "source": "wmh.engine.prompts.BASE_ENV_PROMPT",
            "output_contract": "wmh.core.render.OUTPUT_CONTRACT",
        }
    retrieval = (
        RetrievalInfo(corpus="none", embedder="none", embed_dim=0, top_k=0, index_dir="")
        if no_rag
        else RetrievalInfo(
            corpus="train",
            embedder="HashingEmbedder",
            embed_dim=cast("int", args.embed_dim),
            top_k=cast("int", args.top_k),
            index_dir=str(output_path.parent / "index"),
        )
    )
    run: JsonObject = {
        "status": status,
        "started_at": started_at,
        "updated_at": _now(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "output_path": str(output_path),
        "progress_path": cast("str", args.progress),
        "resumed_cached_rows": cast("int", args.resumed_cached_rows),
    }
    return BenchmarkReport(
        model_name=model_name,
        serving={
            "provider": "openai_compatible_vllm",
            "endpoint": cast("str", args.endpoint),
            "machine": cast("str", args.serving_machine),
            "gpu_ids": cast("str", args.serving_gpus),
            "vllm_version": cast("str", args.vllm_version),
            "command": cast("str", args.serving_command),
            "log_path": cast("str", args.serving_log),
            "max_model_len": cast("int", args.serving_max_model_len),
            "timeout_seconds": cast("float", args.timeout_seconds),
            "target_retries": cast("int", args.target_retries),
            "target_max_tokens": cast("int", args.target_max_tokens),
            "target_temperature": cast("float", args.target_temperature),
            "chat_template_kwargs": (
                {"enable_thinking": False} if cast("bool", args.disable_thinking) else {}
            ),
        },
        prompt=prompt_info,
        split=split,
        retrieval=retrieval,
        judge={
            "kind": judge_kind,
            "provider": judge_provider,
            "model": judge_model if judge_provider == "bedrock" else model_name,
            "region": cast("str", args.judge_region),
            "env_file": cast("str | None", args.env_file),
        },
        aggregate=_aggregate(results),
        results=results,
        usage=_usage_summary(
            target_tracker,
            judge_tracker,
            resumed_cached_rows=cast("int", args.resumed_cached_rows),
        ),
        baseline=_baseline_summary(baseline_path),
        run=run,
    )


def _write_report(path: Path, report: BenchmarkReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _make_judge(
    args: argparse.Namespace,
    target: OpenAICompatibleHTTPProvider,
    tracker: RunTracker,
) -> Judge:
    judge_provider = cast("str", args.judge_provider)
    judge_kind = cast("str", args.judge_kind)
    if judge_provider == "target":
        provider = MeteredProvider(target, tracker, Phase.JUDGE)
    else:
        judge_model = _normalize_bedrock_model(cast("str", args.judge_model))
        provider = GenericMeteredProvider(
            get_provider(
                ProviderConfig(
                    kind=ProviderKind.BEDROCK,
                    model=judge_model,
                    region=cast("str", args.judge_region),
                )
            ),
            tracker,
            Phase.JUDGE,
        )
    return RubricJudge(provider) if judge_kind == "rubric" else LLMJudge(provider)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", default=TRACE_FILE)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"))
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--target-retries", type=int, default=3)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-rag", action="store_true", help="Disable retrieval entirely.")
    parser.add_argument(
        "--prompt-kind",
        choices=["base-env", "agentworld-swe"],
        default="base-env",
        help="Prompt contract for target model predictions.",
    )
    parser.add_argument(
        "--agentworld-prompt-file",
        default=None,
        help="Path to Qwen-AgentWorld prompts/swe/system_prompt.txt when using agentworld-swe.",
    )
    parser.add_argument("--target-max-tokens", type=int, default=1024)
    parser.add_argument("--target-temperature", type=float, default=0.0)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help=(
            "Pass chat_template_kwargs.enable_thinking=false to Qwen-family vLLM chat "
            "templates so observations are returned in message.content."
        ),
    )
    parser.add_argument("--judge-provider", choices=["bedrock", "target"], default="bedrock")
    parser.add_argument("--judge-kind", choices=["match", "rubric"], default="match")
    parser.add_argument("--judge-model", default="us.anthropic.claude-opus-4-8")
    parser.add_argument("--judge-region", default="us-east-1")
    parser.add_argument("--env-file", default=None, help="Optional dotenv file to load first.")
    parser.add_argument("--output", default=f"{DEFAULT_MODEL_DIR}/eval_30_8_8_llm_judge.json")
    parser.add_argument(
        "--progress",
        default=f"{DEFAULT_MODEL_DIR}/eval_30_8_8_llm_judge.partial.json",
    )
    parser.add_argument("--baseline-path", default=BASELINE_PATH)
    parser.add_argument("--serving-machine", default="")
    parser.add_argument("--serving-gpus", default="")
    parser.add_argument("--vllm-version", default="")
    parser.add_argument("--serving-max-model-len", type=int, default=131072)
    parser.add_argument("--serving-command", default="")
    parser.add_argument("--serving-log", default="")
    parser.add_argument("--limit-steps", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    if (
        "--output" not in sys.argv
        and "--prompt-kind" in sys.argv
        and "agentworld-swe" in sys.argv
        and "--no-rag" not in sys.argv
    ):
        parser.set_defaults(
            output=f"{DEFAULT_AGENTWORLD_RAG_MODEL_DIR}/eval_30_8_8_llm_judge.json",
            progress=f"{DEFAULT_AGENTWORLD_RAG_MODEL_DIR}/eval_30_8_8_llm_judge.partial.json",
        )
    args = parser.parse_args()
    args.agentworld_template_sha256 = ""
    args.agentworld_prompt_sha256 = ""
    args.agentworld_prompt_saved_path = ""
    return args


def main() -> None:
    args = parse_args()
    _load_env_file(cast("str | None", args.env_file))
    if cast("str", args.judge_model).startswith("bedrock/"):
        args.judge_model = _normalize_bedrock_model(cast("str", args.judge_model))
    output_path = Path(cast("str", args.output))
    progress_path = Path(cast("str", args.progress))
    started_at = _now()

    traces = _load_traces(cast("str", args.traces))
    train, val, test = _split(traces)
    split = _split_info(train, val, test)
    _assert_expected_split(split)

    if cast("bool", args.no_rag):
        demos = DemoRetriever(None, [], top_k=0)
    else:
        embedder = HashingEmbedder(dim=cast("int", args.embed_dim))
        retriever = EmbeddingRetriever(embedder)
        retriever.index(train)
        retriever.save(output_path.parent / "index")
        demos = DemoRetriever(EmbeddingRetriever(embedder), train, top_k=cast("int", args.top_k))

    agentworld_system_template = ""
    if cast("str", args.prompt_kind) == "agentworld-swe":
        prompt_file = cast("str | None", args.agentworld_prompt_file)
        if not prompt_file:
            raise ValueError(
                "--agentworld-prompt-file is required with --prompt-kind agentworld-swe"
            )
        (
            agentworld_system_template,
            args.agentworld_template_sha256,
        ) = _load_agentworld_swe_template(
            prompt_file,
        )
        static_no_demo_prompt = _build_agentworld_swe_system_prompt(
            agentworld_system_template, []
        )
        args.agentworld_prompt_sha256 = hashlib.sha256(
            static_no_demo_prompt.encode("utf-8")
        ).hexdigest()
        saved_prompt = output_path.parent / "agentworld_swe_system_prompt.filled.txt"
        saved_prompt.parent.mkdir(parents=True, exist_ok=True)
        saved_prompt.write_text(static_no_demo_prompt, encoding="utf-8")
        args.agentworld_prompt_saved_path = str(saved_prompt)

    target_tracker = RunTracker(run_id=f"qwen-agentworld-{int(time.time())}", kind="target")
    judge_tracker = RunTracker(run_id=f"judge-{int(time.time())}", kind="judge")
    target_tracker.start()
    judge_tracker.start()

    target_backend = OpenAICompatibleHTTPProvider(
        model=cast("str", args.model_name),
        base_url=cast("str", args.endpoint),
        api_key=cast("str | None", args.api_key),
        timeout_seconds=cast("float", args.timeout_seconds),
        retries=cast("int", args.target_retries),
        chat_template_kwargs=(
            {"enable_thinking": False} if cast("bool", args.disable_thinking) else None
        ),
    )
    target_provider = MeteredProvider(target_backend, target_tracker, Phase.SERVE)
    judge = _make_judge(args, target_backend, judge_tracker)

    existing = {} if cast("bool", args.no_resume) else _load_existing(progress_path)
    args.resumed_cached_rows = len(existing)
    results: list[BenchmarkStepResult] = []
    completed = 0
    limit = cast("int", args.limit_steps)

    if cast("bool", args.dry_run):
        report = _build_report(
            args=args,
            split=split,
            results=results,
            target_tracker=target_tracker,
            judge_tracker=judge_tracker,
            started_at=started_at,
            status="dry_run",
        )
        _write_report(progress_path, report)
        return

    for trace in test:
        for step_index, step in enumerate(trace.steps):
            key = f"{trace.trace_id}:{step_index}"
            cached = existing.get(key)
            if cached is not None:
                results.append(cached)
                continue
            if cast("str", args.prompt_kind) == "agentworld-swe":
                predicted = _predict_agentworld_swe_observation(
                    target_provider,
                    agentworld_system_template,
                    demos.demos_for(trace.trace_id, step),
                    step.state_before,
                    step.action,
                    step_index,
                    temperature=cast("float", args.target_temperature),
                    max_tokens=cast("int", args.target_max_tokens),
                )
            else:
                predicted = predict_observation(
                    target_provider,
                    BASE_ENV_PROMPT,
                    step.task,
                    step.state_before,
                    step.action,
                    demos=demos.demos_for(trace.trace_id, step),
                )
            verdict = judge.score(predicted, step.observation, step)
            result = BenchmarkStepResult(
                trace_id=trace.trace_id,
                step_index=step_index,
                rendered_action=render_action(step.action),
                actual=step.observation.content,
                predicted=predicted.content,
                score=verdict.score,
                dimensions=verdict.dimensions,
                critique=verdict.critique,
                is_error_actual=step.observation.is_error,
                is_error_predicted=predicted.is_error,
                task=step.task,
            )
            results.append(result)
            completed += 1
            report = _build_report(
                args=args,
                split=split,
                results=results,
                target_tracker=target_tracker,
                judge_tracker=judge_tracker,
                started_at=started_at,
                status="running",
            )
            _write_report(progress_path, report)
            print(
                f"{len(results)}/{split.step_counts['test']} "
                f"{trace.trace_id}:{step_index} score={result.score:.3f}",
                flush=True,
            )
            if limit and completed >= limit:
                target_tracker.stop()
                judge_tracker.stop()
                limited = _build_report(
                    args=args,
                    split=split,
                    results=results,
                    target_tracker=target_tracker,
                    judge_tracker=judge_tracker,
                    started_at=started_at,
                    status="limited",
                )
                _write_report(progress_path, limited)
                return

    target_tracker.stop()
    judge_tracker.stop()
    final_report = _build_report(
        args=args,
        split=split,
        results=results,
        target_tracker=target_tracker,
        judge_tracker=judge_tracker,
        started_at=started_at,
        status="completed",
    )
    _write_report(output_path, final_report)
    _write_report(progress_path, final_report)
    print(
        "completed "
        f"mean={final_report.aggregate.mean_score:.6f} "
        f"std={final_report.aggregate.score_std:.6f} "
        f"error_acc={final_report.aggregate.error_flag_accuracy:.6f} "
        f"n={final_report.aggregate.n_steps}",
        flush=True,
    )


if __name__ == "__main__":
    main()
