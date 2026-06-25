"""GEPA reflective prompt evolution.

GEPA (arXiv 2507.19457): replay held-out steps through a candidate prompt, score predicted vs.
real observation with the LLM judge (which also returns a natural-language critique), reflect on
those critiques to mutate the prompt, and keep a Pareto frontier of candidates across trace buckets.

We do NOT re-implement the evolutionary search: we drive the GEPA authors' reference engine
(`gepa` on PyPI) through a small `GEPAAdapter`. The adapter is the only integration point — it
replays a candidate prompt over held-out steps, scores each with our `Judge`, and turns the judge
critiques into the reflective dataset the engine feeds back to the reflection LM.

The optimizer stays decoupled from the serving engine: replaying a candidate only needs a
`Provider` (see `predict_observation`), so we do NOT import `wmh.engine` (that would create the
cycle engine -> optimize -> engine). The env-prompt assembly here is a minimal local copy.
TODO: converge `_assemble_env_prompt` with `wmh.engine.prompts.build_env_prompt` during integration
so the optimizer evolves against the exact prompt the world model serves.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import gepa
from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from pydantic import BaseModel, Field

from wmh.core.types import Action, EnvState, JsonObject, JsonValue, Observation, Step, Trace
from wmh.optimize.judge import Judge
from wmh.providers.base import Message, Provider


def _render_json(value: JsonObject) -> str:
    """Stable, sorted-key JSON for an object so equal state/args render identically across runs."""
    return json.dumps(value, sort_keys=True, default=str)


# The single named component GEPA evolves: the specialized env (system) prompt.
ENV_PROMPT_COMPONENT = "env_prompt"


class OptimizeMetrics(BaseModel):
    """Outcome metrics from an optimization run."""

    held_out_accuracy: float = 0.0  # mean judge score on the held-out split
    judge_agreement: float = 0.0  # judge self-consistency / human-agreement proxy
    rollouts_used: int = 0


class OptimizeResult(BaseModel):
    prompt: str  # winning specialized env prompt
    frontier: list[str] = Field(default_factory=list)  # Pareto candidates
    metrics: OptimizeMetrics = Field(default_factory=OptimizeMetrics)


@runtime_checkable
class Optimizer(Protocol):
    def optimize(
        self, train: list[Trace], test: list[Trace], base_prompt: str, budget: int
    ) -> OptimizeResult: ...


# --- env-prompt assembly + prediction helper (provider-only; no engine import) -------------------

# Minimal local mirror of wmh.engine.prompts.BASE_ENV_PROMPT's intent.
# TODO: converge with the engine's base prompt during integration.
_PREDICT_INSTRUCTION = (
    "You ARE the environment. Given the environment state, similar past examples, the originating "
    "task, and the agent's latest action, output ONLY what the environment returns in response. "
    "Predict the consequence; if the action is invalid, return the error the environment would "
    "emit. Never address the agent or explain yourself."
)


def _render_demo(step: Step) -> str:
    action = step.action
    label = action.name or action.content or "(none)"
    return (
        f"- action({action.kind.value}): {label} args={_render_json(action.arguments)}\n"
        f"  -> observation(is_error={step.observation.is_error}): {step.observation.content}"
    )


def _assemble_env_prompt(
    prompt: str,
    task: str | None,
    state: EnvState,
    action: Action,
    demos: list[Step],
) -> tuple[str, str]:
    """Return (system, user) for a world-model completion. Minimal local version of the engine's.

    TODO: replace with `wmh.engine.prompts.build_env_prompt` once the import cycle is resolved.
    """
    system = f"{_PREDICT_INSTRUCTION}\n\n--- SPECIALIZED ENV PROMPT ---\n{prompt}"
    demo_block = "\n".join(_render_demo(d) for d in demos) if demos else "(none)"
    action_label = action.name or action.content or "(none)"
    user = (
        f"TASK: {task or '(none)'}\n\n"
        f"ENV STATE:\n  structured: {_render_json(state.structured)}\n"
        f"  scratchpad: {state.scratchpad}\n\n"
        f"SIMILAR PAST EXAMPLES:\n{demo_block}\n\n"
        f"AGENT ACTION ({action.kind.value}): {action_label} args={_render_json(action.arguments)}"
        "\n\nENVIRONMENT RESPONSE:"
    )
    return system, user


def predict_observation(
    provider: Provider,
    prompt: str,
    task: str | None,
    state: EnvState,
    action: Action,
    demos: list[Step],
) -> Observation:
    """Predict the observation for (state, action) under `prompt`, using only a Provider.

    This is the single rollout primitive GEPA replays. The output contract is plain text: the
    model's completion *is* the observation content (TODO: align with the world model's eventual
    structured error/reward contract in wmh.engine.world_model._parse_observation).
    """
    system, user = _assemble_env_prompt(prompt, task, state, action, demos)
    completion = provider.complete(
        system, [Message(role="user", content=user)], temperature=0.0, max_tokens=1024
    )
    return Observation(content=completion.text.strip())


# --- GEPA adapter --------------------------------------------------------------------------------


@dataclass
class _StepTrajectory:
    """Per-example trace captured during evaluation, consumed by make_reflective_dataset."""

    step: Step
    predicted: Observation
    score: float
    critique: str


class WorldModelGEPAAdapter(GEPAAdapter[Step, _StepTrajectory, Observation]):
    """Bridges the world model to the GEPA engine.

    - DataInst is a held-out `Step` (its `state_before`, `action`, `task` are the input; its
      `observation` is the ground truth).
    - A candidate is `{ENV_PROMPT_COMPONENT: <prompt text>}`.
    - Scores are judge scores in 0..1 (higher is better), aggregated by GEPA via sum/mean.
    """

    def __init__(self, provider: Provider, judge: Judge) -> None:
        self._provider = provider
        self._judge = judge

    def evaluate(
        self,
        batch: list[Step],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[_StepTrajectory, Observation]:
        prompt = candidate[ENV_PROMPT_COMPONENT]
        outputs: list[Observation] = []
        scores: list[float] = []
        trajectories: list[_StepTrajectory] | None = [] if capture_traces else None
        for step in batch:
            try:
                predicted = predict_observation(
                    self._provider, prompt, step.task, step.state_before, step.action, demos=[]
                )
                result = self._judge.score(predicted, step.observation, step)
                score, critique = result.score, result.critique
            except Exception as exc:  # noqa: BLE001 - per-example failure must not abort the run
                predicted = Observation(content="", is_error=True)
                score, critique = 0.0, f"Rollout failed: {exc}"
            outputs.append(predicted)
            scores.append(score)
            if trajectories is not None:
                trajectories.append(
                    _StepTrajectory(step=step, predicted=predicted, score=score, critique=critique)
                )
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[_StepTrajectory, Observation],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, JsonValue]]]:
        records: list[Mapping[str, JsonValue]] = []
        for traj in eval_batch.trajectories or []:
            action = traj.step.action
            records.append(
                {
                    "Inputs": {
                        "task": traj.step.task or "(none)",
                        "state": _state_repr(traj.step.state_before),
                        "action": f"{action.kind.value}: "
                        f"{action.name or action.content or '(none)'} args={action.arguments!r}",
                    },
                    "Generated Outputs": traj.predicted.content,
                    "Feedback": (
                        f"score={traj.score:.2f}. {traj.critique} "
                        f"Expected (real) observation: {traj.step.observation.content}"
                    ),
                }
            )
        # GEPA only ever asks us to update the components it selected; we own a single one.
        return {component: records for component in components_to_update}


def _state_repr(state: EnvState) -> str:
    return f"structured={state.structured!r} scratchpad={state.scratchpad!r}"


# --- reflection LM adapter -----------------------------------------------------------------------

_REFLECTION_SYSTEM = (
    "You improve the system prompt for an LLM that simulates an environment for an AI agent. "
    "Given the current prompt and feedback on where its predicted observations diverged from the "
    "real environment, propose an improved prompt. Keep it general across actions; do not overfit "
    "to a single example."
)


def _reflection_lm(provider: Provider):  # noqa: ANN202 - returns gepa's LanguageModel callable
    """Wrap a Provider as GEPA's reflection LM: `(str | list[dict]) -> str`."""

    def call(prompt: str | list[dict[str, JsonValue]]) -> str:
        text = prompt if isinstance(prompt, str) else _flatten_chat(prompt)
        completion = provider.complete(
            _REFLECTION_SYSTEM,
            [Message(role="user", content=text)],
            temperature=1.0,
            max_tokens=2048,
        )
        return completion.text

    return call


def _flatten_chat(messages: list[dict[str, JsonValue]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


# --- the optimizer -------------------------------------------------------------------------------


class GEPAOptimizer:
    """Reflective prompt evolution against the held-out trace split (drives the `gepa` engine)."""

    def __init__(self, provider: Provider, judge: Judge) -> None:
        self._provider = provider
        self._judge = judge

    def optimize(
        self, train: list[Trace], test: list[Trace], base_prompt: str, budget: int
    ) -> OptimizeResult:
        train_steps = [step for trace in train for step in trace.steps]
        val_steps = [step for trace in test for step in trace.steps]
        # GEPA samples minibatches from the trainset; fall back to val when train is empty.
        trainset = train_steps or val_steps
        valset = val_steps or train_steps
        if not trainset or budget <= 0:
            # Nothing to optimize against (or no budget): the base prompt is the only candidate.
            return OptimizeResult(prompt=base_prompt, frontier=[base_prompt])

        adapter = WorldModelGEPAAdapter(self._provider, self._judge)
        result = gepa.optimize(
            seed_candidate={ENV_PROMPT_COMPONENT: base_prompt},
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=_reflection_lm(self._provider),
            candidate_selection_strategy="pareto",
            max_metric_calls=budget,
            reflection_minibatch_size=min(3, len(trainset)),
            display_progress_bar=False,
            raise_on_exception=False,
            seed=0,
        )

        best = _candidate_text(result.candidates[result.best_idx])
        frontier = _frontier_prompts(result)
        return OptimizeResult(
            prompt=best,
            frontier=frontier,
            metrics=OptimizeMetrics(
                held_out_accuracy=float(result.val_aggregate_scores[result.best_idx]),
                # TODO: judge_agreement needs repeated/independent judging; left at default for now.
                judge_agreement=0.0,
                rollouts_used=int(result.total_metric_calls or 0),
            ),
        )


def _candidate_text(candidate: dict[str, str]) -> str:
    return candidate[ENV_PROMPT_COMPONENT]


def _frontier_prompts(result: gepa.GEPAResult) -> list[str]:
    """Collect the Pareto-frontier candidate prompts (deduped, best first)."""
    frontier_idxs: set[int] = set()
    for idxs in result.per_val_instance_best_candidates.values():
        frontier_idxs.update(idxs)
    if not frontier_idxs:
        frontier_idxs = {result.best_idx}
    ordered = sorted(frontier_idxs, key=lambda i: result.val_aggregate_scores[i], reverse=True)
    prompts: list[str] = []
    for i in ordered:
        text = _candidate_text(result.candidates[i])
        if text not in prompts:
            prompts.append(text)
    return prompts
