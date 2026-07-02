"""The ICL arm of BENCH-B: in-context learning against the tau-bench world model.

The no-gradient control every trained arm (SFT/PPO/REINFORCE++/GRPO/SDPO) is compared against.
A policy LLM proposes tau tool calls; the world model answers; the episode-end reward judge
scores the rollout and its `critique` is what gets learned — injected back into the policy's
context instead of into its weights.

Modes (`--mode`):
- `base`     no memory, one attempt per scenario. With the Qwen policy this IS the base-model row.
- `single`   k attempts per scenario; each retry sees the judge critiques of ITS OWN prior
             attempts (within-scenario self-correction). Reported row = final attempt.
- `collect`  run the TRAIN scenarios once each and write the cross-task memory JSONL
             (task outcome + critique per scenario) that `multi` consumes.
- `multi`    one attempt per scenario with the frozen cross-task memory from `collect`
             injected (the CLaaS paper's ICL arm shape).

Scenario files are the PINNED sets (scenarios_train.jsonl / scenarios_eval.jsonl — see
pin_scenarios.py); rows key on scenario provenance. Policy backends: `bedrock:<model-id>`
(dev stand-in) or `vllm:<model>@<base-url>` (Qwen3.5-9B on the wake/sleep server's vLLM).

Examples:
    uv run python examples/tau-bench/rl/icl.py --mode base --scenarios eval --limit 2
    uv run python examples/tau-bench/rl/icl.py --mode collect --scenarios train --wm haiku
    uv run python examples/tau-bench/rl/icl.py --mode multi --scenarios eval --wm gpt-5.5 \
        --policy vllm:Qwen/Qwen3.5-9B@http://localhost:8001/v1
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from wmh.config import load_config
from wmh.core.parsing import extract_json_object
from wmh.core.render import render_action
from wmh.core.types import Action, ActionKind, EnvState, JsonObject, Step, Trace
from wmh.engine import ingest, split_traces_3way
from wmh.engine.world_model import WorldModel
from wmh.env import DONE_SIGNAL, WorldModelEnv, run_episode
from wmh.optimize.reward import EpisodeScore
from wmh.providers.base import Message, Provider, ProviderConfig, ProviderKind
from wmh.providers.registry import get_provider

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent / "models" / "tau-bench"
_TRACES_PATH = _HERE.parent / "traces.otel.jsonl"
_SCENARIO_FILES = {"train": _HERE / "scenarios_train.jsonl", "eval": _HERE / "scenarios_eval.jsonl"}

HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"  # dated profile id (undated is rejected)
OPUS = "us.anthropic.claude-opus-4-8"  # eval reward judge: third family vs both WM backends
REGION = "us-east-1"
MAX_STEPS = 12
OBS_CHARS = 800  # observation excerpt per history line in the policy prompt
MEMORY_RECORDS = 8  # most-recent cross-task records injected in `multi`

AGENT_SYSTEM = """You are a customer-service agent operating {domain} tools.
Work the task step by step: one tool call at a time, read the result, then decide the next call.

Available tools (name: argument keys):
{tools}

Reply with ONLY one JSON object per turn:
  {{"tool": "<name>", "arguments": {{...}}}}         to call a tool
  {{"done": true, "summary": "<what you did>"}}      when the task is complete or impossible
{memory}"""


class PinnedScenario(BaseModel):
    """One line of a pinned scenario file (see pin_scenarios.py)."""

    task: str
    provenance: list[str]
    domain: str = "unknown"


class MemoryRecord(BaseModel):
    """What one finished episode teaches the next ones."""

    scenario_id: str  # first provenance trace_id
    domain: str
    success: bool
    reward: float
    critique: str


class RowResult(BaseModel):
    """One eval row entry: a scored episode on one scenario."""

    scenario_id: str
    domain: str
    attempt: int
    steps: int
    stop_reason: str
    reward: float
    success: bool
    critique: str
    wm_cost_usd: float | None = None
    seconds: float = 0.0


class _ToolCall(BaseModel):
    tool: str
    arguments: JsonObject = Field(default_factory=dict)


class _Done(BaseModel):
    done: bool
    summary: str = ""


class PolicyAgent:
    """wmh `Agent`: an LLM proposing tau tool calls from task + episode history (+ ICL memory)."""

    def __init__(
        self,
        provider: Provider,
        tools_block: str,
        domain: str,
        memory_block: str = "",
        temperature: float = 0.7,
    ) -> None:
        self._provider = provider
        self._system = AGENT_SYSTEM.format(
            domain=domain,
            tools=tools_block,
            memory=f"\n{memory_block}" if memory_block else "",
        )
        self._temperature = temperature

    def act(self, task: str | None, state: EnvState, history: list[Step]) -> Action:
        lines = [f"TASK:\n{task or '(none)'}", "", "EPISODE SO FAR:"]
        if not history:
            lines.append("(no steps yet)")
        for i, step in enumerate(history, start=1):
            observation = step.observation.content[:OBS_CHARS]
            flag = " [ERROR]" if step.observation.is_error else ""
            lines.append(f"{i}. {render_action(step.action)}\n   -> {flag}{observation}")
        lines.append("\nYour next JSON:")
        completion = self._provider.complete(
            self._system,
            [Message(role="user", content="\n".join(lines))],
            temperature=self._temperature,
            max_tokens=1024,
        )
        return _parse_action(completion.text)


def _parse_action(text: str) -> Action:
    raw = extract_json_object(text)
    if raw is not None:
        try:
            call = _ToolCall.model_validate_json(raw)
            return Action(kind=ActionKind.TOOL_CALL, name=call.tool, arguments=call.arguments)
        except ValidationError:
            pass
        try:
            done = _Done.model_validate_json(raw)
            if done.done:
                return Action(kind=ActionKind.MESSAGE, content=DONE_SIGNAL)
        except ValidationError:
            pass
    # Unparseable replies end the episode rather than feeding garbage to the world model.
    return Action(kind=ActionKind.MESSAGE, content=DONE_SIGNAL)


def _tool_inventory(train: list[Trace]) -> dict[str, str]:
    """domain -> 'name: arg, arg' lines, derived from the TRAIN split only (leak-free)."""
    tools: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for trace in train:
        domain = str(trace.metadata.get("domain") or "unknown")
        for step in trace.steps:
            if step.action.kind is ActionKind.TOOL_CALL and step.action.name:
                tools[domain][step.action.name].update(step.action.arguments)
    blocks: dict[str, str] = {}
    for domain, by_name in tools.items():
        lines = [
            f"- {name}: {', '.join(sorted(args)) or '(no args)'}"
            for name, args in sorted(by_name.items())
        ]
        blocks[domain] = "\n".join(lines)
    return blocks


def _memory_block(records: list[MemoryRecord], domain: str) -> str:
    """Cross-task learnings: same-domain records first, most recent first, capped."""
    ranked = [r for r in records if r.domain == domain] + [r for r in records if r.domain != domain]
    picked = ranked[:MEMORY_RECORDS]
    if not picked:
        return ""
    lines = ["== Learnings from prior tasks (use them; do not repeat mistakes) =="]
    for r in picked:
        outcome = "SUCCEEDED" if r.success else "FAILED"
        lines.append(f"[{outcome} r={r.reward:.2f} {r.domain}] {r.critique}")
    return "\n".join(lines)


def _self_critique_block(prior: list[RowResult]) -> str:
    if not prior:
        return ""
    lines = ["== Feedback on YOUR previous attempts at THIS task =="]
    for r in prior:
        lines.append(f"[attempt {r.attempt}: reward={r.reward:.2f}] {r.critique}")
    return "\n".join(lines)


def _build_wm(kind: str) -> WorldModel:
    """The environment: haiku (training WM) or gpt-5.5 (pinned eval WM). Judge rides along."""
    if kind == "haiku":
        env_provider = get_provider(
            ProviderConfig(kind=ProviderKind.BEDROCK, model=HAIKU, region=REGION)
        )
        judge = env_provider  # cheap critiques while collecting memory / dev
    elif kind == "gpt-5.5":
        env_provider = get_provider(ProviderConfig(kind=ProviderKind.OPENAI, model="gpt-5.5"))
        judge = get_provider(ProviderConfig(kind=ProviderKind.BEDROCK, model=OPUS, region=REGION))
    else:
        raise SystemExit(f"unknown --wm {kind!r}; use haiku or gpt-5.5")
    return WorldModel.load(str(_MODEL_DIR), env_provider, reward_provider=judge)


def _build_policy(spec: str) -> Provider:
    """--policy bedrock:<model-id> | vllm:<model>@<base-url> | openai:<model>."""
    scheme, _, rest = spec.partition(":")
    if scheme == "bedrock":
        return get_provider(ProviderConfig(kind=ProviderKind.BEDROCK, model=rest, region=REGION))
    if scheme == "vllm":
        model, _, url = rest.partition("@")
        if not url:
            raise SystemExit("vllm policy needs vllm:<model>@<base-url>")
        return get_provider(ProviderConfig(kind=ProviderKind.OPENAI, model=model, endpoint=url))
    if scheme == "openai":
        return get_provider(ProviderConfig(kind=ProviderKind.OPENAI, model=rest))
    raise SystemExit(f"unknown policy spec {spec!r}")


def _episode(
    wm: WorldModel,
    policy: Provider,
    scenario: PinnedScenario,
    tools: dict[str, str],
    memory_block: str,
    temperature: float,
    attempt: int,
) -> RowResult:
    agent = PolicyAgent(
        policy,
        tools.get(scenario.domain, tools.get("unknown", "(tools unknown)")),
        scenario.domain,
        memory_block=memory_block,
        temperature=temperature,
    )
    env = WorldModelEnv(wm, score_on_close=True)
    started = time.monotonic()
    result = run_episode(env, agent, task=scenario.task, max_steps=MAX_STEPS)
    score: EpisodeScore = env.last_score
    usage = env.usage
    return RowResult(
        scenario_id=scenario.provenance[0],
        domain=scenario.domain,
        attempt=attempt,
        steps=len(result.steps),
        stop_reason=str(result.stop_reason),
        reward=score.reward,
        success=score.success,
        critique=score.critique,
        wm_cost_usd=usage.total.cost_usd if usage else None,
        seconds=round(time.monotonic() - started, 1),
    )


def _summarize(rows: list[RowResult]) -> str:
    if not rows:
        return "no rows"
    by_domain: dict[str, list[RowResult]] = defaultdict(list)
    for r in rows:
        by_domain[r.domain].append(r)
    success = sum(1 for r in rows if r.success) / len(rows)
    reward = sum(r.reward for r in rows) / len(rows)
    cost = sum(r.wm_cost_usd or 0.0 for r in rows)
    parts = [f"n={len(rows)} success={success:.2%} mean_reward={reward:.3f} wm_cost=${cost:.2f}"]
    for domain, group in sorted(by_domain.items()):
        ds = sum(1 for r in group if r.success) / len(group)
        parts.append(f"  {domain}: n={len(group)} success={ds:.2%}")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["base", "single", "collect", "multi"], required=True)
    parser.add_argument("--scenarios", choices=["train", "eval"], required=True)
    parser.add_argument("--wm", default="haiku", help="haiku (training WM) | gpt-5.5 (eval WM)")
    parser.add_argument(
        "--policy", default=f"bedrock:{HAIKU}", help="bedrock:<id> | vllm:<m>@<url>"
    )
    parser.add_argument("--limit", type=int, default=0, help="cap scenarios (0 = all)")
    parser.add_argument("--attempts", type=int, default=2, help="attempts per scenario (single)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--memory", type=Path, default=_HERE / "icl_memory.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="results JSONL (default: derived)")
    parser.add_argument("--wandb", action="store_true", help="log rows to wandb wmh-rl-transfer")
    args = parser.parse_args()

    scenario_path = _SCENARIO_FILES[args.scenarios]
    scenarios = [
        PinnedScenario.model_validate_json(line)
        for line in scenario_path.read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        scenarios = scenarios[: args.limit]

    config = load_config(str(_MODEL_DIR))
    train, _val, _test = split_traces_3way(ingest(config, file=str(_TRACES_PATH)), 0.8, 0.1)
    tools = _tool_inventory(train)
    wm = _build_wm(args.wm)
    policy = _build_policy(args.policy)

    memory: list[MemoryRecord] = []
    if args.mode == "multi":
        if not args.memory.exists():
            raise SystemExit(f"--mode multi needs {args.memory} (run --mode collect first)")
        memory = [
            MemoryRecord.model_validate_json(line)
            for line in args.memory.read_text().splitlines()
            if line.strip()
        ]

    run = None
    if args.wandb:
        import wandb  # example-local optional dep: uv run --with wandb ...

        run = wandb.init(
            project="wmh-rl-transfer",
            name=f"icl-{args.mode}-{args.scenarios}-wm_{args.wm}",
            config={k: str(v) for k, v in vars(args).items()},
        )

    rows: list[RowResult] = []
    collected: list[MemoryRecord] = []
    for i, scenario in enumerate(scenarios, start=1):
        attempts = args.attempts if args.mode == "single" else 1
        prior: list[RowResult] = []
        for attempt in range(1, attempts + 1):
            if args.mode == "multi":
                block = _memory_block(memory, scenario.domain)
            elif args.mode == "single":
                block = _self_critique_block(prior)
            else:
                block = ""
            row = _episode(wm, policy, scenario, tools, block, args.temperature, attempt)
            prior.append(row)
            print(
                f"[{i}/{len(scenarios)} a{attempt}] {scenario.domain} "
                f"reward={row.reward:.2f} success={row.success} steps={row.steps} "
                f"({row.seconds}s)"
            )
        final = prior[-1]
        rows.append(final)
        if run is not None:
            run.log(
                {
                    "reward": final.reward,
                    "success": int(final.success),
                    "steps": final.steps,
                    "wm_cost_usd": final.wm_cost_usd or 0.0,
                    "scenario_index": i,
                }
            )
        if args.mode == "collect":
            collected.append(
                MemoryRecord(
                    scenario_id=final.scenario_id,
                    domain=final.domain,
                    success=final.success,
                    reward=final.reward,
                    critique=final.critique,
                )
            )

    out = args.out or _HERE / f"icl_{args.mode}_{args.scenarios}_wm-{args.wm}.results.jsonl"
    out.write_text("\n".join(r.model_dump_json() for r in rows) + "\n", encoding="utf-8")
    if args.mode == "collect":
        args.memory.write_text(
            "\n".join(r.model_dump_json() for r in collected) + "\n", encoding="utf-8"
        )
        print(f"memory -> {args.memory} ({len(collected)} records)")
    print(f"\n== icl --mode {args.mode} --scenarios {args.scenarios} --wm {args.wm} ==")
    print(_summarize(rows))
    print(f"rows -> {out}")
    if run is not None:
        run.summary["success_rate"] = sum(1 for r in rows if r.success) / len(rows)
        run.summary["mean_reward"] = sum(r.reward for r in rows) / len(rows)
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
