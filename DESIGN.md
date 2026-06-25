# World Model Harness — Design

> **"Docker as an LLM." "Stop running your evals in sandboxes." "Simulate your environment without running it."**
>
> A frontier LLM acts as the *environment* your agent steps against, reconstructed from your own
> observability traces — no sandbox, no live services, no flaky resets.

---

## 1. Motivation & prior art

Agentic RL, evals, and red-teaming all need an environment to step against. Real environments are
expensive, flaky, hard to reset, and unsafe to explore at scale. The research direction is to replace
them with a **language world model**: an LLM that predicts the next observation given the current
state and an agent action.

We replicate the *idea* of three papers but deliberately swap training for **prompt optimization on a
frontier model (Opus 4.8 / GPT 5.5)**:

| Paper | What we take | What we change |
|---|---|---|
| **Qwen-AgentWorld: Language World Models for General Agents** (arXiv 2606.24597) | LLM-as-environment: given `(observation, action)` predict next state via next-state-prediction reasoning. | They CPT→SFT→RL a 35B/397B model on 10M trajectories. We use a frontier model + an optimized prompt — zero training. |
| **GEPA: Reflective Prompt Evolution Can Outperform RL** (arXiv 2507.19457) | Reflect on traces in natural language, mutate the prompt, keep a Pareto frontier of candidates. ~35× fewer rollouts than GRPO. | This *is* our "training": we evolve the env prompt against a held-out trace split. |
| **DreamGym: Scaling Agent Learning via Experience Synthesis** (arXiv 2511.03773) | Experience model predicts `(s_{t+1}, r_{t+1})` from **interaction history + top-k demos retrieved by cosine similarity** over a replay buffer (offline-init, online-enriched) + task instruction. Eq. (4): `(s_{t+1}, r_{t+1}) = M_exp(R_t | {(s_i,a_i)}, {d_j}, τ)`, `{d_j} = Topk(cos(φ(s_t,a_t), φ(s_i,a_i)))`. | The replay buffer is the user's ingested OTel traces. `M_exp` is a prompted frontier model, not an SFT'd one. |

**The thesis:** your observability traces already contain the environment's dynamics. Index them,
retrieve the most relevant prior `(state, action) → observation` examples at each step, optimize the
prompt that turns them into a faithful next-state prediction, and you get a queryable environment.

---

## 2. The three optimization layers

These compose; each is independently useful.

### (a) Optimized base prompt
A hand-tuned, env-agnostic system prompt that frames the model as "the environment." Establishes the
output contract (return only what the environment would emit), the reasoning style (predict the
*consequence* of the action), and failure semantics (invalid action → error/failure state).

### (b) GEPA — offline prompt evolution
On ingestion we split traces into **train** and **held-out test**. GEPA loop:
1. Replay each held-out step through the current candidate prompt to get a **predicted** observation.
2. Score predicted vs. real observation with an **LLM judge (Opus)** that returns a scalar score **and a
   natural-language critique**.
3. Reflect on the critiques, mutate the prompt text (and/or retrieval/formatting knobs).
4. Keep a **Pareto frontier** of candidate prompts across trace buckets (e.g. per tool / span kind), so
   a prompt good at `cd` and one good at chat both survive.
5. Iterate within a rollout budget; emit the winning prompt + frontier.

### (c) Live retrieval (DreamGym, runtime)
At every `step()` we retrieve the top-k most similar past `(state, action)` demos from the indexed
traces and inject them into the env prompt. Retrieval is **conditioned on the latest state** (current
session state + incoming action), exactly as DreamGym Eq. (4). This is what makes the env faithful to
*this* user's system rather than a generic guess.

---

## 3. Lifecycle

```
                ┌─────────────────────────── BUILD (CLI is the UI) ───────────────────────────┐
  vendor SDK ──▶│ ingest ──▶ normalize ──▶ split(train/test) ──▶ embed/index ──▶ GEPA optimize │──▶ artifact
  file upload ──▶│  (OTel)    (Trace model)   (replay buffer)     (vector store)   (prompt+frontier)│   (.wmh/)
                └──────────────────────────────────────────────────────────────────────────────┘
                                                  │
                ┌──────────────────────────── SERVE (live, local) ───────────────────────────┐
  agent  ──────▶│ POST /sessions ─▶ POST /sessions/{id}/step ─▶ retrieve ─▶ prompt ─▶ provider │──▶ observation
   (or WorldModel.step in-process)                            (top-k demos)  (Opus/GPT)        │
                └──────────────────────────────────────────────────────────────────────────────┘
```

Build phase = encode all telemetry + optimize. Serve phase = retrieve-then-predict per step.

---

## 4. Core interfaces

### 4.1 Trace model (normalized)
OTel spans (any vendor) are normalized into a generic internal schema. One concrete adapter ships:
**official OpenTelemetry GenAI semconv** (`gen_ai.*`). Others (OpenInference, OpenLLMetry, vendor
JSONL) plug in behind the same `TraceAdapter` protocol.

```python
@dataclass
class Step:                      # one (state, action) -> observation transition
    action: Action               # what the agent did (tool call or message)
    observation: Observation     # what the environment returned
    state_before: EnvState       # snapshot prior to the action
    task: str | None             # the originating task/instruction (τ in DreamGym)
    raw_span_ids: list[str]

@dataclass
class Trace:                     # one full agent session, ordered steps
    trace_id: str
    steps: list[Step]
    source: str                  # vendor / file origin
```

### 4.2 WorldModel — the thing agents call
Stateful sessions. `step()` takes an action, returns an observation, and mutates session state.
The env can also write free-text notes to its own per-session "database" to stay consistent
across a session (e.g. "user created file foo.txt", "cwd is /tmp").

```python
class WorldModel:
    @classmethod
    def load(cls, artifact_dir: str, provider: ProviderConfig) -> "WorldModel": ...
    def new_session(self, task: str | None = None, seed_state: EnvState | None = None) -> Session: ...
    def step(self, session_id: str, action: Action) -> Observation: ...   # DreamGym Eq. (4)
    def get_session(self, session_id: str) -> Session: ...

@dataclass
class Session:
    id: str
    task: str | None
    state: EnvState               # structured + free-text scratchpad the env writes to
    history: list[Step]           # interaction history {(s_i, a_i)} fed back into the prompt
```

`Action` = `{kind: "tool_call"|"message", name, arguments, content}`.
`Observation` = `{content, is_error, reward?, metadata}` (reward optional, supports RL use).

### 4.3 Providers — one interface, four backends
Fresh adapters + a single `get_provider(config)` entry point. All four verified on startup with a
cheap ping.

```python
class Provider(Protocol):
    def complete(self, system: str, messages: list[Message], **kw) -> Completion: ...
    def verify(self) -> VerifyResult: ...   # cheap creds/model check
    def embed(self, texts: list[str]) -> list[list[float]]: ...  # for retrieval (may delegate)

# Backends: AnthropicProvider (Opus 4.8), BedrockProvider (Claude 4.8),
#           AzureOpenAIProvider (GPT 5.5), OpenAIProvider (GPT 5.5)
```

### 4.4 Retriever (DreamGym top-k)
```python
class Retriever(Protocol):
    def index(self, traces: list[Trace]) -> None: ...                       # build phase
    def topk(self, state: EnvState, action: Action, k: int) -> list[Step]: ...  # cos(φ(s,a), φ(s_i,a_i))
    def add(self, step: Step) -> None: ...                                   # online enrichment
```

### 4.5 Optimizer (GEPA) & Judge
```python
class Judge(Protocol):
    def score(self, predicted: Observation, actual: Observation,
              context: Step) -> JudgeResult: ...   # {score: float, critique: str}

class Optimizer(Protocol):
    def optimize(self, train: list[Trace], test: list[Trace],
                 base_prompt: str, budget: int) -> OptimizeResult: ...  # {prompt, frontier, metrics}
```

---

## 5. CLI (the ingestion UI)

```
wmh init                         # scaffold .wmh/ project, write config
wmh providers verify             # ping all 4 configured providers, report OK/fail
wmh ingest --file traces.jsonl   # file upload
wmh ingest --vendor <v> ...      # vendor SDK pull
wmh build                        # normalize -> split -> embed/index -> GEPA optimize -> artifact
wmh serve [--port 8000]          # run the local backend
wmh demo                         # LLM-as-agent makes a tool call vs the WorldModel; print prompt+output
wmh step --session <id> --tool <name> --args '{...}'   # one-off step from the CLI
```

## 6. Local backend (FastAPI)

| Method | Path | Body / returns |
|---|---|---|
| POST | `/sessions` | `{task?, seed_state?}` → `{session_id}` |
| POST | `/sessions/{id}/step` | `{action}` → `{observation}` |
| GET  | `/sessions/{id}` | → `{session}` |
| GET  | `/healthz` | provider + artifact status |

The backend is just a thin transport over an in-process `WorldModel`; both paths share the same code.

## 7. Demo (`wmh demo`)
A throwaway **LLM-as-agent** (base prompt + a few sampled trace examples, *no GEPA*) is prompted to
emit one tool call. We feed that tool call to the `WorldModel`, then print: (1) the exact env prompt
sent, (2) the predicted observation. Shows the loop end-to-end without the agent needing to be real.

---

## 8. Artifact layout (`.wmh/`)
```
.wmh/
  config.toml          # providers, k, model ids, split ratio
  traces/              # normalized Trace JSON
  index/               # vector store (embeddings + metadata)
  prompts/
    base.txt           # layer (a)
    optimized.txt      # GEPA winner, layer (b)
    frontier.json      # Pareto candidates
  metrics.json         # GEPA scores, judge agreement, held-out accuracy
```

---

## 9. What the skeleton ships vs. defers

**Ships (interfaces + stubs, importable, typed, `NotImplementedError` bodies):** package layout,
`WorldModel`/`Session`/`Step`/`Trace` dataclasses, `Provider` protocol + 4 adapter stubs +
`get_provider`, `Retriever`/`Judge`/`Optimizer` protocols, `TraceAdapter` + OTel-GenAI adapter stub,
Typer CLI with all commands wired, FastAPI app with all routes, `.wmh/` config schema, README quickstart.

**Defers (real implementations):** actual embedding/index, GEPA loop internals, judge prompts, vendor
SDK pulls, the optimized base prompt content, provider request/response mapping, session persistence.

---

## 10. Open questions for later
- Concrete embedding model per provider (Voyage/OpenAI/Bedrock Titan?) and whether `φ(s,a)` embeds
  raw text or a structured summary.
- Reward semantics: outcome-only (DreamGym default) vs. dense — likely a pluggable `RewardModel`.
- Session persistence backend (in-memory vs. sqlite) for `serve`.
- GEPA mutation operators: prompt-text only, or also retrieval-k / formatting knobs.
- Multi-environment artifacts in one project (terminal + chat + browser) vs. one artifact per env.
