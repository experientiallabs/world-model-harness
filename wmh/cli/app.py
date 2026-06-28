"""`wmh` CLI — ingestion UI and operator console for the harness.

Deliberately small. The lifecycle is:
    providers verify -> build -> list -> serve / demo / play
`build` creates the project artifact directory itself, so there is no separate init step. World
models are named (`--name`), stored under `<root>/models/<name>/`, and listed with `wmh list`.
"""

from __future__ import annotations

import typer
from rich.console import Console

from wmh.config import (
    ARTIFACT_DIR,
    DEFAULT_MODEL_NAME,
    PROVIDER_ENV_VARS,
    HarnessConfig,
    WorldModelStore,
    load_config,
    validate_name,
)
from wmh.providers import ProviderConfig, ProviderKind, verify_all, verify_embedder
from wmh.providers.base import EmbedderKind

app = typer.Typer(help="World Model Harness: a frontier LLM acts as your agent's environment.")
providers_app = typer.Typer(help="Manage and verify LLM providers.")
app.add_typer(providers_app, name="providers")
bench_app = typer.Typer(
    help="Run benchmarks (open-loop fidelity) and view the leaderboard.",
    invoke_without_command=True,
)
app.add_typer(bench_app, name="bench")
_console = Console()
_CHECK = "[green]✓[/green]"

# Module-level singleton: a typer.Argument call can't be a default inline (ruff B008).
_EVAL_FILES = typer.Argument(..., help="OTel trace files to score (one corpus each).")

# Where committed benchmark definitions live ("filesystem as DB"); shared with the infra chat.
BENCHMARKS_DIR = "benchmarks"


def _resolve_benchmarks_dir(benchmarks: str) -> str:
    """Resolve the benchmarks dir, tolerant of the current working directory.

    `--benchmarks` defaults to the relative `"benchmarks"`, which only exists at the repo root — so
    running `wmh bench …` from a subdir (e.g. `tools/<bench>-capture/`) would miss it. When the
    given path doesn't exist as-is but the repo's committed `benchmarks/` does, fall back to that
    (located relative to this package, like `default_bundled_dir` does for `world-models/`). An
    explicit `--benchmarks <path>` that exists is always honored.
    """
    from pathlib import Path

    if Path(benchmarks).exists():
        return benchmarks
    repo_root_benchmarks = Path(__file__).resolve().parent.parent.parent / BENCHMARKS_DIR
    if benchmarks == BENCHMARKS_DIR and repo_root_benchmarks.is_dir():
        return str(repo_root_benchmarks)
    return benchmarks  # leave as-is; the caller surfaces a clear "not found" error


@providers_app.command("verify")
def providers_verify(
    name: str = typer.Option(None, "--name", help="Verify one model's providers (default: all)."),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir."),
) -> None:
    """Ping every configured provider (completion + embed path) and report status.

    Gathers provider configs from the built world models (one `--name`, or all of them, deduped by
    kind+model), so a brand-new project with nothing built yet has nothing to verify. The phi embed
    path of each model is checked too, unless it is the offline (creds-free) hashing embedder.
    """
    store = WorldModelStore(root)
    names = [name] if name is not None else store.list_names()
    if not names:
        _console.print("[yellow]no world models built yet[/yellow]; run `wmh build --name <name>`")
        return
    configs: list[HarnessConfig] = []
    for model_name in names:
        try:
            model_dir = str(store.resolve(model_name))
        except (FileNotFoundError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        configs.append(load_config(model_dir))

    # Dedup completion providers by kind+model across all selected models.
    seen: set[tuple[str, str]] = set()
    providers: list[ProviderConfig] = []
    for config in configs:
        for pc in config.providers:
            key = (pc.kind.value, pc.model)
            if key not in seen:
                seen.add(key)
                providers.append(pc)
    for result in verify_all(providers):
        mark = "[green]ok[/green]" if result.ok else "[red]fail[/red]"
        _console.print(f"{mark} {result.kind.value} ({result.model}) {result.detail}")

    # Verify each distinct provider-backed embed path (skip the offline hashing embedder).
    embed_seen: set[tuple[str, str]] = set()
    for config in configs:
        if config.embed_provider is EmbedderKind.HASHING:
            continue
        embed_config = config.embed_provider_config()
        key = (embed_config.kind.value, embed_config.model)
        if key in embed_seen:
            continue
        embed_seen.add(key)
        result = verify_embedder(embed_config)
        mark = "[green]ok[/green]" if result.ok else "[red]fail[/red]"
        _console.print(f"{mark} embed:{result.kind.value} ({result.model}) {result.detail}")


@app.command("build")
def build(
    name: str = typer.Option(None, "--name", help="Name for this world model."),
    file: str = typer.Option(None, "--file", help="Path to exported traces (OTLP-JSON / JSONL)."),
    vendor: str = typer.Option(None, "--vendor", help="Vendor name to pull traces via SDK."),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir holding all world models."),
    provider: str = typer.Option("bedrock", "--provider", help="Provider that serves the model."),
    model: str = typer.Option("us.anthropic.claude-opus-4-8", help="Serve provider model id."),
    region: str = typer.Option(None, help="AWS region (Bedrock)."),
    gepa_budget: int = typer.Option(50, help="GEPA rollout budget."),
    train_split: float = typer.Option(
        0.8, help="Train/held-out ratio for GEPA's internal split (lower = bigger valset)."
    ),
    embed_provider: str = typer.Option(
        "hashing", help="phi embedder: hashing (offline) | bedrock | openai | azure_openai."
    ),
    embed_model: str = typer.Option(None, help="Embeddings model id / Azure embedding deployment."),
    embed_dim: int = typer.Option(512, help="phi dimensionality (index + query must agree)."),
    interactive: bool = typer.Option(
        None,
        "--interactive/--no-interactive",
        help="Guided creation wizard. Default: on at a TTY when inputs are missing.",
    ),
) -> None:
    """Ingest traces (file upload or vendor SDK pull) and build a named world model.

    Stores the artifact under `<root>/models/<name>/`: ingest -> normalize -> split(train/test) ->
    embed/index -> GEPA optimize -> write. Re-running with the same `--name` rebuilds it.

    With no `--name`/`--file` on an interactive terminal, this launches a guided creation wizard;
    pass `--no-interactive` (or any of those flags) to stay fully scriptable.
    """
    import uuid

    from wmh.cli.ui import BuildParams, RichBuildReporter, build_summary_panel, run_build_wizard
    from wmh.config import ArtifactPaths
    from wmh.engine.build import build as run_build
    from wmh.ingest import VendorPull
    from wmh.providers import get_provider
    from wmh.retrieval import get_embedder
    from wmh.tracking import MeteredProvider, Phase, RunTracker, classify_build_call, save_run

    # Decide whether to run the wizard: explicit flag wins; otherwise auto when at a TTY and the
    # essential inputs (a name and a trace source) were not supplied.
    needs_input = name is None or (file is None and vendor is None)
    use_wizard = interactive if interactive is not None else (_console.is_terminal and needs_input)

    params = BuildParams(
        name=name or DEFAULT_MODEL_NAME,
        file=file,
        vendor=vendor,
        provider=provider,
        model=model,
        region=region,
        gepa_budget=gepa_budget,
        train_split=train_split,
        embed_provider=embed_provider,
        embed_model=embed_model,
        embed_dim=embed_dim,
    )
    if use_wizard:
        params = run_build_wizard(_console, params)
    elif name is None and file is None and vendor is None:
        raise typer.BadParameter("provide --file (or --vendor), or run `wmh build` interactively")

    validate_name(params.name)
    try:
        serve_provider = ProviderKind(params.provider)
    except ValueError:
        kinds = ", ".join(k.value for k in ProviderKind)
        raise typer.BadParameter(
            f"unknown provider {params.provider!r}; choose one of: {kinds}"
        ) from None
    try:
        embed_kind = EmbedderKind(params.embed_provider)
    except ValueError:
        kinds = ", ".join(k.value for k in EmbedderKind)
        raise typer.BadParameter(
            f"unknown embed provider {params.embed_provider!r}; choose one of: {kinds}"
        ) from None
    # A provider-backed embedder needs an embeddings model; fail fast, not deep inside embed().
    if embed_kind is not EmbedderKind.HASHING and not params.embed_model:
        raise typer.BadParameter(
            f"--embed-provider {embed_kind.value} requires --embed-model "
            "(the embeddings model id / Azure embedding deployment)"
        )

    store = WorldModelStore(root)
    model_dir = str(store.model_dir(params.name))
    # Provider wiring (reuse-vs-separate embed config) lives in HarnessConfig.for_build, not here.
    config = HarnessConfig.for_build(
        serve_provider=serve_provider,
        serve_model=params.model,
        region=params.region,
        embed_provider=embed_kind,
        embed_model=params.embed_model,
        embed_dim=params.embed_dim,
        gepa_budget=params.gepa_budget,
        train_split=params.train_split,
    )
    # Fail fast: ping the serve provider (and the embed path, if provider-backed) before spending
    # any rollouts. A missing SDK or bad creds otherwise surfaces only deep inside GEPA, which
    # silently swallows it and "succeeds" with a useless held-out-0.0 model.
    _verify_or_abort(config)

    # Meter the build at the provider boundary: the one serve provider drives GEPA rollouts,
    # reflection, and the judge, so wrapping it captures all build LLM cost/tokens without touching
    # the optimizer. `classify_build_call` splits judge vs GEPA by system prompt.
    tracker = RunTracker(run_id=uuid.uuid4().hex, kind="build")
    metered = MeteredProvider(
        get_provider(config.serve_provider_config()),
        tracker,
        classify=classify_build_call,
    )
    with tracker.timed(), RichBuildReporter(_console, params.name) as reporter:
        run_build(
            config,
            file=params.file,
            vendor=VendorPull() if params.vendor else None,
            root=model_dir,
            serve_provider=metered,
            embedder=get_embedder(config),
            reporter=reporter,
        )
    record = tracker.record_summary()
    save_run(record, ArtifactPaths(model_dir).runs)

    _console.print(build_summary_panel(store.info(params.name), model_dir))
    _console.print(
        f"[bold]run[/bold] {record.run_id[:8]}: {record.duration_seconds:.1f}s, "
        f"{record.total.total_tokens} tokens, ${record.total.cost_usd:.4f} "
        f"({record.total.calls} calls)"
    )
    for phase in (Phase.GEPA, Phase.JUDGE):
        bucket = record.by_phase.get(phase)
        if bucket is not None:
            _console.print(
                f"  {phase.value}: {bucket.total_tokens} tokens, "
                f"${bucket.cost_usd:.4f} ({bucket.calls} calls)"
            )


# The `uv sync` extra that installs each provider's SDK, surfaced when a verify ping fails with a
# missing module so the fix is one copy-paste away.
_PROVIDER_EXTRA: dict[ProviderKind, str] = {
    ProviderKind.ANTHROPIC: "anthropic",
    ProviderKind.BEDROCK: "bedrock",
    ProviderKind.OPENAI: "openai",
    ProviderKind.AZURE_OPENAI: "openai",
}


def _verify_or_abort(config: HarnessConfig) -> None:
    """Ping the serve provider (and any provider-backed embedder) and abort on failure.

    Runs before any rollouts so a missing SDK or bad creds fails loudly and immediately, instead of
    being swallowed inside GEPA and yielding a useless model. Raises `typer.Exit(1)` with an
    actionable hint (the `uv sync` extra for a missing SDK; "check creds / model id" otherwise).
    """
    checks = [(config.serve_provider_config(), False)]
    if config.embed_provider is not EmbedderKind.HASHING:
        checks.append((config.embed_provider_config(), True))

    failed = False
    for cfg, is_embed in checks:
        label = f"embed:{cfg.kind.value}" if is_embed else cfg.kind.value
        _console.print(f"verifying {label}…")
        result = verify_embedder(cfg) if is_embed else verify_all([cfg])[0]
        if result.ok:
            _console.print(f"  {_CHECK} {label} ({result.model}) reachable")
            continue
        failed = True
        _console.print(f"  [red]✗ {label} ({result.model}) failed[/red]: {result.detail}")
        if "No module named" in result.detail:
            extra = _PROVIDER_EXTRA.get(cfg.kind, cfg.kind.value)
            _console.print(f"    [yellow]run `uv sync --extra {extra}` to install the SDK[/yellow]")
        else:
            envs = ", ".join(PROVIDER_ENV_VARS.get(cfg.kind, []))
            hint = f" ({envs})" if envs else ""
            _console.print(
                f"    [yellow]check the model id and that your credentials are set{hint}[/yellow]"
            )
    if failed:
        raise typer.Exit(1)


@app.command("list")
def list_models(root: str = typer.Option(ARTIFACT_DIR, help="Project dir to list.")) -> None:
    """List every world model built under the project dir."""
    from wmh.cli.ui import models_table

    infos = WorldModelStore(root).list_info()
    if not infos:
        _console.print("[yellow]no world models built yet[/yellow]; run `wmh build --name <name>`")
        return
    _console.print(models_table(infos))


@app.command("serve")
def serve(
    name: list[str] = typer.Option(  # noqa: B008 - typer reads option defaults at definition time
        None, "--name", help="World model(s) to serve. Repeatable; default: all built ones."
    ),
    port: int = typer.Option(8000, help="Port for the local backend."),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir to serve from."),
) -> None:
    """Run the local FastAPI backend so agents can step against world models over HTTP.

    Serves every built model by default, or just the `--name` ones. Routes are namespaced:
    `/world_models/{name}/sessions` and `.../step`.
    """
    import uvicorn

    from wmh.serving.server import create_app

    names = list(name) if name else None
    uvicorn.run(create_app(root, names=names), host="127.0.0.1", port=port)


@app.command("eval")
def eval_(  # noqa: A001 - `eval` is the user-facing command name; the builtin isn't used here
    files: list[str] = _EVAL_FILES,
    prompt_file: str = typer.Option(None, "--prompt", help="Prompt file; default=BASE_ENV_PROMPT."),
    provider: str = typer.Option("bedrock", "--provider", help="Provider running the model."),
    model: str = typer.Option("us.anthropic.claude-opus-4-8", help="Model id."),
    region: str = typer.Option(None, help="AWS region (Bedrock)."),
    train_split: float = typer.Option(0.7, help="Train/holdout ratio per file."),
    embed_dim: int = typer.Option(512, help="phi dimensionality for the offline embedder."),
    no_rag: bool = typer.Option(False, "--no-rag", help="Disable retrieval (zero-shot replay)."),
    judge: str = typer.Option("rubric", help="Scorer: rubric (5-dim) | match (functional)."),
    sample_turns: str = typer.Option("all", help="Turns scored per trace: all | sampled (5)."),
    seed: int = typer.Option(0, help="Seed for reproducible turn sampling."),
    out: str = typer.Option(None, help="Optional path to write the full JSON report."),
) -> None:
    """Score reconstruction fidelity: replay held-out steps, judge predicted vs. real observations.

    For each trace file: split train/holdout, replay the holdout through the prompt (with leak-free
    RAG unless --no-rag), and report per-file + overall fidelity (mean±std across steps). The
    measurement loop behind iterating on the env prompt (see docs/base_prompt_iteration.md).
    """
    from pathlib import Path

    from wmh.engine.eval import evaluate_files
    from wmh.engine.prompts import BASE_ENV_PROMPT
    from wmh.optimize.judge import LLMJudge, RubricJudge
    from wmh.providers import ProviderConfig, get_provider
    from wmh.retrieval import HashingEmbedder

    serve_provider = ProviderKind(provider)
    llm = get_provider(ProviderConfig(kind=serve_provider, model=model, region=region))
    prompt = Path(prompt_file).read_text(encoding="utf-8") if prompt_file else BASE_ENV_PROMPT
    embedder = None if no_rag else HashingEmbedder(dim=embed_dim)
    scorer = RubricJudge(llm) if judge == "rubric" else LLMJudge(llm)

    report = evaluate_files(
        [Path(f) for f in files],
        prompt,
        llm,
        scorer,
        embedder=embedder,
        train_split=train_split,
        sample_turns=sample_turns,
        seed=seed,
    )
    for name, rep in report.per_file.items():
        _console.print(f"  {name:28} {rep.summary()}")
    _console.print(
        f"[bold]OVERALL[/bold] fidelity={report.overall_fidelity:.3f}±{report.overall_std:.3f} "
        f"over {report.total_steps} held-out steps"
    )
    if out:
        import json

        Path(out).write_text(
            json.dumps({n: r.model_dump() for n, r in report.per_file.items()}, indent=2),
            encoding="utf-8",
        )
        _console.print(f"wrote full report -> {out}")


@app.command("demo")
def demo(
    name: str = typer.Option(None, "--name", help="World model to demo (default: the only one)."),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir."),
) -> None:
    """Demo the harness: an LLM agent makes a tool call vs the world model; show prompt+output."""
    from wmh.engine.demo import run_demo

    wm, _resolved_name, provider = _load_model(name, root)
    # Seed the demo agent from whatever steps the index holds.
    examples = wm.sample_steps(3)
    result = run_demo(wm, provider, examples)
    _console.print(f"[bold]agent action[/bold]: {result.agent_action.model_dump()}")
    _console.print(f"[bold]env prompt[/bold]:\n{result.env_prompt}")
    _console.print(f"[bold]observation[/bold]: {result.observation.model_dump()}")


@app.command("play")
def play(
    name: str = typer.Option(None, "--name", help="World model to play (default: the only one)."),
    task: str = typer.Option(None, "--task", help="Task to seed the session with."),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir."),
) -> None:
    """Step into the environment yourself: type actions, the world model returns observations."""
    from wmh.cli.ui import run_play_repl

    wm, resolved_name, _provider = _load_model(name, root)
    run_play_repl(_console, wm, resolved_name, task)


@bench_app.callback()
def bench(
    ctx: typer.Context,
    benchmarks: str = typer.Option(BENCHMARKS_DIR, "--benchmarks", help="Benchmark defs dir."),
) -> None:
    """With no subcommand, render the leaderboard over all persisted benchmark runs."""
    if ctx.invoked_subcommand is not None:
        return
    from wmh.bench import build_leaderboard, discover_benchmarks, load_runs, results_dir_for
    from wmh.cli.ui import leaderboard_table

    resolved = _resolve_benchmarks_dir(benchmarks)
    defs = discover_benchmarks(resolved)
    if not defs:
        _console.print(
            f"[yellow]no benchmarks under {resolved}/[/yellow]; "
            "add one as benchmarks/<name>/benchmark.toml"
        )
        return
    runs = [run for d in defs for run in load_runs(results_dir_for(d.dir))]
    rows = build_leaderboard(runs)
    if not rows:
        _console.print(
            "[yellow]no benchmark runs yet[/yellow]; score one with `wmh bench run <name>`"
        )
        return
    _console.print(leaderboard_table(rows))


@bench_app.command("list")
def bench_list(
    benchmarks: str = typer.Option(BENCHMARKS_DIR, "--benchmarks", help="Benchmark defs dir."),
) -> None:
    """List every committed benchmark definition and its eval config."""
    from wmh.bench import discover_benchmarks
    from wmh.cli.ui import benchmarks_table

    resolved = _resolve_benchmarks_dir(benchmarks)
    defs = discover_benchmarks(resolved)
    if not defs:
        _console.print(
            f"[yellow]no benchmarks under {resolved}/[/yellow]; "
            "add one as benchmarks/<name>/benchmark.toml"
        )
        return
    _console.print(benchmarks_table(defs))


@bench_app.command("run")
def bench_run(
    name: str = typer.Argument(..., help="Benchmark name (benchmarks/<name>/benchmark.toml)."),
    model: str = typer.Option(
        None, "--model", help="Built world model whose optimized prompt to score (under --root)."
    ),
    prompt_file: str = typer.Option(None, "--prompt", help="Prompt file to score; default=BASE."),
    benchmarks: str = typer.Option(BENCHMARKS_DIR, "--benchmarks", help="Benchmark defs dir."),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir (for --model)."),
) -> None:
    """Score a world-model prompt against a benchmark, persist the run, and print mean±std.

    The prompt comes from `--model` (a built model's optimized prompt), `--prompt` (a file), or the
    bundled `BASE_ENV_PROMPT`. The benchmark's own `benchmark.toml` fixes the eval config — sample
    turns, rollouts, seeds, judge — so a run is reproducible. Results land under
    `benchmarks/<name>/results/` for the leaderboard.
    """
    import uuid
    from datetime import UTC, datetime
    from pathlib import Path

    from wmh.bench import (
        BenchmarkDef,
        evaluate_files_once,
        load_benchmark,
        results_dir_for,
        run_benchmark,
        save_run,
    )

    bench_dir = Path(_resolve_benchmarks_dir(benchmarks)) / name
    try:
        bench_def: BenchmarkDef = load_benchmark(bench_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    missing = bench_def.missing_traces()
    if missing:
        raise typer.BadParameter(
            f"benchmark {name!r} references missing trace files: "
            + ", ".join(str(p) for p in missing)
        )

    prompt, prompt_label = _resolve_prompt(model, prompt_file, root)

    def on_seed(seed: int, done: int, total: int) -> None:
        _console.print(f"  [dim]scored seed {seed} ({done}/{total})[/dim]")

    _console.print(
        f"[bold]running[/bold] {name} (v{bench_def.version}) "
        f"against [cyan]{prompt_label}[/cyan]: "
        f"{len(bench_def.eval.seeds)} seed(s) × {bench_def.eval.rollouts} rollout(s)"
    )

    def score_once(  # noqa: ANN202 - returns wmh.bench.RolloutScore; signature matches ScoreOnce
        files,  # noqa: ANN001
        prompt: str,
        *,
        sample_turns,  # noqa: ANN001
        rollouts: int,
        temperature: float,
        seed: int,
    ):
        return evaluate_files_once(
            files,
            prompt,
            bench_def.eval.judge,
            sample_turns=sample_turns,
            rollouts=rollouts,
            temperature=temperature,
            seed=seed,
            train_split=bench_def.eval.train_split,
            top_k=bench_def.eval.top_k,
            no_rag=bench_def.eval.no_rag,
            embed_dim=bench_def.eval.embed_dim,
        )

    run = run_benchmark(bench_def, prompt, prompt_label, score_once, on_seed=on_seed)
    run.run_id = uuid.uuid4().hex
    # Microsecond precision so two runs of the same prompt seconds apart still order by recency on
    # the leaderboard (created_at is the primary "latest" key; see wmh.bench.leaderboard._newer).
    run.created_at = datetime.now(UTC).isoformat(timespec="microseconds")
    out = save_run(run, results_dir_for(bench_def.dir))

    _console.print(
        f"[bold]{name}[/bold] fidelity={run.fidelity_mean:.3f} ± {run.fidelity_std:.3f} "
        f"over {len(run.seeds)} seed(s), {run.total_steps} held-out steps"
    )
    _console.print(f"[dim]wrote run {run.run_id[:8]} -> {out}[/dim]")


@bench_app.command("scenario")
def bench_scenario(
    name: str = typer.Argument(..., help="Benchmark name (benchmarks/<name>/benchmark.toml)."),
    model: str = typer.Option(
        None, "--model", help="World model to serve (default: the benchmark name)."
    ),
    serve_model: str = typer.Option(
        None,
        "--serve-model",
        help=(
            "Override the LLM that plays the environment, keeping the model's prompt + demos "
            "(e.g. a Bedrock id like us.anthropic.claude-haiku-4-5-20251001-v1:0). Default: the "
            "model's configured serve model."
        ),
    ),
    trace_index: int = typer.Option(
        None, "--trace", help="Held-out trace to replay (default: the simplest = fewest steps)."
    ),
    benchmarks: str = typer.Option(BENCHMARKS_DIR, "--benchmarks", help="Benchmark defs dir."),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir (for --model)."),
) -> None:
    """Open-loop replay one recorded scenario through the world model, timing + costing each step.

    The world-model half of the scenario comparison. Picks a held-out trace from the benchmark's
    corpus — by default the SIMPLEST (fewest recorded steps), so the demo scenario is short; pass
    `--trace N` for a specific one — and predicts each recorded step **teacher-forced, exactly as
    `wmh eval` does** (from the recorded state + history + leak-free demos from the train split; a
    step never sees the model's own prior predictions), printing each predicted observation as it
    lands. Ends with total time, tokens, cost, and fidelity. Run the SAME scenario against the real
    environment with the matching `tools/<benchmark>-capture/` runner (closed-loop) and compare.
    """
    from pathlib import Path

    from wmh.bench import ScenarioStep, load_benchmark, run_scenario
    from wmh.config import ArtifactPaths, load_config
    from wmh.engine.build import split_traces
    from wmh.engine.prompts import BASE_ENV_PROMPT
    from wmh.ingest import get_adapter
    from wmh.providers import get_provider
    from wmh.retrieval import EmbeddingRetriever, get_embedder
    from wmh.retrieval.leakfree import DemoRetriever

    bench_dir = Path(_resolve_benchmarks_dir(benchmarks)) / name
    try:
        bench_def = load_benchmark(bench_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    missing = bench_def.missing_traces()
    if missing:
        raise typer.BadParameter(
            f"benchmark {name!r} references missing trace files: "
            + ", ".join(str(p) for p in missing)
        )

    # Ingest the benchmark's corpus and split it the SAME way the scorer does: replay a held-out
    # trace (one the model was NOT optimized on) and retrieve leak-free demos from the train split.
    adapter = get_adapter("otel-genai")
    traces = [t for f in bench_def.trace_files() for t in adapter.from_file(str(f))]
    if not traces:
        raise typer.BadParameter(f"benchmark {name!r} ingested no traces")
    train, holdout = split_traces(traces, bench_def.eval.train_split)
    pool = holdout or traces  # tiny corpora may have no held-out trace; fall back to all
    kind = "held-out" if holdout else "(no held-out split; all)"
    if trace_index is None:
        # Default: the simplest scenario — the held-out trace with the fewest recorded steps (ties
        # broken by corpus order). Keeps the demo short without the user hunting for a small trace.
        trace = min(pool, key=lambda t: len(t.steps))
    else:
        if not 0 <= trace_index < len(pool):
            raise typer.BadParameter(
                f"--trace {trace_index} out of range; {name!r} has {len(pool)} {kind} trace(s)"
            )
        trace = pool[trace_index]

    # Resolve the model dir (default: the benchmark name -> the bundled canonical model), then load
    # its serve provider + optimized prompt + embedder — the exact pieces the eval path uses.
    store = WorldModelStore(root)
    try:
        model_dir = store.resolve(model or name)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    config = load_config(str(model_dir))
    paths = ArtifactPaths(model_dir)
    env_prompt = (
        paths.optimized_prompt.read_text(encoding="utf-8")
        if paths.optimized_prompt.exists()
        else BASE_ENV_PROMPT
    )
    serve_config = config.serve_provider_config()
    if serve_model:
        # Swap only the LLM id, keeping the same provider backend + the model's prompt/demos — so we
        # can measure a cheaper/faster model (Haiku, Sonnet) against the same scenario and prompt.
        serve_config = serve_config.model_copy(update={"model": serve_model})
    provider = get_provider(serve_config)
    # Leak-free demos from the TRAIN split only (never the query's own trace), identical to eval.
    retriever = EmbeddingRetriever(get_embedder(config))
    demos = DemoRetriever(retriever, train or traces, top_k=config.top_k)

    n_steps = len(trace.steps)
    _console.print(
        f"\n[bold]world model[/bold] [cyan]{model or name}[/cyan] — open-loop replay of {name} "
        f"scenario [cyan]{trace.trace_id[:8]}[/cyan] ({n_steps} steps), no environment to stand up"
    )
    # The task the agent was pursuing (the recorded instruction), shown briefly for context.
    task = trace.steps[0].task if trace.steps else None
    if task:
        _console.print(f"[bold]task[/bold] [dim]{_clip(task, 240)}[/dim]\n")

    def on_step(step: ScenarioStep) -> None:
        # Light live-fidelity signal: error flag agreed and a non-empty prediction landed.
        ok = step.is_error_predicted == step.is_error_actual and bool(step.predicted.strip())
        mark = "[green]✓[/green]" if ok else "[yellow]≈[/yellow]"
        err = " [red](error)[/red]" if step.is_error_predicted else ""
        # Show the agent's CALL to the environment, then the world model's OBSERVATION back.
        _console.print(
            f"{mark} [dim]step {step.index} ({step.seconds:.2f}s)[/dim]\n"
            f"  [bold blue]agent →[/bold blue] {_clip(step.action, 200)}\n"
            f"  [bold green]world model →[/bold green]{err} {_clip(step.predicted)}"
        )

    report = run_scenario(
        provider, env_prompt, trace, demos, benchmark=name, model=(model or name), on_step=on_step
    )
    _console.print(f"\n[bold]done[/bold]: {report.summary()}")


def _clip(text: str, limit: int = 160) -> str:
    """One-line clip of an observation for the live scenario view."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _resolve_prompt(model: str | None, prompt_file: str | None, root: str) -> tuple[str, str]:
    """Resolve the prompt to score + a human label, from --model, --prompt, or the base prompt."""
    from pathlib import Path

    from wmh.config import ArtifactPaths
    from wmh.engine.prompts import BASE_ENV_PROMPT

    if model is not None and prompt_file is not None:
        raise typer.BadParameter("pass at most one of --model / --prompt")
    if model is not None:
        store = WorldModelStore(root)
        try:
            model_dir = store.resolve(model)
        except (FileNotFoundError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        optimized = ArtifactPaths(model_dir).optimized_prompt
        if not optimized.exists():
            raise typer.BadParameter(f"model {model!r} has no optimized prompt at {optimized}")
        return optimized.read_text(encoding="utf-8"), model
    if prompt_file is not None:
        return Path(prompt_file).read_text(encoding="utf-8"), Path(prompt_file).stem
    return BASE_ENV_PROMPT, "base"


def _resolve_name(store: WorldModelStore, name: str | None) -> str:
    """Resolve which model to run: explicit `--name`, an interactive picker, or the sole model.

    With `--name`, validate it exists. Otherwise, when several models are built on an interactive
    terminal, show a numbered picker; on a non-TTY (or a single model) defer to `store.resolve`,
    which returns the lone model or raises a helpful "pass --name" error. Store errors
    (unknown/ambiguous name) are turned into a clean `typer.BadParameter` rather than a traceback.
    """
    try:
        if name is not None:
            store.resolve(name)  # validates existence, raising a friendly error if missing
            return name
        # Only enumerate full model summaries when we actually need the picker (>1 model on a TTY).
        # `list_names` is cheap (a dir scan); `list_info` reads every config/metrics/frontier file.
        if _console.is_terminal and len(store.list_names()) > 1:
            from wmh.cli.ui import select_model

            return select_model(_console, store.list_info())
        return store.resolve(None).name
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _load_model(name: str | None, root: str):  # noqa: ANN202 - (WorldModel, name, Provider)
    """Resolve + load a named world model (or the single built one) with its serve provider.

    Returns `(world_model, resolved_name, provider)` so callers can reuse the provider without
    re-reading config / reconstructing it.
    """
    from wmh.engine import load_world_model

    store = WorldModelStore(root)
    resolved_name = _resolve_name(store, name)
    # `resolve` (not `model_dir`) so a bundled model loads from `world-models/`; `model_dir` is the
    # writable-only build target and would point at a nonexistent `.wmh/` dir for bundled models.
    world_model, provider = load_world_model(store.resolve(resolved_name))
    return world_model, resolved_name, provider


if __name__ == "__main__":
    app()
