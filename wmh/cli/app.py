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
_console = Console()

# Module-level singleton: a typer.Argument call can't be a default inline (ruff B008).
_EVAL_FILES = typer.Argument(..., help="OTel trace files to score (one corpus each).")


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
    )
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
    rollouts: int = typer.Option(1, help="World-model samples per step (use with --temperature)."),
    temperature: float = typer.Option(0.0, help="World-model sampling temperature for rollouts."),
    sample_turns: str = typer.Option("all", help="Turns scored per trace: all | sampled (5)."),
    seed: int = typer.Option(0, help="Seed for reproducible turn sampling."),
    out: str = typer.Option(None, help="Optional path to write the full JSON report."),
) -> None:
    """Score reconstruction fidelity: replay held-out steps, judge predicted vs. real observations.

    For each trace file: split train/holdout, replay the holdout through the prompt (with leak-free
    RAG unless --no-rag), and report per-file + overall fidelity (mean±std). Use --rollouts N with
    --temperature >0 to sample the world model multiple times per step. The measurement loop behind
    iterating on the env prompt (see docs/base_prompt_iteration.md).
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
        rollouts=rollouts,
        temperature=temperature,
        sample_turns=sample_turns,
        seed=seed,
    )
    for name, rep in report.per_file.items():
        _console.print(f"  {name:28} {rep.summary()}")
    _console.print(
        f"[bold]OVERALL[/bold] fidelity={report.overall_fidelity:.3f}±{report.overall_std:.3f} "
        f"over {report.total_steps} held-out steps (rollouts={report.rollouts})"
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
    world_model, provider = load_world_model(store.model_dir(resolved_name))
    return world_model, resolved_name, provider


if __name__ == "__main__":
    app()
