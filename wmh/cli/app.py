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
from wmh.providers import ProviderConfig, ProviderKind, verify_all

app = typer.Typer(help="World Model Harness: a frontier LLM acts as your agent's environment.")
providers_app = typer.Typer(help="Manage and verify LLM providers.")
app.add_typer(providers_app, name="providers")
_console = Console()


@providers_app.command("verify")
def providers_verify(
    name: str = typer.Option(None, "--name", help="Verify one model's providers (default: all)."),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir."),
) -> None:
    """Ping every configured provider (Anthropic/Bedrock/Azure/OpenAI) and report status.

    Gathers provider configs from the built world models (one `--name`, or all of them, deduped by
    kind+model), so a brand-new project with nothing built yet has nothing to verify.
    """
    store = WorldModelStore(root)
    names = [name] if name is not None else store.list_names()
    if not names:
        _console.print("[yellow]no world models built yet[/yellow]; run `wmh build --name <name>`")
        return
    seen: set[tuple[str, str]] = set()
    providers: list[ProviderConfig] = []
    for model_name in names:
        try:
            model_dir = str(store.resolve(model_name))
        except (FileNotFoundError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        for pc in load_config(model_dir).providers:
            key = (pc.kind.value, pc.model)
            if key not in seen:
                seen.add(key)
                providers.append(pc)
    for result in verify_all(providers):
        mark = "[green]ok[/green]" if result.ok else "[red]fail[/red]"
        _console.print(f"{mark} {result.kind.value} ({result.model}) {result.detail}")


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
    from wmh.cli.ui import BuildParams, RichBuildReporter, build_summary_panel, run_build_wizard
    from wmh.engine.build import build as run_build
    from wmh.ingest import VendorPull

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
    )
    if use_wizard:
        params = run_build_wizard(_console, params)
    elif name is None and file is None and vendor is None:
        raise typer.BadParameter("provide --file (or --vendor), or run `wmh build` interactively")

    validate_name(params.name)
    store = WorldModelStore(root)
    model_dir = str(store.model_dir(params.name))

    serve_provider = ProviderKind(params.provider)
    config = HarnessConfig(
        providers=[ProviderConfig(kind=serve_provider, model=params.model, region=params.region)],
        serve_provider=serve_provider,
        gepa_budget=params.gepa_budget,
    )
    with RichBuildReporter(_console, params.name) as reporter:
        run_build(
            config,
            file=params.file,
            vendor=VendorPull() if params.vendor else None,
            root=model_dir,
            reporter=reporter,
        )
    _console.print(build_summary_panel(store.info(params.name), model_dir))


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
    from wmh.engine.world_model import WorldModel
    from wmh.providers import get_provider

    store = WorldModelStore(root)
    resolved_name = _resolve_name(store, name)
    model_dir = store.model_dir(resolved_name)
    config = load_config(str(model_dir))
    provider = get_provider(config.serve_provider_config())
    return WorldModel.load(str(model_dir), provider), resolved_name, provider


if __name__ == "__main__":
    app()
