"""`wmh` CLI — ingestion UI and operator console for the harness.

Deliberately small. The lifecycle is:
    providers verify -> build -> serve / demo
`build` creates the `.wmh/` artifact directory itself, so there is no separate init step.
"""

from __future__ import annotations

import typer
from rich.console import Console

from wmh.config import ARTIFACT_DIR, HarnessConfig, load_config
from wmh.providers import ProviderKind, verify_all, verify_embedder
from wmh.providers.base import EmbedderKind

app = typer.Typer(help="World Model Harness: a frontier LLM acts as your agent's environment.")
providers_app = typer.Typer(help="Manage and verify LLM providers.")
app.add_typer(providers_app, name="providers")
_console = Console()


@providers_app.command("verify")
def providers_verify(root: str = typer.Option(ARTIFACT_DIR, help="Artifact dir.")) -> None:
    """Ping every configured provider (completion + embed path) and report status."""
    config = load_config(root)
    for result in verify_all(config.providers):
        mark = "[green]ok[/green]" if result.ok else "[red]fail[/red]"
        _console.print(f"{mark} {result.kind.value} ({result.model}) {result.detail}")
    # Verify the phi embed path too, unless it's the offline (creds-free) hashing embedder.
    if config.embed_provider is not EmbedderKind.HASHING:
        result = verify_embedder(config.embed_provider_config())
        mark = "[green]ok[/green]" if result.ok else "[red]fail[/red]"
        _console.print(f"{mark} embed:{result.kind.value} ({result.model}) {result.detail}")


@app.command("build")
def build(
    file: str = typer.Option(None, "--file", help="Path to exported traces (OTLP-JSON / JSONL)."),
    vendor: str = typer.Option(None, "--vendor", help="Vendor name to pull traces via SDK."),
    root: str = typer.Option(ARTIFACT_DIR, help="Artifact dir to create/write."),
    provider: str = typer.Option("bedrock", "--provider", help="Provider that serves the model."),
    model: str = typer.Option("us.anthropic.claude-opus-4-8", help="Serve provider model id."),
    region: str = typer.Option(None, help="AWS region (Bedrock)."),
    gepa_budget: int = typer.Option(50, help="GEPA rollout budget."),
    embed_provider: str = typer.Option(
        "hashing", help="phi embedder: hashing (offline) | bedrock | openai | azure_openai."
    ),
    embed_model: str = typer.Option(None, help="Embeddings model id / Azure embedding deployment."),
    embed_dim: int = typer.Option(512, help="phi dimensionality (index + query must agree)."),
) -> None:
    """Ingest traces (file upload or vendor SDK pull) and build the `.wmh/` artifact.

    Creates `.wmh/` if absent, then: ingest -> normalize -> split(train/test) -> embed/index ->
    GEPA optimize -> write the artifact.
    """
    import uuid

    from wmh.config import ArtifactPaths
    from wmh.engine.build import build as run_build
    from wmh.ingest import VendorPull
    from wmh.providers import get_provider
    from wmh.retrieval import get_embedder
    from wmh.tracking import MeteredProvider, Phase, RunTracker, classify_build_call, save_run

    try:
        serve_provider = ProviderKind(provider)
    except ValueError:
        kinds = ", ".join(k.value for k in ProviderKind)
        raise typer.BadParameter(f"unknown provider {provider!r}; choose one of: {kinds}") from None
    try:
        embed_kind = EmbedderKind(embed_provider)
    except ValueError:
        kinds = ", ".join(k.value for k in EmbedderKind)
        raise typer.BadParameter(
            f"unknown embed provider {embed_provider!r}; choose one of: {kinds}"
        ) from None
    # A provider-backed embedder needs an embeddings model; fail fast, not deep inside embed().
    if embed_kind is not EmbedderKind.HASHING and not embed_model:
        raise typer.BadParameter(
            f"--embed-provider {embed_kind.value} requires --embed-model "
            "(the embeddings model id / Azure embedding deployment)"
        )
    # Provider wiring (reuse-vs-separate embed config) lives in HarnessConfig.for_build, not here.
    config = HarnessConfig.for_build(
        serve_provider=serve_provider,
        serve_model=model,
        region=region,
        embed_provider=embed_kind,
        embed_model=embed_model,
        embed_dim=embed_dim,
        gepa_budget=gepa_budget,
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
    with tracker.timed():
        result = run_build(
            config,
            file=file,
            vendor=VendorPull() if vendor else None,
            root=root,
            serve_provider=metered,
            embedder=get_embedder(config),
        )
    record = tracker.record_summary()
    save_run(record, ArtifactPaths(root).runs)

    _console.print(
        f"[green]built[/green] {root}: held_out_accuracy="
        f"{result.metrics.held_out_accuracy:.3f}, frontier={len(result.frontier)}, "
        f"rollouts={result.metrics.rollouts_used}"
    )
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


@app.command("serve")
def serve(
    port: int = typer.Option(8000, help="Port for the local backend."),
    root: str = typer.Option(ARTIFACT_DIR, help="Artifact dir to serve."),
) -> None:
    """Run the local FastAPI backend so agents can step against the world model over HTTP."""
    import uvicorn

    from wmh.serving.server import create_app

    uvicorn.run(create_app(root), host="127.0.0.1", port=port)


@app.command("demo")
def demo(root: str = typer.Option(ARTIFACT_DIR, help="Artifact dir to demo against.")) -> None:
    """Demo the harness: an LLM agent makes a tool call vs the world model; show prompt+output."""
    from wmh.config import load_config as _load
    from wmh.engine.demo import run_demo
    from wmh.engine.world_model import WorldModel
    from wmh.providers import get_provider

    config = _load(root)
    provider = get_provider(config.serve_provider_config())
    wm = WorldModel.load(root, provider)
    # Seed the demo agent from whatever steps the index holds.
    examples = wm.sample_steps(3)
    result = run_demo(wm, provider, examples)
    _console.print(f"[bold]agent action[/bold]: {result.agent_action.model_dump()}")
    _console.print(f"[bold]env prompt[/bold]:\n{result.env_prompt}")
    _console.print(f"[bold]observation[/bold]: {result.observation.model_dump()}")


if __name__ == "__main__":
    app()
