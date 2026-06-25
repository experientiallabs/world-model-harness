"""`wmh` CLI — ingestion UI and operator console for the harness.

Deliberately small. The lifecycle is:
    providers verify -> build -> serve / demo
`build` creates the `.wmh/` artifact directory itself, so there is no separate init step.
"""

from __future__ import annotations

import typer

app = typer.Typer(help="World Model Harness: a frontier LLM acts as your agent's environment.")
providers_app = typer.Typer(help="Manage and verify LLM providers.")
app.add_typer(providers_app, name="providers")


@providers_app.command("verify")
def providers_verify() -> None:
    """Ping every configured provider (Anthropic/Bedrock/Azure/OpenAI) and report status."""
    raise NotImplementedError


@app.command("build")
def build(
    file: str = typer.Option(None, "--file", help="Path to exported traces (OTLP-JSON / JSONL)."),
    vendor: str = typer.Option(None, "--vendor", help="Vendor name to pull traces via SDK."),
) -> None:
    """Ingest traces (file upload or vendor SDK pull) and build the `.wmh/` artifact.

    Creates `.wmh/` if absent, then: ingest -> normalize -> split(train/test) -> embed/index ->
    GEPA optimize -> write the artifact.
    """
    raise NotImplementedError


@app.command("serve")
def serve(port: int = typer.Option(8000, help="Port for the local backend.")) -> None:
    """Run the local FastAPI backend so agents can step against the world model over HTTP."""
    raise NotImplementedError


@app.command("demo")
def demo() -> None:
    """Demo the harness: an LLM agent makes a tool call vs the world model; show prompt+output."""
    raise NotImplementedError


if __name__ == "__main__":
    app()
