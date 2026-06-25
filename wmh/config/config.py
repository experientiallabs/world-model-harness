"""Project config + the `.wmh/` artifact layout.

`.wmh/` holds everything `wmh build` produces and `wmh serve` / `WorldModel.load` consume.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from wmh.providers.base import ProviderConfig, ProviderKind

ARTIFACT_DIR = ".wmh"

# Env var names each provider backend reads its credentials from (documented for the user).
PROVIDER_ENV_VARS: dict[ProviderKind, list[str]] = {
    ProviderKind.ANTHROPIC: ["ANTHROPIC_API_KEY"],
    ProviderKind.BEDROCK: ["AWS_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
    ProviderKind.AZURE_OPENAI: ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"],
    ProviderKind.OPENAI: ["OPENAI_API_KEY"],
}


class HarnessConfig(BaseModel):
    """Persisted to `.wmh/config.toml` and reloaded by `wmh serve` / `WorldModel.load`."""

    providers: list[ProviderConfig] = Field(default_factory=list)
    serve_provider: ProviderKind = ProviderKind.ANTHROPIC  # serves the live world model
    embed_provider: ProviderKind = ProviderKind.OPENAI  # supplies phi for retrieval
    top_k: int = 5  # demos retrieved per step (DreamGym k)
    train_split: float = 0.8  # train/held-out ratio for GEPA
    gepa_budget: int = 50  # rollout budget for prompt evolution
    trace_adapter: str = "otel-genai"


class ArtifactPaths:
    """Resolves the files under `.wmh/`."""

    def __init__(self, root: str | Path = ARTIFACT_DIR) -> None:
        self.root = Path(root)

    @property
    def config(self) -> Path:
        return self.root / "config.toml"

    @property
    def traces(self) -> Path:
        return self.root / "traces"

    @property
    def index(self) -> Path:
        return self.root / "index"

    @property
    def base_prompt(self) -> Path:
        return self.root / "prompts" / "base.txt"

    @property
    def optimized_prompt(self) -> Path:
        return self.root / "prompts" / "optimized.txt"

    @property
    def frontier(self) -> Path:
        return self.root / "prompts" / "frontier.json"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.json"


def load_config(root: str | Path = ARTIFACT_DIR) -> HarnessConfig:
    # TODO: read config.toml; raise a friendly error if `wmh build` hasn't run yet.
    raise NotImplementedError


def save_config(config: HarnessConfig, root: str | Path = ARTIFACT_DIR) -> None:
    # TODO: write config.toml, creating `.wmh/` if missing.
    raise NotImplementedError
