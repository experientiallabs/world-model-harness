"""Project config + the `.wmh/` artifact layout.

`.wmh/` holds everything `wmh build` produces and `wmh serve` / `WorldModel.load` consume.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import tomli_w
from pydantic import BaseModel, Field, JsonValue, ValidationError

from wmh.core.types import JsonObject
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


def _strip_none(value: JsonValue) -> JsonValue:
    """Drop `None`-valued keys recursively so TOML (which has no null) can represent the config.

    On load, pydantic refills the missing optional fields with their `None` defaults, so dropping
    them here round-trips losslessly.
    """
    if isinstance(value, dict):
        return _strip_none_object(value)
    if isinstance(value, list):
        return [_strip_none(v) for v in value]
    return value


def _strip_none_object(obj: dict[str, JsonValue]) -> JsonObject:
    return {k: _strip_none(v) for k, v in obj.items() if v is not None}


def load_config(root: str | Path = ARTIFACT_DIR) -> HarnessConfig:
    """Read `.wmh/config.toml`. Raises a friendly error if the project hasn't been built yet."""
    paths = ArtifactPaths(root)
    if not paths.root.exists():
        raise FileNotFoundError(
            f"no {ARTIFACT_DIR}/ directory at {paths.root}; run `wmh build` first to create it"
        )
    if not paths.config.exists():
        raise FileNotFoundError(
            f"{paths.config} is missing; run `wmh build` to (re)generate the project config"
        )
    try:
        with paths.config.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(
            f"{paths.config} is not valid TOML ({exc}); re-run `wmh build` to regenerate it"
        ) from exc
    try:
        return HarnessConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            f"{paths.config} does not match the current config schema ({exc}); "
            "re-run `wmh build` to regenerate it"
        ) from exc


def save_config(config: HarnessConfig, root: str | Path = ARTIFACT_DIR) -> None:
    """Write `config` to `.wmh/config.toml`, creating `.wmh/` if missing.

    Writes to a temp file in the same directory and renames into place so an interrupted or
    failed write never leaves a truncated `config.toml` behind.
    """
    paths = ArtifactPaths(root)
    paths.root.mkdir(parents=True, exist_ok=True)
    data = _strip_none_object(config.model_dump(mode="json"))
    tmp = paths.config.with_name(f"{paths.config.name}.tmp")
    with tmp.open("wb") as fh:
        tomli_w.dump(data, fh)
    tmp.replace(paths.config)
