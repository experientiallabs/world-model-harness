"""Named world models on disk.

The project root (`.wmh/` by default) holds many named world models under `models/<name>/`. Each
model directory is a self-contained artifact in the layout `ArtifactPaths` already understands
(config.toml, prompts/, index/, metrics.json). The store turns names into directories, lists what
has been built, and reads a small summary for `wmh list`.

    .wmh/
      models/
        tau2-airline/   <- one artifact (config.toml, prompts/, index/, metrics.json)
        retail-bench/   <- another
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from pydantic import BaseModel

from wmh.config.config import ARTIFACT_DIR, ArtifactPaths, HarnessConfig

# The implicit model name used when the user does not pass `--name`.
DEFAULT_MODEL_NAME = "default"

# A safe, filesystem-friendly model name: no path separators, traversal, or leading dot.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_name(name: str) -> str:
    """Return `name` if it is a safe single path segment, else raise a friendly ValueError."""
    if not _NAME_RE.match(name) or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(
            f"invalid world model name {name!r}: use letters, digits, '.', '_', '-' "
            "(must start with a letter or digit, no path separators)"
        )
    return name


class ModelInfo(BaseModel):
    """A one-line summary of a built world model, for `wmh list`."""

    name: str
    serve_provider: str
    serve_model: str
    held_out_accuracy: float | None = None
    rollouts_used: int | None = None
    frontier_size: int | None = None


class WorldModelStore:
    """Resolves and enumerates named world models under a project root."""

    def __init__(self, root: str | Path = ARTIFACT_DIR) -> None:
        self.root = Path(root)

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    def model_dir(self, name: str) -> Path:
        """The artifact directory for `name` (not guaranteed to exist)."""
        return self.models_dir / validate_name(name)

    def exists(self, name: str) -> bool:
        return ArtifactPaths(self.model_dir(name)).config.exists()

    def list_names(self) -> list[str]:
        """Sorted names of every built model (a directory containing a config.toml)."""
        if not self.models_dir.exists():
            return []
        names = [
            d.name for d in self.models_dir.iterdir() if d.is_dir() and (d / "config.toml").exists()
        ]
        return sorted(names)

    def resolve(self, name: str | None) -> Path:
        """Resolve `name` to a built model's artifact dir for read commands (serve/demo/play).

        With an explicit `name`, require it to exist. With `name=None`, fall back to the single
        built model if there is exactly one; otherwise raise, listing the choices.
        """
        if name is not None:
            if not self.exists(name):
                available = self.list_names()
                hint = f" (have: {', '.join(available)})" if available else ""
                raise FileNotFoundError(
                    f"no world model named {name!r} under {self.models_dir}{hint}; "
                    "run `wmh build --name <name>` first"
                )
            return self.model_dir(name)

        names = self.list_names()
        if not names:
            raise FileNotFoundError(
                f"no world models built under {self.models_dir}; "
                "run `wmh build --name <name>` first"
            )
        if len(names) > 1:
            raise ValueError(
                f"multiple world models built ({', '.join(names)}); pass --name to choose one"
            )
        return self.model_dir(names[0])

    def info(self, name: str) -> ModelInfo:
        """Read a model's config + metrics into a summary (for `wmh list`)."""
        paths = ArtifactPaths(self.model_dir(name))
        with paths.config.open("rb") as fh:
            config = HarnessConfig.model_validate(tomllib.load(fh))
        accuracy: float | None = None
        rollouts: int | None = None
        if paths.metrics.exists():
            metrics = json.loads(paths.metrics.read_text(encoding="utf-8"))
            accuracy = _as_float(metrics.get("held_out_accuracy"))
            rollouts = _as_int(metrics.get("rollouts_used"))
        frontier_size: int | None = None
        if paths.frontier.exists():
            frontier = json.loads(paths.frontier.read_text(encoding="utf-8"))
            if isinstance(frontier, list):
                frontier_size = len(frontier)
        serve = config.serve_provider_config()
        return ModelInfo(
            name=name,
            serve_provider=serve.kind.value,
            serve_model=serve.model,
            held_out_accuracy=accuracy,
            rollouts_used=rollouts,
            frontier_size=frontier_size,
        )

    def list_info(self) -> list[ModelInfo]:
        return [self.info(name) for name in self.list_names()]


def _as_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _as_int(value: object) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None
