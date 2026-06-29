"""Named world models on disk.

The project root (`.wmh/` by default) holds the user's own world models under `models/<name>/`. Each
model directory is a self-contained artifact in the layout `ArtifactPaths` already understands
(config.toml, prompts/, index/, metrics.json). The store turns names into directories, lists what
has been built, and reads a small summary for `wmh list`.

    .wmh/                <- writable: where `wmh build` writes
      models/
        tau2-airline/    <- one artifact (config.toml, prompts/, index/, metrics.json)
        retail-bench/    <- another

"Filesystem as DB": loading a model is just reading its folder. On top of the writable root, the
store ALSO searches a read-only **bundled** dir — the committed top-level `world-models/` holding
canonical example models shipped with the repo (e.g. `world-models/tau-bench/`). The bundled layout
puts model dirs DIRECTLY under it (`world-models/<name>/`), not under a `models/` subdir.

    world-models/        <- read-only: committed example models, shipped with the repo
      tau-bench/         <- one artifact, same self-contained layout

Search precedence is writable-first: a user `wmh build --name tau-bench` shadows the bundled model
of the same name, so the bundled examples never block a rebuild.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from pathlib import Path

from pydantic import BaseModel, JsonValue

from wmh.config.config import ARTIFACT_DIR, ArtifactPaths, HarnessConfig

# The implicit model name used when the user does not pass `--name`.
DEFAULT_MODEL_NAME = "default"

# The committed top-level dir of bundled example models (canonical models shipped with the repo).
BUNDLED_DIR_NAME = "world-models"

# Sentinel distinguishing "caller passed bundled_dir=None to disable" from "caller omitted it"
# (omitted -> use the repo default). A plain `None` default can't tell those two apart.
_UNSET: Path = Path("\0__wmh_unset__")


# Env var to override where bundled models are searched. Point it at a custom model library, or at
# an empty/nonexistent path to disable bundled discovery (e.g. for hermetic tests).
BUNDLED_DIR_ENV = "WMH_BUNDLED_DIR"


def default_bundled_dir() -> Path:
    """Locate the bundled `world-models/` dir: `$WMH_BUNDLED_DIR` if set, else the repo's copy.

    Without the override, `store.py` lives at `<repo>/wmh/config/store.py`, so its third parent is
    the repo root that holds the sibling `world-models/`. Works for an editable/source checkout (how
    the harness is developed and run); a packaged install without the dir simply finds nothing (it's
    gated on existence by the caller), which is the correct, model-less default there.
    """
    override = os.environ.get(BUNDLED_DIR_ENV)
    if override is not None:
        return Path(override)
    return Path(__file__).resolve().parent.parent.parent / BUNDLED_DIR_NAME


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
    """Resolves and enumerates named world models across a writable root + bundled examples.

    Two search locations, writable-first:

    - the **writable** project root (`.wmh/models/<name>/`) where `wmh build` writes, and
    - a read-only **bundled** dir (`world-models/<name>/`) of canonical example models shipped with
      the repo. Defaults to the repo's committed `world-models/`; pass `bundled_dir=None` to disable
      (tests that want an isolated store), or an explicit path to point elsewhere.

    Reads (resolve/list/info) see the union of both, with a writable model shadowing a bundled one
    of the same name. Writes always target the writable root (`model_dir`), so a bundled example
    never blocks a `wmh build --name <same>`.
    """

    def __init__(
        self,
        root: str | Path = ARTIFACT_DIR,
        *,
        bundled_dir: str | Path | None = _UNSET,
    ) -> None:
        self.root = Path(root)
        if bundled_dir is _UNSET:
            bundled = default_bundled_dir()
        elif bundled_dir is None:
            bundled = None
        else:
            bundled = Path(bundled_dir)
        # Keep the bundled dir only if it actually exists; a missing dir means "no bundled models".
        keep = bundled is not None and bundled.is_dir()
        self.bundled_dir: Path | None = bundled if keep else None

    @property
    def models_dir(self) -> Path:
        """The writable models dir (`.wmh/models/`) — where builds are written."""
        return self.root / "models"

    def model_dir(self, name: str) -> Path:
        """The WRITABLE artifact directory for `name` (where a build writes; may not exist).

        This is intentionally writable-only: `wmh build` writes here, shadowing any bundled model of
        the same name. To READ a model that may be bundled, use `resolve` / `dir_for`.
        """
        return self.models_dir / validate_name(name)

    def _bundled_model_dir(self, name: str) -> Path | None:
        """The bundled artifact dir for `name`, or None if there's no bundled dir."""
        return self.bundled_dir / validate_name(name) if self.bundled_dir is not None else None

    def dir_for(self, name: str) -> Path | None:
        """The artifact dir holding `name`'s config (writable wins over bundled), or None.

        This is the single name->directory resolver every read path goes through, so writable and
        bundled models load identically.
        """
        writable = self.model_dir(name)
        if ArtifactPaths(writable).config.exists():
            return writable
        bundled = self._bundled_model_dir(name)
        if bundled is not None and ArtifactPaths(bundled).config.exists():
            return bundled
        return None

    def exists(self, name: str) -> bool:
        return self.dir_for(name) is not None

    def _names_in(self, models_dir: Path) -> set[str]:
        """Names of every artifact (a dir containing config.toml) directly under `models_dir`."""
        if not models_dir.exists():
            return set()
        return {d.name for d in models_dir.iterdir() if d.is_dir() and (d / "config.toml").exists()}

    def list_names(self) -> list[str]:
        """Sorted names of every available model — writable + bundled, deduped by name."""
        names = self._names_in(self.models_dir)
        if self.bundled_dir is not None:
            names |= self._names_in(self.bundled_dir)
        return sorted(names)

    def resolve(self, name: str | None) -> Path:
        """Resolve `name` to a model's artifact dir for read commands (serve/demo/play).

        With an explicit `name`, require it to exist (writable or bundled). With `name=None`, fall
        back to the single available model if there is exactly one; else raise, listing the choices.
        """
        if name is not None:
            found = self.dir_for(name)
            if found is None:
                available = self.list_names()
                hint = f" (have: {', '.join(available)})" if available else ""
                raise FileNotFoundError(
                    f"no world model named {name!r} under {self.models_dir}{hint}; "
                    "run `wmh build --name <name>` first"
                )
            return found

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
        resolved = self.dir_for(names[0])
        assert resolved is not None  # name came from list_names(), so it resolves
        return resolved

    def info(self, name: str) -> ModelInfo:
        """Read a model's config + metrics into a summary (for `wmh list`)."""
        model_dir = self.dir_for(name)
        if model_dir is None:
            raise FileNotFoundError(f"no world model named {name!r}")
        paths = ArtifactPaths(model_dir)
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


def _as_float(value: JsonValue) -> float | None:
    # bool is an int subclass; exclude it so a stray `true` doesn't read as 1.0.
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _as_int(value: JsonValue) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
