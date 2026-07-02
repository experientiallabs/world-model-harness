"""Project-local settings stored under the selected harness root."""

from __future__ import annotations

import tomllib
import uuid
from pathlib import Path

import tomli_w
from pydantic import BaseModel, Field, ValidationError

from wmh.config.config import ARTIFACT_DIR

SETTINGS_FILENAME = "settings.toml"


class TelemetrySettings(BaseModel):
    """Usage telemetry preferences for this harness project."""

    enabled: bool = True
    anonymous_id: str | None = None


class ProjectSettings(BaseModel):
    """Settings that are local to one harness project root."""

    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)


def settings_path(root: str | Path = ARTIFACT_DIR) -> Path:
    return Path(root) / SETTINGS_FILENAME


def load_settings(root: str | Path = ARTIFACT_DIR) -> ProjectSettings:
    path = settings_path(root)
    if not path.exists():
        return ProjectSettings()
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path} is not valid TOML ({exc})") from exc
    try:
        return ProjectSettings.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"{path} does not match the current settings schema ({exc})") from exc


def save_settings(settings: ProjectSettings, root: str | Path = ARTIFACT_DIR) -> None:
    path = settings_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = settings.model_dump(mode="json", exclude_none=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("wb") as fh:
        tomli_w.dump(data, fh)
    tmp.replace(path)


def set_telemetry_enabled(enabled: bool, root: str | Path = ARTIFACT_DIR) -> ProjectSettings:
    settings = load_settings(root)
    settings.telemetry.enabled = enabled
    save_settings(settings, root)
    return settings


def ensure_telemetry_anonymous_id(root: str | Path = ARTIFACT_DIR) -> str:
    settings = load_settings(root)
    if settings.telemetry.anonymous_id is None:
        settings.telemetry.anonymous_id = uuid.uuid4().hex
        save_settings(settings, root)
    return settings.telemetry.anonymous_id
