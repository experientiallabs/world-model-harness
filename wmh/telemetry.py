"""Best-effort anonymous usage telemetry."""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import httpx

from wmh.config import ARTIFACT_DIR
from wmh.config.settings import ensure_telemetry_anonymous_id, load_settings

POSTHOG_PROJECT_API_KEY = "phc_rPFfCufWpxyctR7duEZTTXovP4k5kbHqSqzd4Z4MQJdL"
POSTHOG_HOST = "https://us.i.posthog.com"

TelemetryValue = str | int | float | bool | None
TelemetryProperties = dict[str, TelemetryValue]

_FALSE_VALUES = {"0", "false", "off", "no"}
_TRUE_VALUES = {"1", "true", "on", "yes"}


def capture(
    event: str,
    properties: TelemetryProperties | None = None,
    *,
    root: str | Path = ARTIFACT_DIR,
) -> bool:
    """Send one anonymous metadata-only event. Returns False when skipped or failed."""
    if not _enabled(root):
        return False
    api_key = os.getenv("WMH_POSTHOG_PROJECT_API_KEY", POSTHOG_PROJECT_API_KEY).strip()
    if not api_key:
        return False
    host = os.getenv("WMH_POSTHOG_HOST", POSTHOG_HOST).rstrip("/")
    try:
        distinct_id = ensure_telemetry_anonymous_id(root)
        event_properties: TelemetryProperties = {
            "$process_person_profile": False,
            "wmh_version": _wmh_version(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            **(properties or {}),
        }
        # Never log prompts, traces, actions, observations, paths, models, credentials, or text.
        response = httpx.post(
            f"{host}/i/v0/e/",
            json={
                "api_key": api_key,
                "event": event,
                "distinct_id": distinct_id,
                "properties": event_properties,
            },
            timeout=0.5,
        )
        return 200 <= response.status_code < 300
    except (httpx.HTTPError, OSError, ValueError):
        return False


def _enabled(root: str | Path) -> bool:
    if _env_truthy("DO_NOT_TRACK"):
        return False
    env = os.getenv("WMH_TELEMETRY")
    if env is not None:
        return env.strip().lower() in _TRUE_VALUES
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    try:
        return load_settings(root).telemetry.enabled
    except ValueError:
        return False


def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    normalized = value.strip().lower()
    if normalized in _FALSE_VALUES:
        return False
    return bool(normalized)


def _wmh_version() -> str:
    try:
        return version("world-model-harness")
    except PackageNotFoundError:
        return "0+local"
