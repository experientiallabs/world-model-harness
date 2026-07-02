"""Serve the prebuilt tau-bench WM with a Bedrock Haiku 4.5 backend (training-env config).

Same provider swap as examples/tau-bench/rl/smoke.py — the artifact's built-in Opus
provider is overridden so RL smokes/training don't cost Opus money. Serve + reward
judge both run on haiku, wrapped in a same-model FallbackProvider chain (D18): a
throttle fails over instantly and a hung read fails over at the bedrock client's
600s bound instead of killing the episode (observed live: turn-1 WM steps hanging
past the scaffold's timeout).

Run from the wmh repo root:  uv run python .agents/scripts/serve_tau_haiku.py [port]
"""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

from wmh.engine.world_model import WorldModel
from wmh.providers import get_provider
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.providers.fallback import FallbackProvider
from wmh.serving.server import create_app

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "examples" / "tau-bench" / "models" / "tau-bench"
HAIKU_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"  # dated profile id (required)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    cfg = ProviderConfig(kind=ProviderKind.BEDROCK, model=HAIKU_MODEL, region="us-east-1")
    # Two independent provider instances (each owns its boto client/connection pool),
    # same model: retry-via-failover without changing the env distribution.
    provider = FallbackProvider([get_provider(cfg), get_provider(cfg)])
    wm = WorldModel.load(str(MODEL_DIR), provider, reward_provider=provider)
    app = create_app(world_models={"tau-bench": wm})
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
