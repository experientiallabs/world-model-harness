"""Serve the prebuilt tau-bench WM in EVAL configuration (BENCH-B shared protocol).

Env backend = PINNED GPT-5.5 (OpenAI; different family from the Haiku training WM —
strongest circularity blunting available). Reward judge = Opus 4.8 on Bedrock us-east-1
(D12 pins it for fidelity rows; third family vs both WM backends avoids same-family bias).

Requires OPENAI_API_KEY (gitignored .env at the repo root) and AWS creds for Bedrock.

Run from the wmh repo root:  uv run python .agents/scripts/serve_tau_eval.py [port]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn

from wmh.engine.world_model import WorldModel
from wmh.providers import get_provider
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.serving.server import create_app

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "examples" / "tau-bench" / "models" / "tau-bench"
EVAL_ENV_MODEL = "gpt-5.5"
JUDGE_MODEL = "us.anthropic.claude-opus-4-8"  # the artifact's own serve model id (config.toml)


def _load_dotenv() -> None:
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    _load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing: put it in the gitignored .env at the repo root")
    serve_provider = get_provider(ProviderConfig(kind=ProviderKind.OPENAI, model=EVAL_ENV_MODEL))
    judge_provider = get_provider(
        ProviderConfig(kind=ProviderKind.BEDROCK, model=JUDGE_MODEL, region="us-east-1")
    )
    wm = WorldModel.load(str(MODEL_DIR), serve_provider, reward_provider=judge_provider)
    app = create_app(world_models={"tau-bench": wm})
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
