"""Local FastAPI backend — the live environment agents call over HTTP.

Thin transport over an in-process `WorldModel`; the CLI and the API share the same code path.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from wmh.core.types import Action, EnvState, Observation, Session
from wmh.engine.world_model import WorldModel


class NewSessionRequest(BaseModel):
    task: str | None = None
    seed_state: EnvState | None = None


class NewSessionResponse(BaseModel):
    session_id: str


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Observation


def _load_world_model(artifact_dir: str) -> WorldModel:
    """Load the served WorldModel from `.wmh/` using the configured serve provider."""
    from wmh.config import load_config
    from wmh.providers import get_provider

    config = load_config(artifact_dir)
    provider = get_provider(config.serve_provider_config())
    return WorldModel.load(artifact_dir, provider)


def create_app(artifact_dir: str = ".wmh", world_model: WorldModel | None = None) -> FastAPI:
    """Build the FastAPI app bound to a loaded WorldModel.

    The WorldModel is loaded from `artifact_dir` on construction (so the first request is fast), or
    injected directly via `world_model` for testing.
    """
    app = FastAPI(title="World Model Harness")
    wm = world_model or _load_world_model(artifact_dir)

    def _session_or_404(session_id: str) -> Session:
        try:
            return wm.get_session(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no session {session_id}") from None

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/sessions", response_model=NewSessionResponse)
    def new_session(req: NewSessionRequest) -> NewSessionResponse:
        session = wm.new_session(task=req.task, seed_state=req.seed_state)
        return NewSessionResponse(session_id=session.id)

    @app.get("/sessions/{session_id}", response_model=Session)
    def get_session(session_id: str) -> Session:
        return _session_or_404(session_id)

    @app.post("/sessions/{session_id}/step", response_model=StepResponse)
    def step(session_id: str, req: StepRequest) -> StepResponse:
        _session_or_404(session_id)
        observation = wm.step(session_id, req.action)
        return StepResponse(observation=observation)

    return app
