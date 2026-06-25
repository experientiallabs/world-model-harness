"""Local FastAPI backend — the live environment agents call over HTTP.

Thin transport over an in-process `WorldModel`; the CLI and the API share the same code path.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from wmh.types import Action, EnvState, Observation, Session


class NewSessionRequest(BaseModel):
    task: str | None = None
    seed_state: EnvState | None = None


class NewSessionResponse(BaseModel):
    session_id: str


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Observation


def create_app(artifact_dir: str = ".wmh") -> FastAPI:
    """Build the FastAPI app bound to a loaded WorldModel.

    The WorldModel is loaded on startup (and providers verified) so the first request is fast.
    """
    app = FastAPI(title="World Model Harness")

    # TODO: load WorldModel.load(artifact_dir, provider) into app.state on startup.

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        # TODO: report provider + artifact status.
        raise NotImplementedError

    @app.post("/sessions", response_model=NewSessionResponse)
    def new_session(_req: NewSessionRequest) -> NewSessionResponse:
        raise NotImplementedError

    @app.get("/sessions/{session_id}", response_model=Session)
    def get_session(session_id: str) -> Session:
        raise NotImplementedError

    @app.post("/sessions/{session_id}/step", response_model=StepResponse)
    def step(session_id: str, _req: StepRequest) -> StepResponse:
        raise NotImplementedError

    return app
