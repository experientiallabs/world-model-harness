"""Local FastAPI backend — the live environment agents call over HTTP.

Routes are namespaced by world model name (`/world_models/{name}/...`) so one backend can serve
several named models at once. Each route is a thin transport over an in-process `WorldModel`; the
CLI and the API share the same code path.

The backend is also the *reward* server for RL training: `POST .../sessions/{id}/score` judges the
session's rollout (task + history) with `EpisodeRewardJudge`, returning the scalar episode reward
(GRPO/PPO/REINFORCE++), per-step rewards, and a critique string (SDPO's teacher feedback) — so a
training scaffold gets environment and reward behind one API.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from wmh.core.types import Action, EnvState, Observation, Session
from wmh.engine.world_model import WorldModel
from wmh.optimize.reward import EpisodeScore
from wmh.tracking import RunRecord


class NewSessionRequest(BaseModel):
    task: str | None = None
    seed_state: EnvState | None = None


class NewSessionResponse(BaseModel):
    session_id: str


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Observation


class ModelsResponse(BaseModel):
    world_models: list[str]


class KnowledgeResponse(BaseModel):
    """The model's knowledge base: enabled + every markdown file's content."""

    enabled: bool
    files: dict[str, str]


class KnowledgeFileRequest(BaseModel):
    content: str


def _load_named_models(artifact_dir: str, names: list[str] | None) -> dict[str, WorldModel]:
    """Load the requested world models (or all built ones) from `artifact_dir` by name."""
    from wmh.config import WorldModelStore
    from wmh.engine import load_world_model

    store = WorldModelStore(artifact_dir)
    chosen = names if names is not None else store.list_names()
    if not chosen:
        raise FileNotFoundError(
            f"no world models built under {store.models_dir}; run `wmh build --name <name>` first"
        )
    models: dict[str, WorldModel] = {}
    for name in chosen:
        world_model, _provider = load_world_model(store.resolve(name), telemetry_root=store.root)
        models[name] = world_model
    return models


def create_app(
    artifact_dir: str = ".wmh",
    names: list[str] | None = None,
    world_models: dict[str, WorldModel] | None = None,
) -> FastAPI:
    """Build the FastAPI app serving one or more named WorldModels.

    Models are either injected directly via `world_models` (name -> model, for tests), or loaded
    from `artifact_dir` with `names` selecting which to serve (default: all built ones).
    """
    app = FastAPI(title="World Model Harness")
    models = world_models if world_models is not None else _load_named_models(artifact_dir, names)

    def _model_or_404(name: str) -> WorldModel:
        try:
            return models[name]
        except KeyError:
            available = ", ".join(sorted(models)) or "(none)"
            raise HTTPException(
                status_code=404, detail=f"no world model {name!r}; have: {available}"
            ) from None

    def _session_or_404(wm: WorldModel, session_id: str) -> Session:
        try:
            return wm.get_session(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no session {session_id}") from None

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/world_models", response_model=ModelsResponse)
    def list_world_models() -> ModelsResponse:
        return ModelsResponse(world_models=sorted(models))

    @app.post("/world_models/{world_model_name}/sessions", response_model=NewSessionResponse)
    def new_session(world_model_name: str, req: NewSessionRequest) -> NewSessionResponse:
        wm = _model_or_404(world_model_name)
        session = wm.new_session(task=req.task, seed_state=req.seed_state)
        return NewSessionResponse(session_id=session.id)

    @app.get("/world_models/{world_model_name}/sessions/{session_id}", response_model=Session)
    def get_session(world_model_name: str, session_id: str) -> Session:
        wm = _model_or_404(world_model_name)
        return _session_or_404(wm, session_id)

    @app.get(
        "/world_models/{world_model_name}/sessions/{session_id}/usage", response_model=RunRecord
    )
    def session_usage(world_model_name: str, session_id: str) -> RunRecord:
        """Per-session token/cost/time so far (serve-time observability)."""
        wm = _model_or_404(world_model_name)
        _session_or_404(wm, session_id)
        return wm.session_usage(session_id)

    @app.post(
        "/world_models/{world_model_name}/sessions/{session_id}/step", response_model=StepResponse
    )
    def step(world_model_name: str, session_id: str, req: StepRequest) -> StepResponse:
        wm = _model_or_404(world_model_name)
        _session_or_404(wm, session_id)
        observation = wm.step(session_id, req.action)
        return StepResponse(observation=observation)

    @app.post(
        "/world_models/{world_model_name}/sessions/{session_id}/score",
        response_model=EpisodeScore,
    )
    def score_session(world_model_name: str, session_id: str) -> EpisodeScore:
        """Judge the session's rollout so far: episode reward + per-step rewards + critique."""
        wm = _model_or_404(world_model_name)
        _session_or_404(wm, session_id)
        return wm.score_session(session_id)

    @app.delete("/world_models/{world_model_name}/sessions/{session_id}", response_model=RunRecord)
    def end_session(world_model_name: str, session_id: str) -> RunRecord:
        """End the session (free its memory + metering) and return its final usage record."""
        wm = _model_or_404(world_model_name)
        _session_or_404(wm, session_id)
        return wm.end_session(session_id)

    @app.get("/world_models/{world_model_name}/knowledge", response_model=KnowledgeResponse)
    def get_knowledge(world_model_name: str) -> KnowledgeResponse:
        """Read the model's knowledge base (`enabled=False` for pre-knowledge artifacts)."""
        kb = _model_or_404(world_model_name).knowledge
        if kb is None:
            return KnowledgeResponse(enabled=False, files={})
        return KnowledgeResponse(enabled=True, files=kb.files())

    @app.put(
        "/world_models/{world_model_name}/knowledge/{file_name}",
        response_model=KnowledgeResponse,
    )
    def put_knowledge(
        world_model_name: str, file_name: str, req: KnowledgeFileRequest
    ) -> KnowledgeResponse:
        """Create/replace one knowledge markdown file (the HTTP face of 'edit the folder')."""
        kb = _model_or_404(world_model_name).knowledge
        if kb is None:
            raise HTTPException(
                status_code=409,
                detail=f"world model {world_model_name!r} has no knowledge base; "
                "build it with knowledge enabled (or create a knowledge/ dir in its artifact "
                "and re-serve)",
            )
        try:
            kb.write_file(file_name, req.content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return KnowledgeResponse(enabled=True, files=kb.files())

    return app
