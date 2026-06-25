"""Retrieval over the trace replay buffer (DreamGym Eq. 4).

At each step the world model retrieves the top-k past steps whose (state, action) is most similar to
the current one, by cosine similarity of an embedding `phi`:

    {d_j} = Topk( cos( phi(s_t, a_t), phi(s_i, a_i) ) )

The buffer is initialized offline from ingested traces (`index`) and enriched online as the agent
steps (`add`).
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from wmh.core.types import Action, EnvState, JsonObject, Observation, Step, Trace
from wmh.providers.base import Provider

# A placeholder observation for query-only encoding: topk embeds (state, action), never the result.
_EMPTY_OBS = Observation(content="")


@runtime_checkable
class Retriever(Protocol):
    def index(self, traces: list[Trace]) -> None:
        """Build phase: embed every step's (state, action) and store it in the buffer."""
        ...

    def topk(self, state: EnvState, action: Action, k: int) -> list[Step]:
        """Runtime: return the k most similar prior steps to (state, action)."""
        ...

    def add(self, step: Step) -> None:
        """Online enrichment: add a freshly generated step to the buffer."""
        ...


def _render_json(value: JsonObject) -> str:
    """Stable, human-readable one-liner for a JSON object (sorted keys, no whitespace churn)."""
    if not value:
        return "{}"
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


class EmbeddingRetriever:
    """Default Retriever: dense cosine similarity using a provider's embedding model.

    The replay buffer is an in-memory embedding matrix (rows = steps) kept parallel to a
    ``list[Step]``. ``index`` embeds the whole corpus in one batched ``provider.embed`` call;
    ``add`` embeds a single step for online enrichment. ``topk`` ranks by cosine similarity,
    matching DreamGym Eq. 4's ``Topk(cos(phi(s_t,a_t), phi(s_i,a_i)))``.
    """

    def __init__(self, provider: Provider) -> None:
        self._provider = provider
        # Parallel structures: row i of `_matrix` is the embedding of `_steps[i]`.
        self._steps: list[Step] = []
        self._matrix: NDArray[np.float64] | None = None

    @staticmethod
    def _encode_text(state: EnvState, action: Action) -> str:
        """Render (state, action) into the text we embed for phi(s, a).

        We embed a *structured summary* rather than raw JSON blobs: a labelled, line-oriented
        rendering of the env state (structured config + scratchpad "database") followed by the
        action (kind, tool name, arguments, message). Labels are stable and keys are sorted so
        semantically equal steps render identically, keeping cosine similarity meaningful across
        traces.
        """
        lines = [
            "STATE:",
            f"  structured: {_render_json(state.structured)}",
        ]
        if state.scratchpad:
            lines.append(f"  scratchpad: {state.scratchpad}")
        lines.append(f"ACTION kind={action.kind.value}")
        if action.name is not None:
            lines.append(f"  tool: {action.name}")
        if action.arguments:
            lines.append(f"  arguments: {_render_json(action.arguments)}")
        if action.content is not None:
            lines.append(f"  message: {action.content}")
        return "\n".join(lines)

    def _embed_steps(self, steps: list[Step]) -> NDArray[np.float64]:
        texts = [self._encode_text(s.state_before, s.action) for s in steps]
        vectors = self._provider.embed(texts)
        return np.asarray(vectors, dtype=np.float64)

    def index(self, traces: list[Trace]) -> None:
        """Embed every step of every trace and (re)build the buffer from scratch."""
        steps = [step for trace in traces for step in trace.steps]
        self._steps = steps
        if not steps:
            self._matrix = None
            return
        self._matrix = self._embed_steps(steps)

    def topk(self, state: EnvState, action: Action, k: int) -> list[Step]:
        """Return the up-to-k most similar prior steps by cosine similarity."""
        if k <= 0 or self._matrix is None or not self._steps:
            return []
        query = self._embed_steps(
            [Step(action=action, observation=_EMPTY_OBS, state_before=state)]
        )[0]
        scores = _cosine(query, self._matrix)
        # argsort ascending, take the tail, reverse for descending-similarity order.
        count = min(k, len(self._steps))
        top = np.argsort(scores)[-count:][::-1]
        return [self._steps[int(i)] for i in top]

    def add(self, step: Step) -> None:
        """Append a freshly generated step to the buffer for online enrichment."""
        vector = self._embed_steps([step])
        self._steps.append(step)
        if self._matrix is None:
            self._matrix = vector
        else:
            self._matrix = np.vstack([self._matrix, vector])


def _cosine(query: NDArray[np.float64], matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cosine similarity of `query` against each row of `matrix`. Zero vectors score 0."""
    query_norm = float(np.linalg.norm(query))
    row_norms = np.linalg.norm(matrix, axis=1)
    denom = row_norms * query_norm
    dots = matrix @ query
    # Avoid divide-by-zero: where either vector is zero, similarity is 0.
    return np.divide(dots, denom, out=np.zeros_like(dots), where=denom > 0)
