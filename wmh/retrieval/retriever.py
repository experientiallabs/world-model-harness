"""Retrieval over the trace replay buffer (DreamGym Eq. 4).

At each step the world model retrieves the top-k past steps whose (state, action) is most similar to
the current one, by cosine similarity of an embedding `phi`:

    {d_j} = Topk( cos( phi(s_t, a_t), phi(s_i, a_i) ) )

The buffer is initialized offline from ingested traces (`index`) and enriched online as the agent
steps (`add`).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from wmh.core.types import Action, EnvState, Step, Trace
from wmh.providers.base import Provider


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


class EmbeddingRetriever:
    """Default Retriever: dense cosine similarity using a provider's embedding model.

    The skeleton fixes the interface; the index/search internals (in-memory matrix vs. a real
    vector store) are deferred.
    """

    def __init__(self, provider: Provider) -> None:
        self._provider = provider
        # TODO: hold embeddings matrix + parallel list[Step].

    @staticmethod
    def _encode_text(state: EnvState, action: Action) -> str:
        """Render (state, action) into the text we embed for phi(s, a)."""
        # TODO: decide raw-text vs. structured-summary encoding (see DESIGN open questions).
        raise NotImplementedError

    def index(self, traces: list[Trace]) -> None:
        raise NotImplementedError

    def topk(self, state: EnvState, action: Action, k: int) -> list[Step]:
        raise NotImplementedError

    def add(self, step: Step) -> None:
        raise NotImplementedError
