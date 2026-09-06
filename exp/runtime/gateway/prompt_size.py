# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Pre-dispatch context-window check for the prompt itself.

A prompt that cannot fit any rung's context window is doomed before dispatch:
the provider only 400s it back, after a reservation, a round trip, and with a
provider-specific message. The gateway has no tokenizer for every model, so it
never guesses the exact count. It bounds it from BELOW: at
:data:`MAXIMUM_BYTES_PER_TOKEN` bytes of UTF-8 text per token, real tokenizers
on prose, code, or CJK text all produce MORE tokens than this estimate, so a
prompt whose lower bound already exceeds the largest window on the route is
certain to fail and is refused here with the exact numbers. The bound is a
documented heuristic, not a tokenizer proof, so it is deliberately loose and
the check abstains whenever any rung leaves its window undeclared. A prompt
under the bound is dispatched and left to the provider's precise count; output
budgets are never inspected (a too-small ceiling is an ``incomplete`` answer,
not a refusal).

Only text is counted. Inline media (images, audio, documents) tokenizes by its
own rules and is left to the provider.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from exp.common.models.content import TextContentPart
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayRequest
from exp.runtime.models.providers.errors import ProviderParameterError

if TYPE_CHECKING:
    from exp.runtime.gateway.routing import GatewayRoute

# Conservative UTF-8 bytes per token. English prose runs near 4, code near
# 3.5, CJK near 1.5 to 4.5, and even indentation-heavy code stays well under
# this, so the estimate is a lower bound in practice. It is a heuristic, not a
# tokenizer proof: only text dominated by very long whitespace runs could
# approach it, which is why the margin is twice typical prose rather than the
# tight 5 or 6 a precise count would allow.
MAXIMUM_BYTES_PER_TOKEN = 8

CONTEXT_LENGTH_EXCEEDED_CODE = "context_length_exceeded"


def prompt_text_bytes(request: GatewayRequest) -> int:
    """Count the UTF-8 bytes of every text the model will read.

    Args:
        request: Canonical gateway request.

    Returns:
        Bytes across message text, tool-call arguments, echoed provider items,
        and tool definitions. Inline media parts contribute nothing.
    """
    total = 0
    for message in request.messages:
        if message.content is not None:
            # ``content`` already joins every text part (the decoders mirror
            # text parts into it), so parts are only read when it is absent.
            total += len(message.content.encode("utf-8"))
        else:
            for part in message.content_parts:
                if isinstance(part, TextContentPart):
                    total += len(part.text.encode("utf-8"))
        for call in message.tool_calls:
            total += len(call.name.encode("utf-8"))
            arguments = (
                call.raw_arguments
                if call.raw_arguments is not None
                else json.dumps(call.arguments, separators=(",", ":"))
            )
            total += len(arguments.encode("utf-8"))
        for verbatim in (message.provider_native_item, message.provider_anthropic_block):
            if verbatim is not None:
                total += len(json.dumps(verbatim, separators=(",", ":")).encode("utf-8"))
    for tool in request.tools:
        total += len(tool.name.encode("utf-8"))
        if tool.description:
            total += len(tool.description.encode("utf-8"))
        total += len(json.dumps(tool.parameters, separators=(",", ":")).encode("utf-8"))
    for entry in request.provider_server_tools:
        total += len(json.dumps(entry, separators=(",", ":")).encode("utf-8"))
    return total


def minimum_prompt_tokens(request: GatewayRequest) -> int:
    """Lower-bound the prompt's token count from its text bytes.

    Args:
        request: Canonical gateway request.

    Returns:
        The fewest tokens any realistic tokenizer produces for this text.
    """
    return prompt_text_bytes(request) // MAXIMUM_BYTES_PER_TOKEN


def require_prompt_fits_context_window(route: GatewayRoute, request: GatewayRequest) -> None:
    """Refuse a prompt that is certain to exceed every rung's context window.

    Args:
        route: Resolved ordered route. A rung without a declared window is
            permissive (it may accept anything), so the check abstains unless
            EVERY rung declares one; refusing is only ever certain then.
        request: Canonical request about to be shaped and dispatched.

    Raises:
        ProviderParameterError: The prompt's lower-bound token count exceeds
            the largest declared context window on the route
            (``code='context_length_exceeded'``).
    """
    windows: list[int] = []
    for deployment in route.deployments:
        window = (
            None
            if deployment.capabilities is None
            else deployment.capabilities.context_window_tokens
        )
        if window is None:
            return
        windows.append(window)
    if not windows:
        return
    largest = max(windows)
    text_bytes = prompt_text_bytes(request)
    minimum = text_bytes // MAXIMUM_BYTES_PER_TOKEN
    if minimum <= largest:
        return
    param = "input" if request.surface == GatewayApiSurface.RESPONSES else "messages"
    raise ProviderParameterError(
        message=(
            f"The prompt is at least {minimum:,} tokens ({text_bytes:,} bytes of text), "
            f"but the largest context window on this model route is {largest:,} tokens. "
            "Shorten the conversation or choose a model with a larger context window."
        ),
        param=param,
        code=CONTEXT_LENGTH_EXCEEDED_CODE,
    )
