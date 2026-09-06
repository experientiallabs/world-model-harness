# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Tests for the pre-dispatch context-window refusal."""

from __future__ import annotations

import base64

import pytest

from exp.common.models.content import ImageContentPart, TextContentPart
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.common.models.model import ModelCapabilities, ToolCall
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.gateway.prompt_size import (
    MAXIMUM_BYTES_PER_TOKEN,
    minimum_prompt_tokens,
    prompt_text_bytes,
    require_prompt_fits_context_window,
)
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.models.providers.errors import ProviderParameterError


def _deployment(deployment_id: str, context_window_tokens: int | None) -> ExactModelDeployment:
    return ExactModelDeployment(
        deployment_id=deployment_id,
        source_alias=deployment_id,
        exact_model_id="exact-one",
        connection=f"connection-{deployment_id}",
        provider="openai",
        provider_model="provider-model",
        connection_sha256="b" * 64,
        capabilities_sha256="c" * 64,
        capabilities=(
            None
            if context_window_tokens is None
            else ModelCapabilities(context_window_tokens=context_window_tokens)
        ),
    )


def _route(*deployments: ExactModelDeployment) -> GatewayRoute:
    authorization = AuthorizationSnapshot(
        request_id="request-one",
        organization_id="organization-one",
        identity_id="identity-one",
        virtual_key_id="key-one",
        alias="public-model",
        alias_revision_id="revision-one",
        target=DirectTarget(pool_id="pool-one"),
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        catalog_sha256="a" * 64,
        canonical_request_sha256="d" * 64,
        deadline_monotonic=1.0,
    )
    return GatewayRoute(
        snapshot=ExecutionSnapshot(
            authorization=authorization,
            exact_model_id="exact-one",
            pool_id="pool-one",
            deployment_ids=tuple(item.deployment_id for item in deployments),
            failover_mode="maximize_availability",
        ),
        deployment=deployments[0],
        fallback_deployments=deployments[1:],
        route_reason="direct",
    )


def _request(
    text: str, surface: GatewayApiSurface = GatewayApiSurface.CHAT_COMPLETIONS
) -> GatewayRequest:
    return GatewayRequest(surface=surface, messages=(GatewayMessage(role="user", content=text),))


def test_text_bytes_count_every_text_the_model_reads_and_no_media() -> None:
    """Message text, tool calls, and tool schemas count once each; inline images do not."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(role="system", content="sys"),
            GatewayMessage(
                role="user",
                # The decoders mirror text parts into ``content``; it is counted once.
                content="héllo",
                content_parts=(
                    TextContentPart(text="héllo"),
                    ImageContentPart(
                        media_type="image/png",
                        data=base64.b64encode(b"\x89PNG" * 100).decode("ascii"),
                    ),
                ),
            ),
            GatewayMessage(
                role="assistant",
                content=None,
                tool_calls=(ToolCall(call_id="call_1", name="lookup", arguments={"q": "x"}),),
            ),
        ),
        tools=(
            GatewayToolDefinition(name="lookup", description="d", parameters={"type": "object"}),
        ),
    )
    expected = (
        len(b"sys")
        + len("héllo".encode())
        + len(b"lookup")
        + len(b'{"q":"x"}')
        + len(b"lookup")
        + len(b"d")
        + len(b'{"type":"object"}')
    )
    assert prompt_text_bytes(request) == expected
    assert minimum_prompt_tokens(request) == expected // MAXIMUM_BYTES_PER_TOKEN


def test_raw_tool_arguments_are_counted_verbatim_when_present() -> None:
    """A provider's exact argument text outranks the parsed object for byte counting."""
    call = ToolCall(call_id="call_1", name="f", arguments={"a": 1}, raw_arguments='{"a":   1}')
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="assistant", content=None, tool_calls=(call,)),),
    )
    assert prompt_text_bytes(request) == len(b"f") + len(b'{"a":   1}')


def test_a_prompt_certain_to_overflow_every_window_is_refused_with_the_numbers() -> None:
    """The lower bound exceeds the largest window: refuse before dispatch, naming both numbers."""
    route = _route(_deployment("small", 1_000), _deployment("large", 2_000))
    # 2,001 tokens even at the most generous bytes-per-token; certain to fail on both rungs.
    request = _request("x" * (2_001 * MAXIMUM_BYTES_PER_TOKEN))

    with pytest.raises(ProviderParameterError) as caught:
        require_prompt_fits_context_window(route, request)

    assert caught.value.code == "context_length_exceeded"
    assert caught.value.param == "messages"
    assert "at least 2,001 tokens" in str(caught.value)
    assert "largest context window on this model route is 2,000 tokens" in str(caught.value)


def test_a_prompt_that_might_fit_the_largest_window_dispatches() -> None:
    """Only certainty refuses: a prompt over the small rung but under the large one goes through."""
    route = _route(_deployment("small", 1_000), _deployment("large", 2_000))
    require_prompt_fits_context_window(route, _request("x" * (1_500 * MAXIMUM_BYTES_PER_TOKEN)))
    # Exactly at the bound is not over it.
    require_prompt_fits_context_window(route, _request("x" * (2_000 * MAXIMUM_BYTES_PER_TOKEN)))


def test_routes_without_a_declared_window_never_refuse() -> None:
    """No declaration means no bound: the provider's own count decides."""
    route = _route(_deployment("unknown", None))
    require_prompt_fits_context_window(route, _request("x" * 10_000_000))


def test_one_undeclared_rung_makes_the_whole_route_abstain() -> None:
    """An undeclared rung is permissive elsewhere, so certainty is gone and nothing is refused."""
    route = _route(_deployment("small", 100), _deployment("unknown", None))
    require_prompt_fits_context_window(route, _request("x" * (10_000 * MAXIMUM_BYTES_PER_TOKEN)))


def test_the_responses_surface_names_its_input_field() -> None:
    """The refusal points at the field the caller actually sent."""
    route = _route(_deployment("small", 10))
    request = _request("x" * (11 * MAXIMUM_BYTES_PER_TOKEN), GatewayApiSurface.RESPONSES)
    with pytest.raises(ProviderParameterError) as caught:
        require_prompt_fits_context_window(route, request)
    assert caught.value.param == "input"
