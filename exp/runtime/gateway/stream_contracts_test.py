"""Tests for stream outcome contracts."""

import pytest

from exp.runtime.gateway.stream_contracts import GatewayEvent, GatewayEventKind


@pytest.mark.parametrize("length", [257, 65_536])
def test_stream_started_event_preserves_long_tool_id(length: int) -> None:
    """Provider tool IDs fit the same bound on output as on replay."""
    event = GatewayEvent(
        kind=GatewayEventKind.TOOL_CALL_STARTED,
        sequence_number=0,
        tool_call_index=0,
        tool_call_id="x" * length,
        tool_name="terminal",
    )
    assert event.tool_call_id == "x" * length
