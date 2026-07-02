"""Tests for the Bedrock bash-agent capture loop (stubbed converse client)."""

from __future__ import annotations

from botocore.exceptions import ReadTimeoutError

from environment_capture.agent import BedrockBashAgent
from environment_capture.localexec import LocalBashEnv
from environment_capture.trajectory import JsonValue, Task


def _tool_use(name: str, tool_input: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "thinking..."},
                    {"toolUse": {"toolUseId": "t1", "name": name, "input": tool_input}},
                ],
            }
        },
        "stopReason": "tool_use",
    }


def _parallel_tool_use(
    calls: list[tuple[str, str, dict[str, JsonValue]]],
) -> dict[str, JsonValue]:
    """An assistant turn emitting several tool-use blocks at once (parallel tool calls)."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "doing several things"}]
                + [
                    {"toolUse": {"toolUseId": tid, "name": name, "input": tool_input}}
                    for tid, name, tool_input in calls
                ],
            }
        },
        "stopReason": "tool_use",
    }


class _StubClient:
    def __init__(self, responses: list[dict[str, JsonValue]]) -> None:
        self._responses = responses
        self.calls: list[dict[str, JsonValue]] = []

    def converse(
        self,
        *,
        modelId: str,
        messages: list[JsonValue],
        system: list[JsonValue],
        toolConfig: JsonValue,
        inferenceConfig: JsonValue,
    ) -> dict[str, JsonValue]:
        self.calls.append({"modelId": modelId, "messages": list(messages), "system": system})
        return self._responses[len(self.calls) - 1]


class _FlakyClient:
    """Raises a transient error on the first N converse calls, then delegates to a stub."""

    def __init__(self, error: Exception, fails: int, response: dict[str, JsonValue]) -> None:
        self._error = error
        self._remaining_fails = fails
        self._response = response
        self.calls = 0

    def converse(self, **kwargs: JsonValue) -> dict[str, JsonValue]:
        self.calls += 1
        if self._remaining_fails > 0:
            self._remaining_fails -= 1
            raise self._error
        return self._response


def test_read_timeout_is_retried(tmp_path) -> None:  # noqa: ANN001
    """A hung Bedrock call (read timeout) is transient and must be retried, not fatal."""
    timeout = ReadTimeoutError(endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com")
    client = _FlakyClient(timeout, fails=2, response=_tool_use("submit", {"answer": "42"}))
    agent = BedrockBashAgent(model_id="m", client=client, retry_backoff_s=0.0)
    env = LocalBashEnv(workspace=tmp_path)
    try:
        run = agent.run(Task(task_id="t0", prompt="q", data={}), env)
    finally:
        env.close()
    assert run.final_answer == "42"
    assert client.calls == 3  # two timeouts retried, third succeeds


def test_agent_executes_commands_then_submits(tmp_path) -> None:  # noqa: ANN001
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.txt").write_text("capex 1577")
    client = _StubClient(
        [
            _tool_use("bash", {"command": "grep capex docs/a.txt"}),
            _tool_use("submit", {"answer": "$1577 million"}),
        ]
    )
    agent = BedrockBashAgent(model_id="us.anthropic.claude-opus-4-8", client=client)
    env = LocalBashEnv(workspace=tmp_path)
    try:
        run = agent.run(Task(task_id="t0", prompt="What is capex?", data={}), env)
    finally:
        env.close()

    assert run.final_answer == "$1577 million"
    assert run.model == "us.anthropic.claude-opus-4-8"
    assert len(run.steps) == 1
    assert run.steps[0].action.arguments == {"command": "grep capex docs/a.txt"}
    assert "capex 1577" in run.steps[0].output
    assert run.steps[0].is_error is False
    # The real command output must be fed back to the model as a toolResult.
    second_call_messages = client.calls[1]["messages"]
    assert "capex 1577" in str(second_call_messages)


def test_agent_stops_at_max_steps(tmp_path) -> None:  # noqa: ANN001
    client = _StubClient([_tool_use("bash", {"command": "echo again"}) for _ in range(5)])
    agent = BedrockBashAgent(model_id="m", client=client, max_steps=2)
    env = LocalBashEnv(workspace=tmp_path)
    try:
        run = agent.run(Task(task_id="t0", prompt="loop forever", data={}), env)
    finally:
        env.close()
    assert len(run.steps) == 2
    assert run.final_answer == ""


def test_parallel_tool_uses_all_get_answered(tmp_path) -> None:  # noqa: ANN001
    """Every tool-use block in one turn must be executed and answered (Bedrock requires it).

    Anthropic models emit parallel tool calls; responding to only the first leaves dangling
    toolUse ids and the next Converse call is rejected.
    """
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.txt").write_text("beta")
    client = _StubClient(
        [
            _parallel_tool_use(
                [
                    ("u1", "bash", {"command": "cat a.txt"}),
                    ("u2", "bash", {"command": "cat b.txt"}),
                ]
            ),
            _tool_use("submit", {"answer": "alpha+beta"}),
        ]
    )
    agent = BedrockBashAgent(model_id="m", client=client)
    env = LocalBashEnv(workspace=tmp_path)
    try:
        run = agent.run(Task(task_id="t0", prompt="read both", data={}), env)
    finally:
        env.close()

    assert [s.action.arguments["command"] for s in run.steps] == ["cat a.txt", "cat b.txt"]
    assert run.final_answer == "alpha+beta"
    # The follow-up user turn carries a toolResult for BOTH ids, in order.
    follow_up = client.calls[1]["messages"][-1]
    assert follow_up["role"] == "user"
    result_ids = [block["toolResult"]["toolUseId"] for block in follow_up["content"]]
    assert result_ids == ["u1", "u2"]


def test_custom_system_prompt_is_sent(tmp_path) -> None:  # noqa: ANN001
    """A benchmark with a different environment (e.g. a db, not docs/) can frame the agent."""
    client = _StubClient([_tool_use("submit", {"answer": "done"})])
    agent = BedrockBashAgent(
        model_id="m", client=client, system_prompt="Explore the SQLite db at ./database.db."
    )
    env = LocalBashEnv(workspace=tmp_path)
    try:
        agent.run(Task(task_id="t0", prompt="q", data={}), env)
    finally:
        env.close()
    assert client.calls[0]["system"] == [{"text": "Explore the SQLite db at ./database.db."}]


def test_plain_text_reply_is_the_final_answer(tmp_path) -> None:  # noqa: ANN001
    client = _StubClient(
        [
            {
                "output": {
                    "message": {"role": "assistant", "content": [{"text": "The answer is 42."}]}
                },
                "stopReason": "end_turn",
            }
        ]
    )
    agent = BedrockBashAgent(model_id="m", client=client)
    env = LocalBashEnv(workspace=tmp_path)
    try:
        run = agent.run(Task(task_id="t0", prompt="q", data={}), env)
    finally:
        env.close()
    assert run.steps == []
    assert run.final_answer == "The answer is 42."
