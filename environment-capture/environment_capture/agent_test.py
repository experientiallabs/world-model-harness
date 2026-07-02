"""Tests for the Bedrock bash-agent capture loop (stubbed converse client)."""

from __future__ import annotations

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
        self.calls.append({"modelId": modelId, "messages": list(messages)})
        return self._responses[len(self.calls) - 1]


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
