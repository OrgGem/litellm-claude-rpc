from types import SimpleNamespace

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock
from claude_agent_sdk.types import StreamEvent

from litellm_claude_rpc.contracts import Config, GatewayError
from litellm_claude_rpc.driver import ClaudeDriver


def result(**extra):
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="s",
        **extra,
    )


def client(messages):
    async def query(prompt):
        assert prompt == "hello"

    async def receive():
        for message in messages:
            yield message

    return SimpleNamespace(query=query, receive_response=receive)


async def test_partial_and_final_sdk_text_not_duplicated():
    driver = ClaudeDriver(Config(), None, "")
    driver.client = client(
        [
            StreamEvent(
                "u",
                "s",
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "héllo"},
                },
            ),
            AssistantMessage(
                content=[TextBlock("héllo")],
                model="fixture",
                message_id="step-1",
                usage={"input_tokens": 3, "output_tokens": 4},
            ),
            result(usage={"input_tokens": 3, "output_tokens": 4}),
        ]
    )
    events = [e async for e in driver.stream("hello")]
    assert "".join(e.text for e in events) == "héllo"
    assert events[-1].usage["total_tokens"] == 7


async def test_cumulative_result_usage_is_not_invented_as_turn_usage():
    driver = ClaudeDriver(Config(), None, "")
    driver.client = client(
        [
            AssistantMessage(content=[TextBlock("hello")], model="fixture"),
            result(usage={"input_tokens": 9999, "output_tokens": 8888}),
        ]
    )
    events = [e async for e in driver.stream("hello")]
    assert events[-1].usage is None


async def test_tool_activity_rejected_even_if_cli_misconfigured():
    driver = ClaudeDriver(Config(), None, "")
    driver.client = client(
        [
            AssistantMessage(
                content=[ToolUseBlock(id="t", name="Read", input={})], model="fixture"
            )
        ]
    )
    with pytest.raises(GatewayError, match="unexpected_tool_activity"):
        _ = [e async for e in driver.stream("hello")]


async def test_cancelled_result_is_not_success():
    driver = ClaudeDriver(Config(), None, "")
    driver.client = client([result(terminal_reason="aborted_streaming")])
    with pytest.raises(GatewayError, match="backend_turn_failed"):
        _ = [e async for e in driver.stream("hello")]


async def test_stale_terminal_without_assistant_is_rejected():
    driver = ClaudeDriver(Config(), None, "")
    driver.client = client([result()])
    with pytest.raises(GatewayError, match="backend_turn_failed"):
        _ = [e async for e in driver.stream("hello")]


async def test_doctor_succeeds_when_logged_in_and_rejects_logged_out(monkeypatch):
    from litellm_claude_rpc import driver as module

    logged_in = True

    class Process:
        returncode = 0

        def __init__(self, args):
            self.args = args

        async def communicate(self):
            return (
                b"2.1.321 (Claude Code)"
                if "--version" in self.args
                else (b'{"loggedIn":true}' if logged_in else b'{"loggedIn":false}')
            ), b""

    async def spawn(*args, **kwargs):
        return Process(args)

    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", spawn)
    status = await module.doctor(Config())
    assert status["status"] == "ready"
    assert status["cli_version"] == "2.1.321"
    logged_in = False
    with pytest.raises(GatewayError, match="cli_login_required"):
        await module.doctor(Config())
