import importlib.util
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock
from openai import AsyncOpenAI
from test_sessions import Agent

import litellm_claude_rpc
from litellm_claude_rpc.app import App
from litellm_claude_rpc.contracts import Config, Event, GatewayError
from litellm_claude_rpc.driver import ClaudeDriver, is_supported_cli_version
from litellm_claude_rpc.provider import (
    ClaudeLLM,
    ProviderError,
    claude_rpc_provider,
)
from litellm_claude_rpc.responses_context import responses_request_context
from litellm_claude_rpc.responses_guard import claude_rpc_responses_guard
from litellm_claude_rpc.runtime import Runtime
from litellm_claude_rpc.sessions import SessionPool

KEY = "sk-fixture-key-123456789012345"
REPO_ROOT = Path(__file__).resolve().parents[1]


class RichAgent(Agent):
    def __init__(self, model, system):
        super().__init__(model, system)
        self.system = system

    async def stream(self, prompt):
        self.turns += 1
        if "trigger-error" in prompt:
            yield Event(text="partial ")
            raise GatewayError("backend_turn_failed")
        yield Event(text=f"sys={self.system}|" if self.system else "")
        yield Event(text=f"turn-{self.turns}:{prompt}")
        if "no-usage" in prompt:
            yield Event(done=True, usage=None)
        else:
            yield Event(
                done=True,
                usage={
                    "prompt_tokens": 12,
                    "completion_tokens": 4,
                    "total_tokens": 16,
                    "prompt_tokens_details": {
                        "cached_tokens": 5,
                        "cache_write_tokens": 3,
                    },
                },
            )


def test_supported_cli_versions_and_env_config(monkeypatch):
    for ok in ("2.1.321", "2.1.325 (Claude Code)", "2.1.400-rc.1"):
        assert is_supported_cli_version(ok)
    for bad in ("2.1.320", "2.0.999", "3.0.0", "", None):
        assert not is_supported_cli_version(bad)

    assert isinstance(claude_rpc_provider, ClaudeLLM)
    assert "claude_rpc_provider" in litellm_claude_rpc.__all__
    monkeypatch.setenv("CLAUDE_MODEL_MAP", '{"claude-fast":"claude-sonnet-4-6"}')
    monkeypatch.setenv("CLAUDE_RPC_LIMITS", '{"concurrency":3,"max_sessions":6}')
    monkeypatch.setenv("CLAUDE_RPC_ALLOW_PROMPT_FLATTENING", "true")
    monkeypatch.setenv("LITELLM_MASTER_KEY", KEY)
    cfg = Config.from_env()
    assert cfg.models == {"claude-fast": "claude-sonnet-4-6"}
    assert cfg.concurrency == 3
    assert cfg.max_sessions == 6
    assert cfg.flatten is True
    assert cfg.api_key == KEY


def test_example_proxy_config_and_shim():
    cfg = yaml.safe_load((REPO_ROOT / "config.example.yaml").read_text("utf-8"))
    assert cfg["model_list"][0]["litellm_params"]["use_chat_completions_api"] is True
    spec = importlib.util.spec_from_file_location(
        "claude_proxy_handler", REPO_ROOT / "proxy_handler.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.claude_rpc_provider is claude_rpc_provider
    assert module.claude_rpc_responses_guard is claude_rpc_responses_guard


async def test_driver_preserves_cache_read_and_write_usage_details():
    driver = ClaudeDriver(Config(), None, "")

    async def query(prompt):
        assert prompt == "hello"

    async def receive():
        yield AssistantMessage(
            content=[TextBlock("hello")],
            model="fixture",
            message_id="step-1",
            usage={
                "input_tokens": 4,
                "output_tokens": 3,
                "cache_read_input_tokens": 6,
                "cache_creation_input_tokens": 2,
            },
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="s",
        )

    driver.client = SimpleNamespace(query=query, receive_response=receive)
    events = [e async for e in driver.stream("hello")]
    assert events[-1].usage == {
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
        "prompt_tokens_details": {"cached_tokens": 6, "cache_write_tokens": 2},
    }


async def test_responses_api_and_developer_role_http_sdk():
    pool = SessionPool(Config(api_key=KEY, timeout=1.0), RichAgent)
    app = App(pool)
    app.ready = True
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as http:
            sdk = AsyncOpenAI(
                api_key=KEY,
                base_url="http://test/v1",
                http_client=http,
                max_retries=0,
            )
            try:
                chat = await sdk.chat.completions.create(
                    model="claude-default",
                    messages=[
                        {"role": "developer", "content": "dev-rule"},
                        {"role": "user", "content": "hi"},
                    ],
                )
                assert "sys=dev-rule|" in chat.choices[0].message.content
                assert chat.usage.prompt_tokens_details.cached_tokens == 5

                res = await sdk.responses.create(
                    model="claude-default",
                    instructions="resp-rule",
                    input="hello-resp",
                    store=False,
                )
                assert res.status == "completed"
                assert "sys=resp-rule|" in res.output_text
                assert "hello-resp" in res.output_text
                dumped = res.model_dump()
                assert dumped["usage"]["input_tokens"] == 12
                assert dumped["usage"]["input_tokens_details"]["cached_tokens"] == 5
                assert (
                    dumped["usage"]["input_tokens_details"]["cache_write_tokens"] == 3
                )

                followup = await sdk.responses.create(
                    model="claude-default",
                    input=[
                        res.output[0].model_dump(),
                        {
                            "role": "assistant",
                            "phase": "commentary",
                            "content": [{"type": "output_text", "text": "note"}],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": "no-usage"}],
                        },
                    ],
                    store=False,
                )
                assert followup.status == "completed"
                assert followup.usage is None
                assert '"phase": "commentary"' in followup.output_text

                stream = await sdk.responses.create(
                    model="claude-default",
                    input="stream-hi",
                    store=False,
                    stream=True,
                )
                events = [e async for e in stream]
                assert events[0].type == "response.created"
                assert events[-1].type == "response.completed"
                assert events[-1].response.usage.input_tokens == 12

                for bad_headers, bad_body in (
                    ({}, {"store": True}),
                    ({}, {"previous_response_id": "resp_1"}),
                    (
                        {},
                        {
                            "tools": [
                                {"type": "function", "name": "f", "parameters": {}}
                            ]
                        },
                    ),
                    ({"X-Conversation-ID": "chat-1"}, {}),
                ):
                    r = await http.post(
                        "/v1/responses",
                        headers={"Authorization": "Bearer " + KEY, **bad_headers},
                        json={"model": "claude-default", "input": "x", **bad_body},
                    )
                    assert r.status_code == 400, r.text

                failed_stream = await http.post(
                    "/v1/responses",
                    headers={"Authorization": "Bearer " + KEY},
                    json={
                        "model": "claude-default",
                        "input": "trigger-error",
                        "stream": True,
                    },
                )
                assert failed_stream.status_code == 200
                assert "event: response.failed" in failed_stream.text
                assert "event: response.completed" not in failed_stream.text
            finally:
                await sdk.close()
    finally:
        await pool.close()


async def test_claude_responses_guard_hooks():
    runtime = Runtime(SessionPool(Config(api_key=KEY), RichAgent))
    old_runtime = claude_rpc_provider.runtime
    claude_rpc_provider.runtime = runtime
    try:
        data = {
            "model": "claude_rpc/claude-default",
            "input": "check",
            "instructions": "rule",
            "tools": [],
            "proxy_server_request": {
                "body": {
                    "model": "claude-default",
                    "input": "check",
                    "instructions": "rule",
                    "tools": [],
                }
            },
        }
        out = await claude_rpc_responses_guard.async_pre_call_hook(
            {}, None, data, "aresponses"
        )
        assert out["store"] is False
        ctx = responses_request_context.get()
        assert ctx is not None
        resp = await claude_rpc_provider.acompletion(
            "claude_rpc/claude-default",
            [
                {"role": "system", "content": "rule"},
                {"role": "user", "content": "check"},
            ],
        )
        assert resp.usage.prompt_tokens == 12
        assert ctx.usage_known is True
        synthetic = {
            "object": "response",
            "usage": {
                "input_tokens": 12,
                "output_tokens": 4,
                "total_tokens": 16,
                "input_tokens_details": {},
                "output_tokens_details": {},
            },
        }
        await claude_rpc_responses_guard.async_post_call_success_hook({}, {}, synthetic)
        assert synthetic["usage"]["input_tokens_details"]["cache_write_tokens"] == 3
        with pytest.raises(ProviderError):
            await claude_rpc_responses_guard.async_pre_call_hook(
                {},
                None,
                {
                    "model": "claude_rpc/claude-default",
                    "input": "x",
                    "store": True,
                    "proxy_server_request": {
                        "body": {"model": "claude-default", "input": "x", "store": True}
                    },
                },
                "aresponses",
            )
    finally:
        responses_request_context.set(None)
        claude_rpc_provider.runtime = old_runtime
        runtime.close()


async def test_claude_max_tokens_propagates_to_chat_and_responses():
    class MaxTokensDriver(ClaudeDriver):
        async def connect(self):
            async def query(prompt):
                pass

            async def receive():
                yield AssistantMessage(
                    content=[TextBlock("partial cut")],
                    model="fixture",
                    message_id="step-max",
                    stop_reason="max_tokens",
                    usage={
                        "input_tokens": 4,
                        "output_tokens": 8,
                        "cache_creation_input_tokens": 2,
                    },
                )
                yield ResultMessage(
                    subtype="success",
                    duration_ms=1,
                    duration_api_ms=1,
                    is_error=False,
                    num_turns=1,
                    session_id="s-max",
                    stop_reason="max_tokens",
                    usage={"input_tokens": 4, "output_tokens": 8},
                )

            self.client = SimpleNamespace(
                query=query,
                receive_response=receive,
                disconnect=lambda: __import__("asyncio").sleep(0),
            )

    cfg = Config(api_key=KEY, timeout=2.0)
    pool = SessionPool(cfg, lambda m, s: MaxTokensDriver(cfg, m, s))
    app = App(pool)
    app.ready = True
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as http:
            sdk = AsyncOpenAI(
                api_key=KEY,
                base_url="http://test/v1",
                http_client=http,
                max_retries=0,
            )
            try:
                chat = await sdk.chat.completions.create(
                    model="claude-default",
                    messages=[{"role": "user", "content": "hi"}],
                )
                assert chat.choices[0].finish_reason == "length"
                assert chat.choices[0].message.content == "partial cut"

                chat_stream = await sdk.chat.completions.create(
                    model="claude-default",
                    messages=[{"role": "user", "content": "hi"}],
                    stream=True,
                )
                chat_chunks = [c async for c in chat_stream]
                assert [
                    c.choices[0].finish_reason
                    for c in chat_chunks
                    if c.choices and c.choices[0].finish_reason
                ] == ["length"]

                res = await sdk.responses.create(
                    model="claude-default",
                    input="hi",
                    store=False,
                )
                assert res.status == "incomplete"
                assert res.incomplete_details.reason == "max_output_tokens"
                assert res.output[0].status == "incomplete"
                assert res.output_text == "partial cut"

                res_stream = await sdk.responses.create(
                    model="claude-default",
                    input="hi",
                    store=False,
                    stream=True,
                )
                events = [e async for e in res_stream]
                assert events[-1].type == "response.incomplete"
                assert events[-1].response.status == "incomplete"
                assert events[-1].response.incomplete_details.reason == "max_output_tokens"
                assert events[-1].response.output[0].status == "incomplete"
            finally:
                await sdk.close()
    finally:
        await pool.close()

    rt = Runtime(SessionPool(cfg, lambda m, s: MaxTokensDriver(cfg, m, s)))
    llm = ClaudeLLM(rt)
    try:
        comp = await llm.acompletion(
            "claude-default", [{"role": "user", "content": "hi"}]
        )
        assert comp.choices[0].finish_reason == "length"
        chunks = [
            c
            async for c in llm.astreaming(
                "claude-default", [{"role": "user", "content": "hi"}]
            )
        ]
        assert chunks[-1]["finish_reason"] == "length"
    finally:
        await __import__("asyncio").to_thread(rt.close)
