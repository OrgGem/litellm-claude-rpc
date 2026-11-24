import asyncio
import hashlib
import json
import shutil
import time
from pathlib import Path

import httpx
import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock
from claude_agent_sdk._internal.sdk_mcp_bridge import SdkMcpBridge
from jsonschema import Draft202012Validator

from litellm_claude_rpc import native_v2
from litellm_claude_rpc.app import App
from litellm_claude_rpc.contracts import Config, GatewayError


def context():
    now = time.monotonic()
    return {
        "key_scope_id": "key-1",
        "key_epoch": "epoch-1",
        "environment": "test",
        "request_id": "req-1",
        "chain_id": "chain-1",
        "locator": "1" * 64,
        "session_scope": "2" * 64,
        "profile_id": "profile",
        "profile_revision": "rev-1",
        "limits": {
            "inference_seconds": 10,
            "native_deadline": now + 10,
            "chain_deadline": now + 10,
            "tool_wait_seconds": 5,
            "cleanup_seconds": 2,
            "max_handoffs": 2,
        },
    }


def profile():
    return {
        "profile_id": "profile",
        "profile_revision": "rev-1",
        "transport": "stdio",
        "models": {"claude-default": "fixture-model"},
    }


def turn_input(tools=True):
    return {
        "model_alias": "claude-default",
        "resolved_model": "fixture-model",
        "messages": [{"role": "user", "content": "hello"}],
        "history_digest": "3" * 64,
        "toolset_digest": "4" * 64,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        if tools
        else [],
        "round_policy": {"tool_choice": "auto", "parallel_tool_calls": False},
        "reasoning_output": "off",
        "reasoning_effort": None,
    }


def results(ref, locator="1" * 64):
    return {
        "handoff_id": "handoff-1",
        "request_id": "req-2",
        "locator": locator,
        "content_digest": "5" * 64,
        "results": [
            {"public_id": "call-public", "native_ref": ref, "content": "tool result"}
        ],
        "round_policy": {"tool_choice": "auto", "parallel_tool_calls": False},
    }


def result(**kwargs):
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="native-session",
        **kwargs,
    )


class FakeWire:
    def __init__(self, options, config):
        self.options, self.closed, self.killed = options, False, False

    async def close(self):
        self.closed = True

    def kill(self):
        self.killed = True


class FakeSDK:
    instances = []

    def __init__(self, options, transport):
        self.options, self.transport = options, transport
        self.bridge = None
        self.call = None
        self.reply = None
        self.closed = False
        self.prompts = []
        self.__class__.instances.append(self)

    async def connect(self):
        if isinstance(self.options.mcp_servers, dict):
            server = self.options.mcp_servers["agent_rpc"]["instance"]
            self.bridge = SdkMcpBridge("agent_rpc", server)
            init = await self.bridge.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "fixture", "version": "1"},
                    },
                }
            )
            assert "result" in init
            await self.bridge.handle(
                {"jsonrpc": "2.0", "method": "notifications/initialized"}
            )

    async def query(self, prompt):
        self.prompts.append(prompt)
        if self.bridge:
            self.call = asyncio.create_task(
                self.bridge.handle(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {"name": "echo", "arguments": {"text": "hello"}},
                    }
                )
            )

    async def receive_response(self):
        if self.bridge:
            # The observational native ID deliberately differs from the generated callback ref.
            yield AssistantMessage(
                content=[
                    ToolUseBlock(
                        id="sdk-id",
                        name="mcp__agent_rpc__echo",
                        input={"text": "hello"},
                    )
                ],
                model="fixture",
                stop_reason="tool_use",
                message_id="msg-1",
                usage={
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "cache_read_input_tokens": 1,
                    "cache_creation_input_tokens": 0,
                },
            )
            self.reply = await self.call
        yield AssistantMessage(
            content=[TextBlock("done")],
            model="fixture",
            stop_reason="end_turn",
            message_id="msg-2",
            usage={"input_tokens": 4, "output_tokens": 1},
        )
        yield result(stop_reason="end_turn")

    async def disconnect(self):
        if self.bridge:
            await self.bridge.aclose()
        self.closed = True


@pytest.fixture
def candidate(monkeypatch):
    async def allow_candidate(self):
        return None

    FakeSDK.instances = []
    monkeypatch.setattr(native_v2.ClaudeNativeDriver, "_ensure_ready", allow_candidate)
    # Synthetic proof exercises lifecycle only; name/argument equality is not
    # native evidence. Production's verifier is permanently fail-closed.
    monkeypatch.setattr(
        native_v2.ClaudeNativeDriver,
        "_verified_native_batch_boundary",
        lambda self, turn: True,
    )
    monkeypatch.setattr(native_v2, "SDKWire", FakeWire)
    monkeypatch.setattr(native_v2, "ClaudeSDKClient", FakeSDK)
    return native_v2.ClaudeNativeDriver(Config())


async def batch(driver, turn):
    iterator = driver.events(turn)
    seen = []
    while True:
        event = await asyncio.wait_for(anext(iterator), 5)
        seen.append(event)
        if event["type"] == "tool_batch_ready":
            return iterator, seen


def test_frozen_contract_snapshot_and_all_fixtures():
    path = Path(native_v2.__file__).parent / "contract_v0_1"
    manifest = json.loads((path / "manifest.json").read_bytes())
    assert manifest["contract_hash"] == native_v2.CONTRACT_HASH
    assert all(
        hashlib.sha256((path / name).read_bytes()).hexdigest() == digest
        for name, digest in manifest["files"].items()
    )
    validator = Draft202012Validator(native_v2._contract())
    for case in json.loads((path / "fixtures.json").read_bytes()):
        assert validator.is_valid(case["document"]) == case["valid"], case["name"]


async def test_public_driver_gate_and_offline_manifest():
    driver = native_v2.ClaudeNativeDriver()
    manifest = await driver.capabilities({"profile_id": "profile"}, "fixture-model")
    native_v2._validate(native_v2._contract(), "CapabilityManifest", manifest)
    assert all(
        not item["live_verified"] and item["state"] == "blocked"
        for item in manifest["features"].values()
    )
    with pytest.raises(GatewayError, match="claude_live_acceptance_pending"):
        await driver.open_session(context(), profile())


async def test_native_batch_gate_is_closed_even_if_observations_match(
    candidate, monkeypatch
):
    monkeypatch.setattr(
        native_v2.ClaudeNativeDriver,
        "_verified_native_batch_boundary",
        lambda self, turn: False,
    )
    session = await candidate.open_session(context(), profile())
    turn = await candidate.begin_turn(session, turn_input())
    events = [event async for event in candidate.events(turn)]
    assert not any(event["type"] == "tool_batch_ready" for event in events)
    assert events[-1]["payload"]["code"] == "tool_batch_boundary_unverified"
    await candidate.close_session(session, time.monotonic() + 5)


async def test_mismatched_observation_rejected_before_handoff(candidate, monkeypatch):
    async def mismatch(self):
        yield AssistantMessage(
            content=[
                ToolUseBlock(
                    id="sdk-id", name="mcp__agent_rpc__echo", input={"text": "OTHER"}
                )
            ],
            model="fixture",
            stop_reason="tool_use",
        )
        await self.call

    monkeypatch.setattr(FakeSDK, "receive_response", mismatch)
    session = await candidate.open_session(context(), profile())
    turn = await candidate.begin_turn(session, turn_input())
    events = [event async for event in candidate.events(turn)]
    assert not any(event["type"] == "tool_batch_ready" for event in events)
    assert events[-1]["payload"]["code"] == "unexpected_tool_activity"
    await candidate.close_session(session, time.monotonic() + 5)


async def test_mcp_callback_handoff_close_resume_and_usage(candidate):
    session = await candidate.open_session(context(), profile())
    turn = await candidate.begin_turn(session, turn_input())
    iterator, events = await batch(candidate, turn)
    assert [e["type"] for e in events] == [
        "usage",
        "tool_call_start",
        "tool_call_ready",
        "tool_batch_ready",
    ]
    ready, boundary = events[2]["payload"], events[3]["payload"]
    ref = ready["native_ref"]
    assert ref != "sdk-id" and boundary["native_refs"] == [ref]
    assert ready["arguments"] == {"text": "hello"}
    for event in events:
        native_v2._validate(native_v2._contract(), "AdapterEvent", event)
    assert not FakeSDK.instances[0].call.done()
    await iterator.aclose()
    assert not FakeSDK.instances[0].call.done(), (
        "subscriber close must not cancel native turn"
    )
    await candidate.submit_tool_results(turn, results(ref))
    with pytest.raises(GatewayError, match="tool_result_conflict"):
        await candidate.submit_tool_results(turn, results(ref))
    remaining = [event async for event in candidate.events(turn)]
    assert [e["type"] for e in remaining] == [
        "usage",
        "text_delta",
        "native_turn_ended",
    ]
    assert (
        remaining[0]["payload"]["input_tokens"] is None
    )  # absent cache counters are unknown
    assert remaining[-1]["payload"] == {
        "kind": "native_terminal",
        "finish_reason": "stop",
        "usage_completeness": "partial",
    }
    assert FakeSDK.instances[0].reply["result"]["content"][0]["text"] == "tool result"
    with pytest.raises(GatewayError, match="tool_history_unavailable"):
        await candidate.begin_turn(session, turn_input())
    await candidate.close_session(session, time.monotonic() + 5)
    assert FakeSDK.instances[0].closed


async def test_candidate_config_only_whitelists_declared_sdk_bridge(candidate):
    session = await candidate.open_session(context(), profile())
    turn = await candidate.begin_turn(session, turn_input())
    iterator, _ = await batch(candidate, turn)
    options = FakeSDK.instances[0].options
    assert options.tools == []
    assert options.skills == [] and options.plugins == [] and options.hooks == {}
    assert options.allowed_tools == ["mcp__agent_rpc__echo"]
    assert options.strict_mcp_config is True and options.setting_sources == []
    assert set(options.mcp_servers) == {"agent_rpc"}
    assert options.extra_args == {"no-session-persistence": None}
    assert (await options.can_use_tool("Bash", {}, None)).behavior == "deny"
    await iterator.aclose()
    await candidate.cancel(turn, "test", time.monotonic() + 5)


async def test_two_identical_callbacks_stay_isolated_by_session(candidate):
    first = await candidate.open_session(context(), profile())
    second_context = context()
    second_context["locator"] = "9" * 64
    second = await candidate.open_session(second_context, profile())
    t1, t2 = await asyncio.gather(
        candidate.begin_turn(first, turn_input()),
        candidate.begin_turn(second, turn_input()),
    )
    i1, e1 = await batch(candidate, t1)
    i2, e2 = await batch(candidate, t2)
    ref1, ref2 = (
        e1[-1]["payload"]["native_refs"][0],
        e2[-1]["payload"]["native_refs"][0],
    )
    assert ref1 != ref2
    with pytest.raises(GatewayError, match="unknown_tool_session"):
        await candidate.submit_tool_results(t1, results(ref1, "9" * 64))
    with pytest.raises(GatewayError, match="tool_result_conflict"):
        await candidate.submit_tool_results(t1, results(ref2))
    await candidate.submit_tool_results(t1, results(ref1))
    await candidate.submit_tool_results(t2, results(ref2, "9" * 64))
    await i1.aclose()
    await i2.aclose()
    await asyncio.gather(
        candidate.close_session(first, time.monotonic() + 5),
        candidate.close_session(second, time.monotonic() + 5),
    )


async def test_cancel_pending_fences_result_and_closes_sdk(candidate):
    session = await candidate.open_session(context(), profile())
    turn = await candidate.begin_turn(session, turn_input())
    iterator, events = await batch(candidate, turn)
    ref = events[-1]["payload"]["native_refs"][0]
    await iterator.aclose()
    await candidate.cancel(turn, "client_cancelled", time.monotonic() + 5)
    with pytest.raises(GatewayError, match="turn_closed"):
        await candidate.submit_tool_results(turn, results(ref))
    assert FakeSDK.instances[0].closed
    assert session.owner.done()


async def test_reuse_completed_text_turn_only_with_exact_prefix(candidate):
    session = await candidate.open_session(context(), profile())
    request = turn_input(False)
    turn = await candidate.begin_turn(session, request)
    events = [event async for event in candidate.events(turn)]
    assert events[-1]["type"] == "native_turn_ended"
    later = turn_input(False)
    later["messages"] += [
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "/literal"},
    ]
    second = await candidate.begin_turn(session, later)
    assert [e async for e in candidate.events(second)][-1][
        "type"
    ] == "native_turn_ended"
    assert FakeSDK.instances[0].prompts[1].startswith("Literal user message")
    changed = turn_input(False)
    changed["messages"] += [
        {"role": "assistant", "content": "wrong"},
        {"role": "user", "content": "next"},
    ]
    with pytest.raises(GatewayError, match="history_mismatch"):
        await candidate.begin_turn(session, changed)
    await candidate.close_session(session, time.monotonic() + 5)


async def test_stale_cancel_does_not_close_new_turn(candidate, monkeypatch):
    session = await candidate.open_session(context(), profile())
    first_input = turn_input(False)
    first = await candidate.begin_turn(session, first_input)
    assert [event async for event in candidate.events(first)][-1][
        "type"
    ] == "native_turn_ended"
    entered = asyncio.Event()

    async def waiting(self):
        entered.set()
        await asyncio.Event().wait()
        if False:
            yield None

    monkeypatch.setattr(FakeSDK, "receive_response", waiting)
    next_input = turn_input(False)
    next_input["messages"] += [
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "next"},
    ]
    second = await candidate.begin_turn(session, next_input)
    await asyncio.wait_for(entered.wait(), 3)
    await candidate.cancel(first, "late", time.monotonic() + 5)
    assert not session.closed and session.active is second and not second.finished
    await candidate.cancel(second, "current", time.monotonic() + 5)
    assert session.closed


async def test_unsupported_continuation_policy_does_not_consume_future(candidate):
    session = await candidate.open_session(context(), profile())
    turn = await candidate.begin_turn(session, turn_input())
    iterator, events = await batch(candidate, turn)
    ref = events[-1]["payload"]["native_refs"][0]
    bad = results(ref)
    bad["round_policy"]["tool_choice"] = "required"
    with pytest.raises(GatewayError, match="unsupported_round_policy"):
        await candidate.submit_tool_results(turn, bad)
    assert not FakeSDK.instances[0].call.done()
    await candidate.submit_tool_results(turn, results(ref))
    await iterator.aclose()
    remaining = [event async for event in candidate.events(turn)]
    assert remaining[-1]["type"] == "native_turn_ended"
    await candidate.close_session(session, time.monotonic() + 5)


async def test_early_missing_terminal_is_error_not_stop(candidate, monkeypatch):
    async def incomplete(self):
        yield AssistantMessage(
            content=[TextBlock("partial")],
            model="fixture",
            message_id="id",
            usage={"input_tokens": 7, "output_tokens": 2},
        )

    monkeypatch.setattr(FakeSDK, "receive_response", incomplete)
    session = await candidate.open_session(context(), profile())
    turn = await candidate.begin_turn(session, turn_input(False))
    events = [event async for event in candidate.events(turn)]
    assert [e["type"] for e in events] == ["usage", "text_delta", "backend_error"]
    assert events[0]["payload"]["output_tokens"] == 2
    assert events[0]["payload"]["completeness"] == "partial"
    assert events[-1]["payload"]["code"] == "missing_terminal_result"
    assert [event async for event in candidate.events(turn)] == []
    await candidate.close_session(session, time.monotonic() + 5)


async def test_missing_native_stop_reason_is_not_synthesized(candidate, monkeypatch):
    async def incomplete(self):
        yield AssistantMessage(content=[TextBlock("partial")], model="fixture")
        yield result()  # successful SDK result with no stop_reason is ambiguous

    monkeypatch.setattr(FakeSDK, "receive_response", incomplete)
    session = await candidate.open_session(context(), profile())
    turn = await candidate.begin_turn(session, turn_input(False))
    events = [event async for event in candidate.events(turn)]
    assert events[-1]["type"] == "backend_error"
    assert events[-1]["payload"]["code"] == "unsupported_stop_reason"
    await candidate.close_session(session, time.monotonic() + 5)


async def test_second_parallel_callback_fails_closed(candidate, monkeypatch):
    async def query(self, prompt):
        self.prompts.append(prompt)
        self.call = asyncio.create_task(
            self.bridge.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "echo", "arguments": {"text": "hello"}},
                }
            )
        )
        self.other = asyncio.create_task(
            self.bridge.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "echo", "arguments": {"text": "hello"}},
                }
            )
        )

    async def wait_for_callbacks(self):
        # The fake peer has not emitted a batch boundary; do not publish either call.
        await asyncio.gather(self.call, self.other)
        if False:
            yield None

    monkeypatch.setattr(FakeSDK, "query", query)
    monkeypatch.setattr(FakeSDK, "receive_response", wait_for_callbacks)
    session = await candidate.open_session(context(), profile())
    turn = await candidate.begin_turn(session, turn_input())
    events = [event async for event in candidate.events(turn)]
    assert [e["type"] for e in events] == ["backend_error"]
    assert events[0]["payload"]["code"] == "unexpected_tool_activity"
    await asyncio.gather(
        FakeSDK.instances[0].call, FakeSDK.instances[0].other, return_exceptions=True
    )
    await candidate.close_session(session, time.monotonic() + 5)


async def test_connection_cancel_settles_waiter(candidate, monkeypatch):
    entered = asyncio.Event()

    async def blocked_connect(self):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(FakeSDK, "connect", blocked_connect)
    session = await candidate.open_session(context(), profile())
    started = asyncio.create_task(candidate.begin_turn(session, turn_input(False)))
    await asyncio.wait_for(entered.wait(), 3)
    await candidate.close_session(session, time.monotonic() + 5)
    with pytest.raises(GatewayError, match="backend_startup_failed"):
        await asyncio.wait_for(started, 3)
    assert session.owner.done() and FakeSDK.instances[0].closed


async def test_startup_respects_native_deadline(candidate, monkeypatch):
    async def blocked_connect(self):
        await asyncio.Event().wait()

    monkeypatch.setattr(FakeSDK, "connect", blocked_connect)
    request_context = context()
    request_context["limits"]["native_deadline"] = time.monotonic() + 0.05
    session = await candidate.open_session(request_context, profile())
    with pytest.raises(GatewayError, match="deadline_exceeded") as error:
        await asyncio.wait_for(candidate.begin_turn(session, turn_input(False)), 3)
    assert error.value.status == 504
    await candidate.close_session(session, time.monotonic() + 5)
    assert session.owner.done() and FakeSDK.instances[0].closed


async def test_close_after_startup_failure_does_not_cancel_cleanup(
    candidate, monkeypatch
):
    entered, release = asyncio.Event(), asyncio.Event()

    async def failed_connect(self):
        raise RuntimeError("fixture startup failure")

    async def slow_disconnect(self):
        entered.set()
        await release.wait()
        self.closed = True

    monkeypatch.setattr(FakeSDK, "connect", failed_connect)
    monkeypatch.setattr(FakeSDK, "disconnect", slow_disconnect)
    session = await candidate.open_session(context(), profile())
    with pytest.raises(GatewayError, match="backend_startup_failed"):
        await candidate.begin_turn(session, turn_input(False))
    await asyncio.wait_for(entered.wait(), 3)
    closing = asyncio.create_task(
        candidate.close_session(session, time.monotonic() + 5)
    )
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    await asyncio.wait_for(closing, 3)
    assert session.owner.done() and FakeSDK.instances[0].closed


async def test_native_timeout_is_sanitized_error(candidate, monkeypatch):
    async def waiting(self):
        await asyncio.Event().wait()
        if False:
            yield None

    monkeypatch.setattr(FakeSDK, "receive_response", waiting)
    request_context = context()
    request_context["limits"]["inference_seconds"] = 0.05
    session = await candidate.open_session(request_context, profile())
    turn = await candidate.begin_turn(session, turn_input(False))
    events = [event async for event in candidate.events(turn)]
    assert events == [
        {
            "type": "backend_error",
            "invocation_id": turn.invocation_id,
            "sequence": 0,
            "payload": {"code": "deadline_exceeded", "status": 504},
        }
    ]
    await candidate.close_session(session, time.monotonic() + 5)


async def test_standalone_v1_host_rejects_tools_without_spawning():
    class Pool:
        config = Config(api_key="sk-fixture-key-123456789012345")

        def turn(self, request):
            raise AssertionError("unsupported tools must not reach the backend")

    app = App(Pool())
    app.ready = True
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer " + Pool.config.api_key},
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-default",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": turn_input()["tools"],
            },
        )
    assert response.status_code == 400


@pytest.mark.parametrize(
    "mutation",
    [
        lambda request: request["round_policy"].update(tool_choice="required"),
        lambda request: request["round_policy"].update(parallel_tool_calls=True),
        lambda request: request.update(reasoning_output="summary"),
        lambda request: request["tools"][0]["function"]["parameters"].update(
            {"$ref": "https://invalid"}
        ),
        lambda request: request["tools"][0]["function"].update(strict=True),
    ],
)
async def test_unsupported_features_fail_before_sdk_creation(candidate, mutation):
    session = await candidate.open_session(context(), profile())
    request = turn_input()
    mutation(request)
    with pytest.raises(GatewayError):
        await candidate.begin_turn(session, request)
    assert not FakeSDK.instances
    await candidate.close_session(session, time.monotonic() + 5)


async def test_close_before_owner_first_step_settles_begin_waiter(candidate):
    session = await candidate.open_session(context(), profile())
    begin = asyncio.create_task(candidate.begin_turn(session, turn_input(False)))
    await asyncio.sleep(0)
    assert session.owner is not None and not session.owner_started
    await candidate.close_session(session, time.monotonic() + 5)
    with pytest.raises(GatewayError, match="backend_startup_failed"):
        await asyncio.wait_for(begin, 2)
    assert session.owner.done() and not FakeSDK.instances


async def test_cleanup_failure_is_reported_to_host(candidate, monkeypatch):
    async def failed_close(self):
        raise OSError("secret detail must not reach host")

    monkeypatch.setattr(FakeWire, "close", failed_close)
    session = await candidate.open_session(context(), profile())
    turn = await candidate.begin_turn(session, turn_input(False))
    assert [event async for event in candidate.events(turn)][-1][
        "type"
    ] == "native_turn_ended"
    workspace = Path(session.temp.name)
    try:
        with pytest.raises(GatewayError, match="cleanup_failed") as error:
            await candidate.close_session(session, time.monotonic() + 5)
        assert "secret" not in str(error.value)
        assert session.owner.done() and workspace.exists()
        # Do not let a finalizer remove a still-required isolation profile if a
        # caller drops the quarantined session after failed cleanup.
        session.temp = None
        assert workspace.exists()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


async def test_backpressure_preserves_error_after_queued_content(
    candidate, monkeypatch
):
    async def flood(self):
        for _ in range(5):
            yield AssistantMessage(content=[TextBlock("x")], model="fixture")

    monkeypatch.setattr(FakeSDK, "receive_response", flood)
    candidate.config = Config(queue_size=1)
    session = await candidate.open_session(context(), profile())
    turn = await candidate.begin_turn(session, turn_input(False))
    await asyncio.wait_for(asyncio.shield(session.owner), 5)
    events = [event async for event in candidate.events(turn)]
    assert [event["type"] for event in events] == ["text_delta", "backend_error"]
    assert events[-1]["payload"]["code"] == "backpressure_exceeded"
    assert [event["sequence"] for event in events] == [0, 1]
    await candidate.close_session(session, time.monotonic() + 5)


async def test_terminal_releases_session_even_when_queue_is_full(
    candidate, monkeypatch
):
    release = asyncio.Event()

    async def textless_response(self):
        yield AssistantMessage(content=[], model="fixture")
        await release.wait()
        yield result(stop_reason="end_turn")

    monkeypatch.setattr(FakeSDK, "receive_response", textless_response)
    candidate.config = Config(queue_size=1)
    session = await candidate.open_session(context(), profile())
    turn = await candidate.begin_turn(session, turn_input(False))
    stream = candidate.events(turn)
    try:
        asyncio.get_running_loop().call_soon(release.set)
        async with asyncio.timeout(5):
            assert (await anext(stream))["type"] == "native_turn_ended"
            assert session.active is None
    finally:
        await stream.aclose()
        await candidate.close_session(session, time.monotonic() + 5)
