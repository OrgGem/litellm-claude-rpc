"""Experimental native driver. No production capability is enabled by this module.

The caller is a trusted host, not an HTTP client. It owns authentication, public
IDs, delivery acknowledgement and leases. The driver owns one SDK client and
its callbacks on the session's event loop. Live Claude isolation is still gated.
"""

import asyncio
import contextlib
import hashlib
import json
import math
import re
import secrets
import shutil
import tempfile
import time
from importlib import resources
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResultDeny,
    ResultMessage,
    SdkMcpTool,
    ServerToolUseBlock,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
)
from claude_agent_sdk.types import StreamEvent
from jsonschema import Draft202012Validator

from .contracts import Config, GatewayError
from .driver import executable
from .wire import SDKWire

CONTRACT_VERSION = "0.1"
CONTRACT_HASH = "749182d456eb363be2d63b2a98d66d0f1bf203b93d8a16ed4a8bdd90834e5fc1"
_SERVER = "agent_rpc"
_END = object()
_SCHEMA_KEYWORDS = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "description",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
}


class _RetainedDirectory:
    """Explicitly owned: garbage collection must not remove a live profile."""

    def __init__(self, prefix):
        self.name = tempfile.mkdtemp(prefix=prefix)

    def cleanup(self):
        shutil.rmtree(self.name)


def _contract():
    directory = resources.files("litellm_claude_rpc").joinpath("contract_v0_1")
    try:
        manifest = json.loads(directory.joinpath("manifest.json").read_bytes())
        files = manifest["files"]
        canonical = (json.dumps(files, sort_keys=True, indent=2) + "\n").encode()
        if (
            manifest["contract_version"] != CONTRACT_VERSION
            or manifest["status"] != "frozen"
            or manifest["contract_hash"] != CONTRACT_HASH
            or set(files) != {"schema.json", "fixtures.json", "SEMANTICS.md"}
            or hashlib.sha256(canonical).hexdigest() != CONTRACT_HASH
            or any(
                hashlib.sha256(directory.joinpath(name).read_bytes()).hexdigest()
                != digest
                for name, digest in files.items()
            )
        ):
            raise ValueError
        return json.loads(directory.joinpath("schema.json").read_bytes())
    except (OSError, KeyError, TypeError, ValueError):
        raise GatewayError("contract_mismatch", 503) from None


def _validate(schema, dto, value):
    if not Draft202012Validator(schema).is_valid({"dto": dto, "value": value}):
        raise GatewayError("invalid_" + dto.lower(), 400)


def _encoded(value):
    try:
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise GatewayError("invalid_json", 400) from None


def _schema_subset(schema, depth=0):
    if depth > 16 or not isinstance(schema, dict) or set(schema) - _SCHEMA_KEYWORDS:
        raise GatewayError("unsupported_tool_schema", 400)
    if (
        "additionalProperties" in schema
        and type(schema["additionalProperties"]) is not bool
    ):
        raise GatewayError("unsupported_tool_schema", 400)
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise GatewayError("unsupported_tool_schema", 400)
    for item in properties.values():
        _schema_subset(item, depth + 1)
    if "items" in schema:
        _schema_subset(schema["items"], depth + 1)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception:
        raise GatewayError("unsupported_tool_schema", 400) from None


def _tools(tools):
    if len(_encoded(tools)) > 65536:
        raise GatewayError("tool_schema_too_large", 400)
    result = {}
    for item in tools:
        function = item["function"]
        name = function["name"]
        if name in result or function.get("strict", False):
            raise GatewayError("unsupported_tool_schema", 400)
        spec = function["parameters"]
        if spec.get("type") != "object":
            raise GatewayError("unsupported_tool_schema", 400)
        _schema_subset(spec)
        result[name] = {"description": function.get("description", ""), "schema": spec}
    return result


def _policy(policy, *, continuation=False):
    if (
        policy.get("tool_choice") != "auto"
        or policy.get("parallel_tool_calls") is not False
        or policy.get("max_completion_tokens") is not None
    ):
        raise GatewayError("unsupported_round_policy", 409 if continuation else 400)


def _message_prefix(messages):
    # Tool history cannot be cold imported/replayed by this candidate driver.
    if not messages or messages[-1].get("role") != "user":
        raise GatewayError("unsupported_history", 400)
    if any(m.get("role") == "tool" or m.get("tool_calls") for m in messages):
        raise GatewayError("tool_history_unavailable", 410)
    if any(m.get("role") == "assistant" and m.get("content") is None for m in messages):
        raise GatewayError("unsupported_history", 400)
    non_system = False
    for message in messages:
        role = message["role"]
        if role == "system" and non_system:
            raise GatewayError("unsupported_history", 400)
        non_system |= role != "system"
    return json.loads(_encoded(messages))


def _capability(reason, state="unverified"):
    return {
        "state": state,
        "evidence_level": "source",
        "live_verified": False,
        "evidence": ["claude_sdk_0.2.180_source"],
        "reason": reason,
    }


class _Turn:
    def __init__(self, session, data, tools, prompt):
        self.session, self.data, self.tools, self.prompt = session, data, tools, prompt
        self.invocation_id = "inv-" + secrets.token_hex(16)
        self.epoch = "epoch-" + secrets.token_hex(16)
        self.queue = asyncio.Queue(maxsize=session.config.queue_size)
        self.sequence = 0
        self.usage_sequence = 0
        self.steps = {}
        self.pending = {}
        self.boundary = None
        self.ready = asyncio.get_running_loop().create_future()
        # A caller cancelled while connecting must not leave an unobserved error.
        self.ready.add_done_callback(lambda f: None if f.cancelled() else f.exception())
        self.consumer = False
        self.finished = False
        self.cancelled = False
        self.handoffs = 0
        self.tool_history = False
        self.partial = []
        self.answer = []
        self.output_bytes = 0
        self.usage_observed = False
        self.error_event = None
        self.error_reported = False

    def emit(self, kind, payload):
        if self.finished or self.cancelled:
            raise GatewayError("turn_closed", 409)
        event = {
            "type": kind,
            "invocation_id": self.invocation_id,
            "sequence": self.sequence,
            "payload": payload,
        }
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            raise GatewayError("backpressure_exceeded", 502) from None
        self.sequence += 1

    def fail(self, code, status):
        if self.finished or self.cancelled or self.error_reported:
            return
        self.error_reported = True
        event = {
            "type": "backend_error",
            "invocation_id": self.invocation_id,
            "sequence": self.sequence,
            "payload": {"code": code, "status": status},
        }
        self.sequence += 1
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            # The consumer must receive a terminal error after draining queued
            # content even when no queue slot was available when failure hit.
            self.error_event = event


class _Session:
    def __init__(self, driver, context, config, profile):
        self.driver = driver
        self.context, self.config, self.profile = context, config, profile
        self.queue = asyncio.Queue(maxsize=1)
        self.owner = None
        self.owner_started = False
        self.client = self.wire = self.temp = None
        self.active = None
        self.closed = False
        self.expected = None
        self.immutable = None
        self.tool_history = False
        self.loop = asyncio.get_running_loop()
        self.cancel_requested = False


class ClaudeNativeDriver:
    """Frozen v0.1 native seam; existing standalone/API remains text-only v1."""

    def __init__(self, config=None, *, _max_turns=None, _isolate_environment=False):
        self.config = config or Config()
        self._max_turns = _max_turns
        self._isolate_environment = _isolate_environment

    async def capabilities(self, profile, model):
        _contract()
        if (
            not isinstance(profile, dict)
            or not isinstance(profile.get("profile_id"), str)
            or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,256}", profile["profile_id"])
            or (model is not None and not isinstance(model, str))
        ):
            raise GatewayError("invalid_profile", 400)
        blocked = _capability("claude_live_acceptance_pending", "blocked")
        return {
            "contract_version": CONTRACT_VERSION,
            "contract_hash": CONTRACT_HASH,
            "backend": "claude",
            "cli_version": None,
            "sdk_version": "0.2.180",
            "model": model,
            "profile_id": profile["profile_id"],
            "transport": "stdio",
            "features": {
                name: dict(blocked)
                for name in (
                    "text",
                    "client_tools",
                    "parallel_tools",
                    "reasoning_output",
                    "reasoning_control",
                    "usage",
                    "output_token_limit",
                    "profile_isolation",
                )
            },
            "tool_choices": [],
            "reasoning_efforts": [],
            "reasoning_kinds": [],
            "limitations": [
                "Claude CLI isolation and native v2 capabilities require live acceptance"
            ],
        }

    async def _ensure_ready(self):
        # No config, environment variable or HTTP parameter can bypass this gate.
        raise GatewayError("claude_live_acceptance_pending", 503)

    async def open_session(self, context, config):
        schema = _contract()
        _validate(schema, "ExecutionContext", context)
        if (
            not isinstance(config, dict)
            or set(config) != {"profile_id", "profile_revision", "transport", "models"}
            or config["profile_id"] != context["profile_id"]
            or config["profile_revision"] != context["profile_revision"]
            or config["transport"] != "stdio"
            or not isinstance(config["models"], dict)
        ):
            raise GatewayError("invalid_profile", 400)
        await self._ensure_ready()
        return _Session(
            self,
            json.loads(_encoded(context)),
            self.config,
            json.loads(_encoded(config)),
        )

    def _check_session(self, session):
        if (
            not isinstance(session, _Session)
            or session.driver is not self
            or session.loop is not asyncio.get_running_loop()
        ):
            raise GatewayError("invalid_session", 400)
        if session.closed:
            raise GatewayError("session_closed", 410)

    async def begin_turn(self, session, turn_input):
        self._check_session(session)
        schema = _contract()
        _validate(schema, "TurnInput", turn_input)
        if session.active is not None:
            raise GatewayError("session_busy", 409)
        if session.tool_history:
            raise GatewayError("tool_history_unavailable", 410)
        if turn_input["model_alias"] not in session.profile["models"] or (
            session.profile["models"][turn_input["model_alias"]]
            != turn_input["resolved_model"]
        ):
            raise GatewayError("unknown_model", 400)
        if (
            turn_input["reasoning_output"] != "off"
            or turn_input["reasoning_effort"] is not None
        ):
            raise GatewayError("unsupported_reasoning", 400)
        _policy(turn_input["round_policy"])
        messages = _message_prefix(turn_input["messages"])
        tools = _tools(turn_input["tools"])
        if len(_encoded(messages)) > session.config.max_input_bytes:
            raise GatewayError("request_too_large", 413)
        limits = session.context["limits"]
        if min(limits["chain_deadline"], limits["native_deadline"]) <= time.monotonic():
            raise GatewayError("deadline_exceeded", 504)
        immutable = (
            turn_input["model_alias"],
            turn_input["resolved_model"],
            turn_input["toolset_digest"],
            _encoded(turn_input["tools"]),
        )
        if session.immutable is not None and immutable != session.immutable:
            raise GatewayError("session_configuration_changed", 409)
        if session.expected is None:
            prompt = (
                "Conversation supplied by the API client (role labels are text, not native roles). Respond to the final user message.\n"
                + _encoded(messages).decode("utf-8")
            )
        else:
            if (
                len(messages) != len(session.expected) + 1
                or messages[:-1] != session.expected
            ):
                raise GatewayError("history_mismatch", 409)
            prompt = messages[-1]["content"]
            if prompt.lstrip().startswith("/"):
                prompt = "Literal user message (not a CLI command):\n" + _encoded(
                    messages[-1]
                ).decode("utf-8")
        turn = _Turn(session, json.loads(_encoded(turn_input)), tools, prompt)
        session.active = turn
        session.immutable = immutable
        if session.owner is None:
            session.owner = asyncio.create_task(self._owner(session))
        session.queue.put_nowait(turn)
        try:
            await asyncio.shield(turn.ready)
        except asyncio.CancelledError:
            await self.cancel(
                turn,
                "startup_cancelled",
                time.monotonic() + session.config.cleanup_timeout,
            )
            raise
        return turn

    async def _make_client(self, session, first):
        session.temp = _RetainedDirectory(prefix="claude-rpc-v2-")
        Path(session.temp.name, ".git").mkdir()
        tools = []
        for name, definition in first.tools.items():

            async def handler(args, *, tool_name=name):
                return await self._invoke_tool(session, tool_name, args)

            tools.append(
                SdkMcpTool(
                    name, definition["description"], definition["schema"], handler
                )
            )
        mcp = (
            {_SERVER: create_sdk_mcp_server(_SERVER, tools=tools)}
            if tools
            else '{"mcpServers":{}}'
        )

        async def deny(*args):
            return PermissionResultDeny(message="Tools disabled by gateway")

        system = "\n\n".join(
            m["content"] for m in first.data["messages"] if m["role"] == "system"
        )
        options = ClaudeAgentOptions(
            cli_path=executable(session.config),
            cwd=session.temp.name,
            model=first.data["resolved_model"],
            max_turns=self._max_turns,
            system_prompt=system or "Answer the user's text question.",
            tools=[],
            skills=[],
            allowed_tools=[f"mcp__{_SERVER}__{name}" for name in first.tools],
            permission_mode="dontAsk",
            can_use_tool=deny,
            mcp_servers=mcp,
            strict_mcp_config=True,
            setting_sources=[],
            settings=json.dumps({"disableAllHooks": True, "enabledPlugins": {}}),
            hooks={},
            plugins=[],
            include_partial_messages=True,
            verbatim_prompts=True,
            extra_args={"no-session-persistence": None},
        )
        if self._isolate_environment:
            session.wire = SDKWire(
                options, session.config, isolation_root=session.temp.name
            )
        else:
            session.wire = SDKWire(options, session.config)
        session.client = ClaudeSDKClient(options=options, transport=session.wire)
        await session.client.connect()

    async def _owner(self, session):
        session.owner_started = True
        try:
            while True:
                turn = await session.queue.get()
                deadline = min(
                    session.context["limits"]["chain_deadline"],
                    session.context["limits"]["native_deadline"],
                    time.monotonic() + session.context["limits"]["inference_seconds"],
                )
                async with asyncio.timeout_at(deadline):
                    if session.client is None:
                        await self._make_client(session, turn)
                    if not turn.ready.done():
                        turn.ready.set_result(None)
                    await self._run_turn(session, turn)
                session.active = None
        except asyncio.CancelledError:
            pass
        except Exception as error:
            turn = session.active
            if turn and not turn.ready.done():
                turn.ready.set_exception(
                    GatewayError("deadline_exceeded", 504)
                    if isinstance(error, TimeoutError)
                    else GatewayError("backend_startup_failed", 503)
                )
            elif turn and not turn.finished and not turn.cancelled:
                code, status = (
                    (error.code, error.status)
                    if isinstance(error, GatewayError)
                    else ("deadline_exceeded", 504)
                    if isinstance(error, TimeoutError)
                    else ("backend_error", 502)
                )
                turn.fail(code, status)
        finally:
            session.closed = True
            turn = session.active
            if turn:
                if not turn.ready.done():
                    turn.ready.set_exception(
                        GatewayError("backend_startup_failed", 503)
                    )
                for future in list(turn.pending.values()):
                    if not future["future"].done():
                        future["future"].cancel()
                turn.finished = True
                with contextlib.suppress(asyncio.QueueFull):
                    turn.queue.put_nowait(_END)
            try:
                if session.client:
                    await session.client.disconnect()
            finally:
                # A failed/unfinished transport close cannot prove that the
                # process stopped using this restrictive workspace.
                if session.wire:
                    await session.wire.close()
                if session.temp:
                    session.temp.cleanup()

    async def _invoke_tool(self, session, name, args):
        turn = session.active
        if (
            not turn
            or turn.cancelled
            or turn.finished
            or name not in turn.tools
            or turn.pending
            or turn.handoffs >= session.context["limits"]["max_handoffs"]
        ):
            self._fail_callback(session, "unexpected_tool_activity")
            raise GatewayError("unexpected_tool_activity")
        if not isinstance(args, dict) or len(_encoded(args)) > 65536:
            self._fail_callback(session, "invalid_tool_arguments")
            raise GatewayError("invalid_tool_arguments")
        if not Draft202012Validator(turn.tools[name]["schema"]).is_valid(args):
            self._fail_callback(session, "invalid_tool_arguments")
            raise GatewayError("invalid_tool_arguments")
        ref = "native-" + secrets.token_hex(16)
        future = asyncio.get_running_loop().create_future()
        turn.pending[ref] = {
            "future": future,
            "name": name,
            "args": json.loads(_encoded(args)),
        }
        try:
            self._publish_if_ready(turn)
            async with asyncio.timeout(session.context["limits"]["tool_wait_seconds"]):
                return await future
        except TimeoutError:
            self._fail_callback(session, "tool_result_timeout", 504)
            raise GatewayError("tool_result_timeout", 504) from None
        except GatewayError as error:
            self._fail_callback(session, error.code, error.status)
            raise
        finally:
            turn.pending.pop(ref, None)

    @staticmethod
    def _fail_callback(session, code, status=502):
        turn = session.active
        if turn and not turn.finished and not turn.cancelled:
            turn.fail(code, status)
        if session.owner and not session.owner.done() and not session.cancel_requested:
            session.cancel_requested = True
            session.owner.cancel()

    def _publish_if_ready(self, turn):
        if turn.boundary is None or len(turn.pending) != 1:
            return
        ref, pending = next(iter(turn.pending.items()))
        if (
            turn.boundary["name"] != f"mcp__{_SERVER}__{pending['name']}"
            or turn.boundary["arguments"] != pending["args"]
        ):
            raise GatewayError("unexpected_tool_activity")
        # Equal name/args only rule out obvious inconsistency. They are not a
        # native correlation or proof that a parallel batch is complete.
        if not self._verified_native_batch_boundary(turn):
            raise GatewayError("tool_batch_boundary_unverified", 503)
        if pending.get("published"):
            raise GatewayError("duplicate_tool_boundary")
        pending["published"] = True
        turn.handoffs += 1
        turn.tool_history = True
        turn.emit(
            "tool_call_start", {"native_ref": ref, "index": 0, "name": pending["name"]}
        )
        turn.emit(
            "tool_call_ready",
            {
                "native_ref": ref,
                "index": 0,
                "name": pending["name"],
                "arguments": pending["args"],
            },
        )
        turn.emit("tool_batch_ready", {"native_refs": [ref]})
        turn.boundary = None

    def _verified_native_batch_boundary(self, turn):
        # No proof exists for the pinned SDK/CLI. This is deliberately not a
        # config/env option; only a future reviewed implementation may change it.
        return False

    def _usage(self, turn, message):
        raw = message.usage
        if not isinstance(raw, dict) or not message.message_id:
            return
        keys = (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
        if any(k in raw and (type(raw[k]) is not int or raw[k] < 0) for k in keys):
            raise GatewayError("invalid_backend_usage")
        step = turn.steps.setdefault(
            message.message_id, "step-" + secrets.token_hex(16)
        )
        values = [raw.get(k) for k in keys]
        inputs, outputs, cache_read, cache_write = values
        total_input = (
            inputs + cache_read + cache_write
            if all(v is not None for v in (inputs, cache_read, cache_write))
            else None
        )
        turn.usage_observed |= total_input is not None or outputs is not None
        observation = {
            "invocation_id": turn.invocation_id,
            "step_id": step,
            "epoch": turn.epoch,
            "event_id": "usage-" + secrets.token_hex(16),
            "sequence": turn.usage_sequence,
            "scope": "step",
            "semantics": "snapshot",
            "coverage": None,
            "source": "claude_sdk_0.2.180_unverified",
            "unit": "tokens",
            "completeness": "partial"
            if total_input is not None or outputs is not None
            else "unknown",
            "final": False,
            "input_tokens": total_input,
            "output_tokens": outputs,
            "cache_read_tokens": cache_read,
            "cache_write_tokens": cache_write,
            "reasoning_tokens": None,
        }
        turn.usage_sequence += 1
        turn.emit("usage", observation)

    async def _run_turn(self, session, turn):
        await session.client.query(turn.prompt)
        assistant_seen = False
        terminal = None
        async for message in session.client.receive_response():
            if isinstance(message, StreamEvent):
                event = message.event
                if event.get("type") == "content_block_start":
                    block = event.get("content_block", {})
                    if block.get("type") == "server_tool_use":
                        raise GatewayError("unexpected_tool_activity")
                    if block.get("type") == "tool_use" and block.get("name") not in {
                        f"mcp__{_SERVER}__{n}" for n in turn.tools
                    }:
                        raise GatewayError("unexpected_tool_activity")
                delta = event.get("delta", {})
                if (
                    event.get("type") == "content_block_delta"
                    and delta.get("type") == "text_delta"
                ):
                    text = delta.get("text")
                    if not isinstance(text, str):
                        raise GatewayError("invalid_backend_event")
                    turn.output_bytes += len(text.encode("utf-8"))
                    if turn.output_bytes > session.config.max_output_bytes:
                        raise GatewayError("output_too_large")
                    turn.partial.append(text)
                    turn.answer.append(text)
                    turn.emit("text_delta", {"text": text})
            elif isinstance(message, AssistantMessage):
                assistant_seen = True
                self._usage(turn, message)
                if any(
                    isinstance(block, ServerToolUseBlock) for block in message.content
                ):
                    raise GatewayError("unexpected_tool_activity")
                blocks = [
                    block
                    for block in message.content
                    if isinstance(block, ToolUseBlock)
                ]
                if blocks:
                    if (
                        len(blocks) != 1
                        or message.stop_reason != "tool_use"
                        or blocks[0].name
                        not in {f"mcp__{_SERVER}__{n}" for n in turn.tools}
                        or turn.boundary is not None
                    ):
                        raise GatewayError("unexpected_tool_activity")
                    if not isinstance(blocks[0].input, dict):
                        raise GatewayError("invalid_backend_event")
                    turn.boundary = {
                        "name": blocks[0].name,
                        "arguments": json.loads(_encoded(blocks[0].input)),
                    }
                text = "".join(
                    b.text for b in message.content if isinstance(b, TextBlock)
                )
                if turn.partial:
                    if text != "".join(turn.partial):
                        raise GatewayError("inconsistent_backend_text")
                elif text:
                    turn.output_bytes += len(text.encode("utf-8"))
                    if turn.output_bytes > session.config.max_output_bytes:
                        raise GatewayError("output_too_large")
                    turn.answer.append(text)
                    turn.emit("text_delta", {"text": text})
                turn.partial.clear()
                if blocks:
                    self._publish_if_ready(turn)
            elif isinstance(message, SystemMessage):
                if message.subtype.startswith("hook_"):
                    raise GatewayError("unexpected_hook_activity")
                if message.subtype == "init":
                    declared = set(message.data.get("tools") or [])
                    expected = {f"mcp__{_SERVER}__{n}" for n in turn.tools}
                    servers = message.data.get("mcp_servers") or []
                    if declared - expected or any(
                        not isinstance(s, dict) or s.get("name") != _SERVER
                        for s in servers
                    ):
                        raise GatewayError("unsafe_backend_profile", 503)
            elif isinstance(message, ResultMessage):
                if (
                    not assistant_seen
                    or message.is_error
                    or message.subtype != "success"
                    or message.permission_denials
                    or turn.pending
                    or turn.boundary is not None
                    or message.terminal_reason
                    in ("aborted_streaming", "aborted_tools", "max_turns")
                ):
                    raise GatewayError("backend_turn_failed")
                reason = {
                    "end_turn": "stop",
                    "stop_sequence": "stop",
                    "max_tokens": "length",
                    "refusal": "content_filter",
                }.get(message.stop_reason)
                if reason is None:
                    raise GatewayError("unsupported_stop_reason")
                terminal = {
                    "kind": "native_terminal",
                    "finish_reason": reason,
                    "usage_completeness": "partial"
                    if turn.usage_observed
                    else "unknown",
                }
                break
        if terminal is None:
            raise GatewayError("missing_terminal_result")
        session.expected = [
            *turn.data["messages"],
            {"role": "assistant", "content": "".join(turn.answer)},
        ]
        session.tool_history = turn.tool_history
        turn.emit("native_turn_ended", terminal)
        turn.finished = True
        # A finished turn needs no sentinel once its queue is drained. Never
        # block session release on a consumer draining an extra queue slot.
        with contextlib.suppress(asyncio.QueueFull):
            turn.queue.put_nowait(_END)

    async def events(self, turn):
        if (
            not isinstance(turn, _Turn)
            or turn.session.driver is not self
            or turn.session.loop is not asyncio.get_running_loop()
        ):
            raise GatewayError("invalid_turn", 400)
        if turn.consumer:
            raise GatewayError("turn_consumer_exists", 409)
        turn.consumer = True
        try:
            while True:
                if turn.finished and turn.queue.empty():
                    if turn.error_event is not None:
                        event, turn.error_event = turn.error_event, None
                        yield event
                    return
                item = await turn.queue.get()
                if item is _END:
                    if turn.error_event is not None:
                        event, turn.error_event = turn.error_event, None
                        yield event
                    return
                yield item
        finally:
            # HTTP subscriber ownership is separate from the native owner task.
            turn.consumer = False

    async def submit_tool_results(self, turn, results):
        if (
            not isinstance(turn, _Turn)
            or turn.session.driver is not self
            or turn.session.loop is not asyncio.get_running_loop()
        ):
            raise GatewayError("invalid_turn", 400)
        _validate(_contract(), "ToolResults", results)
        _policy(results["round_policy"], continuation=True)
        if turn.cancelled or turn.finished or turn.session.closed:
            raise GatewayError("turn_closed", 410)
        if results["locator"] != turn.session.context["locator"]:
            raise GatewayError("unknown_tool_session", 410)
        items = results["results"]
        if (
            len(items) != 1
            or len(_encoded(items)) > 131072
            or items[0]["native_ref"] not in turn.pending
        ):
            raise GatewayError("tool_result_conflict", 409)
        pending = turn.pending[items[0]["native_ref"]]
        future = pending["future"]
        if not pending.get("published") or future.done():
            raise GatewayError("tool_result_conflict", 409)
        # No await between checking and resolving: duplicate delivery loses atomically.
        future.set_result(
            {
                "content": [{"type": "text", "text": items[0]["content"]}],
                "is_error": False,
            }
        )

    async def cancel(self, turn, reason, cleanup_deadline):
        if (
            not isinstance(turn, _Turn)
            or turn.session.driver is not self
            or turn.session.loop is not asyncio.get_running_loop()
        ):
            raise GatewayError("invalid_turn", 400)
        if (
            not isinstance(reason, str)
            or type(cleanup_deadline) not in (int, float)
            or not math.isfinite(cleanup_deadline)
        ):
            raise GatewayError("invalid_cancellation", 400)
        # A late cancellation for an already completed turn must not close the
        # next native turn that happens to reuse the same session.
        if turn.finished or turn.session.active is not turn:
            return
        turn.cancelled = True
        await self._close(turn.session, cleanup_deadline)

    async def _close(self, session, cleanup_deadline):
        already_closing = session.closed
        session.closed = True
        if (
            session.owner
            and not session.owner.done()
            and not session.cancel_requested
            and not already_closing
        ):
            session.cancel_requested = True
            session.owner.cancel()
        if session.owner:
            done, _ = await asyncio.wait(
                {session.owner}, timeout=max(0, cleanup_deadline - time.monotonic())
            )
            if not done and session.wire:
                with contextlib.suppress(Exception):
                    session.wire.kill()
            # If cancellation preceded the owner's first instruction, its
            # finally block cannot run and must not strand begin_turn's waiter.
            if session.active and not session.active.ready.done():
                session.active.ready.set_exception(
                    GatewayError(
                        "backend_startup_failed" if done else "cleanup_timeout", 503
                    )
                )
            if not done:
                raise GatewayError("cleanup_timeout", 503)
            outcome = (await asyncio.gather(session.owner, return_exceptions=True))[0]
            if isinstance(outcome, BaseException) and (
                not isinstance(outcome, asyncio.CancelledError) or session.owner_started
            ):
                raise GatewayError("cleanup_failed", 503) from None

    async def close_session(self, session, cleanup_deadline):
        if (
            not isinstance(session, _Session)
            or session.driver is not self
            or session.loop is not asyncio.get_running_loop()
        ):
            raise GatewayError("invalid_session", 400)
        if type(cleanup_deadline) not in (int, float) or not math.isfinite(
            cleanup_deadline
        ):
            raise GatewayError("invalid_cancellation", 400)
        await self._close(session, cleanup_deadline)
