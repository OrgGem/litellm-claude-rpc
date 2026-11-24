"""Claude Agent SDK 0.2.180, restricted CLI 2.1.x profile."""

import asyncio
import json
import re
import tempfile
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResultDeny,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
)
from claude_agent_sdk.types import StreamEvent

from .contracts import Event, GatewayError
from .wire import SDKWire

SDK_VERSION = "0.2.180"
CLI_VERSION = "2.1.321"
_CLI_VERSION_RE = re.compile(r"^2\.1\.(\d+)(?:[-+][A-Za-z0-9.]+)?\Z")


def is_supported_cli_version(raw: str | None) -> bool:
    if not isinstance(raw, str) or not raw.strip():
        return False
    token = raw.strip().split()[0]
    match = _CLI_VERSION_RE.match(token)
    return match is not None and int(match.group(1)) >= 321


def executable(config):
    if config.executable:
        return config.executable
    import os

    import claude_agent_sdk

    return str(
        Path(claude_agent_sdk.__file__).parent
        / "_bundled"
        / ("claude.exe" if os.name == "nt" else "claude")
    )


class ClaudeDriver:
    def __init__(self, config, model, system):
        self.config, self.model, self.system = config, model, system
        self.temp = self.client = self.wire = None

    async def connect(self):
        await doctor(self.config)
        self.temp = tempfile.TemporaryDirectory(prefix="claude-rpc-")
        Path(self.temp.name, ".git").mkdir()

        async def deny(*args):
            return PermissionResultDeny(message="Tools disabled by gateway")

        options = ClaudeAgentOptions(
            cli_path=executable(self.config),
            cwd=self.temp.name,
            model=self.model,
            system_prompt=self.system or "Answer the user's text question.",
            tools=[],
            skills=[],
            allowed_tools=[],
            permission_mode="dontAsk",
            can_use_tool=deny,
            mcp_servers='{"mcpServers":{}}',
            strict_mcp_config=True,
            setting_sources=[],
            settings=json.dumps({"disableAllHooks": True, "enabledPlugins": {}}),
            hooks={},
            plugins=[],
            include_partial_messages=True,
            verbatim_prompts=True,
            extra_args={"no-session-persistence": None},
        )
        self.wire = SDKWire(options, self.config)
        self.client = ClaudeSDKClient(options=options, transport=self.wire)
        await self.client.connect()

    async def stream(self, prompt):
        await self.client.query(prompt)
        partial = []
        step_usage = {}
        usage_complete = True
        assistant_seen = False
        stream_stop_reason = None
        assistant_stop_reason = None
        async for message in self.client.receive_response():
            if isinstance(message, StreamEvent):
                event = message.event
                if event.get("type") == "content_block_start" and event.get(
                    "content_block", {}
                ).get("type") in ("tool_use", "server_tool_use"):
                    raise GatewayError("unexpected_tool_activity")
                delta = event.get("delta", {})
                if (
                    event.get("type") == "message_delta"
                    and isinstance(delta, dict)
                    and delta.get("stop_reason") is not None
                ):
                    stream_stop_reason = delta.get("stop_reason")
                if (
                    event.get("type") == "content_block_delta"
                    and delta.get("type") == "text_delta"
                ):
                    text = delta.get("text")
                    if not isinstance(text, str):
                        raise GatewayError("invalid_backend_event")
                    partial.append(text)
                    yield Event(text=text)
            elif isinstance(message, AssistantMessage):
                assistant_seen = True
                if getattr(message, "stop_reason", None) is not None:
                    assistant_stop_reason = message.stop_reason
                raw = message.usage or {}
                keys = (
                    "input_tokens",
                    "output_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                )
                if (
                    message.message_id
                    and isinstance(raw, dict)
                    and all(
                        type(raw.get(k, 0)) is int and raw.get(k, 0) >= 0 for k in keys
                    )
                    and "input_tokens" in raw
                    and "output_tokens" in raw
                ):
                    # Assistant usage is per API step. Repeated blocks with one
                    # message ID share the same usage and must be counted once.
                    cached_read = (
                        raw["cache_read_input_tokens"]
                        if "cache_read_input_tokens" in raw
                        else None
                    )
                    cache_write = (
                        raw["cache_creation_input_tokens"]
                        if "cache_creation_input_tokens" in raw
                        else None
                    )
                    step_usage[message.message_id] = (
                        raw["input_tokens"] + (cached_read or 0) + (cache_write or 0),
                        raw["output_tokens"],
                        cached_read,
                        cache_write,
                    )
                else:
                    usage_complete = False
                if any(isinstance(b, ToolUseBlock) for b in message.content):
                    raise GatewayError("unexpected_tool_activity")
                text = "".join(
                    b.text for b in message.content if isinstance(b, TextBlock)
                )
                if partial:
                    if text != "".join(partial):
                        raise GatewayError("inconsistent_backend_text")
                elif text:
                    yield Event(text=text)
                partial.clear()
            elif isinstance(message, SystemMessage):
                if message.subtype == "init" and (
                    message.data.get("tools") or message.data.get("mcp_servers")
                ):
                    raise GatewayError("unsafe_backend_profile", 503)
                if message.subtype.startswith("hook_"):
                    raise GatewayError("unexpected_hook_activity")
            elif isinstance(message, ResultMessage):
                if (
                    not assistant_seen
                    or message.is_error
                    or message.subtype != "success"
                    or message.permission_denials
                    or message.terminal_reason
                    in ("aborted_streaming", "aborted_tools", "max_turns")
                ):
                    raise GatewayError("backend_turn_failed")
                usage = None
                if usage_complete and step_usage:
                    steps = list(step_usage.values())
                    inputs = sum(item[0] for item in steps)
                    outputs = sum(item[1] for item in steps)
                    details = {}
                    if any(item[2] is not None for item in steps):
                        details["cached_tokens"] = sum(item[2] or 0 for item in steps)
                    if any(item[3] is not None for item in steps):
                        details["cache_write_tokens"] = sum(
                            item[3] or 0 for item in steps
                        )
                    usage = {
                        "prompt_tokens": inputs,
                        "completion_tokens": outputs,
                        "total_tokens": inputs + outputs,
                    }
                    if details:
                        usage["prompt_tokens_details"] = details
                raw_stop = (
                    message.stop_reason
                    if message.stop_reason is not None
                    else (
                        assistant_stop_reason
                        if assistant_stop_reason is not None
                        else stream_stop_reason
                    )
                )
                finish_reason = {
                    None: "stop",
                    "end_turn": "stop",
                    "stop_sequence": "stop",
                    "max_tokens": "length",
                    "refusal": "content_filter",
                }.get(raw_stop)
                if finish_reason is None:
                    raise GatewayError("unsupported_stop_reason")
                yield Event(done=True, usage=usage, finish_reason=finish_reason)

    def kill(self):
        if self.wire:
            self.wire.kill()

    async def close(self):
        try:
            if self.client:
                await self.client.disconnect()
        finally:
            if self.wire:
                await self.wire.close()
            if self.temp:
                self.temp.cleanup()


async def doctor(config):
    """No prompt; no credential contents returned to the gateway."""
    from .wire import child_environment

    async def run(*args):
        proc = await asyncio.create_subprocess_exec(
            executable(config),
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=child_environment(),
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), 15)
            return proc.returncode, out
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()

    code, version = await run("--version")
    decoded = version.decode(errors="replace").strip()
    if code or not is_supported_cli_version(decoded):
        raise GatewayError("unsupported_cli_version", 503)
    code, status = await run("auth", "status", "--json")
    try:
        parsed = json.loads(status) if not code else {}
    except (ValueError, UnicodeError):
        parsed = {}
    if code or not isinstance(parsed, dict) or not parsed.get("loggedIn"):
        raise GatewayError("cli_login_required", 503)
    return {
        "status": "ready",
        "cli_version": decoded.split()[0],
        "sdk_version": SDK_VERSION,
        "model_access_verified": False,
    }
