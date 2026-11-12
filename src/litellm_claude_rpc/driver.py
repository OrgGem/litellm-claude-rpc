"""Claude Agent SDK 0.2.180, restricted CLI 2.1.321 profile."""

import asyncio
import json
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
        async for message in self.client.receive_response():
            if isinstance(message, StreamEvent):
                event = message.event
                if event.get("type") == "content_block_start" and event.get(
                    "content_block", {}
                ).get("type") in ("tool_use", "server_tool_use"):
                    raise GatewayError("unexpected_tool_activity")
                delta = event.get("delta", {})
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
                raw = message.usage or {}
                keys = (
                    "input_tokens",
                    "output_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                )
                if (
                    message.message_id
                    and all(
                        type(raw.get(k, 0)) is int and raw.get(k, 0) >= 0 for k in keys
                    )
                    and "input_tokens" in raw
                    and "output_tokens" in raw
                ):
                    # Assistant usage is per API step. Repeated blocks with one
                    # message ID share the same usage and must be counted once.
                    step_usage[message.message_id] = (
                        raw["input_tokens"]
                        + raw.get("cache_read_input_tokens", 0)
                        + raw.get("cache_creation_input_tokens", 0),
                        raw["output_tokens"],
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
                    inputs = sum(pair[0] for pair in step_usage.values())
                    outputs = sum(pair[1] for pair in step_usage.values())
                    usage = {
                        "prompt_tokens": inputs,
                        "completion_tokens": outputs,
                        "total_tokens": inputs + outputs,
                    }
                yield Event(done=True, usage=usage)

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
    if code or version.decode().split()[0] != CLI_VERSION:
        raise GatewayError("unsupported_cli_version", 503)
    code, status = await run("auth", "status", "--json")
    if code or not json.loads(status).get("loggedIn"):
        raise GatewayError("cli_login_required", 503)
    # Remove only with a version-bound Windows acceptance report proving
    # tools/hooks isolation AND the persistent SDK turn boundary. Login alone
    # must not silently promote this development build to a ready backend.
    raise GatewayError("claude_live_acceptance_pending", 503)
