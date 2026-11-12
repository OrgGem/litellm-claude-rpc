import sys

from claude_agent_sdk import ClaudeAgentOptions

from litellm_claude_rpc.contracts import Config
from litellm_claude_rpc.wire import SDKWire, child_environment


async def test_owned_process_is_reaped_on_close(tmp_path):
    wire = SDKWire(ClaudeAgentOptions(cwd=str(tmp_path)), Config(cleanup_timeout=2))
    wire.builder._build_command = lambda: [
        sys.executable,
        "-u",
        "-c",
        'import time; print("{\\"type\\":\\"fixture\\"}", flush=True); time.sleep(120)',
    ]
    await wire.connect()
    proc = wire.proc
    assert (await anext(wire.read_messages())) == {"type": "fixture"}
    await wire.close()
    assert proc.returncode is not None
    assert wire.tree.handle is None


def test_child_environment_drops_ambient_secrets_and_injection(monkeypatch):
    for name in (
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "NODE_OPTIONS",
        "PYTHONPATH",
        "LITELLM_MASTER_KEY",
        "CLAUDE_CONFIG_DIR",
    ):
        monkeypatch.setenv(name, "must-not-pass")
    env = child_environment()
    assert not any(value == "must-not-pass" for value in env.values())
    assert any(k.upper() == "PATH" for k in env)
