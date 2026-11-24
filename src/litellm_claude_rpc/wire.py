"""SDK transport with bounded frames and owned Windows process trees.

Only command construction uses a pinned SDK internal API; the SDK owns the
control protocol, permission callbacks, parsing of message types and turns.
"""

import asyncio
import os
from pathlib import Path

from claude_agent_sdk._internal.transport import Transport
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

from .contracts import GatewayError, strict_json
from .process import ProcessTree


def child_environment(isolation_root=None):
    # Deliberately omit credential/provider overrides, debug, hooks, NODE_OPTIONS,
    # Python injection and telemetry variables. Official CLI reads its own login.
    allowed = {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "TEMP",
        "TMP",
        "HOME",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMFILES",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    }
    env = {k: v for k, v in os.environ.items() if k.upper() in allowed}
    if isolation_root is not None:
        for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP"):
            folder = Path(isolation_root) / ("isolated-" + key.lower())
            folder.mkdir(exist_ok=True)
            env[key] = str(folder)
    env.update(
        CLAUDE_CODE_ENTRYPOINT="sdk-py",
        CLAUDE_CODE_SDK_READS_SESSION_STATE="1",
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
    )
    return env


class SDKWire(Transport):
    def __init__(self, options, config, *, isolation_root=None):
        async def empty():
            if False:
                yield {}

        self.builder = SubprocessCLITransport(empty(), options)
        self.options, self.config = options, config
        self.isolation_root = isolation_root
        self.proc = self.tree = None
        self.lock = asyncio.Lock()

    async def connect(self):
        spawn = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *self.builder._build_command(),
                cwd=self.options.cwd,
                env=child_environment(self.isolation_root),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                limit=4 * 1024 * 1024,
                start_new_session=os.name != "nt",
                creationflags=4 if os.name == "nt" else 0,
            )
        )
        try:
            self.proc = await asyncio.shield(spawn)
            self.tree = ProcessTree(self.proc.pid)
            self.tree.resume()
        except BaseException:
            self.proc = await spawn
            await self.close()
            raise

    async def write(self, data):
        async with self.lock:
            self.proc.stdin.write(data.encode("utf-8"))
            await self.proc.stdin.drain()

    async def read_messages(self):
        while True:
            try:
                line = await self.proc.stdout.readline()
                if not line:
                    return
                if not line.endswith(b"\n") or len(line) > 4 * 1024 * 1024:
                    raise ValueError()
                message = strict_json(line.decode("utf-8"))
                if not isinstance(message, dict):
                    raise ValueError()
            except (ValueError, UnicodeError):
                raise GatewayError("invalid_backend_frame") from None
            yield message

    def is_ready(self):
        return self.proc is not None and self.proc.returncode is None

    async def end_input(self):
        if self.proc and self.proc.stdin:
            self.proc.stdin.close()

    def kill(self):
        if self.tree:
            self.tree.kill()
        elif self.proc and self.proc.returncode is None:
            self.proc.kill()

    async def close(self):
        self.kill()
        if self.proc:
            try:
                await asyncio.wait_for(self.proc.wait(), self.config.cleanup_timeout)
            finally:
                if self.tree:
                    await self.tree.wait(self.config.cleanup_timeout)
                self.proc = None
