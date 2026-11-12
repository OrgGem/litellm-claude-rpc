# LiteLLM Claude RPC

Development implementation for localhost Windows. This is **not a live-accepted release**.

## Install and run

```powershell
uv sync --frozen --extra dev
uv run --frozen python -m litellm_claude_rpc doctor
$env:LITELLM_MASTER_KEY = 'sk-replace-with-a-distinct-random-secret'
uv run --frozen python -m litellm_claude_rpc serve --port 4103
```

`doctor` never sends a prompt, copies credentials, or starts an interactive login.
The API binds only `127.0.0.1`; startup fails when readiness fails. Model listing
is not a quota or entitlement check. The CLI executable may be selected by
`--cli-bin`; arguments are arrays, never a shell command.

## API and sessions

Bearer-authenticated `GET /v1/models` and `POST /v1/chat/completions` accept
`model`, `messages`, `stream`, and `stream_options.include_usage` only. Roles are
system (prefix only), user and assistant; final message must be user. Content is
a string or text-only blocks. Unicode and empty strings are preserved. Tools,
images/audio, developer role, sampling/token limits and Responses are rejected.
Slash-leading user input is wrapped as literal text to avoid invoking CLI workflows.

History is converted to a labeled JSON transcript. Role precedence is not
identical to OpenAI. The native system prompt is also used where supported.
No implicit truncation or retries are performed by the gateway. CLI-internal
network behavior is controlled by that CLI and is not an exactly-once guarantee.

Send a stable `X-Conversation-ID` (1–128 ASCII letters, digits, `_ . : -`) and
complete messages each time. Only an exact completed-history prefix plus one new
user message can reuse a healthy session. A history/system/model change boots a
new session. No header means a fresh session. Concurrent requests with the same
ID serialize. Errors, timeout, cancellation, unfinished delivery or a closed
stream discard the session. Restart forgets all mappings.

Defaults: 2 active turns, 4 live sessions, 16 admitted requests, 600-second idle
TTL, 300-second total deadline including waits, 10-second cleanup budget,
128 KiB request, 2 MiB output, 64 events per bounded queue. Overflow fails the
turn. Session state and history hashes live only in RAM. Prompt/response text is
not logged by this host; stderr is discarded. Owned temporary workspaces are
removed on session closure. No user CLI store or credentials are deleted.

One gateway key represents one local owner. This is not a multi-tenant service.
The root router hashes conversation IDs before forwarding; direct-host calls
therefore have a separate conversation namespace from router calls.

## In-process LiteLLM provider

`ClaudeLLM` in `litellm_claude_rpc.provider` implements sync/async completion and
streaming. Construct `Runtime(SessionPool(config, factory))`, pass it to the
provider and call `runtime.close()` on shutdown. The production factory is
`lambda model, system: ClaudeDriver(config, model, system)` from `driver.py`.
`optional_params.conversation_id` enables the same reuse contract. Keep one
runtime per provider instance; do not create a runtime per request. Only consume
streaming iterators in their owning scope, and close them when abandoning output.

## Validation

```powershell
uv run --frozen --extra dev pytest -q
uv run --frozen --extra dev ruff check src tests
uv run --frozen --extra dev ruff format --check src tests
uv build
```

Tests use fake SDK/ACP peers and real owned child processes. They do not prove
CLI login, model access, tool/hook suppression, or live warm-session behavior.
Before release, complete the Windows matrix in `docs/ACCEPTANCE.md`.

This repository is independent of the router and other plugins. Contract,
session, host, runtime and process modules are maintained locally so installation
does not depend on sibling repositories. Process/runtime patterns were adapted
from the workspace's Antigravity adapter; protocol drivers are backend-specific.

## Claude status

Pinned Python Agent SDK **0.2.180**, bundled Claude Code **2.1.321**. The SDK owns
Claude's control protocol; a version-bound custom transport uses the SDK command
builder and Windows job objects to contain the entire process tree before resume.
SDK calls and connection cleanup run in one owner task. `tools=[]`, `skills=[]`,
strict empty MCP, empty settings sources, disabled hooks and no session persistence
are set explicitly. An unexpected tool/hook event fails the session.

Local doctor on 2026-11-13: **cli_login_required**. No real model call was made.
Log in manually with the bundled official CLI:

```powershell
& .venv/Lib/site-packages/claude_agent_sdk/_bundled/claude.exe auth login
```

The gateway does not drive login or inspect its credentials. Retention outside
its own temp directory remains CLI/provider-owned; `--no-session-persistence`
is requested but does not imply deletion of all provider-side records.
Usage comes from deduplicated assistant API-step counters; cumulative result
counters are not reported as per-turn usage. Missing counters yield no usage.

## Operational configuration and LiteLLM usage limitation

All host limits can be overridden with `CLAUDE_RPC_LIMITS`, a JSON object
using Config field names: timeout, cleanup_timeout, idle_ttl, concurrency,
max_sessions, max_pending, queue_size, max_input_bytes and max_output_bytes.
Unknown keys, booleans in numeric fields, nonfinite values and nonpositive
values are rejected before starting the host.

Pinned LiteLLM can synthesize unknown token counts. The in-process provider
therefore disables LiteLLM's estimated streaming usage before the wrapper is
created. The shared router omits all non-stream usage and returns null stream
usage for new backends until true upstream provenance is preserved. This also
omits genuine counters at those surfaces; `include_usage` is consequently
partial there. Direct standalone HTTP hosts return real counters when known,
otherwise no usage (or null in a requested final usage chunk).

Even after login, this development build returns `claude_live_acceptance_pending`.
Production remains gated until a version-bound Windows acceptance report proves
tool/hook isolation and the persistent SDK turn boundary. The driver rejects a
terminal result before any assistant message, but this alone cannot associate an
entire delayed assistant/result pair with its originating turn. Warm reuse stays
unreleased until that case is resolved. There is no runtime bypass flag.
