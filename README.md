# LiteLLM Claude RPC

OpenAI-compatible gateway (`Chat Completions` + `Responses API`) and LiteLLM
`CustomLLM` provider backed by restricted Claude CLI (`claude-agent-sdk` 0.2.180,
CLI 2.1.321+) sessions.

## Install and run

```powershell
uv sync --frozen --extra dev
uv run --frozen python -m litellm_claude_rpc doctor
uv run --frozen python -m litellm_claude_rpc serve --port 4103
```

`doctor` checks CLI version and login status without sending prompts or copying
credentials. When `LITELLM_MASTER_KEY` / `CLAUDE_RPC_API_KEY` is unset, `serve`
generates a session API key for the current process. CLI flags include `--host`,
`--port`, `--cli-bin`, `--model`, `--model-name`, `--max-concurrent`, `--timeout`
and `--allow-prompt-flattening`.

You can also host via LiteLLM Proxy directly with `config.example.yaml` and
`proxy_handler.py`:

```powershell
$env:LITELLM_MASTER_KEY = 'sk-replace-with-a-distinct-random-secret'
litellm --config config.example.yaml --port 4000
```

## API and sessions

Bearer-authenticated `GET /v1/models`, `POST /v1/chat/completions`,
`POST /v1/responses` (normal JSON and `stream: true` SSE), and `GET /health/ready`:

- **Chat Completions**: supports `system`, `developer`, `user`, `assistant` roles
  and string or `[{"type": "text", ...}]` content blocks. Send a stable
  `X-Conversation-ID` (1–128 ASCII letters, digits, `_ . : -`) to reuse a warm
  session when completed history is an exact prefix plus one new user message.
- **Responses API**: stateless (`store: false`), accepts top-level `instructions`,
  string or message-list `input` (`input_text` / `output_text`, assistant `phase`),
  and rejects `store: true`, `previous_response_id`, tools, multimodal, and
  sampling parameters.
- **Usage**: deduplicates per-step assistant counters and exposes
  `prompt_tokens_details.cached_tokens` (`cache_read_input_tokens`) and
  `cache_write_tokens` (`cache_creation_input_tokens`) when supplied by the SDK.
  Unknown counters remain `None`/omitted rather than estimated.

## Environment configuration

| Variable | Default |
| --- | --- |
| `CLAUDE_MODEL_MAP` | `{"claude-default":null}` |
| `CLAUDE_RPC_EXECUTABLE` | SDK-bundled `claude` binary |
| `CLAUDE_RPC_LIMITS` | `{}` (JSON override for Config limit fields) |
| `CLAUDE_RPC_REQUEST_TIMEOUT` | `300` |
| `CLAUDE_RPC_MAX_CONCURRENT` | `2` |
| `CLAUDE_RPC_ALLOW_PROMPT_FLATTENING` | `false` |
| `LITELLM_MASTER_KEY` / `CLAUDE_RPC_API_KEY` | auto-generated on `serve` |

## In-process LiteLLM provider

Import `claude_rpc_provider` or `ClaudeLLM` from `litellm_claude_rpc`. Calling
`claude_rpc_provider` lazily initializes `Config.from_env()` and an owned session
runtime on first request; call `close()` / `await aclose()` on shutdown. You can
also pass an explicit `Runtime(SessionPool(config, factory))` to `ClaudeLLM(runtime)`.

Native V2 (`ClaudeNativeDriver`) and `AnswerOnlyDriver` keep their production
acceptance gates closed (`claude_live_acceptance_pending` / empty receipts).
