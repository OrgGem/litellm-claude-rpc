# Candidate native v2 seam

`litellm_claude_rpc.native_v2.ClaudeNativeDriver` implements the frozen 0.1
native seam for integration experiments. Its contract snapshot is bundled with
the package and checked by hash. The public driver **always rejects session
creation** with `claude_live_acceptance_pending`; there is no user-facing switch
to activate it. Existing standalone HTTP and `ClaudeLLM` remain text-only v1.

Fixture tests use the actual SDK in-process MCP bridge with a fake SDK peer and
fake transport, never an authenticated CLI. They do not establish live MCP
invocation, serial batch boundaries, tools/hooks/profile isolation, or reasoning
and token semantics. In particular, the bridge fails closed even when a single
`tool_use` stop boundary matches callback name/arguments; no SDK correlation or
complete native batch boundary has been proven. Lifecycle tests monkeypatch a
private verifier to simulate proof; no runtime configuration can enable it. The
candidate does not advertise parallel calls, reasoning
output/control, output-token limits, or complete usage. Observed usage is partial
until its scope is independently verified. It keeps the inference permit and
counts tool wait against the active budget while native pause is unverified.
Tool history cannot yet be imported/reused after a completed tool turn.

The host owns authentication, public tool IDs, delivery acknowledgements and
pending leases; the driver owns SDK callback futures and the native client task.
If transport cleanup fails, the isolated temporary workspace is retained and
the host must quarantine its native lease until process termination is proven.
Do not construct native sessions or expose the candidate as a production backend
until the version-bound isolation and live acceptance gates are satisfied.
