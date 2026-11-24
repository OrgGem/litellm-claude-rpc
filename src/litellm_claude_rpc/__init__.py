"""Claude CLI provider. Importing this package never starts a CLI."""

__version__ = "0.2.0.dev0"

from .provider import ClaudeLLM, ClaudeRPCProvider, claude_rpc_provider

__all__ = ["ClaudeLLM", "ClaudeRPCProvider", "claude_rpc_provider", "__version__"]
