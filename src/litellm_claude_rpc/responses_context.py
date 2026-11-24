"""Request-local information shared between the proxy hook and provider."""

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class ResponsesRequestContext:
    alias: str
    instructions: str | None
    phases: tuple
    roles: tuple
    usage_known: bool | None = None
    usage: dict | None = None


responses_request_context = ContextVar(
    "claude_rpc_responses_request_context", default=None
)
