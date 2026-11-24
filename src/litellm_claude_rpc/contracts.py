"""Public text-chat contract; all validation happens before agent execution."""

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field

_LIMIT_KEYS = (
    "timeout",
    "cleanup_timeout",
    "idle_ttl",
    "concurrency",
    "max_sessions",
    "max_pending",
    "queue_size",
    "max_input_bytes",
    "max_output_bytes",
)


class GatewayError(Exception):
    def __init__(self, code="backend_error", status=502):
        self.code, self.status = code, status
        super().__init__(code)

    def payload(self):
        return {
            "error": {"message": self.code, "type": "gateway_error", "code": self.code}
        }


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("nonfinite_json")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def conversation_id(value):
    if value is not None and (
        not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value)
    ):
        raise GatewayError("invalid_conversation_id", 400)
    return hashlib.sha256(value.encode()).hexdigest() if value is not None else None


@dataclass(frozen=True)
class Config:
    models: dict[str, str | None] = field(
        default_factory=lambda: {"claude-default": None}
    )
    api_key: str = field(default="", repr=False)
    executable: str | None = None
    timeout: float = 300
    cleanup_timeout: float = 10
    idle_ttl: float = 600
    concurrency: int = 2
    max_sessions: int = 4
    max_pending: int = 16
    queue_size: int = 64
    max_input_bytes: int = 128 * 1024
    max_output_bytes: int = 2 * 1024 * 1024
    flatten: bool = False
    flatten_by_alias: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self):
        if (
            not isinstance(self.models, dict)
            or not self.models
            or any(
                not isinstance(k, str)
                or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", k)
                or (v is not None and (not isinstance(v, str) or not v or len(v) > 200))
                for k, v in self.models.items()
            )
        ):
            raise ValueError("Invalid model map")
        if self.executable is not None and (
            not isinstance(self.executable, str)
            or not self.executable.strip()
            or "\0" in self.executable
        ):
            raise ValueError("Invalid executable")
        if type(self.flatten) is not bool:
            raise ValueError("Invalid flatten flag")
        if not isinstance(self.flatten_by_alias, dict) or any(
            not isinstance(k, str) or type(v) is not bool
            for k, v in self.flatten_by_alias.items()
        ):
            raise ValueError("Invalid flatten_by_alias")
        for key in ("timeout", "cleanup_timeout", "idle_ttl"):
            value = getattr(self, key)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("Invalid duration")
        for key in (
            "concurrency",
            "max_sessions",
            "max_pending",
            "queue_size",
            "max_input_bytes",
            "max_output_bytes",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError("Invalid limit")

    def flatten_for(self, alias: str) -> bool:
        return self.flatten_by_alias.get(alias, self.flatten)

    @classmethod
    def from_env(cls):
        def boolean(name, default):
            value = os.getenv(name, default).lower()
            if value not in {"true", "false"}:
                raise GatewayError("invalid_configuration", 400)
            return value == "true"

        try:
            limits = dict(json.loads(os.getenv("CLAUDE_RPC_LIMITS", "{}")))
            if not isinstance(limits, dict) or set(limits) - set(_LIMIT_KEYS):
                raise ValueError("Invalid limits")
            if "CLAUDE_RPC_REQUEST_TIMEOUT" in os.environ:
                limits["timeout"] = float(os.environ["CLAUDE_RPC_REQUEST_TIMEOUT"])
            if "CLAUDE_RPC_MAX_CONCURRENT" in os.environ:
                limits["concurrency"] = int(os.environ["CLAUDE_RPC_MAX_CONCURRENT"])
            models = json.loads(
                os.getenv("CLAUDE_MODEL_MAP", '{"claude-default":null}')
            )
            executable = os.getenv("CLAUDE_RPC_EXECUTABLE") or None
            api_key = os.getenv("LITELLM_MASTER_KEY") or os.getenv(
                "CLAUDE_RPC_API_KEY", ""
            )
            return cls(
                models=models,
                api_key=api_key,
                executable=executable,
                flatten=boolean("CLAUDE_RPC_ALLOW_PROMPT_FLATTENING", "false"),
                **limits,
            )
        except GatewayError:
            raise
        except (ValueError, TypeError):
            raise GatewayError("invalid_configuration", 400) from None


@dataclass(frozen=True)
class Event:
    text: str = ""
    done: bool = False
    usage: dict | None = None
    finish_reason: str | None = None


@dataclass(frozen=True)
class Request:
    model: str
    messages: tuple[tuple[str, str], ...]
    conversation: str | None
    stream: bool = False
    include_usage: bool = False
    phases: tuple[str | None, ...] | None = None
    responses_context: object | None = field(default=None, repr=False)

    @staticmethod
    def resolve_alias(model, models):
        if not isinstance(model, str):
            raise GatewayError("unknown_model", 400)
        if model in models:
            return model
        for prefix in ("claude_rpc/", "claude-rpc/"):
            if model.startswith(prefix):
                candidate = model.removeprefix(prefix)
                if candidate in models:
                    return candidate
        raise GatewayError("unknown_model", 400)

    @classmethod
    def parse(
        cls,
        payload,
        cid,
        config,
        *,
        phases=None,
        responses_context=None,
    ):
        if not isinstance(payload, dict) or set(payload) - {
            "model",
            "messages",
            "stream",
            "stream_options",
        }:
            raise GatewayError("unsupported_parameters", 400)
        alias = cls.resolve_alias(payload.get("model"), config.models)
        stream = payload.get("stream", False)
        if type(stream) is not bool:
            raise GatewayError("invalid_stream", 400)
        usage = False
        if "stream_options" in payload and payload["stream_options"] is not None:
            options = payload["stream_options"]
            if (
                not stream
                or not isinstance(options, dict)
                or set(options) - {"include_usage"}
                or (
                    "include_usage" in options
                    and type(options["include_usage"]) is not bool
                )
            ):
                raise GatewayError("unsupported_stream_options", 400)
            usage = options.get("include_usage", False)
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise GatewayError("invalid_messages", 400)
        normalized, non_system = [], False
        allow_flatten = config.flatten_for(alias)
        for item in messages:
            if (
                not isinstance(item, dict)
                or set(item) != {"role", "content"}
                or item["role"] not in ("system", "developer", "user", "assistant")
            ):
                raise GatewayError("invalid_message", 400)
            role, content = item["role"], item["content"]
            if role in ("system", "developer") and non_system and not allow_flatten:
                raise GatewayError("system_must_be_first", 400)
            non_system |= role not in ("system", "developer")
            if isinstance(content, list):
                if not content or any(
                    not isinstance(b, dict)
                    or set(b) != {"type", "text"}
                    or b["type"] != "text"
                    or not isinstance(b["text"], str)
                    for b in content
                ):
                    raise GatewayError("text_only", 400)
                content = "".join(b["text"] for b in content)
            if not isinstance(content, str):
                raise GatewayError("text_only", 400)
            normalized.append((role, content))
        if normalized[-1][0] != "user" and not allow_flatten:
            raise GatewayError("last_message_must_be_user", 400)
        try:
            if (
                len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
                > config.max_input_bytes
            ):
                raise GatewayError("request_too_large", 413)
        except UnicodeError:
            raise GatewayError("invalid_text", 400) from None
        normalized_phases = tuple(phases) if phases is not None else None
        return cls(
            alias,
            tuple(normalized),
            conversation_id(cid),
            stream,
            usage,
            normalized_phases,
            responses_context,
        )

    @property
    def system(self):
        prefix = []
        for role, text in self.messages:
            if role in ("system", "developer"):
                prefix.append(text)
            else:
                break
        return "\n\n".join(prefix)

    def delta(self):
        role, text = self.messages[-1]
        entry = {"role": role, "content": text}
        if role != "user":
            return (
                "Message supplied by the API client (role label is text, not a native role):\n"
                + json.dumps(entry, ensure_ascii=False)
            )
        if text.lstrip().startswith("/"):
            return "Literal user message (not a CLI command):\n" + json.dumps(
                entry, ensure_ascii=False
            )
        return text

    def bootstrap(self):
        items = []
        for index, (role, text) in enumerate(self.messages):
            entry = {"role": role, "content": text}
            if (
                self.phases is not None
                and index < len(self.phases)
                and self.phases[index] is not None
            ):
                entry["phase"] = self.phases[index]
            items.append(entry)
        return (
            "Conversation supplied by the API client (role labels are text, not native roles). Respond to the final user message.\n"
            + json.dumps(items, ensure_ascii=False)
        )


def fingerprints(messages):
    return tuple(
        hashlib.sha256(json.dumps(m, ensure_ascii=False).encode()).digest()
        for m in messages
    )
