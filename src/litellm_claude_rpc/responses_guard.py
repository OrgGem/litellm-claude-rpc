"""Validate Responses requests before LiteLLM maps them to chat completions."""

from litellm.integrations.custom_logger import CustomLogger

from .contracts import Config, GatewayError
from .provider import ProviderError, claude_rpc_provider
from .responses_context import ResponsesRequestContext, responses_request_context

SUPPORTED_RESPONSES_KEYS = {
    "model",
    "input",
    "instructions",
    "stream",
    "store",
    "litellm_metadata",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "text",
    "background",
    "truncation",
}


def validate_input(value):
    if isinstance(value, str):
        return
    if not isinstance(value, list) or not value:
        raise GatewayError("invalid_messages", 400)
    for message in value:
        if (
            not isinstance(message, dict)
            or set(message) - {"type", "role", "content", "id", "status", "phase"}
            or message.get("type", "message") != "message"
            or not isinstance(message.get("role"), str)
            or message.get("role") not in {"user", "assistant", "system", "developer"}
            or ("id" in message and not isinstance(message["id"], str))
            or message.get("status", "completed") != "completed"
            or message.get("phase") not in (None, "commentary", "final_answer")
            or (message.get("phase") is not None and message.get("role") != "assistant")
        ):
            raise GatewayError("invalid_message", 400)
        content = message.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list) or not content:
            raise GatewayError("text_only", 400)
        accepted_types = (
            {"input_text", "output_text"}
            if message["role"] == "assistant"
            else {"input_text"}
        )
        for part in content:
            if (
                not isinstance(part, dict)
                or set(part) - {"type", "text", "annotations", "logprobs"}
                or not isinstance(part.get("type"), str)
                or part.get("type") not in accepted_types
                or not isinstance(part.get("text"), str)
                or part.get("annotations", []) != []
                or (
                    part.get("logprobs") is not None
                    and not isinstance(part["logprobs"], list)
                )
            ):
                raise GatewayError("text_only", 400)


def validate_responses_payload(payload):
    if not isinstance(payload, dict):
        raise GatewayError("invalid_request", 400)
    if any(
        key not in SUPPORTED_RESPONSES_KEYS and value is not None
        for key, value in payload.items()
    ):
        raise GatewayError("unsupported_parameters", 400)
    if not isinstance(payload.get("model"), str) or not payload["model"]:
        raise GatewayError("unknown_model", 400)
    if payload.get("store") is not None and payload["store"] is not False:
        raise GatewayError("unsupported_parameters", 400)
    if payload.get("tools") not in (None, []):
        raise GatewayError("unsupported_parameters", 400)
    if payload.get("tool_choice") not in (None, "auto", "none"):
        raise GatewayError("unsupported_parameters", 400)
    if (
        payload.get("parallel_tool_calls") is not None
        and type(payload["parallel_tool_calls"]) is not bool
    ):
        raise GatewayError("invalid_request", 400)
    if payload.get("text") not in (None, {}, {"format": {"type": "text"}}):
        raise GatewayError("unsupported_parameters", 400)
    if payload.get("background") is not None and payload["background"] is not False:
        raise GatewayError("unsupported_parameters", 400)
    if payload.get("truncation") not in (None, "disabled"):
        raise GatewayError("unsupported_parameters", 400)
    if payload.get("stream") is not None and type(payload["stream"]) is not bool:
        raise GatewayError("invalid_stream", 400)
    if payload.get("instructions") is not None and not isinstance(
        payload["instructions"], str
    ):
        raise GatewayError("invalid_request", 400)
    validate_input(payload.get("input"))


def input_to_messages(value, instructions=None):
    messages = []
    phases = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
        phases.append(None)
    if isinstance(value, str):
        messages.append({"role": "user", "content": value})
        phases.append(None)
    else:
        for item in value:
            content = item["content"]
            if isinstance(content, list):
                content = "".join(part["text"] for part in content)
            messages.append({"role": item["role"], "content": content})
            phases.append(item.get("phase"))
    return messages, tuple(phases)


def responses_usage_dict(usage):
    if not isinstance(usage, dict):
        return None
    prompt_details = dict(usage.get("prompt_tokens_details") or {})
    completion_details = dict(usage.get("completion_tokens_details") or {})
    return {
        "input_tokens": usage["prompt_tokens"],
        "output_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "input_tokens_details": prompt_details,
        "output_tokens_details": completion_details,
    }


def preserve_usage(response, context):
    if context is None or context.usage_known is None:
        return
    if context.usage_known is False or context.usage is None:
        if isinstance(response, dict):
            response["usage"] = None
        else:
            response.usage = None
        return

    src_prompt = context.usage.get("prompt_tokens_details") or {}
    src_completion = context.usage.get("completion_tokens_details") or {}
    if isinstance(response, dict):
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return
        input_details = usage.get("input_tokens_details")
        output_details = usage.get("output_tokens_details")
        if isinstance(input_details, dict) and "cached_tokens" not in src_prompt:
            input_details.pop("cached_tokens", None)
        if "cache_write_tokens" in src_prompt:
            if not isinstance(input_details, dict):
                input_details = {}
                usage["input_tokens_details"] = input_details
            input_details["cache_write_tokens"] = src_prompt["cache_write_tokens"]
        if (
            isinstance(output_details, dict)
            and "reasoning_tokens" not in src_completion
        ):
            output_details.pop("reasoning_tokens", None)
    else:
        usage = response.usage
        if usage is None:
            return
        if usage.input_tokens_details is not None:
            if "cached_tokens" not in src_prompt:
                usage.input_tokens_details.cached_tokens = None
            if "cache_write_tokens" in src_prompt:
                usage.input_tokens_details.cache_write_tokens = src_prompt[
                    "cache_write_tokens"
                ]
        if (
            usage.output_tokens_details is not None
            and "reasoning_tokens" not in src_completion
        ):
            usage.output_tokens_details.reasoning_tokens = None


class ClaudeResponsesGuard(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        responses_request_context.set(None)
        if call_type != "aresponses":
            return data
        try:
            runtime = claude_rpc_provider.runtime
            config = (
                runtime.backend.config
                if runtime is not None
                else claude_rpc_provider.config or Config.from_env()
            )
        except (GatewayError, ValueError):
            return data
        model = data.get("model")
        if not isinstance(model, str):
            return data
        alias = model.removeprefix("claude_rpc/").removeprefix("claude-rpc/")
        if alias not in config.models:
            return data

        snapshot = data.get("proxy_server_request", {}).get("body")
        if not isinstance(snapshot, dict):
            raise ProviderError(GatewayError("invalid_request", 400))
        try:
            validate_responses_payload(snapshot)
            validate_input(data.get("input"))
        except GatewayError as error:
            raise ProviderError(error) from None

        input_items = data["input"]
        phases = (
            tuple(item.get("phase") for item in input_items)
            if isinstance(input_items, list)
            else ()
        )
        roles = (
            tuple(item["role"] for item in input_items)
            if isinstance(input_items, list)
            else ()
        )
        responses_request_context.set(
            ResponsesRequestContext(
                alias, data.get("instructions") or None, phases, roles
            )
        )
        for key in (
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "text",
            "background",
            "truncation",
        ):
            data.pop(key, None)
        data["store"] = False
        return data

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        context = responses_request_context.get()
        if isinstance(response, dict) and response.get("object") == "response":
            preserve_usage(response, context)
        elif getattr(response, "object", None) == "response":
            preserve_usage(response, context)
        return response

    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict, response, request_data
    ):
        context = responses_request_context.get()
        async for event in response:
            if isinstance(event, dict) and event.get("type") in (
                "response.completed",
                "response.incomplete",
            ):
                preserve_usage(event["response"], context)
            elif getattr(event, "type", None) in (
                "response.completed",
                "response.incomplete",
            ):
                preserve_usage(event.response, context)
            yield event


claude_rpc_responses_guard = ClaudeResponsesGuard()
