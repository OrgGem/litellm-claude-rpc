import asyncio
import contextlib
import threading
import uuid

from litellm import Choices, CustomLLM, Message, ModelResponse
from litellm.llms.custom_llm import CustomLLMError
from litellm.types.utils import Usage as LiteLLMUsage

from .contracts import Config, GatewayError, Request
from .responses_context import responses_request_context

_FRAMEWORK_OPTIONS = {
    "api_base",
    "custom_prompt_dict",
    "model_response",
    "print_verbose",
    "encoding",
    "api_key",
    "logging_obj",
    "optional_params",
    "acompletion",
    "litellm_params",
    "logger_fn",
    "headers",
    "timeout",
    "client",
    "stream",
    "stream_options",
}


def usage_model(usage):
    if usage is None:
        return None
    if (
        not isinstance(usage, dict)
        or type(usage.get("prompt_tokens")) is not int
        or usage["prompt_tokens"] < 0
        or type(usage.get("completion_tokens")) is not int
        or usage["completion_tokens"] < 0
        or type(usage.get("total_tokens")) is not int
        or usage["total_tokens"] < 0
    ):
        raise GatewayError("invalid_usage")
    extra = {}
    prompt_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict) and prompt_details:
        extra["prompt_tokens_details"] = dict(prompt_details)
    completion_details = usage.get("completion_tokens_details")
    if isinstance(completion_details, dict) and completion_details:
        extra["completion_tokens_details"] = dict(completion_details)
    return LiteLLMUsage(
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        **extra,
    )


class ProviderError(CustomLLMError):
    def __init__(self, error):
        self.gateway_error = error
        super().__init__(status_code=error.status, message=error.code)


class ClaudeLLM(CustomLLM):
    def __init__(self, runtime=None, *, config=None):
        super().__init__()
        self.runtime = runtime
        self.config = config
        self._owns_runtime = runtime is None
        self._lock = threading.RLock()
        self._closed = False

    def _ensure(self):
        with self._lock:
            if self._closed:
                raise GatewayError("shutting_down", 503)
            if self.runtime is not None:
                return self.runtime
            self.config = self.config or Config.from_env()
            from .driver import ClaudeDriver
            from .runtime import Runtime
            from .sessions import SessionPool

            pool = SessionPool(
                self.config,
                lambda model, system: ClaudeDriver(self.config, model, system),
            )
            self.runtime = Runtime(pool)
            return self.runtime

    def _request(
        self, model, messages, optional_params=None, request_id=None, **kwargs
    ):
        runtime = self._ensure()
        config = runtime.backend.config
        if any(
            key not in _FRAMEWORK_OPTIONS and value is not None
            for key, value in kwargs.items()
        ):
            raise GatewayError("unsupported_parameters", 400)
        if optional_params is not None and not isinstance(optional_params, dict):
            raise GatewayError("unsupported_parameters", 400)
        params = dict(optional_params or {})
        max_retries = params.pop("max_retries", 0)
        if max_retries is not None and (
            type(max_retries) is not int or max_retries != 0
        ):
            raise GatewayError("retries_not_supported", 400)
        cid = params.pop("conversation_id", None)
        headers = kwargs.get("headers") or {}
        if not isinstance(headers, dict) or any(
            not isinstance(k, str) for k in headers
        ):
            raise GatewayError("invalid_request", 400)
        header_cid = next(
            (v for k, v in headers.items() if k.lower() == "x-conversation-id"), None
        )
        if cid is None and header_cid is not None:
            cid = header_cid
        if "stream" not in params and kwargs.get("stream") is not None:
            params["stream"] = kwargs["stream"]
        if "stream_options" not in params and kwargs.get("stream_options") is not None:
            params["stream_options"] = kwargs["stream_options"]
        if "stream_options" in params and not params.get("stream"):
            # LiteLLM SDK may pass stream_options alongside streaming() without
            # setting optional_params["stream"]=True.
            options = params.pop("stream_options")
            if options is not None and (
                not isinstance(options, dict)
                or set(options) - {"include_usage"}
                or (
                    "include_usage" in options
                    and type(options["include_usage"]) is not bool
                )
            ):
                raise GatewayError("unsupported_stream_options", 400)

        alias = Request.resolve_alias(model, config.models)
        phases = None
        effective_config = config
        context = responses_request_context.get()
        if context is not None:
            if context.alias != alias or not isinstance(messages, list):
                raise GatewayError("invalid_request", 400)
            body_messages = messages
            if context.instructions is not None:
                if not messages or messages[0] != {
                    "role": "system",
                    "content": context.instructions,
                }:
                    raise GatewayError("invalid_request", 400)
                body_messages = messages[1:]
            if any(phase is not None for phase in context.phases):
                if len(body_messages) != len(context.roles) or any(
                    not isinstance(item, dict) or item.get("role") != role
                    for item, role in zip(body_messages, context.roles)
                ):
                    raise GatewayError("invalid_request", 400)
                prefix_len = 1 if context.instructions is not None else 0
                phases = (None,) * prefix_len + tuple(context.phases)

        return Request.parse(
            {"model": alias, "messages": messages, **params},
            cid,
            effective_config,
            phases=phases,
            responses_context=context,
        )

    @staticmethod
    def _record_context_usage(request, terminal):
        if request.responses_context is not None and terminal is not None:
            known = terminal.usage is not None
            request.responses_context.usage_known = known
            request.responses_context.usage = terminal.usage if known else None

    def events(self, request):
        runtime = self._ensure()
        job = runtime.submit(runtime.start(request)).result()
        try:
            while (event := runtime.submit(runtime.next(job)).result()) is not None:
                yield event
        finally:
            runtime.release(job).result()

    async def aevents(self, request):
        runtime = await asyncio.to_thread(self._ensure)
        # If cancelled while start is completing, recover and cancel that job too.
        start = runtime.submit(runtime.start(request))
        try:
            job = await asyncio.shield(asyncio.wrap_future(start))
        except asyncio.CancelledError:
            job = await asyncio.wrap_future(start)
            await asyncio.wrap_future(runtime.release(job))
            raise
        try:
            while (
                event := await asyncio.wrap_future(runtime.submit(runtime.next(job)))
            ) is not None:
                yield event
        finally:
            await asyncio.wrap_future(runtime.release(job))

    @staticmethod
    def response(model, text, terminal):
        finish_reason = (
            terminal.finish_reason if terminal is not None else None
        ) or "stop"
        result = ModelResponse(
            id="chatcmpl-" + uuid.uuid4().hex,
            model=model,
            choices=[
                Choices(
                    index=0,
                    finish_reason=finish_reason,
                    message=Message(role="assistant", content=text),
                )
            ],
        )
        parsed_usage = usage_model(terminal.usage)
        if parsed_usage is not None:
            result.usage = parsed_usage
        elif hasattr(result, "usage"):
            delattr(result, "usage")
        return result

    def completion(self, model, messages, *args, optional_params=None, **kwargs):
        try:
            request = self._request(model, messages, optional_params, **kwargs)
            text, terminal = [], None
            with contextlib.closing(self.events(request)) as events:
                for event in events:
                    if event.done:
                        terminal = event
                    else:
                        text.append(event.text)
            self._record_context_usage(request, terminal)
            return self.response(model, "".join(text), terminal)
        except GatewayError as exc:
            raise ProviderError(exc) from None

    async def acompletion(self, model, messages, *args, optional_params=None, **kwargs):
        try:
            request = await asyncio.to_thread(
                self._request, model, messages, optional_params, **kwargs
            )
            text, terminal = [], None
            async with contextlib.aclosing(self.aevents(request)) as events:
                async for event in events:
                    if event.done:
                        terminal = event
                    else:
                        text.append(event.text)
            self._record_context_usage(request, terminal)
            return self.response(model, "".join(text), terminal)
        except GatewayError as exc:
            raise ProviderError(exc) from None

    @staticmethod
    def chunk(event):
        parsed = usage_model(event.usage)
        return {
            "text": event.text,
            "is_finished": event.done,
            "finish_reason": (event.finish_reason or "stop") if event.done else None,
            "index": 0,
            "tool_use": None,
            "usage": parsed.model_dump(exclude_none=True)
            if parsed is not None
            else None,
        }

    def _streaming(self, model, messages, *args, optional_params=None, **kwargs):
        try:
            request = self._request(model, messages, optional_params, **kwargs)
            with contextlib.closing(self.events(request)) as events:
                for event in events:
                    if event.done:
                        self._record_context_usage(request, event)
                    yield self.chunk(event)
        except GatewayError as exc:
            raise ProviderError(exc) from None

    async def _astreaming(self, model, messages, *args, optional_params=None, **kwargs):
        try:
            request = await asyncio.to_thread(
                self._request, model, messages, optional_params, **kwargs
            )
            async with contextlib.aclosing(self.aevents(request)) as events:
                async for event in events:
                    if event.done:
                        self._record_context_usage(request, event)
                    yield self.chunk(event)
        except GatewayError as exc:
            raise ProviderError(exc) from None

    def streaming(self, *args, **kwargs):
        self._suppress_estimated_usage(kwargs)
        return self._streaming(*args, **kwargs)

    def astreaming(self, *args, **kwargs):
        self._suppress_estimated_usage(kwargs)
        return self._astreaming(*args, **kwargs)

    @staticmethod
    def _suppress_estimated_usage(kwargs):
        # LiteLLM 1.109 synthesizes a token estimate on stream exhaustion when
        # include_usage is true. Disable that wrapper feature before it is built.
        logging_obj = kwargs.get("logging_obj")
        if logging_obj is not None:
            logging_obj.stream_options = {"include_usage": False}

    def close(self):
        with self._lock:
            self._closed = True
            runtime = self.runtime if self._owns_runtime else None
            if self._owns_runtime:
                self.runtime = None
        if runtime is not None:
            runtime.close()

    async def aclose(self):
        await asyncio.to_thread(self.close)


ClaudeRPCProvider = ClaudeLLM
claude_rpc_provider = ClaudeLLM()
