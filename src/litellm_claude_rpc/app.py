"""Authenticated ASGI host; commit a reusable turn only after HTTP delivery."""

import asyncio
import contextlib
import hmac
import json
import time
import uuid

from .contracts import GatewayError, Request, strict_json
from .responses_guard import (
    input_to_messages,
    responses_usage_dict,
    validate_responses_payload,
)


def completion(model, text, usage, request_id, finish_reason="stop"):
    result = {
        "id": "chatcmpl-" + request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        result["usage"] = usage
    return result


def _responses_completion_state(finish_reason):
    if finish_reason == "length":
        return "incomplete", {"reason": "max_output_tokens"}, "response.incomplete"
    if finish_reason == "content_filter":
        return "incomplete", {"reason": "content_filter"}, "response.incomplete"
    return "completed", None, "response.completed"


def _response_object(
    response_id,
    created_at,
    model,
    status,
    output,
    *,
    instructions=None,
    parallel_tool_calls=True,
    tool_choice="auto",
    usage=None,
    error=None,
    incomplete_details=None,
):
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "error": error,
        "incomplete_details": incomplete_details,
        "instructions": instructions,
        "model": model,
        "output": output,
        "parallel_tool_calls": parallel_tool_calls,
        "tool_choice": tool_choice,
        "tools": [],
        "usage": usage,
    }


def _output_message(item_id, text, status="completed"):
    return {
        "id": item_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


class App:
    def __init__(self, pool, preflight=None):
        self.pool, self.config, self.preflight = pool, pool.config, preflight
        key = self.config.api_key
        if (
            not isinstance(key, str)
            or len(key) < 24
            or not key.isascii()
            or any(c.isspace() for c in key)
        ):
            raise ValueError("Set a 24+ character ASCII gateway key")
        self.ready = False

    async def _run_with_disconnect(self, receive, coro):
        async def disconnected():
            while (await receive())["type"] != "http.disconnect":
                pass

        task, watcher = (
            asyncio.create_task(coro),
            asyncio.create_task(disconnected()),
        )
        try:
            done, _ = await asyncio.wait(
                {task, watcher}, return_when=asyncio.FIRST_COMPLETED
            )
            if task in done:
                await task
            else:
                task.cancel()
        finally:
            task.cancel()
            watcher.cancel()
            await asyncio.gather(task, watcher, return_exceptions=True)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                event = await receive()
                if event["type"] == "lifespan.startup":
                    try:
                        if self.preflight:
                            await self.preflight()
                        self.ready = True
                        await send({"type": "lifespan.startup.complete"})
                    except Exception:
                        await self.pool.close()
                        await send(
                            {
                                "type": "lifespan.startup.failed",
                                "message": "Backend readiness failed; run doctor",
                            }
                        )
                        return
                elif event["type"] == "lifespan.shutdown":
                    self.ready = False
                    await self.pool.close()
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        started = False
        responses_sse = None
        request_id = uuid.uuid4().hex

        async def bounded_send(message):
            nonlocal started
            async with asyncio.timeout(10):
                await send(message)
            if message["type"] == "http.response.start":
                started = True

        async def respond(status, body):
            encoded = json.dumps(body, ensure_ascii=True, allow_nan=False).encode()
            await bounded_send(
                {
                    "type": "http.response.start",
                    "status": status,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"x-request-id", request_id.encode()),
                    ],
                }
            )
            await bounded_send({"type": "http.response.body", "body": encoded})

        try:
            auth = [
                v for k, v in scope.get("headers", []) if k.lower() == b"authorization"
            ]
            if len(auth) != 1 or not hmac.compare_digest(
                auth[0], ("Bearer " + self.config.api_key).encode()
            ):
                raise GatewayError("invalid_api_key", 401)
            if not self.ready:
                raise GatewayError("not_ready", 503)
            path, method = scope["path"], scope["method"]
            if method == "GET" and path == "/v1/models":
                return await respond(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": alias,
                                "object": "model",
                                "created": 0,
                                "owned_by": "agent-cli",
                            }
                            for alias in self.config.models
                        ],
                    },
                )
            if method == "GET" and path == "/health/ready":
                return await respond(
                    200,
                    {
                        "status": "ready",
                        "model_access_verified": False,
                        "role_semantics": "text-transcript",
                        "session_reuse": "X-Conversation-ID",
                    },
                )
            if method != "POST" or path not in (
                "/v1/chat/completions",
                "/v1/responses",
            ):
                raise GatewayError("not_found", 404)
            ids = [
                v
                for k, v in scope.get("headers", [])
                if k.lower() == b"x-conversation-id"
            ]
            if len(ids) > 1:
                raise GatewayError("invalid_conversation_id", 400)
            cid = ids[0].decode("ascii") if ids else None
            body = bytearray()
            async with asyncio.timeout(10):
                while True:
                    event = await receive()
                    if event["type"] == "http.disconnect":
                        return
                    body.extend(event.get("body", b""))
                    if len(body) > self.config.max_input_bytes:
                        raise GatewayError("request_too_large", 413)
                    if not event.get("more_body", False):
                        break
            try:
                payload = strict_json(body.decode("utf-8"))
            except (ValueError, UnicodeError):
                raise GatewayError("invalid_json", 400) from None

            if path == "/v1/responses":
                if cid is not None:
                    raise GatewayError("invalid_conversation_id", 400)
                validate_responses_payload(payload)
                alias = Request.resolve_alias(payload["model"], self.config.models)
                instructions = payload.get("instructions") or None
                messages, phases = input_to_messages(
                    payload["input"], instructions=instructions
                )
                stream = bool(payload.get("stream", False))
                request = Request.parse(
                    {"model": alias, "messages": messages, "stream": stream},
                    None,
                    self.config,
                    phases=phases,
                )
                parallel = payload.get("parallel_tool_calls")
                if parallel is None:
                    parallel = True
                tool_choice = payload.get("tool_choice") or "auto"
                if stream:
                    responses_sse = {
                        "id": "resp_" + request_id,
                        "created_at": int(time.time()),
                        "model": alias,
                        "instructions": instructions,
                        "parallel_tool_calls": parallel,
                        "tool_choice": tool_choice,
                        "seq": 0,
                    }

                async def generate_responses():
                    async with self.pool.turn(request) as turn:
                        if not stream:
                            async for _ in turn:
                                pass
                            resp_status, incomplete_details, _ = (
                                _responses_completion_state(
                                    turn.terminal.finish_reason
                                )
                            )
                            response_body = _response_object(
                                "resp_" + request_id,
                                int(time.time()),
                                alias,
                                resp_status,
                                [
                                    _output_message(
                                        "msg_" + request_id,
                                        "".join(turn.text),
                                        status=resp_status,
                                    )
                                ],
                                instructions=instructions,
                                parallel_tool_calls=parallel,
                                tool_choice=tool_choice,
                                usage=responses_usage_dict(turn.terminal.usage),
                                incomplete_details=incomplete_details,
                            )
                            await respond(200, response_body)
                        else:
                            await self._stream_responses(
                                turn, responses_sse, bounded_send
                            )
                        turn.commit()

                return await self._run_with_disconnect(receive, generate_responses())

            request = Request.parse(payload, cid, self.config)

            async def generate():
                async with self.pool.turn(request) as turn:
                    if not request.stream:
                        async for _ in turn:
                            pass
                        await respond(
                            200,
                            completion(
                                request.model,
                                "".join(turn.text),
                                turn.terminal.usage,
                                request_id,
                                turn.terminal.finish_reason or "stop",
                            ),
                        )
                    else:
                        first = await anext(turn)
                        await bounded_send(
                            {
                                "type": "http.response.start",
                                "status": 200,
                                "headers": [
                                    (b"content-type", b"text/event-stream"),
                                    (b"cache-control", b"no-cache"),
                                    (b"x-accel-buffering", b"no"),
                                ],
                            }
                        )

                        async def emit(choices, usage=None):
                            chunk = {
                                "id": "chatcmpl-" + request_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": choices,
                            }
                            if request.include_usage:
                                chunk["usage"] = usage
                            await bounded_send(
                                {
                                    "type": "http.response.body",
                                    "body": (
                                        "data: " + json.dumps(chunk) + "\n\n"
                                    ).encode(),
                                    "more_body": True,
                                }
                            )

                        await emit(
                            [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant"},
                                    "finish_reason": None,
                                }
                            ]
                        )
                        event = first
                        while event is not None:
                            if not event.done:
                                await emit(
                                    [
                                        {
                                            "index": 0,
                                            "delta": {"content": event.text},
                                            "finish_reason": None,
                                        }
                                    ]
                                )
                            event = await anext(turn, None)
                        await emit(
                            [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": turn.terminal.finish_reason
                                    or "stop",
                                }
                            ]
                        )
                        if request.include_usage:
                            await emit([], turn.terminal.usage)
                        await bounded_send(
                            {
                                "type": "http.response.body",
                                "body": b"data: [DONE]\n\n",
                                "more_body": False,
                            }
                        )
                    turn.commit()

            await self._run_with_disconnect(receive, generate())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = (
                exc
                if isinstance(exc, GatewayError)
                else GatewayError("internal_error", 500)
            )
            if isinstance(exc, UnicodeError):
                error = GatewayError("invalid_request", 400)
            with contextlib.suppress(OSError, TimeoutError):
                if not started:
                    await respond(error.status, error.payload())
                elif responses_sse is not None:
                    await self._emit_responses_failed(
                        responses_sse, error.code, bounded_send
                    )
                else:
                    await bounded_send(
                        {
                            "type": "http.response.body",
                            "body": (
                                "data: " + json.dumps(error.payload()) + "\n\n"
                            ).encode(),
                            "more_body": False,
                        }
                    )

    async def _emit_responses_failed(self, meta, code, send):
        seq = meta["seq"]
        meta["seq"] += 1
        failed = _response_object(
            meta["id"],
            meta["created_at"],
            meta["model"],
            "failed",
            [],
            instructions=meta["instructions"],
            parallel_tool_calls=meta["parallel_tool_calls"],
            tool_choice=meta["tool_choice"],
            error={"code": "server_error", "message": code},
        )
        payload = {
            "type": "response.failed",
            "sequence_number": seq,
            "response": failed,
        }
        await send(
            {
                "type": "http.response.body",
                "body": (
                    "event: response.failed\ndata: "
                    + json.dumps(payload, ensure_ascii=True)
                    + "\n\n"
                ).encode(),
                "more_body": False,
            }
        )

    async def _stream_responses(self, turn, meta, send):
        first = await anext(turn)
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/event-stream"),
                    (b"cache-control", b"no-cache"),
                    (b"x-accel-buffering", b"no"),
                    (b"x-request-id", meta["id"].removeprefix("resp_").encode()),
                ],
            }
        )
        item_id = "msg_" + meta["id"].removeprefix("resp_")

        async def send_event(event_type, body, more_body=True):
            seq = meta["seq"]
            meta["seq"] += 1
            payload = {"type": event_type, "sequence_number": seq, **body}
            await send(
                {
                    "type": "http.response.body",
                    "body": (
                        f"event: {event_type}\ndata: "
                        + json.dumps(payload, ensure_ascii=True)
                        + "\n\n"
                    ).encode(),
                    "more_body": more_body,
                }
            )

        in_progress = _response_object(
            meta["id"],
            meta["created_at"],
            meta["model"],
            "in_progress",
            [],
            instructions=meta["instructions"],
            parallel_tool_calls=meta["parallel_tool_calls"],
            tool_choice=meta["tool_choice"],
        )
        await send_event("response.created", {"response": in_progress})
        await send_event("response.in_progress", {"response": in_progress})
        await send_event(
            "response.output_item.added",
            {
                "output_index": 0,
                "item": {
                    "id": item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            },
        )
        await send_event(
            "response.content_part.added",
            {
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        )

        event = first
        while event is not None:
            if not event.done and event.text:
                await send_event(
                    "response.output_text.delta",
                    {
                        "item_id": item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": event.text,
                        "logprobs": [],
                    },
                )
            event = await anext(turn, None)

        full_text = "".join(turn.text)
        await send_event(
            "response.output_text.done",
            {
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "text": full_text,
                "logprobs": [],
            },
        )
        await send_event(
            "response.content_part.done",
            {
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": full_text, "annotations": []},
            },
        )
        resp_status, incomplete_details, terminal_event = _responses_completion_state(
            turn.terminal.finish_reason
        )
        completed_item = _output_message(item_id, full_text, status=resp_status)
        await send_event(
            "response.output_item.done",
            {"output_index": 0, "item": completed_item},
        )
        completed_response = _response_object(
            meta["id"],
            meta["created_at"],
            meta["model"],
            resp_status,
            [completed_item],
            instructions=meta["instructions"],
            parallel_tool_calls=meta["parallel_tool_calls"],
            tool_choice=meta["tool_choice"],
            usage=responses_usage_dict(turn.terminal.usage),
            incomplete_details=incomplete_details,
        )
        await send_event(
            terminal_event,
            {"response": completed_response},
            more_body=False,
        )
