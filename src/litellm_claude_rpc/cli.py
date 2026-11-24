"""Local service commands. Doctor never submits a prompt."""

import argparse
import asyncio
import json
import os
import secrets

import uvicorn

from .app import App
from .contracts import Config, GatewayError
from .driver import ClaudeDriver, doctor
from .sessions import SessionPool


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("doctor", "serve"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4103)
    parser.add_argument("--cli-bin", default=None)
    parser.add_argument("--model", default=None, help="Claude model ID")
    parser.add_argument(
        "--model-name", default=None, help="API model alias (default: claude-default)"
    )
    parser.add_argument("--max-concurrent", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument(
        "--allow-prompt-flattening",
        action="store_true",
        help="Accept mid-conversation system/developer messages as flattened JSON text",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.port <= 65535:
            raise ValueError("Invalid port")
        base = Config.from_env()
        models = (
            {args.model_name or "claude-default": args.model}
            if (args.model is not None or args.model_name is not None)
            else base.models
        )
        raw_key = os.environ.get("LITELLM_MASTER_KEY") or os.environ.get(
            "CLAUDE_RPC_API_KEY"
        )
        generated = args.action == "serve" and not raw_key
        if generated:
            raw_key = "sk-" + secrets.token_urlsafe(32)
        config = Config(
            models=models,
            api_key=raw_key or "",
            executable=args.cli_bin or base.executable,
            timeout=args.timeout if args.timeout is not None else base.timeout,
            cleanup_timeout=base.cleanup_timeout,
            idle_ttl=base.idle_ttl,
            concurrency=args.max_concurrent
            if args.max_concurrent is not None
            else base.concurrency,
            max_sessions=base.max_sessions,
            max_pending=base.max_pending,
            queue_size=base.queue_size,
            max_input_bytes=base.max_input_bytes,
            max_output_bytes=base.max_output_bytes,
            flatten=args.allow_prompt_flattening or base.flatten,
        )
        if args.action == "doctor":
            print(json.dumps(asyncio.run(doctor(config))))
            return 0

        async def preflight():
            await doctor(config)

        app = App(
            SessionPool(
                config, lambda model, system: ClaudeDriver(config, model, system)
            ),
            preflight,
        )
        host = f"[{args.host}]" if ":" in args.host else args.host
        print(f"OpenAI base URL: http://{host}:{args.port}/v1", flush=True)
        print("Models: " + ", ".join(config.models), flush=True)
        if generated:
            print(f"Session API key: {raw_key}", flush=True)
        uvicorn.run(
            app, host=args.host, port=args.port, access_log=False, log_level="warning"
        )
        return 0
    except (GatewayError, ValueError, OSError) as exc:
        print(
            json.dumps(
                {
                    "status": "not_ready",
                    "reason": exc.code
                    if isinstance(exc, GatewayError)
                    else "invalid_config_or_cli",
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
