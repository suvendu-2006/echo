"""Manual gate for Phase 1 (TESTS.md T1-07): chat with any OpenAI-compatible
brain through the universal adapter — the PLAN 1.1 gate, human-run.

Needs a real free-tier key in the environment or .env (D-013 — never a flag
value in shell history you care about):

    # Groq (free tier): https://console.groq.com
    GROQ_API_KEY=gsk_...  python -m app.chat_demo \
        --base-url https://api.groq.com/openai/v1 \
        --model llama-3.3-70b-versatile --key-env GROQ_API_KEY "hello there"

    # Or local Ollama (no key at all):
    python -m app.chat_demo --base-url http://localhost:11434/v1 \
        --model gemma3 --name ollama "hello there"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_env  # noqa: E402
from app.providers.base import (  # noqa: E402
    Message,
    ProviderError,
    Request,
)
from app.providers.openai_compat import OpenAICompatProvider  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with any OpenAI-compatible brain.")
    parser.add_argument("--base-url", required=True, help="e.g. https://api.groq.com/openai/v1")
    parser.add_argument("--model", required=True, help="model id, e.g. llama-3.3-70b-versatile")
    parser.add_argument("--key-env", default=None, help="env var holding the API key (optional)")
    parser.add_argument("--name", default="cli", help="provider label for errors")
    parser.add_argument("prompt", nargs="+", help="the message to send")
    return parser.parse_args()


async def main() -> int:
    load_env()  # .env dev secrets (D-013); real environment variables win
    args = parse_args()
    api_key = os.environ.get(args.key_env) if args.key_env else None
    if args.key_env and not api_key:
        print(f"missing key: set {args.key_env} in the environment or .env")
        return 1

    provider = OpenAICompatProvider(
        name=args.name, model=args.model, base_url=args.base_url, api_key=api_key
    )
    request = Request(messages=[Message(role="user", content=" ".join(args.prompt))])
    try:
        async for event in provider.stream(request):
            if event.type == "text":
                print(event.text, end="", flush=True)
            elif event.type == "usage" and event.usage:
                print(
                    f"\n[tokens: {event.usage.prompt_tokens} in / "
                    f"{event.usage.completion_tokens} out]"
                )
            elif event.type == "error":
                print(f"\n[error] {event.error}")
                return 1
    except ProviderError as exc:
        print(f"\n[provider error] {exc}")
        return 1
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
