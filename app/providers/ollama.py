"""Ollama auto-discovery (PLAN Phase 1.3) — the local brains, found live.

Ollama publishes its pulled models at ``GET {base}/api/tags`` (default
port 11434). This module answers the three questions the rest of Echo
asks about the local runtime:

    list_ollama_models()          ->  ["llama3:latest", "gemma3:12b", ...]
    ollama_running()              ->  True iff Ollama is up with a model pulled
    ollama_compat_provider(name)  ->  the universal OpenAI-compatible adapter
                                      (PLAN step 1.2, DECISIONS.md D-002)
                                      aimed at Ollama's OpenAI-compatible ``/v1``

Law of this module: it NEVER raises. A machine without Ollama running is
normal life, not an error — every discovery failure (connection refused,
timeout, junk body) collapses to [] or False. The wire is injectable
(``fetch``) so tests run fully offline: zero sockets, zero localhost.
"""

from __future__ import annotations

import json
from typing import Awaitable, Callable

#: Where Ollama listens by default (PLAN step 1.3).
DEFAULT_BASE_URL = "http://localhost:11434"

#: The injectable wire: ``async def fetch(url, timeout) -> bytes | str``.
#: The default below speaks httpx; tests swap in fakes and touch no socket.
Fetcher = Callable[[str, float], Awaitable[bytes | str]]


# ---------------------------------------------------------------- discovery


async def _default_fetch(url: str, timeout: float) -> bytes | str:
    """The real wire — httpx, imported lazily (same habit as app.config's
    keyring import: the module stays importable on a stdlib-only box)."""
    import httpx  # lazy: keeps the stdlib-only test suite hermetic

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()  # a 404/500 is just another "no Ollama"
        return response.content


async def list_ollama_models(
    base_url: str = DEFAULT_BASE_URL,
    fetch: Fetcher | None = None,
    timeout: float = 1.5,
) -> list[str]:
    """Live list of pulled model names, in server order — [] when silent.

    Tolerant by contract: a missing "models" key, entries without a
    "name", or a body that isn't JSON all degrade to [] (entries that do
    parse still survive when only some are malformed). Connection and
    timeout errors mean Ollama simply isn't running — [] again, never an
    exception.
    """
    do_fetch = fetch or _default_fetch
    root = base_url.rstrip("/")
    try:
        body = await do_fetch(f"{root}/api/tags", timeout)
        payload = json.loads(body)
    except Exception:
        # Anything wrong on the wire or in the body means "no Ollama to
        # see here" — an expected state, never a crash.
        return []
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return []
    return [
        entry["name"]
        for entry in models
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    ]


async def ollama_running(
    base_url: str = DEFAULT_BASE_URL,
    fetch: Fetcher | None = None,
) -> bool:
    """True iff Ollama is reachable AND has at least one model pulled."""
    return bool(await list_ollama_models(base_url, fetch=fetch))


# ---------------------------------------------------- the universal bridge


def ollama_compat_provider(
    model: str,
    base_url: str = DEFAULT_BASE_URL,
) -> "OpenAICompatProvider":
    """Point the universal OpenAI-compatible adapter (PLAN 1.2) at Ollama.

    The sibling adapter is imported lazily inside this body: discovery
    must work even while that module is being built or refactored, and
    Echo's import graph stays light until a provider is actually needed.
    Ollama ignores API keys, so a fixed placeholder satisfies the
    universal constructor without any real secret existing (D-013).
    """
    from app.providers.openai_compat import OpenAICompatProvider

    root = base_url.rstrip("/")
    return OpenAICompatProvider(
        name="ollama",
        model=model,
        base_url=f"{root}/v1",
        api_key="ollama",
    )
