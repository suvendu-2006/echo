"""Token accounting for the context manager (H5-1) — measure BEFORE every call.

Claude's crown-jewel discipline (PLAN principle 4: "context is sacred"): no
provider ever receives an oversized request, so every request is counted
against the model's window before it leaves the harness. This module holds the
three accounting primitives the pipeline and the compact engine share:

- :class:`WindowRegistry` — best-effort model name → context window sizes.
  The seed table is a set of labels, not gospel; anything unknown falls back
  to ``DEFAULT_CONTEXT_WINDOW`` from the frozen provider seam
  (``app.providers.base``). Best-effort matching also handles Ollama-style
  suffixed names (``llama3.1:8b-instruct`` still maps to the ``llama3.1``
  window).
- :class:`TokenCounter` (protocol) + :class:`EstimateCounter` — the zero-dependency
  estimator: ``ceil(chars / 4)`` per text, messages + tool specs + a fixed reply
  headroom per request. A tiktoken-backed counter can replace it at the seam
  later without touching any stage (it only needs the two protocol methods).
- :func:`budget_for` — the H5-1 split of one window: system+wallet 20% /
  tool menu 15% / history 55% / result headroom 10%, as ints that sum to
  exactly the window (floors first, remainder dealt to the largest shares).

House rules carried over from the rest of the harness: no ``assert`` (a
bad window raises ``ValueError`` — survives ``python -O`` and bandit B101),
no dependencies beyond the stdlib.
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

from app.providers.base import DEFAULT_CONTEXT_WINDOW, Message, Request

#: Tokens always reserved for the model's REPLY inside one window. Counted as
#: part of every request so prompt + completion can never overflow together.
REPLY_HEADROOM_TOKENS = 512

#: Best-effort window labels for models Echo may meet (H5-1 seed). Unknown
#: models fall back to ``DEFAULT_CONTEXT_WINDOW`` — never a guess that could
#: overflow a smaller real window.
KNOWN_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "llama-3.3-70b": 128_000,
    "llama3.1": 131_072,
    "gemma3": 128_000,
    "glm-4.7": 128_000,
    "deepseek-chat": 65_536,
    "qwen2.5": 32_768,
}

#: The frozen H5-1 budget split. Keys: "system" (system prompt + wallet
#: memories), "tool_menu", "history", "result_headroom" (tool results the
#: current step will add). Values are fractions of the window.
_BUDGET_SHARES: dict[str, float] = {
    "system": 0.20,
    "tool_menu": 0.15,
    "history": 0.55,
    "result_headroom": 0.10,
}


def _json_tokens(value: object) -> str:
    """Stable JSON form of schema-ish values (counting input, never shipped)."""
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(value)  # non-serializable junk still counts, just coarser


@runtime_checkable
class TokenCounter(Protocol):
    """The counting seam (H5-1). ``EstimateCounter`` is the stdlib default."""

    def count_text(self, text: str) -> int:
        """Tokens for one piece of text."""
        ...

    def count_request(self, request: Request) -> int:
        """Tokens for a whole request: messages + tool specs + reply headroom."""
        ...


class EstimateCounter:
    """Zero-dependency token estimator: ``ceil(len(chars) / 4)``.

    Deliberately crude and deliberately consistent — the pipeline only ever
    compares estimates from THIS counter against budgets from THIS module, so
    a systematic bias cancels out. Swap in a tiktoken-backed counter at the
    seam for real models; every stage keeps working.
    """

    def count_text(self, text: str) -> int:
        """``ceil(len(text) / 4)`` — integer ceil, 0 for empty text."""
        return -(-len(text or "") // 4)

    def count_messages(self, messages: list[Message]) -> int:
        """Tokens for projected messages (roles, content, tool calls, ids).

        Extra convenience beyond the protocol — history fitting counts message
        groups WITHOUT the reply headroom (``count_request`` adds that once
        for the whole request, not once per group).
        """
        total = 0
        for message in messages:
            total += self.count_text(message.role)
            total += self.count_text(message.content or "")
            for call in message.tool_calls:
                total += self.count_text(call.name)
                total += self.count_text(_json_tokens(call.arguments))
            if message.tool_call_id:
                total += self.count_text(message.tool_call_id)
            if message.name:
                total += self.count_text(message.name)
        return total

    def count_request(self, request: Request) -> int:
        """Messages + tool specs + :data:`REPLY_HEADROOM_TOKENS` (H5-1)."""
        total = self.count_messages(request.messages)
        for spec in request.tools:
            total += self.count_text(spec.name)
            total += self.count_text(spec.description)
            total += self.count_text(_json_tokens(spec.parameters))
        return total + REPLY_HEADROOM_TOKENS


class WindowRegistry:
    """Known model → context window sizes, with a safe fallback (H5-1).

    Lookup order: exact name → case-insensitive exact → longest known label
    contained in the model name (``llama3.1:8b-instruct`` → the ``llama3.1``
    window) → the default. The substring step is best-effort by design; the
    fallback is the conservative ``DEFAULT_CONTEXT_WINDOW``, never a guess
    larger than reality.
    """

    def __init__(
        self,
        windows: dict[str, int] | None = None,
        default: int | None = None,
    ) -> None:
        self._windows: dict[str, int] = dict(KNOWN_CONTEXT_WINDOWS)
        if windows:
            self._windows.update({str(name): int(size) for name, size in windows.items()})
        self._default = int(default) if default is not None else DEFAULT_CONTEXT_WINDOW

    def window_for(self, model: str) -> int:
        """Window size for ``model``; never raises, never returns <= 0."""
        if not model:
            return self._default
        if model in self._windows:
            return self._windows[model]
        lowered = model.lower()
        for name, size in self._windows.items():
            if name.lower() == lowered:
                return size
        matches = [
            (name, size)
            for name, size in self._windows.items()
            if name.lower() in lowered
        ]
        if matches:
            # Longest label wins; alphabetical breaks ties deterministically.
            _, size = max(matches, key=lambda item: (len(item[0]), item[0]))
            return size
        return self._default

    @property
    def known(self) -> dict[str, int]:
        """A copy of the known table (for settings screens / debugging)."""
        return dict(self._windows)


def budget_for(window: int) -> dict[str, int]:
    """Split one context window per the frozen H5-1 contract.

    Returns ``{"system": .., "tool_menu": .., "history": .., "result_headroom": ..}``
    — 20% / 15% / 55% / 10% as ints summing to EXACTLY ``window`` (each share
    floored, the remainder dealt one token at a time to the largest shares).
    Raises ``ValueError`` for a non-positive window — a real exception, never
    an assert (survives ``python -O``, bandit B101).
    """
    if window <= 0:
        raise ValueError(f"context window must be positive, got {window!r}")
    shares = {name: int(window * fraction) for name, fraction in _BUDGET_SHARES.items()}
    remaining = window - sum(shares.values())
    order = sorted(_BUDGET_SHARES, key=lambda name: (-_BUDGET_SHARES[name], name))
    index = 0
    while remaining > 0:
        shares[order[index % len(order)]] += 1
        remaining -= 1
        index += 1
    return shares
