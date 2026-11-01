"""The before/after hook contract (H4-3) — callbacks as gates (ADK pattern).

The engine calls every hook synchronously (no awaits inside ``decide``), so one
decision is atomic on the event loop and the engine's scratch attributes are
race-free for a single-decision flow.

BeforeHook — ``hook(call, danger, ctx)`` and the return value decides:

- ``None``            no rewrite; evaluation continues.
- ``dict``            REPLACES ``call.arguments`` on the ToolCall (rewrite in
  place). Later hooks — and the deny/ask/allow rules — see the rewritten
  arguments, because hooks run before every rule (H4-2 order).
- ``allow_result(fake)`` (a ``BlockResult`` marker) — block the call and record
  the wrapped dict as a fake result on ``engine.last_block_result``; the
  dispatcher consults that seam and returns it WITHOUT running the tool.
- falsy (``False``, ``0``, ``""``) — block the call (plain deny, no fake result).
- raise — block the call: fail-closed DENY with the error logged.
- truthy non-dict (``True``, a stray coroutine from an async hook, ...) —
  misbehaving hook: fail-closed DENY with the error logged.

AfterHook — ``hook(call, danger, ctx, result) -> None``: observes the permission
decision (``result`` = ``{"decision": ..., "reason": ..., "tool": ...}``).
Return values are ignored and exceptions are logged + swallowed: an after-hook
can never change a decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.harness.interfaces import Danger, HarnessContext
from app.providers.base import ToolCall


@dataclass(frozen=True)
class BlockResult:
    """Marker a before-hook returns to block and substitute a fake result."""

    result: dict


def allow_result(result: dict) -> BlockResult:
    """The ADK-style block-and-return-fake-result helper (H4-3).

    A before-hook returns ``allow_result({...})`` to BLOCK the call while
    handing back a clean fake result the dispatcher may return without running
    the tool. (The name follows the ADK convention: this is the result the
    engine "allows through" instead of the real tool output.) The engine copies
    the dict, so later mutation of the caller's object cannot leak in.
    """
    return BlockResult(result=dict(result))


@runtime_checkable
class BeforeHook(Protocol):
    """``(call, danger, ctx) -> dict | BlockResult | None`` — see module doc."""

    def __call__(
        self, call: ToolCall, danger: Danger, ctx: HarnessContext
    ) -> "dict | BlockResult | None": ...


@runtime_checkable
class AfterHook(Protocol):
    """``(call, danger, ctx, result: dict) -> None`` — observes, never decides."""

    def __call__(self, call: ToolCall, danger: Danger, ctx: HarnessContext, result: dict) -> None: ...
