"""Failover chain (PLAN step 1.5) — a provider dies, the conversation continues.

Wraps an ordered list of adapters (first = preferred) behind the same
``stream(request)`` seam as a single adapter (app/providers/base.py), so the
harness loop can talk to ``ProviderChain.stream`` without knowing failover
exists.

THE FAILOVER CONTRACT (recorded here for DECISIONS.md D-015):
    * Try the active adapter. If it RAISES ProviderUnavailable (or
      ConnectionError/OSError) BEFORE yielding any event -> silently move to
      the next adapter and restart the current step from scratch (call
      ``on_switch``). This is the only allowed switch moment: the plan
      explicitly forbids hot-swapping providers mid-stream.
    * If a provider fails AFTER it already yielded at least one event
      (mid-stream death) -> DO NOT switch; yield
      ``StreamEvent(type="error", error=<message>)`` and stop. Content
      already reached the user; a silent restart would duplicate or
      contradict it.
    * Any other ProviderError before content (e.g. config/auth) also triggers
      trying the next adapter — but record the reason in the switch callback.
    * Forward all events of the successful adapter unchanged and in order
      (including its end event). Exactly one end, or one error, total.
    * If every adapter is exhausted -> yield ``StreamEvent(type="error",
      error=<summary of each failure>)`` and stop (never raise out of
      ``stream()``).

Behaviour this module adds on top of the contract above:
    * STICKY takeover: the active position persists across ``stream()`` calls
      — once adapter N has taken over, later calls start at N, never back at
      the head of the list (a flapping preferred provider must not hijack
      every second request).
    * A yielded ``type="error"`` event is the adapter's own failure report
      (the base contract allows a single "error" instead of a raise). It is
      forwarded unchanged and is terminal — never a failover signal.
    * ``stream()`` never raises out of itself for adapter failures: even an
      unexpected non-provider exception is treated like any other failure
      (fail over before content, error event after content). Async
      cancellation (CancelledError) is not an adapter failure and passes
      through untouched.

``on_switch(from_adapter, to_adapter, reason)`` is an optional sync callback
for logging/UI. It fires exactly when the active position moves — including
moves that later adapters do not survive. No switch is ever silent *to the
callback*; "silently" in the contract only means no error event reaches the
user for the dead provider that was never heard from.
"""

from __future__ import annotations

from typing import AsyncIterator, Callable, Sequence

from app.providers.base import (
    ProviderAdapter,
    ProviderConfigError,
    ProviderError,
    ProviderUnavailable,
    Request,
    StreamEvent,
)


# ------------------------------------------------------------------- helpers


def _failure_reason(exc: Exception) -> str:
    """One-line human-readable reason for a failed adapter attempt.

    Ordered most-specific first: the unavailable/config subclasses of
    ProviderError must be recognised before the base class.
    """
    if isinstance(exc, ProviderUnavailable):
        return f"unavailable: {exc}"
    if isinstance(exc, ProviderConfigError):
        return f"config error: {exc}"
    if isinstance(exc, ProviderError):
        return f"provider error: {exc}"
    if isinstance(exc, (ConnectionError, OSError)):
        return f"connection failed: {exc}"
    return f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------- the chain


class ProviderChain:
    """Ordered adapters, first = preferred; dead head -> silent takeover.

    Implements the failover contract (see module docstring). ``.active`` is
    the currently preferred adapter and is STICKY across calls.
    """

    def __init__(
        self,
        adapters: Sequence[ProviderAdapter],
        on_switch: Callable[[ProviderAdapter, ProviderAdapter, str], None] | None = None,
    ) -> None:
        if not adapters:
            raise ValueError("ProviderChain needs at least one adapter")
        self._adapters: list[ProviderAdapter] = list(adapters)
        self._on_switch = on_switch
        self._index = 0  # sticky: only ever moves forward

    @property
    def active(self) -> ProviderAdapter:
        """The currently preferred adapter (sticky across stream() calls)."""
        return self._adapters[self._index]

    def descriptions(self) -> list[str]:
        """One line per adapter — "name (model) — <capability>" — chain order."""
        lines: list[str] = []
        for adapter in self._adapters:
            cap = adapter.capability
            skills: list[str] = []
            if cap.tools:
                skills.append("native tools")
            if cap.vision:
                skills.append("vision")
            skills.append(f"{cap.context_window}-token context")
            lines.append(f"{adapter.name} ({adapter.model}) — {', '.join(skills)}")
        return lines

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Stream one completion with silent failover. See module docstring."""
        failures: list[str] = []
        while self._index < len(self._adapters):
            adapter = self._adapters[self._index]
            served = False  # did this adapter hand us at least one event?
            gen: AsyncIterator[StreamEvent] | None = None
            try:
                gen = adapter.stream(request)
                async for event in gen:
                    served = True
                    yield event
                    if event.type in ("end", "error"):
                        return  # terminal: the step is over
            except Exception as exc:  # adapter failures only; see docstring
                if served:
                    # Mid-stream death: content already reached the user.
                    # Never restart, never switch — report and stop.
                    yield StreamEvent(
                        type="error",
                        error=(
                            f"{adapter.name} died mid-stream: {exc} — step not"
                            " restarted (content already sent)"
                        ),
                    )
                    return
                # Died before any content: the one allowed switch moment.
                reason = _failure_reason(exc)
                failures.append(f"{adapter.name}: {reason}")
                if self._index + 1 < len(self._adapters):
                    nxt = self._adapters[self._index + 1]
                    self._index += 1
                    if self._on_switch is not None:
                        self._on_switch(adapter, nxt, reason)
                    continue
                yield StreamEvent(
                    type="error",
                    error="all providers failed — " + "; ".join(failures),
                )
                return
            finally:
                if gen is not None:
                    closer = getattr(gen, "aclose", None)
                    if closer is not None:
                        await closer()
