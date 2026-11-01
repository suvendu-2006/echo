"""The loop heart (H2) — Echo's think→act→observe cycle.

``AgentLoop.run_turn`` is the single most important function in the project
(PLAN H2). One turn:

1. LOG-FIRST (dsh discipline): the user event is appended before anything
   else can fail.
2. Repeat: ``pipeline.build(ctx, log.replay(session), dispatcher.specs())``
   → ``ctx.llm.stream(request)`` → accumulate text / tool calls / usage.
3. TERMINATION CONTRACT: the first reply with no tool calls ends the turn —
   its assistant event is the last thing logged and yielded.
4. Each tool call: log ``tool_call`` → permission round-trip → ``dispatch``
   → log ``tool_result``. Everything that enters the log is also yielded,
   in order — a caller sees exactly what the sacred log saw.

Guards (H2-2) can always force a stop — max steps, wall clock, repeated
identical calls — and end the turn with a ``system`` event
(``result={"status": "aborted", "reason": ...}``), never an exception.
Provider failures (the ProviderError family, or a terminal ``error`` stream
event from a failover chain) become the same clean abort; partial stream
text stays in memory, per the only-final-events-enter-the-log rule. Tool
errors are messages, not crashes (H2-3): an ERROR dispatch result is logged
and fed back verbatim so the model self-corrects (H2-4's loop detection is
the retry bound).

The danger seam (contract note for the H3 dispatcher)
-----------------------------------------------------
``ToolSpec`` carries no danger classification and the frozen dispatcher
protocol exposes only ``specs()``/``dispatch()`` — yet the permission
engine's ``decide(call, danger, ctx)`` needs one. Solved without touching
the frozen seam via an OPTIONAL dispatcher capability:

    dispatcher.danger_of(call: ToolCall) -> Danger    # sync, total

The real H3 dispatcher ships it; test fakes provide it. If the attribute is
absent, or calling it raises or returns a non-Danger value, the loop fails
safe to ``Danger.CONFIRM`` — the stricter class, so an unclassifiable tool
at worst asks, never silently allows.

Purity: this module imports only ``app.harness.interfaces`` and
``app.providers.base`` — never the concrete tool, permission, or context
implementations (lint-tested).
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import AsyncIterator, Callable

from app.harness.interfaces import (
    Danger,
    Decision,
    HarnessContext,
    HarnessEvent,
    LoopGuards,
    PermissionEngine,
    RequestPipeline,
    ToolDispatcher,
)
from app.providers.base import ProviderError, ToolCall, Usage


# ------------------------------------------------------------ small helpers


def _loop_key(call: ToolCall) -> tuple[str, str]:
    """H2-4 identity of a tool call: (name, canonical JSON of arguments)."""
    try:
        arguments = json.dumps(call.arguments, sort_keys=True)
    except (TypeError, ValueError):
        arguments = repr(call.arguments)  # non-JSON args still compare, just coarser
    return (call.name, arguments)


def _danger_of(dispatcher: ToolDispatcher, call: ToolCall) -> Danger:
    """Resolve a call's danger class — the optional-capability seam.

    See the module docstring: ``dispatcher.danger_of(call) -> Danger`` when
    the dispatcher can say; otherwise (absent, raising, or nonsense) the
    fail-safe default ``Danger.CONFIRM``.
    """
    helper = getattr(dispatcher, "danger_of", None)
    if helper is None:
        return Danger.CONFIRM
    try:
        return Danger(helper(call))
    except Exception:
        return Danger.CONFIRM


# ----------------------------------------------------------------- the loop


class AgentLoop:
    """The think→act→observe cycle (H2-1 .. H2-4) — the project's heart.

    Every collaborator is injected: the harness context (sacred log +
    provider seam), the request pipeline, the tool dispatcher, the
    permission engine, the guards, and the wall clock (injectable so tests
    can travel in time).
    """

    def __init__(
        self,
        ctx: HarnessContext,
        pipeline: RequestPipeline,
        dispatcher: ToolDispatcher,
        engine: PermissionEngine,
        guards: LoopGuards = LoopGuards(),
        clock: Callable[[], float] = time.monotonic,
        on_delta: Callable[[str], object] | None = None,
    ) -> None:
        self.ctx = ctx
        self.pipeline = pipeline
        self.dispatcher = dispatcher
        self.engine = engine
        self.guards = guards
        self.clock = clock
        # Optional live-streaming hook (H7-3 door): fired per provider text
        # chunk. Sync or async — never allowed to break the loop (fail-safe).
        self.on_delta = on_delta

    async def _fire_delta(self, text: str) -> None:
        if self.on_delta is None:
            return
        try:
            result = self.on_delta(text)
            if inspect.isawaitable(result):
                await result
        except Exception:  # a broken UI hook must never kill the loop
            pass

    async def run_turn(self, user_text: str) -> AsyncIterator[HarnessEvent]:
        """Run one user turn to completion, yielding every logged event.

        Log-first, then the build→stream→dispatch cycle until a tool-free
        reply (the termination contract) or a guard/provider stop (a
        terminal ``system`` event). Never raises out of the generator.
        """
        # LOG-FIRST: nothing may happen in this turn before this append.
        user_event = HarnessEvent(type="user", content=user_text)
        self.ctx.log.append(self.ctx.session_id, user_event)
        yield user_event

        start = self.clock()
        steps = 0
        repeats: dict[tuple[str, str], int] = {}

        try:
            while True:
                # Guards first (H2-2): the loop can ALWAYS be forced to stop.
                if steps >= self.guards.max_steps:
                    yield self._abort(f"max_steps limit reached ({self.guards.max_steps})")
                    return
                elapsed = self.clock() - start
                if elapsed > self.guards.wall_clock_seconds:
                    yield self._abort(
                        f"wall_clock limit exceeded ({elapsed:.1f}s > "
                        f"{self.guards.wall_clock_seconds:.1f}s)"
                    )
                    return

                steps += 1
                request = await self.pipeline.build(
                    self.ctx,
                    self.ctx.log.replay(self.ctx.session_id),
                    self.dispatcher.specs(),
                )

                text_parts: list[str] = []
                calls: list[ToolCall] = []
                usage = Usage()
                async for chunk in self.ctx.llm.stream(request):
                    if chunk.type == "text":
                        text_parts.append(chunk.text)
                        await self._fire_delta(chunk.text)
                    elif chunk.type == "tool_call" and chunk.tool_call is not None:
                        calls.append(chunk.tool_call)
                    elif chunk.type == "usage" and chunk.usage is not None:
                        usage.prompt_tokens += chunk.usage.prompt_tokens
                        usage.completion_tokens += chunk.usage.completion_tokens
                    elif chunk.type == "error":
                        # A failover chain reports exhaustion as a terminal
                        # error event — same handling as a raised ProviderError.
                        raise ProviderError(chunk.error or "provider error event")

                # One assistant event per cycle; meta carries this cycle's
                # tool calls so events_to_messages pairs the tool results.
                assistant = HarnessEvent(
                    type="assistant",
                    content="".join(text_parts),
                    usage=usage,
                    meta={"assistant_tool_calls": list(calls)},
                )
                self.ctx.log.append(self.ctx.session_id, assistant)
                yield assistant

                if not calls:
                    return  # TERMINATION CONTRACT: a tool-free reply ends the turn.

                for call in calls:
                    call_event = HarnessEvent(type="tool_call", tool_call=call)
                    self.ctx.log.append(self.ctx.session_id, call_event)
                    yield call_event

                    # Loop detection (H2-4): identical tool+args counting.
                    key = _loop_key(call)
                    repeats[key] = repeats.get(key, 0) + 1
                    count = repeats[key]
                    if count >= self.guards.loop_abort_after:
                        yield self._abort(
                            f"loop_aborted: '{call.name}' repeated {count}x "
                            "with identical arguments"
                        )
                        return
                    if count >= self.guards.loop_break_after:
                        re_focus = HarnessEvent(
                            type="system",
                            content=user_text,
                            meta={"goal": user_text, "note": "loop detected — re-focus"},
                        )
                        self.ctx.log.append(self.ctx.session_id, re_focus)
                        yield re_focus

                    danger = _danger_of(self.dispatcher, call)
                    deny_hint = "denied by permission engine"
                    try:
                        decision = await self.engine.decide(call, danger, self.ctx)
                        if decision is Decision.ASK:
                            deny_hint = "denied at the ask event"
                            decision = await self.engine.ask(call, danger, self.ctx)
                    except Exception as exc:  # a broken guard fails closed
                        decision = Decision.DENY
                        deny_hint = f"permission engine error (fail-closed): {exc!r}"

                    if decision is Decision.ALLOW:
                        try:
                            result = await self.dispatcher.dispatch(call, self.ctx)
                        except Exception as exc:  # protocol says never; belt and braces
                            result = {"status": "ERROR", "hint": f"dispatcher crashed: {exc!r}"}
                    else:
                        result = {"status": "DENIED", "hint": deny_hint}

                    result_event = HarnessEvent(
                        type="tool_result", result=result, meta={"for_call": call}
                    )
                    self.ctx.log.append(self.ctx.session_id, result_event)
                    yield result_event
        except ProviderError as exc:
            yield self._abort(f"provider error: {exc}")
            return
        except Exception as exc:  # fail-safe: nothing escapes the loop heart
            yield self._abort(f"unexpected error: {exc!r}")
            return

    def _abort(self, reason: str) -> HarnessEvent:
        """Build + log the terminal ``system`` event for a forced stop (H2-2)."""
        event = HarnessEvent(
            type="system",
            content=f"turn aborted: {reason}",
            result={"status": "aborted", "reason": reason},
        )
        self.ctx.log.append(self.ctx.session_id, event)
        return event
