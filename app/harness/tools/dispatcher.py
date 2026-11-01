"""ToolDispatcher (H3-4) — the desk every tool call passes through.

Satisfies the FROZEN ``interfaces.ToolDispatcher`` protocol (never edits it)
plus the optional capability the H2 loop consumes (see loop.py's danger-seam
note)::

    dispatcher.danger_of(call: ToolCall) -> Danger   # sync, total

Unknown tools classify as ``Danger.CONFIRM`` — fail-safe: the stricter class
at worst asks, never silently allows.

``dispatch(call, ctx)`` NEVER raises. One call flows through, in order:

1. lookup (unknown tool -> ERROR envelope);
2. defensive argument validation against the spec's JSON schema — unknown
   arguments, missing required arguments and wrong-typed values all return
   ``{"status": "ERROR", "hint": ...}`` instead of reaching Python code
   (malformed model output is an expected event, never a crash);
3. ``Danger.BLOCK`` -> ``{"status": "DENIED", "hint": "blocked tool"}``
   without asking anyone;
4. SAFE/CONFIRM -> ``engine.decide``; ASK -> ``engine.ask``; DENY -> DENIED.
   A missing engine fail-closes (SAFE still runs; CONFIRM is denied). A
   broken engine fail-closes to DENIED — never to allow;
5. the H4 block-and-return-fake-result seam: if the engine exposes a
   ``last_block_result`` dict after its decision, that dict is returned as
   the tool result WITHOUT running the tool (a before-hook substituted a
   clean fake — see app/harness/permissions/callbacks.py);
6. run: sync tools run via ``asyncio.to_thread`` (the loop never blocks;
   SAFE tools therefore overlap), async tools run on the loop. A declared
   ``ctx: ToolContext`` parameter is injected here (H3-2).

Output discipline (H3-4): every result is stringified and capped — anything
over ``tool_result_max_chars`` is spilled whole to
``<spills_dir>/<timestamp>-<tool>.txt`` (default ``data/spills``) and reduced
to a 2000-char preview plus ``spilled_to``/``full_size`` in the envelope.
Any exception inside a tool becomes ``{"status": "ERROR", "hint": ...}``
(logged, never propagated) — errors are messages, not crashes.

``dispatch_many(calls, ctx)`` is the parallelism rule: SAFE calls gather
concurrently (they are read-only by definition), CONFIRM calls serialize on
an ``asyncio.Lock`` (mutating tools never interleave); the result list keeps
the input order regardless.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import DATA_DIR
from app.harness.interfaces import Danger, Decision, HarnessContext
from app.harness.tools.base import ToolMeta
from app.harness.tools.context import ToolContext
from app.harness.tools.registry import ToolRegistry
from app.providers.base import ToolCall, ToolSpec

log = logging.getLogger(__name__)

#: Preview size handed to the model when a result spills (contract H3-4).
SPILL_PREVIEW_CHARS = 2_000

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def _check_type(value: Any, schema: dict) -> tuple[Any, str | None]:
    """One argument against one schema fragment -> (coerced value | error)."""
    expected = schema.get("type")
    if value is None:
        if schema.get("nullable"):
            return None, None
        return value, f"expected {expected or 'any'}, got null"
    if not expected or expected == "any":
        return value, None
    if expected == "integer":
        if isinstance(value, bool):  # bool IS an int in Python — JSON says no
            return value, "expected integer, got boolean"
        if isinstance(value, int):
            return value, None
        if isinstance(value, float) and value.is_integer():
            return int(value), None  # JSON Schema: 3.0 is a valid integer
        return value, f"expected integer, got {type(value).__name__}"
    if expected == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return value, f"expected number, got {type(value).__name__}"
        return value, None
    python_types = _JSON_TYPES.get(expected)
    if python_types and isinstance(value, python_types):
        if expected == "boolean" or not isinstance(value, bool):
            items = schema.get("items")
            if expected == "array" and isinstance(items, dict) and items.get("type"):
                for element in value:
                    _, error = _check_type(element, items)
                    if error:
                        return value, f"array element {error}"
            return value, None
    return value, f"expected {expected}, got {type(value).__name__}"


def validate_arguments(call_name: str, spec: ToolSpec, arguments: Any) -> tuple[dict, str | None]:
    """Defensively validate ``arguments`` against the spec schema (H3-4).

    Returns ``(kwargs, None)`` when clean (a fresh dict, integral floats
    coerced to int) or ``({}, hint)`` on the first problem: not an object,
    unknown argument, missing required argument, wrong type. Extra safety,
    never a substitute for the permission round.
    """
    if not isinstance(arguments, dict):
        return {}, f"invalid arguments for tool {call_name!r}: expected an object, got {type(arguments).__name__}"
    properties = spec.parameters.get("properties", {})
    required = spec.parameters.get("required", [])
    kwargs: dict = {}
    for key, value in arguments.items():
        if key not in properties:
            return {}, f"invalid arguments for tool {call_name!r}: unknown argument {key!r}"
        coerced, error = _check_type(value, properties[key])
        if error:
            return {}, f"invalid arguments for tool {call_name!r}: argument {key!r} {error}"
        kwargs[key] = coerced
    for key in required:
        if key not in kwargs:
            return {}, f"invalid arguments for tool {call_name!r}: missing required argument {key!r}"
    return kwargs, None


class ToolDispatcher:
    """Validates, asks permissions, runs, caps outputs (H3-4)."""

    def __init__(
        self,
        registry: ToolRegistry,
        engine: Any | None = None,
        log: Any | None = None,
        state: dict | None = None,
        tool_result_max_chars: int = 20_000,
        spills_dir: str | Path | None = None,
    ) -> None:
        self._registry = registry
        self._engine = engine
        self._log = log
        self._state = state if state is not None else {}
        self._tool_result_max_chars = tool_result_max_chars
        self._spills_dir = Path(spills_dir) if spills_dir is not None else DATA_DIR / "spills"
        self._confirm_lock = asyncio.Lock()  # CONFIRM tools never interleave

    # ------------------------------------------------------- frozen protocol

    def specs(self, context_window: int | None = None) -> list[ToolSpec]:
        """The (pruned) tool menu — the registry decides, we just forward."""
        return self._registry.specs(context_window)

    # ------------------------------------------------- the danger seam (H2)

    def danger_of(self, call: ToolCall) -> Danger:
        """Sync, total: the call's danger class; unknown -> CONFIRM (fail-safe).

        The H2 loop's optional capability — see loop.py's ``_danger_of``.
        """
        try:
            return self._registry.danger_of(call.name)
        except Exception:
            return Danger.CONFIRM  # unknown/unclassifiable: at worst it asks

    # ------------------------------------------------------------------ run

    async def dispatch(self, call: ToolCall, ctx: HarnessContext | None) -> dict:
        """One tool call -> envelope; NEVER raises (H2-3 error protocol)."""
        try:
            return await self._dispatch(call, ctx)
        except Exception as exc:
            log.warning("tool %s crashed: %s", getattr(call, "name", "?"), exc, exc_info=True)
            return {"status": "ERROR", "hint": f"tool crashed: {exc}"}

    async def _dispatch(self, call: ToolCall, ctx: HarnessContext | None) -> dict:
        if not isinstance(call.name, str) or call.name not in self._registry:
            return {"status": "ERROR", "hint": f"unknown tool: {call.name!r}"}
        fn = self._registry.get(call.name)
        meta: ToolMeta = fn._echo_tool

        kwargs, error = validate_arguments(call.name, meta.spec, call.arguments)
        if error is not None:
            return {"status": "ERROR", "hint": error}

        if meta.danger is Danger.BLOCK:
            # Registered for visibility only — never dispatched, never asked.
            return {"status": "DENIED", "hint": "blocked tool"}

        allowed, denied_hint = await self._permission_round(call, meta, ctx)

        # H4 block-and-return-fake-result seam: a before-hook that blocked
        # left a clean fake result on the engine — return it, run nothing.
        # Checked after the decision, whatever the decision was (a block is
        # a DENY whose "result" is the fake, not the DENIED envelope).
        block_result = getattr(self._engine, "last_block_result", None)
        if isinstance(block_result, dict):
            return dict(block_result)

        if not allowed:
            return {"status": "DENIED", "hint": denied_hint}

        return await self._run(fn, meta, kwargs, ctx)

    async def _permission_round(
        self, call: ToolCall, meta: ToolMeta, ctx: HarnessContext | None
    ) -> tuple[bool, str]:
        """Engine round -> (allowed, denied_hint). Deny/fail-closed everywhere."""
        if self._engine is None:
            if meta.danger is Danger.SAFE:
                return True, ""
            return False, "no permission engine configured (fail-closed)"
        deny_hint = "denied by permission engine"
        try:
            decision = await self._engine.decide(call, meta.danger, ctx)
            if decision is Decision.ASK:
                deny_hint = "denied at the ask event"
                decision = await self._engine.ask(call, meta.danger, ctx)
        except Exception as exc:  # a broken guard fails closed
            log.warning("permission engine failed for %s; deny (fail-closed): %s", call.name, exc)
            return False, f"permission engine error (fail-closed): {exc!r}"
        if decision is Decision.ALLOW:
            return True, ""
        return False, deny_hint  # non-ALLOW from ask() is a deny too (fail-closed)

    async def _run(
        self, fn: Any, meta: ToolMeta, kwargs: dict, ctx: HarnessContext | None
    ) -> dict:
        if meta.takes_ctx:
            kwargs["ctx"] = self._tool_context(ctx)
        if asyncio.iscoroutinefunction(fn):
            result = await fn(**kwargs)
        else:
            result = await asyncio.to_thread(fn, **kwargs)  # the loop never blocks
        text = _stringify(result)
        if len(text) > self._tool_result_max_chars:
            return self._spill(fn.__name__, text)
        return {"status": "OK", "result": text}

    def _tool_context(self, ctx: HarnessContext | None) -> ToolContext:
        """Build the H3-2 window: session_id/state/log, dispatcher fallbacks."""
        session_id = getattr(ctx, "session_id", "") if ctx is not None else ""
        state = getattr(ctx, "state", None)
        log = getattr(ctx, "log", None)
        return ToolContext(
            session_id=session_id if isinstance(session_id, str) else str(session_id),
            state=state if isinstance(state, dict) else self._state,
            log=log if log is not None else self._log,
        )

    def _spill(self, tool_name: str, text: str) -> dict:
        """Write the full result to disk, hand back only a 2000-char preview."""
        try:
            self._spills_dir.mkdir(parents=True, exist_ok=True)
            path = self._spill_path(tool_name)
            path.write_text(text, encoding="utf-8")
        except OSError as exc:
            log.warning("spill to %s failed: %s", self._spills_dir, exc)
            return {
                "status": "OK",
                "result": text[:SPILL_PREVIEW_CHARS] + "\n[output truncated; spill failed]",
            }
        log.info("tool %s output spilled to %s (%d chars)", tool_name, path, len(text))
        return {
            "status": "OK",
            "result": text[:SPILL_PREVIEW_CHARS],
            "spilled_to": str(path),
            "full_size": len(text),
        }

    def _spill_path(self, tool_name: str) -> Path:
        """``<timestamp>-<tool>.txt`` — unique even within one microsecond."""
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in tool_name)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = self._spills_dir / f"{stamp}-{safe_name}.txt"
        counter = 2
        while path.exists():
            path = self._spills_dir / f"{stamp}-{safe_name}-{counter}.txt"
            counter += 1
        return path

    # --------------------------------------------------------- parallelism

    async def dispatch_many(self, calls: list[ToolCall], ctx: HarnessContext | None) -> list[dict]:
        """Many calls -> envelopes, input order preserved (H3-4).

        SAFE calls gather concurrently (read-only tools overlap); every other
        danger class serializes on one lock — mutating tools never interleave.
        """
        results: list[dict | None] = [None] * len(calls)
        safe_indices: list[int] = []
        confirm_indices: list[int] = []
        for index, call in enumerate(calls):
            if self.danger_of(call) is Danger.SAFE:
                safe_indices.append(index)
            else:
                confirm_indices.append(index)

        async def run_one(index: int) -> None:
            results[index] = await self.dispatch(calls[index], ctx)

        async def run_serial() -> None:
            for index in confirm_indices:
                async with self._confirm_lock:
                    await run_one(index)

        await asyncio.gather(*(run_one(i) for i in safe_indices), run_serial())
        return [result or {"status": "ERROR", "hint": "dispatch produced no result"} for result in results]


def _stringify(result: Any) -> str:
    """Every tool output enters the context as text (H3-4 output discipline)."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    return str(result)
