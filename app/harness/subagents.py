"""Subagents (H6) — context firewalls: fresh-context workers, short summaries.

The parent delegates grinding to a worker that starts with a BLANK slate
(master inspiration: Claude Code's Task tool). The worker sees none of the
parent's history — its context is firewalled — and the parent sees none of
the worker's grinding: only a summary capped at 2000 chars comes back
(H6-2). The full worker transcript is never lost, though: every child event
is appended to the SAME sacred log under its own child session id
(``parent_session_id`` set, H6-3), so a kill mid-worker orphans nothing and
``replay(child_id)`` reconstructs exactly what the worker did.

The three pieces:

- :class:`RestrictedDispatcher` — a wrapper around the parent's real
  dispatcher that only permits the tools named in the task's allowlist;
  everything else gets ``{"status": "DENIED", "hint": "not available to this
  subagent"}`` before the parent's dispatcher is ever reached. It satisfies
  the frozen ``ToolDispatcher`` protocol (``specs``/``dispatch``) plus the
  loop's optional ``danger_of`` capability, so the child's ``AgentLoop`` runs
  unchanged.
- :class:`SubagentRunner` — the orchestrator. ``run(description,
  tool_names, parent_ctx)`` enforces the limits (H6-3: depth <= 2 — a session
  already at depth 2 may not spawn; concurrency <= 4 via ``asyncio.Semaphore``
  — a 5th simultaneous task WAITS, never errors, never hangs), creates the
  child session, builds a fresh child ``HarnessContext`` via :func:`make_ctx`
  (fresh state, ``depth = parent.depth + 1``, the parent's trace tap so H7
  sees child sessions too), runs one ``AgentLoop`` turn to completion, and
  returns the concatenated assistant text — hard-capped at 2000 chars with an
  explicit ``[truncated]`` marker. ``run`` NEVER raises: every failure path
  (depth limit, crashed provider, broken factory) comes back as an
  ``ERROR``-marked string, the same messages-not-crashes discipline as H2-3.
- :func:`spawn_task` + the ``task`` tool — the model-facing door. ``spawn_task``
  resolves the runner and the parent :class:`HarnessContext` from the injected
  :class:`ToolContext`'s ``state`` (the dispatcher shares that dict by
  reference — see :func:`wire_state`) and delegates to the runner. The tool is
  ``@tool(danger=Danger.CONFIRM)``: spawning spends tokens, so the permission
  engine gets a say before any worker starts (contract H6).

Wiring (how ChatService hands the tool its engine, Phase 3 integration)::

    runner = SubagentRunner(log=log, llm_factory=..., pipeline_factory=..., engine=engine)
    ctx = make_ctx(session_id=sid, llm=chain, log=log, tools=dispatcher, perms=engine)
    wire_state(ctx, runner)          # the task tool finds runner + parent ctx here
    registry.register(task)          # the parent's menu gains "task"

Child contexts are wired the same way automatically, so depth-1 workers may
spawn depth-2 grandchildren (the depth limit is what stops the recursion).

Purity: this module imports only the frozen interfaces, the H0 context
constructor, the H2 loop, and the H3 tool seams — never the H5 context
manager or permission internals.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable

from app.harness.context import make_ctx
from app.harness.interfaces import (
    Danger,
    HarnessContext,
    HarnessEvent,
    LoopGuards,
    PermissionEngine,
    RequestPipeline,
    SessionLogLike,
    ToolDispatcher,
)
from app.harness.loop import AgentLoop
from app.harness.tools.base import tool
from app.harness.tools.context import ToolContext
from app.providers.base import ToolCall, ToolSpec

__all__ = [
    "MAX_CONCURRENT_SUBAGENTS",
    "MAX_DEPTH",
    "SUMMARY_MAX_CHARS",
    "SUBAGENT_SYSTEM_PROMPT",
    "RestrictedDispatcher",
    "SubagentRunner",
    "spawn_task",
    "task",
    "wire_state",
]


# ------------------------------------------------------------------ constants

#: H6-2: the parent never receives more than this many chars of summary.
SUMMARY_MAX_CHARS = 2_000

#: Appended (only) when a summary was cut — an explicit, greppable marker.
TRUNCATION_MARKER = "[truncated]"

#: H6-3: nesting bound. Children may exist at depths 1 and 2; a session that
#: is itself at depth 2 may not spawn (its spawn attempt returns an ERROR).
MAX_DEPTH = 2

#: H6-3: at most this many subagent runs in flight; a 5th waits on the
#: semaphore rather than erroring or hanging.
MAX_CONCURRENT_SUBAGENTS = 4

#: The child's own system prompt (H6-1): logged as the child session's first
#: event, so every provider request the worker makes carries it and a replay
#: shows exactly what the worker was told.
SUBAGENT_SYSTEM_PROMPT = (
    "You are an Echo subagent: a focused worker spawned for exactly one task. "
    "You have no memory of any parent conversation — the task description is "
    "your complete briefing. Use only the tools in your menu. Work until the "
    "task is done, then reply with a concise final summary: that summary is "
    "the only part of your work the parent will ever see."
)

#: ``ctx.state`` key holding the :class:`SubagentRunner` (set by
#: :func:`wire_state`; the dispatcher shares this dict with the tool by reference).
RUNNER_STATE_KEY = "echo.subagent_runner"

#: ``ctx.state`` key holding the parent :class:`HarnessContext` — the bridge
#: from the tool-side :class:`ToolContext` (which has no depth/tools) to the
#: harness-side context :func:`spawn_task` needs (set by :func:`wire_state`).
PARENT_CTX_STATE_KEY = "echo.subagent_parent_ctx"


def _parse_allowlist(tool_names: str) -> frozenset[str]:
    """``"a, b"`` -> ``{"a", "b"}``; empty/blank -> no tools at all."""
    if not tool_names:
        return frozenset()
    return frozenset(name.strip() for name in tool_names.split(",") if name.strip())


# -------------------------------------------------------- restricted dispatcher


class RestrictedDispatcher:
    """The child's tool desk: the parent's dispatcher behind an allowlist (H6-1).

    Satisfies the frozen ``interfaces.ToolDispatcher`` protocol and the loop's
    optional ``danger_of`` capability, so the child's ``AgentLoop`` cannot tell
    it apart from the real desk — it just sees a smaller menu and gets a clean
    ``DENIED`` envelope for anything off-list (a message the model adapts to,
    never a crash, per the H2-3 error protocol).

    Dispatched calls DELEGATE to the parent's real dispatcher (validation,
    permission round, output capping all still apply) with the CHILD's
    context — so results are logged under the child session and the child's
    scratch state stays its own.
    """

    def __init__(self, parent: ToolDispatcher | None, allowed: frozenset[str]) -> None:
        self._parent = parent
        self._allowed = allowed

    # ------------------------------------------------------- frozen protocol

    def specs(self, context_window: int | None = None) -> list[ToolSpec]:
        """Only the allowlisted slice of the parent's (pruned) menu."""
        if self._parent is None:
            return []
        return [spec for spec in self._parent.specs(context_window) if spec.name in self._allowed]

    async def dispatch(self, call: ToolCall, ctx: HarnessContext | None) -> dict:
        """Restriction check FIRST, then delegate; never raises (H2-3)."""
        if call.name not in self._allowed:
            return {"status": "DENIED", "hint": "not available to this subagent"}
        if self._parent is None:
            return {"status": "ERROR", "hint": "no dispatcher available to this subagent"}
        return await self._parent.dispatch(call, ctx)

    # ------------------------------------------------- the danger seam (H2)

    def danger_of(self, call: ToolCall) -> Danger:
        """Pass through to the parent's classifier when it has one; CONFIRM otherwise.

        Truthful even for off-list tools — the restriction is enforced by
        :meth:`dispatch`, and the fail-safe default mirrors the loop's
        ``_danger_of`` (unknown/unclassifiable at worst asks, never allows).
        """
        helper = getattr(self._parent, "danger_of", None)
        if helper is None:
            return Danger.CONFIRM
        try:
            return Danger(helper(call))
        except Exception:
            return Danger.CONFIRM


# ------------------------------------------------------------------- the runner


class SubagentRunner:
    """Spawns, runs, and summarizes child workers (H6). Never raises from run.

    Constructor surface (this is what ChatService wires in Phase 3
    integration)::

        SubagentRunner(log, llm_factory, pipeline_factory, engine)

    - ``log`` — the ONE sacred log (child sessions live in the same file).
    - ``llm_factory`` — zero-arg callable returning the provider seam for a
      child (a fresh adapter per run; hand children a cheaper chain here).
    - ``pipeline_factory`` — zero-arg callable building the child's request
      pipeline the same way the parent's was built (tests inject fakes).
    - ``engine`` — the SAME permission engine the parent's loop uses; the
      child's tool calls go through the normal decide/ask round.
    """

    def __init__(
        self,
        log: SessionLogLike,
        llm_factory: Callable[[], object],
        pipeline_factory: Callable[[], RequestPipeline],
        engine: PermissionEngine | None,
        *,
        guards: LoopGuards | None = None,
        clock: Callable[[], float] | None = None,
        max_depth: int = MAX_DEPTH,
        max_concurrency: int = MAX_CONCURRENT_SUBAGENTS,
    ) -> None:
        self._log = log
        self._llm_factory = llm_factory
        self._pipeline_factory = pipeline_factory
        self._engine = engine
        self._guards = guards if guards is not None else LoopGuards()
        self._clock = clock if clock is not None else time.monotonic
        self._max_depth = max_depth
        self._semaphore = asyncio.Semaphore(max_concurrency)

    # ------------------------------------------------------------------ run

    async def run(self, description: str, tool_names: str = "", parent_ctx: HarnessContext | None = None) -> str:
        """One child worker, start to finish -> capped summary (never raises).

        Depth is checked before anything is spawned (a session already at
        ``max_depth`` gets an immediate ``nesting limit`` ERROR and no child
        session is created); concurrency waits on the semaphore — a run above
        the limit is delayed, never dropped. Both checks happen BEFORE any
        model call, so exceeding a limit can never hang.
        """
        if parent_ctx is None:
            return "[subagent ERROR] no parent context given to the subagent runner"
        if parent_ctx.depth + 1 > self._max_depth:
            return (
                f"[subagent ERROR] nesting limit: this session is at depth "
                f"{parent_ctx.depth}; subagents may nest at most {self._max_depth} deep"
            )
        allowed = _parse_allowlist(tool_names)
        async with self._semaphore:
            return await self._run_child(description, allowed, parent_ctx)

    async def _run_child(
        self, description: str, allowed: frozenset[str], parent_ctx: HarnessContext
    ) -> str:
        """The spawn itself: child session -> child ctx -> one AgentLoop turn."""
        child_id = uuid.uuid4().hex
        # The SAME sacred log (D-006): the child session is registered with its
        # parent link before anything can fail, so even a crash right here
        # leaves a replayable (if empty) record that a spawn was attempted.
        self._log.create_session(child_id, parent_session_id=parent_ctx.session_id)
        try:
            child_llm = self._llm_factory()
            restricted = RestrictedDispatcher(parent_ctx.tools, allowed)
            child_ctx = make_ctx(
                session_id=child_id,
                llm=child_llm,
                log=self._log,
                tools=restricted,
                perms=self._engine,
                trace=parent_ctx.trace,
                parent_session_id=parent_ctx.session_id,
                depth=parent_ctx.depth + 1,
            )
            # Fresh state (the firewall) with just the wiring a grandchild's
            # task tool would need — nested spawns obey the same depth limit.
            child_ctx.state[RUNNER_STATE_KEY] = self
            child_ctx.state[PARENT_CTX_STATE_KEY] = child_ctx

            # Log-first (D-006): the child's own system prompt is its first
            # event, before the loop appends the task as the user event.
            self._log.append(
                child_id, HarnessEvent(type="system", content=SUBAGENT_SYSTEM_PROMPT)
            )

            pipeline = self._pipeline_factory()
            loop = AgentLoop(
                child_ctx,
                pipeline,
                restricted,
                self._engine,
                guards=self._guards,
                clock=self._clock,
            )

            parts: list[str] = []
            abort_reason: str | None = None
            async for event in loop.run_turn(description):
                if event.type == "assistant":
                    parts.append(event.content)
                elif (
                    event.type == "system"
                    and event.result is not None
                    and event.result.get("status") == "aborted"
                ):
                    abort_reason = str(event.result.get("reason", "aborted"))

            summary = "".join(parts)
            if len(summary) > SUMMARY_MAX_CHARS:
                summary = summary[:SUMMARY_MAX_CHARS] + TRUNCATION_MARKER
            if abort_reason is not None:
                notice = f"[subagent ERROR] the child run aborted: {abort_reason}"
                return f"{summary}\n{notice}" if summary else notice
            return summary
        except Exception as exc:  # nothing escapes a spawn — errors are messages
            return f"[subagent ERROR] the subagent crashed before finishing: {exc!r}"


# -------------------------------------------------------------- the task tool


def wire_state(ctx: HarnessContext, runner: SubagentRunner) -> None:
    """Register ``runner`` + ``ctx`` in ``ctx.state`` so the task tool can find them.

    One wiring point for ChatService (and for :meth:`SubagentRunner._run_child`,
    which does the same for every child). The dispatcher shares ``ctx.state``
    with the injected :class:`ToolContext` by reference, so this is the only
    bridge the tool-side seam needs to reach the harness-side context.
    """
    ctx.state[RUNNER_STATE_KEY] = runner
    ctx.state[PARENT_CTX_STATE_KEY] = ctx


async def spawn_task(ctx: ToolContext, description: str, tool_names: str = "") -> str:
    """Spawn one subagent run from inside a tool call (contract H6).

    ``ctx`` is the :class:`ToolContext` the dispatcher injected. The runner and
    the parent :class:`HarnessContext` are resolved from ``ctx.state`` (wired
    via :func:`wire_state`); the child session is created in the SAME log under
    ``ctx.session_id``. Missing wiring is an ERROR message, never a raise.
    """
    state = ctx.state if isinstance(ctx.state, dict) else {}
    runner = state.get(RUNNER_STATE_KEY)
    if not isinstance(runner, SubagentRunner):
        return "[subagent ERROR] no subagent runner is wired into this session's state"
    parent_ctx = state.get(PARENT_CTX_STATE_KEY)
    if not isinstance(parent_ctx, HarnessContext):
        return "[subagent ERROR] no parent context is wired into this session's state"
    return await runner.run(description, tool_names, parent_ctx)


@tool(danger=Danger.CONFIRM)
async def task(description: str, tool_names: str = "", *, ctx: ToolContext) -> str:
    """Spawn a subagent worker on a task; returns its summary (max 2000 chars).

    The worker starts with a completely fresh context — it cannot see this
    conversation — and only its final summary comes back. `tool_names` is a
    comma-separated allowlist of tools the worker may use (empty means it
    works without tools). Spawning spends tokens, so it asks first.
    """
    return await spawn_task(ctx, description, tool_names)
