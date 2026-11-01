"""The request pipeline (H5-2) — composable stages, never one monolith.

The frozen seam (``interfaces.RequestPipeline``) is one method:
``async build(ctx, events, tools) -> Request``. Underneath, the pipeline is a
LIST of stages, each with the same shape::

    async def stage(ctx, events, tools, draft) -> draft

``draft`` is a small dataclass (:class:`Draft`) holding the request under
construction — ``messages``, ``tools``, ``meta``. Stages receive it, shape it,
return it; the pipeline runs them in list order and owns the two hard
guarantees that no stage may break:

1. **No provider ever receives an oversized request.** After the stages run,
   ``build`` counts the finished request; if it still exceeds the window
   (pathological input — e.g. one gigantic message), history is hard-truncated
   to the leading system messages + the last user turn, then the tool list and
   finally the user content itself are trimmed until it fits. The truncated
   version is RETURNED (never an assert, never a raise).

2. **The frozen stage order** (provable by unit test with recording stages):
   memory-inject → tool-menu → history → compact-check.

Stage notes
-----------
- :class:`MemoryInjectStage` is the **Phase 6 seam**: a memory object exposing
  ``async relevant(query, k) -> list[str]``. Because the frozen order puts it
  BEFORE :class:`HistoryStage` (whose job is events → messages), the memory
  stage does not append to ``draft.messages`` inside the default pipeline —
  it fetches the facts (query = the latest user event's text) and stashes
  them in ``draft.meta["memory_facts"]``; HistoryStage then places them as a
  system message right after the latest user turn. When composed AFTER a
  history stage (custom pipelines), ``draft.messages`` is already populated
  and the stage injects directly. Both paths end in the same place: memory
  sits next to what the model is currently answering.
- :class:`ToolMenuStage` is pass-through shaping only: the caller hands in an
  already-pruned menu (H3-3 ``registry.specs(context_window)``), the stage
  attaches it to the draft.
- :class:`HistoryStage` projects events → messages via the canonical
  ``interfaces.events_to_messages`` and fits them into the history budget:
  newest-first keep, oldest dropped first, the latest user turn NEVER dropped,
  and assistant tool_calls never orphaned (events are grouped into atomic
  units — an assistant and its tool results are kept or dropped together).
  Anything dropped is announced by ONE leading system note. A ``compact``
  event in the replayed log (H5-3) replaces everything it covers with its
  summary as the FIRST system message.
- :class:`CompactCheckStage` hands the finished draft to the compact engine
  (H5-3, ``compact.py``). The ≥75%-of-window decision lives in the engine —
  one source of truth for the threshold — this stage is the pipeline seam
  that invokes it after history fitting.

Purity: this module imports only the frozen interfaces, the provider data
model, and its package sibling ``budget.py`` — never concrete tools,
permissions, providers, or the compact engine (which is injected).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import Any

from app.harness.ctx.budget import EstimateCounter, TokenCounter, budget_for
from app.harness.interfaces import HarnessEvent, events_to_messages
from app.providers.base import Message, Request, ToolCall, ToolSpec

logger = logging.getLogger("echo.harness")

#: The ONE leading note inserted when history fitting had to drop anything
#: (text fixed by the H5-2 contract).
ELISION_NOTE = "[older history elided — summary follows]"

#: The note marking a pathological hard-truncate (window still exceeded after
#: compaction — the pipeline's last resort, H5-2).
HARD_TRUNCATE_NOTE = "[older history elided — context exceeded the model window]"

#: Header of the injected memory system message (Phase 6 wallet seam).
MEMORY_HEADER = "[memory]"


@dataclass
class Draft:
    """The request under construction — one stage's input and output."""

    messages: list[Message] = field(default_factory=list)
    tools: list[ToolSpec] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# ------------------------------------------------- pure projection helpers
#
# These live here (not in compact.py) so both HistoryStage and the compact
# engine share ONE history-fitting implementation — zero drift between the
# normal path and the post-compact rebuild.


def last_user_index(events: list[HarnessEvent]) -> int:
    """Index of the latest ``user`` event; -1 when there is none."""
    for index in range(len(events) - 1, -1, -1):
        if events[index].type == "user":
            return index
    return -1


def group_events(events: list[HarnessEvent]) -> list[list[HarnessEvent]]:
    """Group events into atomic units so tool calls are never orphaned.

    An event starts a new group unless it is a ``tool_call``/``tool_result``
    — those attach to the current group (the assistant that issued them).
    Keeping or dropping whole groups therefore keeps assistant↔result pairs
    together in BOTH directions: no assistant survives without its results,
    and no result message survives without its assistant.
    """
    groups: list[list[HarnessEvent]] = []
    for event in events:
        if groups and event.type in ("tool_call", "tool_result"):
            groups[-1].append(event)
        else:
            groups.append([event])
    return groups


def _count_messages(counter: TokenCounter, messages: list[Message]) -> int:
    """Count messages via the counter, tolerating protocol-minimal counters.

    ``EstimateCounter.count_messages`` is used when available; otherwise a
    conservative fallback sums the same fields through ``count_text``.
    """
    helper = getattr(counter, "count_messages", None)
    if callable(helper):
        return int(helper(messages))
    total = 0
    for message in messages:
        total += counter.count_text(message.role)
        total += counter.count_text(message.content or "")
        for call in message.tool_calls:
            total += counter.count_text(call.name)
        if message.tool_call_id:
            total += counter.count_text(message.tool_call_id)
    return total


def group_tokens(group: list[HarnessEvent], counter: TokenCounter) -> int:
    """Token cost of one atomic group once projected to messages."""
    return _count_messages(counter, events_to_messages(group))


def fit_history(
    groups: list[list[HarnessEvent]], counter: TokenCounter, budget: int
) -> tuple[list[list[HarnessEvent]], int]:
    """Keep the newest run of groups that fits the budget; drop the oldest.

    Returns ``(kept_groups_in_order, dropped_count)``. Rules:

    - newest-first keep: the walk starts at the newest group and stops at the
      first that does not fit — the oldest side is what gets dropped;
    - the newest group is kept even when alone over budget (a request with an
      empty history helps nobody);
    - the group holding the LATEST user event is mandatory — force-kept even
      over budget (the model must always see what it is answering; the
      pipeline's window guard is the backstop for that pathology).

    Group atomicity (see :func:`group_events`) makes the keep-together rule
    structural: an assistant and its tool results can never be split.
    """
    last_user_group = -1
    for index, group in enumerate(groups):
        if any(event.type == "user" for event in group):
            last_user_group = index
    kept_indices: list[int] = []
    total = 0
    for index in range(len(groups) - 1, -1, -1):
        cost = group_tokens(groups[index], counter)
        if not kept_indices or total + cost <= budget or index == last_user_group:
            kept_indices.append(index)
            total += cost
        else:
            break
    if last_user_group >= 0 and last_user_group not in kept_indices:
        kept_indices.append(last_user_group)
    kept_indices.sort()
    kept = [groups[index] for index in kept_indices]
    return kept, len(groups) - len(kept)


def split_by_compact(
    events: list[HarnessEvent],
) -> tuple[str, list[HarnessEvent]]:
    """Apply compact-event coverage (H5-3): ``(summary_text, surviving_events)``.

    A ``compact`` event's ``meta["covers_seq"]`` is a high-water mark: every
    event with seq <= covers_seq was summarized into the compact event and is
    replaced by its ``content`` here. The log is gapless per session (H0-2
    gate: seqs 0..n-1), so a replayed event's list index IS its seq; a skipped
    corrupt tail row shifts coverage by one — a benign degradation (one extra
    old event survives or is summarized, never a recent one lost).

    The NEWEST compact event wins (older compacts are inside its coverage).
    A compact event without ``covers_seq`` (hand-crafted) covers everything
    before itself. ``split_by_compact`` never mutates the input.
    """
    compact_index = -1
    for index, event in enumerate(events):
        if event.type == "compact":
            compact_index = index  # last one wins
    if compact_index < 0:
        return "", list(events)
    compact = events[compact_index]
    covers = compact.meta.get("covers_seq")
    if not isinstance(covers, int):
        covers = compact_index - 1
    survivors = [event for index, event in enumerate(events) if index > covers]
    summary = compact.content or str(compact.meta.get("goal") or "")
    return summary, survivors


def memory_message(facts: Sequence[str]) -> Message:
    """The injected wallet-memory system message (Phase 6 seam shape)."""
    body = "\n".join(f"- {fact}" for fact in facts)
    return Message(role="system", content=f"{MEMORY_HEADER}\n{body}")


def insert_after_last_user(
    messages: list[Message], message: Message
) -> list[Message]:
    """Insert ``message`` right after the latest user message (a new list).

    With no user message at all, the message is appended. Used for the memory
    injection point: next to what the model is currently answering.
    """
    index = -1
    for position, existing in enumerate(messages):
        if existing.role == "user":
            index = position
    if index < 0:
        return [*messages, message]
    return [*messages[: index + 1], message, *messages[index + 1 :]]


def _coerce_tool_call(value: Any) -> ToolCall | None:
    """Rebuild a ToolCall from the shapes a replayed log can carry.

    In-memory events hold real ToolCall objects; a JSON round-trip preserves
    plain dicts; anything else (e.g. a repr string left by a ``default=str``
    serialization) is unrecoverable and becomes None — the projection then
    degrades gracefully instead of crashing.
    """
    if isinstance(value, ToolCall):
        return value
    if isinstance(value, dict):
        return ToolCall(
            id=str(value.get("id") or ""),
            name=str(value.get("name") or ""),
            arguments=value.get("arguments") or {},
            raw_arguments=str(value.get("raw_arguments") or ""),
            parse_error=value.get("parse_error"),
        )
    return None


def normalize_event(event: HarnessEvent) -> HarnessEvent:
    """Coerce meta-embedded tool calls to ToolCall objects (a new event).

    The frozen ``events_to_messages`` expects ``meta["assistant_tool_calls"]``
    and ``meta["for_call"]`` to be ToolCall objects, but a replayed log may
    hold dicts (clean JSON) or degraded strings. Normalizing here keeps ONE
    projection path working over every shape; unrecoverable entries are
    dropped/blanked rather than crashing the pipeline. Never mutates the
    input event.
    """
    meta = event.meta or {}
    calls = meta.get("assistant_tool_calls")
    for_call = meta.get("for_call")
    calls_need = calls is not None and any(
        not isinstance(call, ToolCall) for call in (calls or [])
    )
    for_call_needs = for_call is not None and not isinstance(for_call, ToolCall)
    if not calls_need and not for_call_needs:
        return event
    new_meta = dict(meta)
    if calls_need:
        new_meta["assistant_tool_calls"] = [
            coerced for coerced in (_coerce_tool_call(call) for call in (calls or [])) if coerced
        ]
    if for_call_needs:
        new_meta["for_call"] = _coerce_tool_call(for_call)  # None if unrecoverable
    return HarnessEvent(
        type=event.type,
        content=event.content,
        tool_call=event.tool_call,
        result=event.result,
        usage=event.usage,
        meta=new_meta,
    )


def project_history(
    events: list[HarnessEvent],
    counter: TokenCounter,
    budget: int,
    *,
    summary: str = "",
    memory_facts: Sequence[str] = (),
) -> tuple[list[Message], int]:
    """The ONE history assembler, shared by HistoryStage and the compact engine.

    ``[compact summary?][elision note?][fitted history][memory facts?]`` —
    the compact summary (H5-3) is the FIRST system message, the fit-elision
    note follows when fitting dropped anything, and memory lands after the
    latest user turn. Returns ``(messages, dropped_group_count)``. Events are
    normalized (meta tool calls coerced to ToolCall objects) before
    projection, so replayed logs project no matter which meta shape they
    carry.
    """
    events = [normalize_event(event) for event in events]
    kept, dropped = fit_history(group_events(events), counter, budget)
    messages: list[Message] = []
    if summary:
        messages.append(Message(role="system", content=summary))
    if dropped:
        messages.append(Message(role="system", content=ELISION_NOTE))
    for group in kept:
        messages.extend(events_to_messages(group))
    if memory_facts:
        messages = insert_after_last_user(messages, memory_message(memory_facts))
    return messages, dropped


# ------------------------------------------------------------------ stages


class MemoryInjectStage:
    """The Phase 6 memory seam (H5-2) — default no-op until memory lands.

    ``memory`` must expose ``async relevant(query, k) -> list[str]``. The
    query is the latest user event's text; with no user event (and no
    facts), the stage does nothing. A failing memory backend fails OPEN —
    logged, never fatal (memory must not break the loop).

    Placement: inside the default pipeline the stage runs BEFORE
    HistoryStage, so it stashes facts in ``draft.meta["memory_facts"]`` and
    HistoryStage places them after the latest user turn; composed after a
    history stage it injects directly into ``draft.messages``. Either way
    the facts end up next to the latest user turn — which also means memory
    re-injection after a compact falls out naturally (every rebuild goes
    through the same placement helper).
    """

    def __init__(self, memory: Any = None, k: int = 5) -> None:
        self._memory = memory
        self._k = int(k)

    async def stage(
        self,
        ctx: Any,
        events: list[HarnessEvent],
        tools: list[ToolSpec],
        draft: Draft,
    ) -> Draft:
        if self._memory is None:
            return draft
        query_index = last_user_index(list(events or []))
        if query_index < 0:
            return draft  # nothing to anchor a memory query to
        query = events[query_index].content
        try:
            fetched = await self._memory.relevant(query, self._k)
        except Exception:  # noqa: BLE001 — memory fails open, never fatal
            logger.warning("memory backend failed; continuing without memories", exc_info=True)
            return draft
        facts = [str(fact) for fact in list(fetched or []) if str(fact).strip()]
        if not facts:
            return draft
        draft.meta["memory_facts"] = facts
        if draft.messages:
            # Custom composition: history already built — inject directly.
            draft.messages = insert_after_last_user(draft.messages, memory_message(facts))
        return draft


class ToolMenuStage:
    """Attach the (already-pruned) tool menu — pass-through shaping only.

    Pruning itself is H3-3's job (``registry.specs(context_window)``); this
    stage is where the menu enters the request.
    """

    async def stage(
        self,
        ctx: Any,
        events: list[HarnessEvent],
        tools: list[ToolSpec],
        draft: Draft,
    ) -> Draft:
        draft.tools = list(tools or [])
        return draft


class HistoryStage:
    """Events → fitted messages (H5-2) — newest-first keep, never orphaned.

    ``budget`` is the history slice of the window (:func:`budget_for`). On
    top of fitting, this stage applies compact coverage: a ``compact`` event
    in the replayed log replaces everything it covers with its summary as the
    FIRST system message (see :func:`split_by_compact`) — so after a compact,
    replay alone rebuilds the trimmed conversation. Memory facts stashed by
    an earlier MemoryInjectStage are placed after the latest user turn.
    """

    def __init__(self, counter: TokenCounter, budget: int) -> None:
        self._counter = counter
        self._budget = int(budget)

    async def stage(
        self,
        ctx: Any,
        events: list[HarnessEvent],
        tools: list[ToolSpec],
        draft: Draft,
    ) -> Draft:
        summary, survivors = split_by_compact(list(events or []))
        messages, dropped = project_history(
            survivors,
            self._counter,
            self._budget,
            summary=summary,
            memory_facts=draft.meta.get("memory_facts", ()),
        )
        draft.messages = messages
        draft.meta["history_groups_dropped"] = dropped
        return draft


class CompactCheckStage:
    """The auto-compact gate (H5-3) — hand the finished draft to the engine.

    Delegation is unconditional; the ≥75%-of-window decision lives in the
    compact engine (single source of truth for the threshold), which returns
    the draft unchanged when there is nothing to do.
    """

    def __init__(self, compact_engine: Any) -> None:
        self._engine = compact_engine

    async def stage(
        self,
        ctx: Any,
        events: list[HarnessEvent],
        tools: list[ToolSpec],
        draft: Draft,
    ) -> Draft:
        return await self._engine.maybe_compact(ctx, events, draft)


# ---------------------------------------------------------------- pipeline


class RequestPipeline:
    """The H5-2 request builder — a composable stage list plus hard guards.

    Constructor::

        RequestPipeline(window, counter=None, *, memory=None,
                        compact_engine=None, stages=None)

    - ``window``: the active model's context window (from WindowRegistry).
    - ``counter``: a TokenCounter; defaults to :class:`EstimateCounter`.
    - ``memory`` / ``compact_engine``: the Phase 6 seam and the H5-3 engine;
      when ``stages`` is None the FROZEN default order is wired from them —
      ``[MemoryInjectStage(memory), ToolMenuStage(), HistoryStage(...)]`` plus
      ``CompactCheckStage(compact_engine)`` when an engine is given.
    - ``stages``: a custom stage list (recording stages in tests, exotic
      orders in experiments). Stages may be objects with an ``async stage``
      method or bare async callables of the same signature.

    ``build`` runs the stages in list order, counts the finished request, and
    hard-truncates (leading system messages + last user turn, then tools,
    then content) if the window is STILL exceeded — the pipeline never
    returns an oversized request, and never raises for one.
    """

    def __init__(
        self,
        window: int,
        counter: TokenCounter | None = None,
        *,
        memory: Any = None,
        compact_engine: Any = None,
        stages: Sequence[Any] | None = None,
    ) -> None:
        if window <= 0:
            raise ValueError(f"context window must be positive, got {window!r}")
        self.window = int(window)
        self.counter: TokenCounter = counter if counter is not None else EstimateCounter()
        if stages is None:
            stages = [
                MemoryInjectStage(memory),
                ToolMenuStage(),
                HistoryStage(self.counter, budget_for(self.window)["history"]),
            ]
            if compact_engine is not None:
                stages.append(CompactCheckStage(compact_engine))
        self.stages = list(stages)

    async def build(
        self, ctx: Any, events: list[HarnessEvent], tools: list[ToolSpec]
    ) -> Request:
        """Run every stage in order, then enforce the window (H5-1 guarantee)."""
        draft = Draft()
        for stage in self.stages:
            step = getattr(stage, "stage", stage)
            draft = await step(ctx, events, tools, draft)
            if draft is None:
                raise RuntimeError(
                    f"pipeline stage {stage!r} returned None; stages must return the draft"
                )
        request = Request(messages=list(draft.messages), tools=list(draft.tools))
        if self.counter.count_request(request) > self.window:
            request = self._hard_truncate(request)
        return request

    # ------------------------------------------------------------- last resort

    def _hard_truncate(self, request: Request) -> Request:
        """Pathological fallback: leading system messages + last user turn.

        Order of eviction: history first (keep the head system messages — the
        compact summary and its goal survive — plus ONE hard-truncate note and
        the last user turn), then the tool menu from the tail, then the user
        message's own content. Terminates: content shrinks toward empty and
        ``count_text("")`` is 0.
        """
        system_head: list[Message] = []
        for message in request.messages:
            if message.role != "system":
                break
            system_head.append(message)
        if not any(message.content == HARD_TRUNCATE_NOTE for message in system_head):
            system_head.append(Message(role="system", content=HARD_TRUNCATE_NOTE))
        last_user = next(
            (message for message in reversed(request.messages) if message.role == "user"),
            None,
        )
        tools = list(request.tools)

        def fitted(user_content: str) -> Request:
            messages = list(system_head)
            if last_user is not None:
                messages.append(dataclass_replace(last_user, content=user_content))
            return Request(messages=messages, tools=list(tools))

        content = last_user.content if last_user is not None else ""
        while tools and self.counter.count_request(fitted(content)) > self.window:
            tools.pop()
        while content and self.counter.count_request(fitted(content)) > self.window:
            overflow = self.counter.count_request(fitted(content)) - self.window
            cut = max(16, 4 * overflow + 16)  # 4 chars ≈ 1 token + margin
            content = content[:-cut] if cut < len(content) else ""
        logger.warning(
            "request exceeded the %d-token window after compaction; hard-truncated "
            "history to the last user turn", self.window,
        )
        return fitted(content)
