"""Auto-compact (H5-3) — cheap compactions first, then summarize (H5-4).

Claude's discipline: compact at 75% of the window BEFORE the model drowns.
``CompactEngine.maybe_compact(ctx, events, draft) -> draft`` is the single
entry point (invoked by the pipeline's ``CompactCheckStage``); it escalates:

1. **Do nothing** while the estimated request stays under the threshold
   (default 75% of the window) — the draft comes back untouched.
2. **Cheap first (H5-4)**: rewrite OLD ``tool_result`` events in the DRAFT
   (never the log — the log is append-only; full outputs live there and in
   spill files) to ``{"status": "ELIDED", "note": ...}``, keeping the newest
   3 intact. Recount; if that alone drops the estimate below the threshold,
   no summary is produced and nothing is appended.
3. **Summarize**: hand everything except a protected recent tail to the
   injectable ``Summarizer`` (zero-model fallback:
   :class:`ExtractiveSummarizer`), append ONE ``compact`` event via
   ``ctx.log`` — ``meta`` keeps ``goal`` (latest user text), the last-5
   touched files from ``ctx.state`` (if present), ``elided_tool_results``,
   and ``covers_seq`` — then rebuild the draft with the summary as the first
   system message.

Coverage & replay (the load-bearing trick): the ``compact`` event's
``meta["covers_seq"]`` is a high-water mark — every event with seq <= it was
summarized. The log is gapless per session (H0-2 gate: seqs 0..n-1), so a
replayed event's list index IS its seq; on every later replay the pipeline's
HistoryStage drops covered events and re-places the summary. Compaction is
therefore reversible-by-replay and the log stays the single source of truth
(D-006).

Protected everywhere, by construction:

- the LATEST user turn is never covered and never fit-dropped — "what was I
  doing?" cannot happen (the goal also rides in every summary: the covered
  prefix always contains the session's first user event, which the
  ExtractiveSummarizer echoes verbatim);
- the coverage boundary never splits an assistant from its tool results
  (:func:`safe_coverage_end` advances the cut past any group that would leave
  an orphaned ``role="tool"`` message);
- the summarize-path rebuild shrinks its history slice in steps until the
  request is under the threshold again — a compact always actually compacts.

Memory re-injection after a compact falls out naturally: the frozen stage
order runs MemoryInjectStage BEFORE HistoryStage, so the memory stage stashes
facts in ``draft.meta`` and every rebuild (cheap or summarized) re-places
them right after the latest user turn via the shared projection helper —
the memory stage never needs to know compaction happened.

``draft`` is duck-typed (any dataclass with ``.messages``/``.tools``/``.meta``
— the pipeline's ``Draft``); rebuilds go through ``dataclasses.replace`` so
this module never imports the pipeline back.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import replace as dataclass_replace
from typing import Any

from app.harness.ctx.budget import TokenCounter, budget_for
from app.harness.ctx.processors import (
    Draft,
    ELISION_NOTE,
    last_user_index,
    project_history,
    split_by_compact,
)
from app.harness.interfaces import HarnessEvent
from app.providers.base import Request

logger = logging.getLogger("echo.harness")

#: Compact when the estimated request reaches this fraction of the window.
COMPACT_THRESHOLD = 0.75

#: Newest tool results always kept intact by the cheap pass (H5-4).
KEEP_NEWEST_TOOL_RESULTS = 3

#: Events after the coverage boundary that compaction never touches.
DEFAULT_KEEP_RECENT_EVENTS = 12


class CompactEngine:
    """The 75% gate: cheap-first elision, then a summarized compact event.

    Constructor::

        CompactEngine(log, counter, summarizer, window, *,
                      threshold=0.75, keep_recent_events=12,
                      keep_tool_results=3)

    ``log`` is the session log the engine was wired with (``ctx.log`` is
    preferred at call time when present — same object in production).
    ``summarizer`` satisfies the frozen ``Summarizer`` protocol; the
    :class:`ExtractiveSummarizer` is the always-available zero-model default.
    """

    def __init__(
        self,
        log: Any,
        counter: TokenCounter,
        summarizer: Any,
        window: int,
        *,
        threshold: float = COMPACT_THRESHOLD,
        keep_recent_events: int = DEFAULT_KEEP_RECENT_EVENTS,
        keep_tool_results: int = KEEP_NEWEST_TOOL_RESULTS,
    ) -> None:
        if window <= 0:
            raise ValueError(f"context window must be positive, got {window!r}")
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"compact threshold must be in (0, 1], got {threshold!r}")
        self._log = log
        self._counter = counter
        self._summarizer = summarizer
        self._window = int(window)
        self._threshold = float(threshold)
        self._keep_recent = max(1, int(keep_recent_events))
        self._keep_tool_results = max(0, int(keep_tool_results))
        self._history_budget = budget_for(self._window)["history"]

    # ------------------------------------------------------------------ api

    @property
    def threshold_tokens(self) -> int:
        """The ≥-trigger in tokens: ``threshold * window`` (floor)."""
        return int(self._threshold * self._window)

    async def maybe_compact(
        self, ctx: Any, events: list[HarnessEvent], draft: Draft
    ) -> Draft:
        """Escalate cheap → summarize while the draft is ≥ 75% of the window.

        Returns the draft unchanged (same object) below the threshold; a new
        draft (via ``dataclasses.replace``) after a cheap elision or a
        summarized compact. Appends at most ONE log event per call.
        """
        events = list(events or [])
        estimate = self._counter.count_request(
            Request(messages=list(draft.messages), tools=list(draft.tools))
        )
        if estimate < self.threshold_tokens:
            return draft
        memory_facts = tuple(draft.meta.get("memory_facts", ()) or ())

        # ---- H5-4: cheap first — elide old tool results, DRAFT only.
        _prior_summary, base_events = split_by_compact(events)
        elided, n_elided = elide_old_tool_results(base_events, self._keep_tool_results)
        cheap_messages, _dropped = project_history(
            elided,
            self._counter,
            self._history_budget,
            summary=_prior_summary,
            memory_facts=memory_facts,
        )
        cheap_estimate = self._counter.count_request(
            Request(messages=cheap_messages, tools=list(draft.tools))
        )
        if cheap_estimate < self.threshold_tokens:
            logger.info(
                "compact(cheap): elided %d old tool result(s) — no summary needed", n_elided
            )
            return dataclass_replace(
                draft,
                messages=cheap_messages,
                meta={**draft.meta, "compaction": "cheap", "elided_tool_results": n_elided},
            )

        # ---- Summarize: cover everything except the protected recent tail.
        cut = self._coverage_end(events)
        if cut <= 0:
            # Nothing is old enough to cover (e.g. the whole session is the
            # current turn) — return the cheap elision as the best effort.
            logger.info(
                "compact: nothing coverable yet (%d events); serving cheap elision", len(events)
            )
            return dataclass_replace(
                draft,
                messages=cheap_messages,
                meta={**draft.meta, "compaction": "cheap", "elided_tool_results": n_elided},
            )
        existing_covers = _newest_coverage_mark(events)
        if existing_covers is not None and cut - 1 <= existing_covers:
            # A compact event already covers at least this much — appending
            # another would summarize nothing new (the mandatory tail alone
            # exceeds the threshold; the pipeline's window guard handles the
            # remainder). Serve the cheap elision instead of spamming the log.
            return dataclass_replace(
                draft,
                messages=cheap_messages,
                meta={**draft.meta, "compaction": "cheap", "elided_tool_results": n_elided},
            )
        covered = events[:cut]
        summary = await self._summarizer.summarize(covered)
        if not str(summary).strip():
            summary = "(compaction produced no summary)"  # coverage must not lose the goal
        user_index = last_user_index(events)
        goal = events[user_index].content if user_index >= 0 else ""
        compact_event = HarnessEvent(
            type="compact",
            content=str(summary),
            meta={
                "goal": goal,
                "touched_files": self._touched_files(ctx),
                "elided_tool_results": n_elided,
                "covers_seq": cut - 1,  # events[:cut] are summarized by this event
            },
        )
        log = getattr(ctx, "log", None) or self._log
        compact_seq = log.append(ctx.session_id, compact_event)
        if compact_seq != len(events):
            # Coverage marks assume the gapless per-session seq of H0-2; a
            # mismatch means the replay was not the full session — log it and
            # continue (the mark stays self-consistent for this projection).
            logger.warning(
                "compact: event seq %d but %d events replayed; coverage marks "
                "assume a gapless log", compact_seq, len(events),
            )
        logger.info(
            "compact: summarized %d event(s) into a compact event (%d tool result(s) "
            "elided) for session %s", cut, n_elided, getattr(ctx, "session_id", "?"),
        )
        messages = self._rebuild_after_compact(
            events[cut:], str(summary), memory_facts, list(draft.tools)
        )
        return dataclass_replace(
            draft,
            messages=messages,
            meta={
                **draft.meta,
                "compaction": "summarize",
                "elided_tool_results": n_elided,
                "compact_covered_events": cut,
            },
        )

    # -------------------------------------------------------------- internals

    def _coverage_end(self, events: list[HarnessEvent]) -> int:
        """How many leading events the summary may cover (boundary-safe).

        Proposal: everything except the ``keep_recent_events`` newest — but
        never reaching past the latest user turn, and never splitting an
        assistant from its tool results (:func:`safe_coverage_end`).
        """
        proposed = len(events) - self._keep_recent
        user_index = last_user_index(events)
        if user_index >= 0:
            proposed = min(proposed, user_index)
        return safe_coverage_end(events, max(0, proposed))

    def _rebuild_after_compact(
        self,
        survivors: list[HarnessEvent],
        summary: str,
        memory_facts: tuple[str, ...],
        tools: list[Any],
    ) -> list[Any]:
        """Post-compact draft: summary first, survivors shrunk until under 75%.

        The history slice shrinks in steps (full → half → mandatory-only) so a
        compact always actually brings the estimate below the threshold; the
        latest user turn survives every step (the fit keeps it mandatory).
        """
        messages: list[Any] = []
        for factor in (1.0, 0.5, 0.0):
            messages, _dropped = project_history(
                survivors,
                self._counter,
                int(self._history_budget * factor),
                summary=summary,
                memory_facts=memory_facts,
            )
            estimate = self._counter.count_request(
                Request(messages=messages, tools=list(tools))
            )
            if estimate < self.threshold_tokens:
                break
        return messages

    def _touched_files(self, ctx: Any) -> list[str]:
        """Last-5-touched files from ``ctx.state`` when present (H5-3 meta)."""
        raw = (getattr(ctx, "state", None) or {}).get("touched_files")
        if not raw:
            return []
        if isinstance(raw, str):
            return [raw]
        try:
            return [str(item) for item in list(raw)[-5:]]
        except TypeError:  # not iterable — treat as one value
            return [str(raw)]


# ------------------------------------------------- pure compaction helpers


def _newest_coverage_mark(events: list[HarnessEvent]) -> int | None:
    """``covers_seq`` of the newest compact event; None when there is none.

    A compact event without an int mark counts as covering everything before
    itself (the same fallback :func:`split_by_compact` applies), so callers
    never double-summarize behind a hand-crafted compact.
    """
    for index in range(len(events) - 1, -1, -1):
        event = events[index]
        if event.type != "compact":
            continue
        covers = event.meta.get("covers_seq")
        return covers if isinstance(covers, int) else index - 1
    return None


def elide_old_tool_results(
    events: list[HarnessEvent], keep_newest: int = KEEP_NEWEST_TOOL_RESULTS
) -> tuple[list[HarnessEvent], int]:
    """Rewrite old ``tool_result`` events to ELIDED stubs (H5-4, DRAFT only).

    Returns ``(new_event_list, elided_count)``. The newest ``keep_newest``
    tool results stay intact (the model may still be acting on them); older
    ones become ``{"status": "ELIDED", "note": ...}`` with their pairing meta
    preserved so the projection still attaches them to the right tool_call id.
    The input list and its events are NEVER mutated — the sacred log keeps
    every full output (plus spill files).
    """
    result_indexes = [i for i, event in enumerate(events) if event.type == "tool_result"]
    to_elide = (
        set(result_indexes[:-keep_newest]) if len(result_indexes) > keep_newest else set()
    )
    rebuilt: list[HarnessEvent] = []
    count = 0
    for index, event in enumerate(events):
        if index not in to_elide:
            rebuilt.append(event)
            continue
        call = (event.meta or {}).get("for_call")
        name = getattr(call, "name", None) or "tool"
        rebuilt.append(
            HarnessEvent(
                type="tool_result",
                result={
                    "status": "ELIDED",
                    "note": (
                        f"output of '{name}' elided by auto-compact; the full result "
                        "remains in the sacred log (and spill files)"
                    ),
                },
                meta=dict(event.meta or {}),
            )
        )
        count += 1
    return rebuilt, count


def _matching_assistant_index(events: list[HarnessEvent], result_index: int) -> int:
    """Index of the assistant event that issued this tool result; -1 if none."""
    result = events[result_index]
    call = (result.meta or {}).get("for_call")
    call_id = getattr(call, "id", None) or (result.meta or {}).get("for_call_id")
    for index in range(result_index - 1, -1, -1):
        event = events[index]
        if event.type != "assistant":
            continue
        for issued in (event.meta or {}).get("assistant_tool_calls", []) or []:
            issued_id = (
                issued.get("id") if isinstance(issued, dict) else getattr(issued, "id", None)
            )
            if issued_id == call_id:
                return index
    return -1


def safe_coverage_end(events: list[HarnessEvent], proposed: int) -> int:
    """Smallest cut n >= proposed whose coverage never orphans a tool result.

    Coverage ``events[:n]`` is safe when every tool_result AFTER the cut finds
    its issuing assistant after the cut too (a kept result whose assistant was
    summarized away would project to a ``role="tool"`` message with a dangling
    tool_call_id — providers reject that). The cut only ever moves FORWARD
    (cover a little more) and one forward pass is sufficient: when the cut
    advances past result i, every earlier index is inside coverage.
    """
    cut = max(0, min(int(proposed), len(events)))
    for index in range(cut, len(events)):
        if events[index].type != "tool_result":
            continue
        assistant_index = _matching_assistant_index(events, index)
        if 0 <= assistant_index < cut:
            cut = index + 1
    return cut


class ExtractiveSummarizer:
    """The zero-model fallback ``Summarizer`` — no provider calls, ever.

    Head of the first user event (the goal) + what happened (tool calls by
    name) + tail of the latest assistant/user content, headed by an explicit
    marker that the summary is extractive. Keeps the session's goal alive
    across compactions: the covered prefix always contains the first user
    event, so every summary re-echoes it.
    """

    def __init__(
        self,
        head_chars: int = 500,
        tail_chars: int = 500,
        max_tool_names: int = 8,
    ) -> None:
        self._head = int(head_chars)
        self._tail = int(tail_chars)
        self._max_names = int(max_tool_names)

    async def summarize(self, events: list[HarnessEvent]) -> str:
        events = list(events or [])
        lines: list[str] = ["[extractive summary — model summarizer unavailable]"]
        first_user = next(
            (event.content for event in events if event.type == "user" and event.content), ""
        )
        if first_user:
            lines.append(f"Goal (first user request): {first_user[: self._head]}")
        calls: Counter[str] = Counter()
        for event in events:
            if event.type == "tool_call" and event.tool_call is not None:
                calls[event.tool_call.name] += 1
        if not calls:  # histories without tool_call events: count assistant meta
            for event in events:
                if event.type != "assistant":
                    continue
                for issued in (event.meta or {}).get("assistant_tool_calls", []) or []:
                    name = (
                        issued.get("name")
                        if isinstance(issued, dict)
                        else getattr(issued, "name", None)
                    )
                    if name:
                        calls[str(name)] += 1
        if calls:
            total = sum(calls.values())
            top = sorted(calls, key=lambda name: (-calls[name], name))[: self._max_names]
            breakdown = ", ".join(f"{name} x{calls[name]}" for name in top)
            lines.append(f"Activity: {total} tool call(s): {breakdown}")
        tail_event = next(
            (
                event
                for event in reversed(events)
                if event.type in ("assistant", "user") and event.content
            ),
            None,
        )
        if tail_event is not None:
            body = tail_event.content[-self._tail :]
            prefix = "..." if len(tail_event.content) > self._tail else ""
            lines.append(f"Latest activity ({tail_event.type}): {prefix}{body}")
        if len(lines) == 1:
            lines.append("(nothing to summarize)")
        return "\n".join(lines)
