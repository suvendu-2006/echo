"""Echo context manager (H5) — token budget, processor pipeline, auto-compact.

Package named ``ctx/`` (not ``context/``) on purpose: ``context/`` would
shadow H0's ``app/harness/context.py`` module (contract deviation recorded
2026-11-01 in docs/phase3-contract.md).

Layout:
- ``budget``     — WindowRegistry, TokenCounter/EstimateCounter, budget_for (H5-1)
- ``processors`` — Draft + the composable RequestPipeline stages (H5-2)
- ``compact``    — CompactEngine + ExtractiveSummarizer (H5-3/H5-4)
"""

from app.harness.ctx.budget import (
    KNOWN_CONTEXT_WINDOWS,
    REPLY_HEADROOM_TOKENS,
    EstimateCounter,
    TokenCounter,
    WindowRegistry,
    budget_for,
)
from app.harness.ctx.compact import (
    COMPACT_THRESHOLD,
    DEFAULT_KEEP_RECENT_EVENTS,
    KEEP_NEWEST_TOOL_RESULTS,
    CompactEngine,
    ExtractiveSummarizer,
    elide_old_tool_results,
    safe_coverage_end,
)
from app.harness.ctx.processors import (
    ELISION_NOTE,
    HARD_TRUNCATE_NOTE,
    MEMORY_HEADER,
    CompactCheckStage,
    Draft,
    HistoryStage,
    MemoryInjectStage,
    RequestPipeline,
    ToolMenuStage,
    fit_history,
    group_events,
    insert_after_last_user,
    memory_message,
    project_history,
    split_by_compact,
)

__all__ = [
    # budget (H5-1)
    "KNOWN_CONTEXT_WINDOWS",
    "REPLY_HEADROOM_TOKENS",
    "EstimateCounter",
    "TokenCounter",
    "WindowRegistry",
    "budget_for",
    # processors (H5-2)
    "CompactCheckStage",
    "Draft",
    "ELISION_NOTE",
    "HARD_TRUNCATE_NOTE",
    "HistoryStage",
    "MEMORY_HEADER",
    "MemoryInjectStage",
    "RequestPipeline",
    "ToolMenuStage",
    "fit_history",
    "group_events",
    "insert_after_last_user",
    "memory_message",
    "project_history",
    "split_by_compact",
    # compact (H5-3/H5-4)
    "COMPACT_THRESHOLD",
    "DEFAULT_KEEP_RECENT_EVENTS",
    "KEEP_NEWEST_TOOL_RESULTS",
    "CompactEngine",
    "ExtractiveSummarizer",
    "elide_old_tool_results",
    "safe_coverage_end",
]
