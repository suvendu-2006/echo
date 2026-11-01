"""The frozen harness interfaces (PLAN Phase 3) — every stage builds against these.

This file is the Phase 3 equivalent of app/providers/base.py: changing it is a
DECISIONS.md event, never a casual edit. Concrete implementations live in the
stage modules (events.py, session_log.py, loop.py, tools/, permissions/,
context/, subagents.py) and must satisfy these protocols — the lint test in
tests/test_phase3_contract.py enforces conformance.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from app.providers.base import Message, Request, ToolCall, ToolSpec, Usage


# ------------------------------------------------------------------ danger


class Danger(Enum):
    """Mandatory classification for every tool (H3-3, H4). Untagged = unregistered."""

    SAFE = "safe"        # read-only; may run in parallel with other SAFE tools
    CONFIRM = "confirm"  # the permission engine must allow (and may ask the user) first
    BLOCK = "block"      # registered for visibility, never dispatched


class Decision(Enum):
    """What the permission engine says about one tool call (H4)."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


# ------------------------------------------------------- H0: events + seams


@dataclass
class HarnessEvent:
    """One entry in the sacred log — the single source of truth (D-006).

    Types: "user" | "assistant" | "tool_call" | "tool_result" | "system" |
    "compact" | "status". Exactly one payload field is meaningful per type.
    """

    type: str
    content: str = ""                     # user/assistant/system text
    tool_call: ToolCall | None = None     # "tool_call"
    result: dict | None = None            # "tool_result" (status/result/hint) and compact meta
    usage: Usage | None = None            # "assistant" may carry spend
    meta: dict = field(default_factory=dict)  # goal re-injection, child links, etc.

    @staticmethod
    def _conv_out(value: object) -> object:
        """Serialize meta values: ToolCall objects → dicts (JSON-safe round-trip, D-018)."""
        if isinstance(value, ToolCall):
            return {
                "id": value.id,
                "name": value.name,
                "arguments": value.arguments,
                "raw_arguments": value.raw_arguments,
                "parse_error": value.parse_error,
            }
        if isinstance(value, list):
            return [HarnessEvent._conv_out(v) for v in value]
        if isinstance(value, dict):
            return {k: HarnessEvent._conv_out(v) for k, v in value.items()}
        return value

    @staticmethod
    def _conv_in(value: object) -> object:
        """Rebuild meta values: dicts shaped like ToolCalls → ToolCall objects (D-018)."""
        if isinstance(value, dict) and isinstance(value.get("id"), str) and isinstance(value.get("name"), str):
            return ToolCall(
                id=value["id"],
                name=value["name"],
                arguments=value.get("arguments") or {},
                raw_arguments=value.get("raw_arguments", ""),
                parse_error=value.get("parse_error"),
            )
        if isinstance(value, list):
            return [HarnessEvent._conv_in(v) for v in value]
        if isinstance(value, dict):
            return {k: HarnessEvent._conv_in(v) for k, v in value.items()}
        return value

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "content": self.content,
            "tool_call": (
                {
                    "id": self.tool_call.id,
                    "name": self.tool_call.name,
                    "arguments": self.tool_call.arguments,
                    "raw_arguments": self.tool_call.raw_arguments,
                    "parse_error": self.tool_call.parse_error,
                }
                if self.tool_call
                else None
            ),
            "result": self._conv_out(self.result) if self.result is not None else None,
            "usage": (
                {
                    "prompt_tokens": self.usage.prompt_tokens,
                    "completion_tokens": self.usage.completion_tokens,
                }
                if self.usage
                else None
            ),
            "meta": self._conv_out(self.meta),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HarnessEvent":
        call = data.get("tool_call")
        usage = data.get("usage")
        return cls(
            type=data["type"],
            content=data.get("content", ""),
            tool_call=ToolCall(
                id=call["id"],
                name=call["name"],
                arguments=call.get("arguments") or {},
                raw_arguments=call.get("raw_arguments", ""),
                parse_error=call.get("parse_error"),
            )
            if call
            else None,
            result=data.get("result"),
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            )
            if usage
            else None,
            meta=cls._conv_in(data.get("meta") or {}),
        )


@runtime_checkable
class SessionLogLike(Protocol):
    """The sacred log (H0-2): append-only, replayable, crash-safe (D-006)."""

    def create_session(self, session_id: str, parent_session_id: str | None = None) -> None: ...

    def append(self, session_id: str, event: HarnessEvent) -> int:
        """Append one event; returns its per-session sequence number. Log FIRST."""
        ...

    def replay(self, session_id: str) -> list[HarnessEvent]:
        """Full ordered history; a corrupt tail entry is skipped, never fatal."""
        ...

    def list_sessions(self) -> list[dict]: ...

    def close(self) -> None: ...


@runtime_checkable
class TraceSink(Protocol):
    """Where every log event is mirrored (H7-1). Fast, sync, never raises."""

    def emit(self, session_id: str, event: HarnessEvent) -> None: ...


@dataclass
class HarnessContext:
    """The narrow seam everything plugs through (H0-3, D-006): ctx.llm/.tools/.perms/.state."""

    session_id: str
    llm: object                                   # adapter or ProviderChain (the provider seam)
    log: SessionLogLike
    state: dict = field(default_factory=dict)     # scratch state, tools may read/write
    tools: "ToolDispatcher | None" = None
    perms: "PermissionEngine | None" = None
    trace: TraceSink | None = None
    parent_session_id: str | None = None          # set for subagent child sessions (H6)
    depth: int = 0                                # subagent nesting depth (H6-3)


# ------------------------------------------------- H2/H3/H4/H5: the contracts


@runtime_checkable
class ToolDispatcher(Protocol):
    """The tool desk (H3): validates, asks permissions, runs, caps outputs."""

    def specs(self, context_window: int | None = None) -> list[ToolSpec]: ...

    async def dispatch(self, call: ToolCall, ctx: HarnessContext) -> dict:
        """Returns {"status": "OK"|"ERROR"|"DENIED", "result"/"hint": ...} — never raises."""
        ...


@runtime_checkable
class PermissionEngine(Protocol):
    """The guard (H4): pattern rules, evaluation order hooks→deny→ask→allow→mode; deny wins."""

    async def decide(self, call: ToolCall, danger: Danger, ctx: HarnessContext) -> Decision: ...

    async def ask(self, call: ToolCall, danger: Danger, ctx: HarnessContext) -> Decision:
        """The ask-event hook (H4-4): UI popup or policy; returns ALLOW/DENY only."""
        ...


@runtime_checkable
class RequestPipeline(Protocol):
    """The request builder (H5-2): composable stages — memory → tool menu → history → compact."""

    async def build(
        self, ctx: HarnessContext, events: list[HarnessEvent], tools: list[ToolSpec]
    ) -> Request: ...


@runtime_checkable
class Summarizer(Protocol):
    """Turns old history into a compact summary (H5-3). Injectable; cheap-first."""

    async def summarize(self, events: list[HarnessEvent]) -> str: ...


@dataclass
class LoopGuards:
    """The loop's hard stops (H2-2) — the loop can ALWAYS be forced to stop."""

    max_steps: int = 25
    wall_clock_seconds: float = 300.0
    loop_break_after: int = 3       # same tool+args hash ×N → goal re-injection
    loop_abort_after: int = 4       # ×N+1 → abort the turn cleanly
    tool_result_max_chars: int = 20_000


def events_to_messages(events: list[HarnessEvent]) -> list[Message]:
    """The canonical event→provider-message projection (used by pipelines).

    user/system → role messages; assistant → content (+tool_calls); tool_result →
    role="tool" with tool_call_id from the matching tool_call meta.
    """
    messages: list[Message] = []
    pending_calls: dict[str, ToolCall] = {}
    for event in events:
        if event.type in ("user", "system"):
            messages.append(Message(role=event.type, content=event.content))
        elif event.type == "assistant":
            messages.append(
                Message(role="assistant", content=event.content, tool_calls=event.meta.get("assistant_tool_calls", []))
            )
            for call in event.meta.get("assistant_tool_calls", []):
                pending_calls[call.id] = call
        elif event.type == "tool_result":
            call = event.meta.get("for_call")
            messages.append(
                Message(
                    role="tool",
                    content=str(event.result.get("result", event.result)) if event.result else "",
                    tool_call_id=call.id if call else "",
                    name=call.name if call else None,
                )
            )
    return messages
