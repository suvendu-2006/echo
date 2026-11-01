"""The provider seam (PLAN Phase 1.1) — every AI brain plugs in here.

This contract is FROZEN for the whole project (DECISIONS.md D-002): adapters
implement it, everything else consumes it. Changing it is a DECISIONS.md
event, never a casual edit.

The one method:
    adapter.stream(request)  ->  async iterator of StreamEvent, ending with type == "end"

Event order a caller may rely on:
    ("text" | "tool_call" | "usage")*   then   "end"    — or a single "error".

Adapters RAISE (never yield) ProviderUnavailable for a dead/unreachable
backend BEFORE producing any content — that is the failover signal
(PLAN step 1.5, app/providers/failover.py). Other failures raise
ProviderError (runtime) or ProviderConfigError (bad key / bad args).
Adapters must parse tool arguments defensively — malformed model output is
an expected event, never a crash (PLAN step 1.2 "battle scars").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol, runtime_checkable

DEFAULT_CONTEXT_WINDOW = 8192


# ---------------------------------------------------------------- data model


@dataclass
class ToolCall:
    """A tool the model wants to run. Arguments are pre-parsed defensively."""

    id: str
    name: str
    arguments: dict = field(default_factory=dict)  # parsed JSON args ({} if unparseable)
    raw_arguments: str = ""                        # the untouched string, for repair/debug
    parse_error: str | None = None                 # set when arguments could not be parsed


@dataclass
class Message:
    """One conversation turn, provider-neutral."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)  # role == "assistant"
    tool_call_id: str | None = None                           # role == "tool" (result of this call)
    name: str | None = None                                   # tool name for role == "tool"


@dataclass
class ToolSpec:
    """A tool offered to the model: JSON-schema description."""

    name: str
    description: str
    parameters: dict = field(default_factory=dict)  # JSON schema for arguments


@dataclass
class Request:
    """Everything a provider needs for one completion."""

    messages: list[Message]
    tools: list[ToolSpec] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int | None = None


@dataclass
class Usage:
    """Token spend for one call — the $0 budget's heartbeat (PLAN 1.5 / H5-1)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class StreamEvent:
    """One chunk of a streamed reply. Exactly one payload field is meaningful."""

    type: str                    # "text" | "tool_call" | "usage" | "end" | "error"
    text: str = ""               # type == "text": the delta
    tool_call: ToolCall | None = None  # type == "tool_call"
    usage: Usage | None = None   # type == "usage"
    error: str = ""              # type == "error": human-readable reason


@dataclass
class Capability:
    """What a model can do — probed at activation (PLAN step 1.4)."""

    provider: str
    model: str
    tools: bool = False          # native tool calling?
    vision: bool = False
    context_window: int = DEFAULT_CONTEXT_WINDOW


# ----------------------------------------------------------------- failures


class ProviderError(Exception):
    """A provider failed at runtime (bad reply, rejected request)."""


class ProviderConfigError(ProviderError):
    """Configuration problem: missing key, malformed base URL, bad model name."""


class ProviderUnavailable(ProviderError):
    """Backend is unreachable/overloaded — the failover-eligible failure."""


# ------------------------------------------------------------------ the seam


@runtime_checkable
class ProviderAdapter(Protocol):
    """The seam every brain implements. See module docstring for the contract."""

    name: str
    model: str
    capability: Capability

    def stream(self, request: Request) -> AsyncIterator[StreamEvent]: ...


class MockProvider:
    """Reference adapter — replays a scripted stream. The contract's living doc.

    ``script`` is a list of StreamEvent and/or Exception instances, yielded/
    raised in order per call. The script must include its own "end" event.
    Used by tests and by the other Phase 1 modules; costs nothing, hits nothing.
    """

    def __init__(
        self,
        name: str = "mock",
        model: str = "mock-1",
        script: list | None = None,
        capability: Capability | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.capability = capability or Capability(provider=name, model=model)
        self._script = list(script or [])
        self.calls = 0  # how many times stream() was started

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        self.calls += 1
        for item in self._script:
            if isinstance(item, Exception):
                raise item
            yield item
