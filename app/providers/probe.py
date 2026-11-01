"""Capability probe + prompt-tools fallback (PLAN step 1.4).

Two pieces, both built on base.py types only:

    probe_capabilities(adapter)   ->  Capability   # the 5-second test call
    TextProtocolAdapter(inner)    ->  adapter      # the "TOOL: name(args)" fallback

The probe asks a model to make one tool call and watches the stream for a
tool_call event: native tool calling proven or denied. A probe failure is
information, never a crash — timeout or any adapter failure simply probes
as tools=False (and a wrapped fallback still gets hands, see below).

TextProtocolAdapter wraps any adapter whose model cannot call tools
natively. It teaches the model a plain-text protocol —

    TOOL: <tool_name> {"arg": "value"}    (one call per line, JSON arguments)

— then reads such lines back out of the model's text at the inner end
event and re-emits them as real tool_call events. Weak local models still
get hands (PLAN step 1.4 "prompt-tools fallback").
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator

from app.providers.base import (
    Capability,
    Message,
    ProviderAdapter,
    Request,
    StreamEvent,
    ToolCall,
    ToolSpec,
)

#: The probe's whole conversation: one nudge, one tool on the menu.
PROBE_PROMPT = "Reply with the tool call for a 5-second capability test."

#: The one tool offered during the probe (echoes its ``text`` argument back).
PROBE_TOOL = ToolSpec(
    name="echo_test",
    description="Repeats its argument back.",
    parameters={"type": "object", "properties": {"text": {"type": "string"}}},
)

#: Prepended by TextProtocolAdapter as a system message — the fallback's
#: whole grammar. Tool specs themselves still ride along in request.tools.
PROTOCOL_INSTRUCTION = (
    "You can use tools. To call one, output a line exactly in this form: "
    'TOOL: <tool_name> {"arg": "value"} — one call per line, JSON arguments required.'
)

#: One protocol line: leading whitespace tolerated, a name, then arguments.
_TOOL_LINE_RE = re.compile(r"^[ \t]*TOOL:[ \t]*(\S+)[ \t]*(.*)$")


# --------------------------------------------------------------- the probe


async def probe_capabilities(adapter: ProviderAdapter, timeout: float = 5.0) -> Capability:
    """The 5-second test call at activation: does this model do native tools?

    Sends a minimal request offering one tool and consumes the stream under
    ``asyncio.wait_for(timeout)``. ``tools`` is True iff at least one
    tool_call event arrives. Everything else (provider / model /
    context_window) is taken from ``adapter.capability`` when it has one;
    vision is never probed and stays False.

    Never raises: a timeout or any adapter failure probes as tools=False —
    a probe failure is information, not a crash. (A cancelled probe task
    still propagates cancellation, as is right.)
    """
    request = Request(messages=[Message(role="user", content=PROBE_PROMPT)], tools=[PROBE_TOOL])
    saw_tool_call = False

    async def _consume() -> None:
        nonlocal saw_tool_call
        async for event in adapter.stream(request):
            if event.type == "tool_call":
                saw_tool_call = True

    try:
        await asyncio.wait_for(_consume(), timeout)
    except Exception:  # noqa: BLE001 — timeout, dead backend, junk reply: all probe as False
        saw_tool_call = False

    return _capability_from(adapter, tools=saw_tool_call)


def _capability_from(adapter: ProviderAdapter, *, tools: bool) -> Capability:
    """Build the probe result: identity from the adapter, tools from the probe."""
    reported = getattr(adapter, "capability", None)
    if isinstance(reported, Capability):
        return Capability(
            provider=reported.provider,
            model=reported.model,
            tools=tools,
            vision=False,
            context_window=reported.context_window,
        )
    return Capability(
        provider=str(getattr(adapter, "name", "unknown")),
        model=str(getattr(adapter, "model", "unknown")),
        tools=tools,
    )


# ---------------------------------------------------------- the fallback


class TextProtocolAdapter:
    """Gives weak local models hands via the ``TOOL: name(args)`` protocol.

    The wrapper CREATES tool ability: ``capability`` mirrors the inner
    adapter's with tools forced True.

    Requests WITHOUT tools pass through untouched. Requests WITH tools get
    a system message prepended (``PROTOCOL_INSTRUCTION``) and the wrapper
    buffers the model's text; at the inner end event every ``TOOL:`` line
    is parsed defensively. Parsed lines become real tool_call events and
    are suppressed from the text. Broken ones are reported as tool_call
    events with ``parse_error`` set and the raw line preserved — AND stay
    visible in the text, so the harness can react and the user still sees
    what the model said.

    Output shape: text first, then native tool_call/usage events the inner
    adapter produced (arrival order), then the extracted tool_call events,
    then exactly one end (inner's, or synthesized if inner never sent one).
    An inner ``error`` event flushes the raw text and is forwarded as-is
    with no end — error replaces end. An inner EXCEPTION flushes the raw
    text and propagates unchanged: the failover signal (PLAN step 1.5)
    must survive the wrapper.
    """

    def __init__(self, inner: ProviderAdapter) -> None:
        self._inner = inner

    # ------------------------------------------------------- the seam

    @property
    def name(self) -> str:
        return str(getattr(self._inner, "name", "text-protocol"))

    @property
    def model(self) -> str:
        return str(getattr(self._inner, "model", "unknown"))

    @property
    def capability(self) -> Capability:
        """Inner's capability with tools=True — this wrapper creates tool ability."""
        reported = getattr(self._inner, "capability", None)
        if isinstance(reported, Capability):
            return Capability(
                provider=reported.provider,
                model=reported.model,
                tools=True,
                vision=reported.vision,
                context_window=reported.context_window,
            )
        return Capability(provider=self.name, model=self.model, tools=True)

    def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        return self._stream(request)

    async def _stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        if not request.tools:
            # No tools asked for: the wrapper is invisible.
            async for event in self._inner.stream(request):
                yield event
            return

        prepared = Request(
            messages=[Message(role="system", content=PROTOCOL_INSTRUCTION), *request.messages],
            tools=list(request.tools),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        text_chunks: list[str] = []
        native: list[StreamEvent] = []  # inner tool_call / usage, arrival order
        end_event: StreamEvent | None = None
        error_event: StreamEvent | None = None

        try:
            async for event in self._inner.stream(prepared):
                if event.type == "text":
                    text_chunks.append(event.text)
                elif event.type == "end":
                    end_event = event
                    break
                elif event.type == "error":
                    error_event = event
                    break
                else:
                    native.append(event)
        except Exception:
            # Inner died mid-call: flush what was already said so nothing is
            # lost, then let the failure propagate (failover must see it).
            if text_chunks:
                yield StreamEvent(type="text", text="".join(text_chunks))
            raise

        if error_event is not None:
            # The call failed: flush the raw text (a partial stream is no place
            # for protocol parsing) and forward the error. Error replaces end.
            if text_chunks:
                yield StreamEvent(type="text", text="".join(text_chunks))
            yield error_event
            return

        visible_text, calls = _extract_tool_calls("".join(text_chunks))
        if visible_text:
            yield StreamEvent(type="text", text=visible_text)
        for event in native:
            yield event
        for call in calls:
            yield StreamEvent(type="tool_call", tool_call=call)
        if end_event is not None:
            yield end_event
        else:
            # Inner violated the contract (no end event); keep exactly one.
            yield StreamEvent(type="end")


def _extract_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Pull ``TOOL:`` lines out of accumulated text, defensively, never crashing.

    Returns ``(visible_text, calls)``. Lines are matched only at line start
    (prose mentioning TOOL: mid-sentence is left alone). A parsed line is
    suppressed from the text; a broken line stays visible AND yields a
    ToolCall with ``parse_error`` set. Ids are ``text-1``, ``text-2``, ...
    in document order across parsed and broken calls alike.
    """
    kept: list[str] = []
    calls: list[ToolCall] = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        match = _TOOL_LINE_RE.match(body)
        if match is None:
            kept.append(line)
            continue
        name, rest = match.group(1), match.group(2).strip()
        call_id = f"text-{len(calls) + 1}"
        arguments, parse_error = _parse_tool_arguments(rest)
        if parse_error is None:
            calls.append(ToolCall(id=call_id, name=name, arguments=arguments, raw_arguments=rest))
        else:
            calls.append(ToolCall(id=call_id, name=name, raw_arguments=body, parse_error=parse_error))
            kept.append(line)
    return "".join(kept), calls


def _parse_tool_arguments(rest: str) -> tuple[dict, str | None]:
    """Parse the arguments part of a ``TOOL:`` line into ``(arguments, parse_error)``.

    ``parse_error`` is None iff ``arguments`` is a parsed JSON object (dict);
    a JSON value of any other type counts as broken. Never raises.
    """
    if not rest:
        return {}, "missing JSON arguments after the tool name"
    start, end = rest.find("{"), rest.rfind("}")
    if start == -1 or end <= start:
        return {}, f"no JSON object found after the tool name (got {rest!r})"
    try:
        parsed = json.loads(rest[start : end + 1])
    except ValueError as exc:  # json.JSONDecodeError and anything else odd
        return {}, f"invalid JSON arguments: {exc}"
    if not isinstance(parsed, dict):
        return {}, f"arguments must be a JSON object, got {type(parsed).__name__}"
    return parsed, None
