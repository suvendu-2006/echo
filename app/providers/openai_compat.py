"""The universal OpenAI-compatible adapter (PLAN step 1.2, DECISIONS.md D-002).

One class, config-driven, speaks for every $0 brain that answers on
``POST {base_url}/chat/completions`` with ``stream=true``: OpenAI, Groq,
NVIDIA, Gemini-compat, Z.ai, DeepSeek, Ollama. Swapping brains = one
config line (that is the whole point of D-002).

Contract (FROZEN, app/providers/base.py + D-002):
    adapter.stream(request)  ->  async iterator of StreamEvent,
                                 ending with exactly one type == "end"

Battle scars baked in (PLAN 1.2):
- Streamed tool-call arguments are accumulated BY INDEX — the id/name
  typically arrive only in the first chunk while later chunks carry
  argument string fragments. Matching is on the ``index`` field, never
  on id/name.
- Model output is parsed defensively: malformed argument JSON becomes a
  ``ToolCall.parse_error`` (``arguments == {}``, raw string preserved) and
  a malformed SSE chunk is skipped. Expected events, never crashes.

The transport is injectable: pass ``open_stream`` (called with the JSON
request payload; must return an async iterator of raw SSE bytes) to run
fully offline — tests never touch a socket. The default httpx transport
raises ProviderUnavailable for connect errors, HTTP 429 and 5xx BEFORE
yielding any event — that is the failover signal (PLAN step 1.5) — and
ProviderError (body snippet included) for any other non-200 status.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Callable

import httpx

from app.providers.base import (
    Capability,
    Message,
    ProviderConfigError,
    ProviderError,
    ProviderUnavailable,
    Request,
    StreamEvent,
    ToolCall,
    ToolSpec,
    Usage,
)

#: Seconds before a stalled backend counts as dead (failover-eligible).
DEFAULT_TIMEOUT_SECONDS = 60.0

#: Body snippet cap in error messages — never echo a whole HTML error page.
MAX_ERROR_BODY_CHARS = 300

#: Opens the raw SSE byte stream for one request payload. A callable taking
#: the payload dict and returning an async iterator of bytes; a bare async
#: iterator (or a callable returning an awaitable that resolves to one) is
#: tolerated too — tests and scripts appreciate the laxity.
StreamOpener = Callable[[dict], AsyncIterator[bytes]]


def _coerce_int(value: object) -> int:
    """Defensively coerce a usage counter — fields may be absent or garbage."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


class OpenAICompatProvider:
    """One adapter for every OpenAI-compatible backend (PLAN 1.2, D-002).

    Attributes:
        name: Provider label ("groq", "ollama", ...) — used in errors and
            as the fallback prefix for id-less tool calls.
        model: Model id sent as the wire ``model`` field.
        capability: What this brain can do — defaults to
            ``Capability(provider=name, model=model)``; pass one to claim
            tools/vision/context_window up front (the PLAN 1.4 probe may
            refine it later).

    ``open_stream`` (optional) replaces the HTTP transport: called with the
    JSON request payload it must return an async iterator of raw SSE byte
    chunks. The default posts to ``{base_url}/chat/completions`` with
    ``stream=true`` via httpx and maps failures per the base.py contract.
    """

    def __init__(
        self,
        *,
        name: str,
        model: str,
        base_url: str,
        api_key: str | None = None,
        capability: Capability | None = None,
        open_stream: StreamOpener | None = None,
    ) -> None:
        if not base_url:
            raise ProviderConfigError(f"{name}: base_url is required")
        self.name = name
        self.model = model
        self.capability = capability or Capability(provider=name, model=model)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._open_stream: StreamOpener = open_stream or self._default_open_stream

    # ------------------------------------------------------------- the seam

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Yield the backend's reply as StreamEvents, ending with one "end".

        Event order a caller may rely on: ("text" | "tool_call")*, then
        "usage" when the backend reported spend, then a single "end".
        Transport failures RAISE (never yield) — see module docstring.
        """
        payload = self._build_payload(request)
        source = await self._start_stream(payload)

        buffer = b""
        pending: dict[int, dict[str, str]] = {}  # tool index -> accumulated parts
        final_usage: Usage | None = None
        done = False

        def absorb_tool_fragment(fragment: dict) -> None:
            """Merge one delta tool_call fragment into ``pending`` — BY INDEX.

            The id/name usually ride only on the first chunk; later chunks
            bring argument string fragments. Never assume id/name present.
            """
            index = fragment.get("index")
            if not isinstance(index, int) or isinstance(index, bool):
                index = 0  # single-call providers often omit the index
            entry = pending.setdefault(index, {"id": "", "name": "", "args": ""})
            if isinstance(fragment.get("id"), str) and fragment["id"]:
                entry["id"] = fragment["id"]
            function = fragment.get("function")
            if not isinstance(function, dict):
                return
            name = function.get("name")
            if isinstance(name, str) and name:
                entry["name"] = name
            args = function.get("arguments")
            if isinstance(args, str):
                entry["args"] += args  # argument string fragments, in order
            elif isinstance(args, dict):
                entry["args"] += json.dumps(args)  # rare: object, not string

        def flush_tool_calls() -> list[StreamEvent]:
            """Finish every accumulated call — defensive JSON parse, no crash."""
            events: list[StreamEvent] = []
            for index in sorted(pending):
                entry = pending[index]
                raw = entry["args"]
                arguments: dict = {}
                parse_error: str | None = None
                if raw.strip():
                    try:
                        parsed = json.loads(raw)
                    except ValueError as exc:
                        parse_error = f"invalid JSON: {exc}"
                    else:
                        if isinstance(parsed, dict):
                            arguments = parsed
                        else:
                            parse_error = (
                                f"expected a JSON object, got {type(parsed).__name__}"
                            )
                events.append(
                    StreamEvent(
                        type="tool_call",
                        tool_call=ToolCall(
                            id=entry["id"] or f"{self.name}-toolcall-{index}",
                            name=entry["name"] or "unknown",
                            arguments=arguments,
                            raw_arguments=raw,
                            parse_error=parse_error,
                        ),
                    )
                )
            pending.clear()
            return events

        def process_line(raw_line: str) -> list[StreamEvent]:
            """Handle one SSE line; return events to yield (text / flushes)."""
            nonlocal done, final_usage
            line = raw_line.strip()
            if not line or line.startswith(":"):
                return []  # event separator or keep-alive comment
            if not line.startswith("data:"):
                return []  # event:/id:/retry: and anything else unconsumed
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                done = True
                return []
            try:
                chunk = json.loads(data)
            except ValueError:
                return []  # malformed SSE payload — skipped, never fatal
            if not isinstance(chunk, dict):
                return []
            usage = chunk.get("usage")
            if isinstance(usage, dict):
                final_usage = Usage(
                    prompt_tokens=_coerce_int(usage.get("prompt_tokens")),
                    completion_tokens=_coerce_int(usage.get("completion_tokens")),
                )
            events: list[StreamEvent] = []
            choices = chunk.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                choice = first if isinstance(first, dict) else {}
                delta = choice.get("delta")
                if isinstance(delta, dict):
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        events.append(StreamEvent(type="text", text=content))
                    for fragment in delta.get("tool_calls") or []:
                        if isinstance(fragment, dict):
                            absorb_tool_fragment(fragment)
                if choice.get("finish_reason") is not None:
                    events.extend(flush_tool_calls())
            return events

        async for raw_chunk in source:
            if isinstance(raw_chunk, str):  # tolerate sloppy openers
                raw_chunk = raw_chunk.encode("utf-8")
            buffer += raw_chunk
            while True:
                head, separator, tail = buffer.partition(b"\n")
                if not separator:
                    break
                buffer = tail
                for event in process_line(head.decode("utf-8", errors="replace")):
                    yield event
                if done:
                    break
            if done:
                break

        if not done and buffer.strip():
            # Stream ended without a trailing newline — process the tail too.
            for event in process_line(buffer.decode("utf-8", errors="replace")):
                yield event

        for event in flush_tool_calls():  # no-op if finish_reason already flushed
            yield event
        if final_usage is not None:
            yield StreamEvent(type="usage", usage=final_usage)
        yield StreamEvent(type="end")

    # ------------------------------------------------------- wire conversion

    def _build_payload(self, request: Request) -> dict:
        """Translate the neutral Request into chat-completions wire format."""
        messages: list[dict] = []
        for message in request.messages:
            entry: dict = {"role": message.role, "content": message.content}
            if message.role == "assistant" and message.tool_calls:
                entry["tool_calls"] = [
                    self._tool_call_to_wire(call) for call in message.tool_calls
                ]
            elif message.role == "tool":
                entry["tool_call_id"] = message.tool_call_id or ""
            messages.append(entry)
        payload: dict = {"model": self.model, "messages": messages, "stream": True}
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ]
        return payload

    @staticmethod
    def _tool_call_to_wire(call: ToolCall) -> dict:
        """ToolCall back to wire format. The raw argument string wins: it is
        what the model emitted, and providers expect their own bytes back."""
        raw_arguments = call.raw_arguments or json.dumps(call.arguments)
        return {
            "id": call.id,
            "type": "function",
            "function": {"name": call.name, "arguments": raw_arguments},
        }

    # ----------------------------------------------------------- transports

    async def _start_stream(self, payload: dict) -> AsyncIterator[bytes]:
        """Open the raw byte stream, tolerating the opener shapes above."""
        opener = self._open_stream
        candidate = opener(payload) if callable(opener) else opener
        if inspect.isawaitable(candidate):
            candidate = await candidate
        return candidate  # type: ignore[return-value]

    def _headers(self) -> dict[str, str]:
        """Request headers. The api_key's only consumer on the wire."""
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _default_open_stream(self, payload: dict) -> AsyncIterator[bytes]:
        """The real transport: httpx POST .../chat/completions (stream=true).

        An async generator on purpose: the request fires on first iteration,
        so every failure below raises BEFORE any byte — and therefore before
        any StreamEvent — is produced (base.py contract, PLAN 1.5 failover).
        """
        url = f"{self._base_url}/chat/completions"
        client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS)
        try:
            wire_request = client.build_request(
                "POST", url, json=payload, headers=self._headers()
            )
            response = await client.send(wire_request, stream=True)
        except httpx.HTTPError as exc:  # connect errors, DNS, timeouts
            await client.aclose()
            raise ProviderUnavailable(
                f"{self.name}: cannot reach {url} ({exc})"
            ) from exc
        if response.status_code != 200:
            body = (await response.aread()).decode("utf-8", errors="replace")
            await response.aclose()
            await client.aclose()
            raise self._status_failure(response.status_code, body)

        async def raw_bytes() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()
                await client.aclose()

        return raw_bytes()

    @staticmethod
    def _status_failure(status_code: int, body: str) -> ProviderError:
        """Map a non-200 status to the contract failure (offline-testable).

        429 and 5xx mean dead/overloaded — ProviderUnavailable, the
        failover-eligible signal (PLAN 1.5). Everything else is a runtime
        ProviderError carrying a body snippet for debugging.
        """
        snippet = " ".join(body.split())[:MAX_ERROR_BODY_CHARS]
        detail = f"HTTP {status_code}" + (f": {snippet}" if snippet else "")
        if status_code == 429 or status_code >= 500:
            return ProviderUnavailable(detail)
        return ProviderError(detail)
