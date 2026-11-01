"""The tool efficiency engine (PLAN 4.4) — every token must earn its place.

``per-query tool pruning, parallel independent calls, result caching/squeezing``
— mapped to the three pieces below. The $0 budget's hard currency is context
tokens: everything this module does either removes a redundant call (cache) or
shrinks what an already-made call costs in the window (squeeze); parallelism
turns wall-clock into a free resource instead of a second model round-trip.

1. Per-query tool pruning is the registry's job (H3-3, ``specs(window)``) and
   already landed; this module adds the runtime halves:
2. ``ToolResultCache`` — LRU + TTL memo of SAFE (read-only) tool results,
   keyed by ``(tool_name, canonical-json-arguments)``. The same lookup asked
   twice costs ONE dispatch. Mutating tools are NEVER cached — their second
   call is not the same event (a "safe" cache is a correctness property, not
   a hint), and ERROR envelopes are not cached either (a transient failure
   must not be pinned for the TTL).
3. ``Squeezer`` — head+tail with an elision marker: a huge tool result keeps
   its start and end, the middle is dropped from the context.
4. ``dispatch_with_efficiency(...)`` — the wrapper the loop calls instead of
   ``dispatcher.dispatch_many``: cache lookup per call (hit →
   ``{"status": "OK", "cached": True, "result": ...}``), identical SAFE calls
   inside one batch collapse to one dispatch, misses go through
   ``dispatch_many`` (SAFE parallel / CONFIRM serialized — unchanged), every
   result squeezed on the way out, SAFE/OK results stored.

Cacheability is decided by ``dispatcher.danger_of`` (the H2 danger seam);
a dispatcher WITHOUT it, or one that raises, is treated as CONFIRM —
unclassifiable tools are never cached (fail-safe, like everywhere else).
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from typing import Any, Callable

from app.harness.interfaces import Danger, HarnessContext
from app.providers.base import ToolCall

#: Monotonic clock seam (tests inject a controllable fake).
Clock = Callable[[], float]

#: Marker length budget inside Squeezer — fits any realistic elision count.
_SQUEEZE_RESERVE = 80

#: Cap on a malformed dispatcher envelope echoed into a hint.
_MAX_ENVELOPE_HINT = 200


def canonical_arguments(arguments: dict) -> str:
    """Arguments as canonical JSON: sorted keys, no whitespace.

    ``{"b": 1, "a": 2}`` and ``{"a": 2, "b": 1}`` are the same cache entry.
    Non-JSON-native values degrade via ``str`` — ToolCall arguments are
    JSON-parsed by construction, this is defense, not a crash path.
    """
    return json.dumps(arguments or {}, sort_keys=True, separators=(",", ":"), default=str)


def cache_key(tool_name: str, arguments: dict) -> tuple[str, str]:
    """The cache key: ``(tool_name, canonical_json_args)``."""
    return (tool_name, canonical_arguments(arguments))


class ToolResultCache:
    """LRU + TTL memo for tool results (PLAN 4.4, piece 2).

    - LRU: ``max_entries`` oldest-touched entries are evicted on overflow.
    - TTL: entries older than ``ttl_seconds`` (by the injectable monotonic
      ``clock``) are expired on access — a stale read-only answer is a wrong
      answer.
    - Values are stored per ``cache_key``; ``get`` of a missing/expired key
      returns None (cached values are envelopes — always dicts).
    """

    def __init__(
        self,
        max_entries: int = 128,
        ttl_seconds: float = 600.0,
        clock: Clock | None = None,
    ) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise RuntimeError(f"max_entries must be a positive int, got {max_entries!r}")
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)) or ttl_seconds <= 0:
            raise RuntimeError(f"ttl_seconds must be a positive number, got {ttl_seconds!r}")
        self._max_entries = max_entries
        self._ttl_seconds = float(ttl_seconds)
        self._clock: Clock = clock or time.monotonic
        self._entries: "OrderedDict[tuple[str, str], tuple[float, Any]]" = OrderedDict()

    def get(self, key: tuple[str, str]) -> Any:
        """The cached value, or None on miss/expiry (refreshes LRU order)."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if self._clock() - stored_at >= self._ttl_seconds:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return value

    def put(self, key: tuple[str, str], value: Any) -> None:
        """Store one entry (newest); evicts the least-recently-used on overflow."""
        self._entries[key] = (self._clock(), value)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)


class Squeezer:
    """Head+tail squeeze for oversized tool results (PLAN 4.4, piece 3).

    A pure function: text in, shorter text out. Under ``max_chars`` the text
    passes through untouched; above it, the head and tail survive and the
    elided middle is announced by an exact-count marker. The model keeps the
    shape of the answer (its beginning and its end) at a fixed token cost.
    """

    def __init__(self, max_chars: int = 2000) -> None:
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
            raise RuntimeError(f"max_chars must be a positive int, got {max_chars!r}")
        self._max_chars = max_chars

    def squeeze(self, text: object) -> str:
        """Squeeze ``text`` to at most ``max_chars`` (idempotent for small text)."""
        if not isinstance(text, str):
            text = str(text)
        if len(text) <= self._max_chars:
            return text
        head_keep = max(1, (self._max_chars - _SQUEEZE_RESERVE) * 2 // 3)
        tail_keep = max(1, (self._max_chars - _SQUEEZE_RESERVE) - head_keep)
        head = text[:head_keep]
        while True:  # converges in one pass; keeps the marker's count exact
            tail = text[-tail_keep:] if tail_keep else ""
            elided = len(text) - head_keep - tail_keep
            squeezed = f"{head}\n[... {elided} characters elided ...]\n{tail}"
            if len(squeezed) <= self._max_chars or tail_keep <= 0:
                return squeezed
            tail_keep = max(0, tail_keep - (len(squeezed) - self._max_chars))


def _danger_of(dispatcher: Any, call: ToolCall) -> Danger:
    """The dispatcher's verdict; absent/broken/unclassifiable → CONFIRM.

    CONFIRM is the safe default on every seam in Echo (the dispatcher does
    the same for unknown tools): here it means "possibly mutating → never
    cached, always dispatched".
    """
    getter = getattr(dispatcher, "danger_of", None)
    if not callable(getter):
        return Danger.CONFIRM
    try:
        danger = getter(call)
    except Exception:
        return Danger.CONFIRM
    return danger if isinstance(danger, Danger) else Danger.CONFIRM


def _clean_envelope(raw: object) -> dict:
    """A dispatcher envelope we can hand on; malformed ones become ERROR."""
    if isinstance(raw, dict) and isinstance(raw.get("status"), str):
        return dict(raw)
    hint = f"dispatcher returned a malformed envelope: {raw!r}"[:_MAX_ENVELOPE_HINT]
    return {"status": "ERROR", "hint": hint}


def _squeezed_view(envelope: dict, squeezer: "Squeezer | None", *, cached: bool) -> dict:
    """A fresh dict view of ``envelope`` with the result text squeezed."""
    view = dict(envelope)
    if squeezer is not None and isinstance(view.get("result"), str):
        view["result"] = squeezer.squeeze(view["result"])
    if cached:
        view["cached"] = True
    return view


async def dispatch_with_efficiency(
    dispatcher: Any,
    calls: list[ToolCall],
    ctx: HarnessContext | None,
    cache: ToolResultCache | None = None,
    squeezer: Squeezer | None = None,
) -> list[dict]:
    """``dispatcher.dispatch_many`` with the PLAN 4.4 efficiency applied.

    Per call, in input order:

    - SAFE + cache hit → ``{"status": "OK", "cached": True, "result": ...}``
      (no dispatch happens);
    - SAFE + identical call already pending in THIS batch → collapsed onto
      the first call's result (marked ``"cached": True`` — one dispatch
      serves all identical SAFE calls);
    - everything else is a miss → forwarded to ``dispatcher.dispatch_many``
      (which keeps the parallelism rule: SAFE concurrent, CONFIRM serialized).

    Results are returned in input order, each squeezed through ``squeezer``
    (when given) — including cache hits, so the squeezer's budget applies to
    every envelope that leaves this function. Only SAFE + OK misses are
    stored in the cache: mutating tools and ERROR envelopes never enter it.

    ``cache=None`` (default) means no cross-call memoization — identical
    calls within one batch still collapse; ``squeezer=None`` means no
    squeezing. The dispatcher's own contract holds: it never raises per call.
    """
    results: list[dict | None] = [None] * len(calls)

    # ---- pass 1: cache hits, batch-local dedupe targets, collect the misses.
    miss_calls: list[ToolCall] = []
    miss_targets: list[list[int]] = []  # parallel to miss_calls: result slots
    miss_keys: list[tuple[str, str] | None] = []  # parallel; None = not cacheable
    pending: dict[tuple[str, str], list[int]] = {}

    for index, call in enumerate(calls):
        cacheable = _danger_of(dispatcher, call) is Danger.SAFE
        key = cache_key(call.name, call.arguments) if cacheable else None
        if cacheable and cache is not None:
            hit = cache.get(key)
            if hit is not None:
                results[index] = _squeezed_view(hit, squeezer, cached=True)
                continue
        if cacheable and key in pending:
            pending[key].append(index)  # identical SAFE call in this batch
            continue
        targets = [index]
        if cacheable:
            pending[key] = targets
        miss_calls.append(call)
        miss_targets.append(targets)
        miss_keys.append(key)

    # ---- pass 2: dispatch the misses (SAFE parallel / CONFIRM serialized
    # inside dispatch_many — the parallelism rule lives there, unchanged).
    if miss_calls:
        envelopes = await dispatcher.dispatch_many(miss_calls, ctx)
        for targets, key, call, raw in zip(miss_targets, miss_keys, miss_calls, envelopes):
            envelope = _clean_envelope(raw)
            store = (
                cache is not None
                and key is not None
                and envelope.get("status") == "OK"
            )
            if store:
                cache.put(key, dict(envelope))
            for offset, index in enumerate(targets):
                # slot 0 is the dispatched call itself; the rest are batch
                # duplicates served from its result (marked cached).
                results[index] = _squeezed_view(envelope, squeezer, cached=offset > 0)

    return [result or {"status": "ERROR", "hint": "dispatch produced no result"} for result in results]
