"""Harness events (H0-1) — the frozen model's re-export + JSON round-trip.

The ``HarnessEvent`` dataclass itself is FROZEN in app/harness/interfaces.py
(D-006); this module is the convenient import point for stage modules and adds
the serialization helpers the sacred log needs for its payload column
(docs/phase3-contract.md, stage H0).
"""

from __future__ import annotations

import json

from app.harness.interfaces import HarnessEvent, events_to_messages

__all__ = [
    "HarnessEvent",
    "events_to_messages",
    "event_to_json",
    "event_from_json",
]


def event_to_json(event: HarnessEvent) -> str:
    """Serialize one event to the JSON stored in the log's payload column.

    ``default=str`` is deliberate (log-first discipline): a result or meta dict
    that smuggles in a non-JSON value (a Path, a datetime) must never crash the
    append path — it degrades to its string form instead.
    """
    return json.dumps(event.to_dict(), ensure_ascii=False, default=str)


def event_from_json(raw: str) -> HarnessEvent:
    """Rebuild one event from a stored payload.

    Raises ValueError (bad JSON, or valid JSON that is not an object),
    KeyError (missing "type") or TypeError (wrong shape) — ``SessionLog.replay``
    turns those into corrupt-row skips, never crashes.
    """
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(
            f"event payload must be a JSON object, got {type(data).__name__}"
        )
    return HarnessEvent.from_dict(data)
