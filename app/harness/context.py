"""HarnessContext — the narrow seam (H0-3). The canonical import point.

Stage modules (loop, tools, permissions, subagents) import the context from
here, not from interfaces.py directly, so :func:`make_ctx` stays the one place
that constructs the context and wires the seams together (D-006).
"""

from __future__ import annotations

from app.harness.interfaces import HarnessContext, SessionLogLike, TraceSink

__all__ = ["HarnessContext", "make_ctx"]


def make_ctx(
    session_id: str,
    llm: object,
    log: SessionLogLike,
    *,
    state: dict | None = None,
    tools: object | None = None,
    perms: object | None = None,
    trace: TraceSink | None = None,
    parent_session_id: str | None = None,
    depth: int = 0,
) -> HarnessContext:
    """Build a HarnessContext and wire its trace tap (H7-1 seam).

    If ``trace`` is given and the log supports taps (``SessionLog.add_trace``),
    the sink is registered so every subsequent append is mirrored to it — one
    wiring point, no loop or tool code ever touches tracing directly. ``state``
    is shared by reference (tools read/write ``ctx.state`` by design).
    """
    if trace is not None:
        add_trace = getattr(log, "add_trace", None)
        if callable(add_trace):
            add_trace(trace)
    return HarnessContext(
        session_id=session_id,
        llm=llm,
        log=log,
        state=state or {},
        tools=tools,
        perms=perms,
        trace=trace,
        parent_session_id=parent_session_id,
        depth=depth,
    )
