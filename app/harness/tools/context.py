"""ToolContext (H3-2) — the opt-in window tools get onto the harness.

A tool that needs harness access declares a final parameter ``ctx: ToolContext``;
the dispatcher builds one per call and injects it. Deliberately narrow: a tool
sees the session id, the shared scratch ``state`` dict, and the sacred log —
nothing else (no provider seam, no dispatcher, no permission engine).

``append_event`` is the log-first helper (D-006): a tool that mutates the
world beyond its return value (spawning work, noting progress) appends its
own event BEFORE returning, so a crash right after still left a trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.harness.interfaces import HarnessEvent, SessionLogLike


@dataclass
class ToolContext:
    """The narrow tool-side seam (H3-2): session_id + state + log, nothing else."""

    session_id: str
    state: dict = field(default_factory=dict)  # shared scratch state (by reference)
    log: SessionLogLike | None = None          # the sacred log (H0-2)

    def append_event(self, event: HarnessEvent) -> int:
        """Append one event to this session's sacred log; returns its seq number.

        Raises RuntimeError when no log is available — a tool that promised to
        log must not silently skip it.
        """
        if self.log is None:
            raise RuntimeError("ToolContext.append_event: no session log available")
        return self.log.append(self.session_id, event)
