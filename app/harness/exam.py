"""The scripted exam harness (H7-2) — Promptfoo-style checks, fully offline.

Promptfoo runs the five core scenarios against a REAL model when a key exists
(``promptfooconfig.yaml`` — the human gate, T3-07). This module is the harness
those same scenarios use offline: a declarative :class:`ExamCase` plus
:func:`run_exam_case`, which drives ONE ``AgentLoop.run_turn`` on a fresh
session and grades the LOGGED events — no network, no key, deterministic
(PLAN H7-2: "every H-gate becomes a test"; rule 5: "every bug becomes a test,
same day").

Judge rules (all optional, all combinable):

- ``must_not_call_tools`` — no ``tool_call`` event may appear (the
  termination contract: a plain question is answered plainly).
- ``must_ask_permission_for`` — the engine's ``ask`` hook must have been
  invoked for that tool name. Observation is liberal: the recorder shadows
  ``engine.ask`` for the run AND reads any ``asked``/``asks`` list the engine
  keeps itself, so both test fakes and the real permission engine are
  judgeable without coupling exam.py to either.
- ``expect_tool_called`` — the tool name appears among the ``tool_call``
  events. The dispatcher's own ``dispatched`` recording, when present, only
  enriches the failure message (called-but-denied vs never-called).
- ``expect_text_contains`` — any assistant content contains the needle
  (case-insensitive).

``run_exam_case`` NEVER raises — a loop that aborts, a loop heart that
crashes, even a tripping judge all become human-readable failure strings.
Exams grade the harness; they must never become the crash.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.harness.interfaces import HarnessEvent

if TYPE_CHECKING:  # typing only — keeps the exam's import graph flat
    from app.harness.interfaces import PermissionEngine, ToolDispatcher


# ------------------------------------------------------------------ the model


@dataclass
class ExamCase:
    """One scripted scenario — declarative, shareable, exactly six fields.

    The five core Promptfoo scenarios (PLAN H7-2: math-without-tools ·
    dangerous-command-asks-permission · fail->strategy-change ·
    trim-survival · draft-before-send) all express themselves with these
    judges; combine them freely.
    """

    name: str
    question: str
    must_not_call_tools: bool = False
    must_ask_permission_for: str | None = None
    expect_text_contains: str | None = None
    expect_tool_called: str | None = None


@dataclass
class ExamResult:
    """The grade for one case: pass/fail, WHY (human-readable), and the tape."""

    passed: bool
    failures: list[str] = field(default_factory=list)
    events: list[HarnessEvent] = field(default_factory=list)


# -------------------------------------------------------------------- the run


async def run_exam_case(
    case: ExamCase,
    loop: object,
    dispatcher: "ToolDispatcher | None" = None,
    engine: "PermissionEngine | None" = None,
) -> ExamResult:
    """Drive ONE ``run_turn`` on a fresh session and grade it. Never raises.

    The caller wires a fresh loop (fresh session id, fresh log) — the exam
    runs exactly ``loop.run_turn(case.question)`` once. ``dispatcher`` and
    ``engine`` are the same objects the loop uses; they are consulted only
    for richer observation/diagnosis (``engine`` for the ask recording,
    ``dispatcher`` for the dispatched-tools note) and are never required.
    """
    failures: list[str] = []
    events: list[HarnessEvent] = []
    recorder = _AskRecorder(engine)
    try:
        events = [event async for event in loop.run_turn(case.question)]
    except Exception as exc:
        failures.append(
            f"{case.name}: run_turn raised (exams grade, they never crash): {exc!r}"
        )
    finally:
        recorder.restore()

    try:
        failures.extend(_judge(case, events, dispatcher, engine, recorder.names))
    except Exception as exc:  # a tripping judge is a failure, not a crash
        failures.append(f"{case.name}: the exam judge itself failed: {exc!r}")
    return ExamResult(passed=not failures, failures=failures, events=events)


# ------------------------------------------------------------------ the judges


def _judge(
    case: ExamCase,
    events: list[HarnessEvent],
    dispatcher: object,
    engine: object,
    shadowed_asks: set[str],
) -> list[str]:
    """Apply every set judge rule; return human-readable failures (never raises
    by itself — run_exam_case keeps even THIS covered)."""
    failures: list[str] = []
    called = _called_tool_names(events)
    assistant_texts = [event.content for event in events if event.type == "assistant"]

    if case.must_not_call_tools and called:
        failures.append(
            f"{case.name}: expected no tool calls, but the model called: "
            f"{', '.join(called)}"
        )

    if case.must_ask_permission_for is not None:
        asked = shadowed_asks | _engine_recorded_asks(engine)
        if case.must_ask_permission_for not in asked:
            failures.append(
                f"{case.name}: expected the engine to ask permission for "
                f"'{case.must_ask_permission_for}', but ask was invoked for: "
                f"{sorted(asked) or 'nothing'}"
            )

    if case.expect_tool_called is not None and case.expect_tool_called not in called:
        failures.append(
            f"{case.name}: expected tool '{case.expect_tool_called}' to be called, "
            f"but tool_call events show: {called or 'none'}{_dispatched_note(dispatcher)}"
        )

    if case.expect_text_contains is not None:
        needle = case.expect_text_contains.lower()
        if not any(needle in text.lower() for text in assistant_texts):
            preview = " | ".join(text[:120] for text in assistant_texts)
            failures.append(
                f"{case.name}: expected some assistant text to contain "
                f"'{case.expect_text_contains}'; assistant text was: "
                f"{preview or '<no assistant text>'!r}"
            )
    return failures


def _called_tool_names(events: Iterable[HarnessEvent]) -> list[str]:
    """Tool names from the logged ``tool_call`` events, in order."""
    return [
        event.tool_call.name
        for event in events
        if event.type == "tool_call" and event.tool_call is not None
    ]


def _dispatched_note(dispatcher: object) -> str:
    """Optional diagnosis from a dispatcher that records what it actually ran."""
    dispatched = getattr(dispatcher, "dispatched", None)
    if dispatched is None:
        return ""
    try:
        names = [call.name for call in dispatched]
    except Exception:
        return ""
    return f" (dispatcher actually ran: {names or 'nothing'})"


def _engine_recorded_asks(engine: object) -> set[str]:
    """Read an engine's own ask recording, liberally (it is a test seam).

    Accepts an ``asked``/``asks`` attribute holding plain names, ``(name,
    danger)`` tuples, or objects with a ``.name`` — the shapes fakes
    naturally grow. Anything unreadable simply contributes nothing.
    """
    names: set[str] = set()
    if engine is None:
        return names
    for attr in ("asked", "asks"):
        entries = getattr(engine, attr, None)
        if entries is None:
            continue
        try:
            collected = list(entries)
        except TypeError:
            continue
        for entry in collected:
            name = _as_tool_name(entry)
            if name:
                names.add(name)
    return names


def _as_tool_name(entry: object) -> str | None:
    """Pull a tool name out of a str / (name, ...) tuple / .name object."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, (tuple, list)) and entry:
        first = entry[0]
        return first if isinstance(first, str) else None
    name = getattr(entry, "name", None)
    return name if isinstance(name, str) else None


class _AskRecorder:
    """Shadow ``engine.ask`` for one exam run, recording the tool names.

    Works with ANY engine — the real one included: the shadow is an instance
    attribute that delegates to the original and is removed afterwards. If
    the engine refuses shadowing (``__slots__``, read-only), we simply rely
    on its own recording instead. Cleanup is best-effort and never raises —
    a recorder bug must not outlive the exam it ran in.
    """

    def __init__(self, engine: object) -> None:
        self.names: set[str] = set()
        self._engine = engine
        self._shadowed = False
        self._had_own = False
        self._own: object = None
        if engine is None:
            return
        original = getattr(engine, "ask", None)
        if not callable(original):
            return
        recorder = self

        async def recording_ask(call: object, danger: object, ctx: object) -> object:
            recorder.names.add(_as_tool_name(call) or str(call))
            return await original(call, danger, ctx)  # delegate, never alter

        try:
            self._had_own = "ask" in vars(engine)
            self._own = vars(engine).get("ask")
            engine.ask = recording_ask
            self._shadowed = True
        except Exception:
            self._shadowed = False  # read-only engine: its own recording rules

    def restore(self) -> None:
        """Undo the instance shadow so the engine leaves the exam unmodified."""
        if not self._shadowed:
            return
        try:
            if self._had_own:
                self._engine.ask = self._own
            else:
                del self._engine.ask
        except Exception:
            pass  # cleanup must never break the exam
