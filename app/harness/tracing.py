"""Tracing (H7-1) — where every logged event is mirrored, never blocking.

The sacred log is the single source of truth (D-006); tracing is the cheap
ELECTRICAL COPY. ``SessionLog.append`` fans every committed event out to the
registered ``TraceSink``s (``SessionLog.add_trace`` / ``make_ctx(trace=...)``),
and one rule rules them all: **tracing must never break the loop** — a dead
backend is a logged warning, never an exception (H7-1, PLAN principle 5's
sibling: observability from Phase 3, not ship time — D-005).

Sinks provided here (docs/phase3-contract.md, stage H7):

- :class:`LocalJSONLTracer` — the default. One JSON line per event under
  ``<dir>/<session_id>.jsonl`` (``data/traces/`` by default). Zero deps, zero
  config, human-auditable, kill-safe (flush per line).
- :class:`LangfuseTracer` — optional. Lazy ``import langfuse`` inside the
  constructor; maps harness events onto Langfuse traces/generations
  best-effort. Credentials are EXPLICIT constructor arguments — keys are
  never read from the environment implicitly; they come from the keychain
  through :func:`tracer_from_settings` (D-013).
- :class:`TraceTap` — fan-out: one event to N sinks, each guarded alone.
- :func:`tracer_from_settings` — the wiring factory: settings key
  ``trace.backend`` = ``local`` | ``langfuse`` (plus
  ``trace.langfuse.host``, optional ``trace.local.dir``), credentials from
  anything with ``.get_key(name)`` under the keychain names
  ``langfuse.public_key`` / ``langfuse.secret_key``. Anything unknown,
  missing, or misbehaving degrades to the local tracer — fail-safe, so the
  harness always has eyes.

Purity: imports only ``app.config`` and ``app.harness.interfaces`` — never
the concrete stage modules.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import IO

from app import config
from app.harness.interfaces import HarnessEvent, TraceSink

logger = logging.getLogger("echo.trace")

#: Settings key selecting the backend: "local" (default) | "langfuse".
TRACE_BACKEND_KEY = "trace.backend"

#: Settings key for the Langfuse host — explicit, never an implicit default.
LANGFUSE_HOST_KEY = "trace.langfuse.host"

#: Optional settings key overriding the local tracer's directory.
LOCAL_DIR_KEY = "trace.local.dir"

#: Keychain entry names (``SecretStore.get_key``) for Langfuse credentials.
LANGFUSE_PUBLIC_KEY_NAME = "langfuse.public_key"
LANGFUSE_SECRET_KEY_NAME = "langfuse.secret_key"

#: Directory (under DATA_DIR) the factory's local tracer writes to.
TRACE_DIR_NAME = "traces"

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


def safe_session_name(session_id: str) -> str:
    """A filesystem-safe trace-file stem for a session id (Windows-safe).

    ``/`` ``\\`` and every other hostile character (NTFS reserves ``<>:"/\\|?*``)
    become ``_`` so a session id can never escape the trace directory or trip
    the filesystem; ordinary ids (uuid-style) pass through untouched. The
    mapping is deterministic — the SAME function names the file in ``emit``
    and in ``read_trace``. (Two pathological ids could collide after
    sanitizing; that trades total isolation for the never-raise guarantee.)
    """
    safe = _UNSAFE_NAME.sub("_", str(session_id))
    return safe or "_empty_session"


# ------------------------------------------------------------ the default sink


class LocalJSONLTracer:
    """The default sink: one JSON line per event, one file per session (H7-1).

    - Line shape: ``{"ts": <unix time>, "session_id": ..., "event": <to_dict()>}``.
    - Lazy: the directory is created on the FIRST event, not at construction —
      a constructed-but-unused tracer touches no disk.
    - Kill-safe: every line is written AND flushed before ``emit`` returns, so
      a hard process kill keeps every flushed line.
    - Never raises: any filesystem failure (unwritable path, disk full, hostile
      name) is swallowed and logged on ``echo.trace``. A broken mirror must
      not shatter the loop.
    - ``close()`` releases the open files (call it before deleting the tmp dir
      on Windows); a later ``emit`` simply re-opens lazily.
    - ``read_trace(session_id)`` reads a session's lines back as dicts for
      tests/audit; corrupt lines are skipped with a warning — the same
      corrupt-tail discipline the sacred log itself follows.
    """

    def __init__(self, dir_path: str | Path) -> None:
        self._dir = Path(dir_path)
        self._handles: dict[str, IO[str]] = {}
        self._lock = threading.Lock()

    @property
    def dir_path(self) -> Path:
        """Where the per-session trace files live (audit tooling)."""
        return self._dir

    def emit(self, session_id: str, event: HarnessEvent) -> None:
        """Append one JSON line for this event. Swallows every failure.

        Open-append-close per line — NO cached handle. Windows locks open
        files (exam bugs #1–#3 are all this one lesson), and a tracer that
        holds a trace file open makes its directory undeletable. One
        open/close per event is negligible at chat rates and even safer
        against kills than flush-per-line.
        """
        line = json.dumps(
            {"ts": time.time(), "session_id": session_id, "event": event.to_dict()},
            ensure_ascii=False,
            default=str,  # log-first discipline: odd payloads degrade, never crash
        )
        try:
            with self._lock:
                self._dir.mkdir(parents=True, exist_ok=True)
                path = self._dir / f"{safe_session_name(session_id)}.jsonl"
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except Exception:
            logger.warning(
                "LocalJSONLTracer: dropped a trace line for session %r",
                session_id,
                exc_info=True,
            )

    def read_trace(self, session_id: str) -> list[dict]:
        """A session's trace lines, oldest first (tests/audit helper).

        Missing file -> ``[]``. Corrupt lines are skipped with a logged
        warning. Pending writes are flushed first so a read never trails a
        recent ``emit`` on the same object.
        """
        key = safe_session_name(session_id)
        path = self._dir / f"{key}.jsonl"
        lines: list[dict] = []
        try:
            with path.open("r", encoding="utf-8") as reader:
                for raw in reader:
                    stripped = raw.strip()
                    if not stripped:
                        continue
                    try:
                        parsed = json.loads(stripped)
                    except (ValueError, TypeError):
                        logger.warning(
                            "LocalJSONLTracer: skipping corrupt trace line in %s", path
                        )
                        continue
                    if isinstance(parsed, dict):
                        lines.append(parsed)
        except FileNotFoundError:
            return []
        except OSError:
            logger.warning(
                "LocalJSONLTracer: could not read trace file %s", path, exc_info=True
            )
        return lines

    def close(self) -> None:
        """No cached handles to release (open-append-close per emit). No-op,
        kept so callers can close a tracer unconditionally. Safe to re-call."""
        return None

    # (the old handle-cache helpers were removed with the cache itself —
    #  open-append-close per emit is the whole story now)


# ----------------------------------------------------------------- fan-out sink


class TraceTap:
    """Fan-out sink (H7-1): one event to N backends, each guarded ALONE.

    ``TraceTap(LocalJSONLTracer(a), LangfuseTracer(...))`` — or a single list,
    ``TraceTap([a, b])``. A child sink that raises is a logged warning; its
    siblings still receive the event. A tap satisfies :class:`TraceSink`
    itself, so taps nest (fan-out of fan-outs).
    """

    def __init__(self, *sinks: TraceSink) -> None:
        children: list[TraceSink] = []
        for sink in sinks:
            if isinstance(sink, (list, tuple)):
                children.extend(candidate for candidate in sink if candidate is not None)
            elif sink is not None:
                children.append(sink)
        self._sinks: list[TraceSink] = children

    def add(self, sink: TraceSink) -> None:
        """Register one more child sink (a ``None`` is a no-op)."""
        if sink is not None:
            self._sinks.append(sink)

    @property
    def sinks(self) -> list[TraceSink]:
        """The children, in dispatch order (wiring audit)."""
        return list(self._sinks)

    def emit(self, session_id: str, event: HarnessEvent) -> None:
        for sink in list(self._sinks):
            try:
                sink.emit(session_id, event)
            except Exception:
                logger.warning(
                    "TraceTap: sink %r failed; the remaining taps still ran",
                    sink,
                    exc_info=True,
                )


# ------------------------------------------------------------- optional backend


class LangfuseTracer:
    """Optional Langfuse sink (H7-1) — every harness event, inspectable.

    Lazy by design: ``import langfuse`` happens inside the constructor, so a
    machine without the package (the default install — it is deliberately not
    in requirements.txt) only fails HERE, with an explicit
    ``pip install langfuse`` hint, and only when settings actually asked for
    Langfuse.

    Credentials are constructor arguments — ``public_key``/``secret_key``
    arrive from the keychain via :func:`tracer_from_settings`, never from the
    environment implicitly (D-013).

    Mapping (best-effort — every emit is guarded, every failure logged):

    ==================  ====================================================
    HarnessEvent        Langfuse
    ==================  ====================================================
    assistant           generation (content in, tool calls out, usage attached)
    user/system         event of the same name
    tool_call           event ``tool_call`` (name + arguments in metadata)
    tool_result         event ``tool_result`` (status envelope in output)
    compact/status      event of the same name
    ==================  ====================================================

    One Langfuse trace per session id, created on first sight. The client
    factory has been renamed across SDK major versions (``client.trace`` in
    v2, ``client.start_trace`` in v3) — both are probed.
    """

    def __init__(self, public_key: str, secret_key: str, host: str) -> None:
        try:
            import langfuse  # lazy on purpose: an optional dependency (H7-1)
        except ImportError as exc:
            raise RuntimeError(
                "LangfuseTracer needs the optional 'langfuse' package — "
                "pip install langfuse (or keep trace.backend=local)"
            ) from exc
        try:
            self._client = langfuse.Langfuse(
                public_key=public_key, secret_key=secret_key, host=host
            )
        except Exception as exc:
            raise RuntimeError(
                f"LangfuseTracer could not build a Langfuse client for host "
                f"{host!r}: {exc!r}"
            ) from exc
        self._traces: dict[str, object] = {}
        self._lock = threading.Lock()

    def emit(self, session_id: str, event: HarnessEvent) -> None:
        """Mirror one event. Swallows every failure — a dead SaaS is a warning."""
        try:
            trace = self._session_trace(session_id)
            if trace is not None:
                self._record(trace, event)
        except Exception:
            logger.warning(
                "LangfuseTracer: event for session %r was not mirrored",
                session_id,
                exc_info=True,
            )

    def flush(self) -> None:
        """Best-effort push of buffered events to the server (never raises)."""
        try:
            flush = getattr(self._client, "flush", None)
            if callable(flush):
                flush()
        except Exception:
            logger.warning("LangfuseTracer: flush failed", exc_info=True)

    def close(self) -> None:
        """Flush and release the client (best-effort; safe to call twice)."""
        self.flush()
        try:
            shutdown = getattr(self._client, "shutdown", None)
            if callable(shutdown):
                shutdown()
        except Exception:
            logger.warning("LangfuseTracer: client shutdown failed", exc_info=True)

    # -------------------------------------------------------------- internals

    def _session_trace(self, session_id: str) -> object | None:
        """Get-or-create the Langfuse trace for a session (v2/v3 tolerant)."""
        with self._lock:
            trace = self._traces.get(session_id)
            if trace is None:
                factory = getattr(self._client, "trace", None) or getattr(
                    self._client, "start_trace", None
                )
                if not callable(factory):
                    return None  # unknown SDK shape: mirror nothing, stay quiet
                trace = factory(
                    id=f"echo-{session_id}", name="echo-session", session_id=session_id
                )
                self._traces[session_id] = trace
            return trace

    def _record(self, trace: object, event: HarnessEvent) -> None:
        """Map one harness event onto a generation/event (best-effort shape)."""
        payload = event.to_dict()
        meta = payload.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        if event.type == "assistant" and callable(getattr(trace, "generation", None)):
            usage = event.usage
            trace.generation(
                name="assistant",
                input=payload.get("content", ""),
                output=meta.get("assistant_tool_calls", []),
                usage=(
                    {
                        "prompt": usage.prompt_tokens,
                        "completion": usage.completion_tokens,
                    }
                    if usage is not None
                    else None
                ),
                metadata=meta,
            )
            return
        factory = getattr(trace, "event", None)
        if callable(factory):
            factory(
                name=payload.get("type") or "harness_event",
                input=payload.get("content", ""),
                output=payload.get("result"),
                metadata={
                    "tool_call": payload.get("tool_call"),
                    "usage": payload.get("usage"),
                    **meta,
                },
            )


# ----------------------------------------------------------------- the factory


def tracer_from_settings(settings: object, secret_store: object) -> TraceSink:
    """Wire the trace backend from settings + keychain (fail-safe, H7-1).

    ``settings`` may be a :class:`app.config.SettingsStore`, a plain dict, or
    anything with ``.get(key, default)``. Keys:

    - ``trace.backend`` — ``local`` (default) | ``langfuse``
    - ``trace.langfuse.host`` — the self-hosted/server Langfuse URL
    - ``trace.local.dir`` — optional override for the local tracer's directory
      (default ``DATA_DIR/traces``)

    ``secret_store`` is anything with ``.get_key(name)``; Langfuse credentials
    live under ``langfuse.public_key`` / ``langfuse.secret_key``.

    NOTHING here raises: an unknown backend, missing credentials/host, a dead
    package, or a misbehaving settings object all degrade to the local JSONL
    tracer with a logged reason — the harness always has eyes.
    """
    backend = _setting(settings, TRACE_BACKEND_KEY, "local").strip().lower()
    if backend == "langfuse":
        public_key = _secret(secret_store, LANGFUSE_PUBLIC_KEY_NAME)
        secret_key = _secret(secret_store, LANGFUSE_SECRET_KEY_NAME)
        host = _setting(settings, LANGFUSE_HOST_KEY, "")
        if public_key and secret_key and host:
            try:
                return LangfuseTracer(
                    public_key=public_key, secret_key=secret_key, host=host
                )
            except Exception:
                logger.warning(
                    "tracer_from_settings: Langfuse unavailable; "
                    "using the local tracer",
                    exc_info=True,
                )
        else:
            logger.warning(
                "%s=langfuse but keychain credentials or host are missing; "
                "using the local tracer",
                TRACE_BACKEND_KEY,
            )
    elif backend != "local":
        logger.warning(
            "unknown %s %r; using the local tracer (fail-safe)",
            TRACE_BACKEND_KEY,
            backend,
        )
    return LocalJSONLTracer(_local_dir(settings))


def _local_dir(settings: object) -> Path:
    """The local tracer's directory: ``trace.local.dir`` or ``DATA_DIR/traces``."""
    override = _setting(settings, LOCAL_DIR_KEY, "")
    if override:
        return Path(override)
    return Path(config.DATA_DIR) / TRACE_DIR_NAME


def _setting(settings: object, key: str, default: str) -> str:
    """Read one settings key from anything shaped like a mapping. Never raises."""
    getter = getattr(settings, "get", None)
    if not callable(getter):
        return default
    try:
        value = getter(key, default)
    except Exception:
        return default
    return default if value is None else str(value)


def _secret(secret_store: object, name: str) -> str:
    """Read one keychain entry. Never raises; missing means empty."""
    getter = getattr(secret_store, "get_key", None)
    if not callable(getter):
        return ""
    try:
        value = getter(name)
    except Exception:
        return ""
    return str(value) if value is not None else ""
