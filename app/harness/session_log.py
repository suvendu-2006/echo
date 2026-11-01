"""The sacred session log (H0-2) — append-only SQLite, crash-safe (D-006).

Design (docs/phase3-contract.md, stage H0; PLAN principle 1 — log-first, dsh):

- **Append-only**: events are never updated or deleted. The log is the single
  source of truth — the process may crash, restart, or resume months later and
  ``replay()`` reconstructs exact state.
- **Crash-safe**: WAL journaling (a sudden kill loses at most the last
  in-flight transaction, never older commits), and every append reads
  ``MAX(seq) + 1`` inside one ``BEGIN IMMEDIATE`` transaction so concurrent
  writers — threads today, other processes on the same file always — can never
  collide or interleave a seq assignment.
- **Corrupt-tail recovery**: a torn final write (a row holding truncated JSON)
  is skipped with a logged warning on replay; older events always survive and
  the log never raises on its own data.
- **Hygiene** carried over from app/config.py: usable as a context manager,
  idempotent ``close()``, and a RuntimeError (never assert — survives
  ``python -O`` and bandit B101) use-after-close guard.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import threading
import time
from collections.abc import Iterator
from pathlib import Path

from app import config
from app.harness.events import HarnessEvent, event_from_json, event_to_json
from app.harness.interfaces import TraceSink

logger = logging.getLogger("echo.harness")

#: Filename under DATA_DIR when no explicit path is given.
DEFAULT_LOG_FILENAME = "harness.db"

#: How long a writer waits for a competing writer on the same file (ms) before
#: giving up — covers two SessionLog instances (or raw connections) racing.
#: Set as a literal PRAGMA below; bandit prefers SQL-shaped strings kept plain.

#: Schema EXACTLY as frozen in docs/phase3-contract.md (stage H0).
_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS harness_events ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " session_id TEXT NOT NULL,"
    " seq INTEGER NOT NULL,"
    " type TEXT NOT NULL,"
    " payload TEXT NOT NULL,"
    " created_at REAL NOT NULL,"
    " UNIQUE(session_id, seq))",
    "CREATE TABLE IF NOT EXISTS harness_sessions ("
    " session_id TEXT PRIMARY KEY,"
    " parent_session_id TEXT,"
    " created_at REAL)",
)


def default_log_path() -> Path:
    """DATA_DIR/harness.db, resolved lazily so ECHO_DATA_DIR is honoured."""
    return config.DATA_DIR / DEFAULT_LOG_FILENAME


class SessionLog:
    """Append-only, replayable, crash-safe event log (satisfies SessionLogLike).

    One instance owns one SQLite connection (``check_same_thread=False``);
    an internal lock serializes transactions across threads, and
    ``BEGIN IMMEDIATE`` serializes across processes. Trace taps (H7-1) can be
    registered via :meth:`add_trace` — every committed append is mirrored to
    them, and a failing tap is swallowed + logged (tracing never breaks the
    loop).
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else default_log_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            str(self._path), check_same_thread=False
        )
        # Autocommit mode: transactions are explicit (BEGIN IMMEDIATE ... COMMIT)
        # so the driver never interleaves its own implicit BEGIN with ours.
        self._conn.isolation_level = None
        self._conn.execute("PRAGMA journal_mode=WAL").fetchone()
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        for statement in _SCHEMA:
            self._conn.execute(statement)
        self._lock = threading.Lock()
        self._traces: list[TraceSink] = []

    # ------------------------------------------------------------- lifecycle

    @property
    def path(self) -> Path:
        """Where the log file lives (audit screen / backup tooling)."""
        return self._path

    def __enter__(self) -> "SessionLog":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            # A real exception, not an assert: survives `python -O` and bandit B101.
            raise RuntimeError("SessionLog used after close()")
        return self._conn

    def close(self) -> None:
        """Release the database file. Safe to call more than once."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    @contextlib.contextmanager
    def _immediate(self) -> Iterator[sqlite3.Connection]:
        """One ``BEGIN IMMEDIATE`` transaction on the shared connection.

        IMMEDIATE takes the database write lock up front, so the seq
        read-and-insert pair cannot interleave with another writer (thread or
        process). Commits on success; on any failure rolls back and re-raises
        the original error.
        """
        conn = self._db()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                logger.warning("SessionLog: rollback after failure also failed", exc_info=True)
            raise

    # ----------------------------------------------------------- the log API

    def create_session(self, session_id: str, parent_session_id: str | None = None) -> None:
        """Register a session (idempotent). Subagent children pass a parent (H6)."""
        with self._lock, self._immediate() as conn:
            conn.execute(
                "INSERT INTO harness_sessions (session_id, parent_session_id, created_at)"
                " VALUES (?, ?, ?) ON CONFLICT(session_id) DO NOTHING",
                (session_id, parent_session_id, time.time()),
            )

    def append(self, session_id: str, event: HarnessEvent) -> int:
        """Append one event; returns its per-session sequence number. Log FIRST.

        The next seq (``COALESCE(MAX(seq), -1) + 1``) is read and the row
        inserted inside one ``BEGIN IMMEDIATE`` transaction — atomic under
        concurrent writers (the H3-12 groundwork). Trace taps fire only after
        the row is durably committed.
        """
        payload = event_to_json(event)  # outside the lock: no DB time wasted
        with self._lock, self._immediate() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), -1) + 1 FROM harness_events"
                " WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            seq = int(row[0])
            conn.execute(
                "INSERT INTO harness_events"
                " (session_id, seq, type, payload, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (session_id, seq, event.type, payload, time.time()),
            )
        self._emit_trace(session_id, event)
        return seq

    def replay(self, session_id: str) -> list[HarnessEvent]:
        """Full ordered history for a session — corrupt rows skipped, never fatal.

        Crash-tail recovery (the H0 kill-test): if the process died mid-write,
        a row can hold truncated JSON. Such rows are dropped with a logged
        warning and every parseable event is returned in seq order — the log
        never raises on its own data.
        """
        conn = self._db()
        with self._lock:
            rows = conn.execute(
                "SELECT seq, payload FROM harness_events"
                " WHERE session_id = ? ORDER BY seq ASC",
                (session_id,),
            ).fetchall()
        events: list[HarnessEvent] = []
        for seq, payload in rows:
            try:
                events.append(event_from_json(payload))
            except (ValueError, TypeError, KeyError) as exc:
                logger.warning(
                    "replay: skipping corrupt event row (session=%s seq=%s): %s",
                    session_id,
                    seq,
                    exc,
                )
        return events

    def list_sessions(self) -> list[dict]:
        """All known sessions, newest first, each with its event count.

        Keys: session_id, parent_session_id, created_at, event_count. Sessions
        that have events but no registry row (a caller that skipped
        create_session) still appear — the events table is the truth.
        """
        conn = self._db()
        with self._lock:
            registered = conn.execute(
                "SELECT session_id, parent_session_id, created_at, rowid"
                " FROM harness_sessions ORDER BY rowid ASC"
            ).fetchall()
            counted = conn.execute(
                "SELECT session_id, COUNT(*), MIN(created_at)"
                " FROM harness_events GROUP BY session_id"
            ).fetchall()
        counts = {sid: count for sid, count, _ in counted}
        sessions = [
            {
                "session_id": sid,
                "parent_session_id": parent,
                "created_at": created_at,
                "event_count": counts.get(sid, 0),
                "_order": rowid,
            }
            for sid, parent, created_at, rowid in registered
        ]
        known = {s["session_id"] for s in sessions}
        for sid, _count, first_seen in counted:
            if sid not in known:
                sessions.append(
                    {
                        "session_id": sid,
                        "parent_session_id": None,
                        "created_at": first_seen,
                        "event_count": counts[sid],
                        "_order": -1,
                    }
                )
        # Newest first; rowid breaks created_at ties by true creation order
        # (time.time() can tie within one clock tick on Windows).
        sessions.sort(key=lambda s: (s["created_at"] or 0.0, s["_order"]), reverse=True)
        for s in sessions:
            del s["_order"]
        return sessions

    # ------------------------------------------------------------- trace taps

    def add_trace(self, sink: TraceSink) -> None:
        """Register a trace tap (H7-1 seam); make_ctx wires these. Identity-deduped."""
        if not any(existing is sink for existing in self._traces):
            self._traces.append(sink)

    def _emit_trace(self, session_id: str, event: HarnessEvent) -> None:
        """Mirror one committed event to every tap — tracing must never break the loop."""
        for sink in list(self._traces):
            try:
                sink.emit(session_id, event)
            except Exception:  # noqa: BLE001 — sink bugs are logged, never fatal
                logger.warning(
                    "trace sink %r failed; event not mirrored", sink, exc_info=True
                )
