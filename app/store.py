"""Session + message store (PLAN Phase 2.4) — the thin, temporary chat memory.

THIN AND TEMPORARY by design: this is the Phase 2 stop-gap so the UI can show
a conversation list while the real system of record does not exist yet. Phase 3
H0-2 supersedes it with the sacred append-only log — every row written here is
expected to be migrated or thrown away wholesale; nothing else may grow a
dependency on this schema.

Same SQLite hygiene as app/config.py (DECISIONS D-00x): WAL journaling from
day one, a context manager, idempotent close, and a real RuntimeError guard
(not an assert — bandit B101) for use-after-close. Two additions the session
tables need that the settings table does not:

- ``PRAGMA foreign_keys = ON`` so deleting a session cascades to its messages.
- ``check_same_thread = False``: ASGI servers and the Starlette test client
  run handlers on a different thread than the one that built the app. SQLite's
  serialized mode makes shared use of one connection safe; the server only
  ever touches the store from its event-loop thread.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import DB_PATH

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS sessions ("
    " id TEXT PRIMARY KEY,"
    " title TEXT NOT NULL,"
    " created_at REAL NOT NULL,"
    " updated_at REAL NOT NULL)",
    "CREATE TABLE IF NOT EXISTS messages ("
    " id TEXT PRIMARY KEY,"
    " session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,"
    " role TEXT NOT NULL,"
    " content TEXT NOT NULL,"
    " created_at REAL NOT NULL)",
)

_SESSION_FIELDS = ("id", "title", "created_at", "updated_at")
_MESSAGE_FIELDS = ("id", "session_id", "role", "content", "created_at")


def _connect(path: Path) -> sqlite3.Connection:
    """Open a session database with app.config.connect's hygiene plus the
    two session-specific pragmas (see module docstring)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL").fetchone()
    conn.execute("PRAGMA foreign_keys=ON")
    for statement in _SCHEMA:
        conn.execute(statement)
    return conn


class SessionStore:
    """Sessions and their messages, persisted in SQLite with WAL journaling.

    Usable as a context manager. On Windows an open connection keeps an
    exclusive lock on the database file, so the connection MUST be closed
    (or the store used via ``with``) before the file can be moved/deleted.
    """

    def __init__(self, path: Path = DB_PATH) -> None:
        self._conn: sqlite3.Connection | None = _connect(path)

    def __enter__(self) -> "SessionStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            # A real exception, not an assert: survives `python -O` and bandit B101.
            raise RuntimeError("SessionStore used after close()")
        return self._conn

    @property
    def journal_mode(self) -> str:
        row = self._db().execute("PRAGMA journal_mode").fetchone()
        return str(row[0]).lower()

    # ------------------------------------------------------------- sessions

    def create_session(self, title: str = "New chat") -> dict[str, Any]:
        """Create an empty session and return its row as a dict."""
        now = time.time()
        session_id = uuid4().hex
        self._db().execute(
            "INSERT INTO sessions (id, title, created_at, updated_at)"
            " VALUES (?, ?, ?, ?)",
            (session_id, title, now, now),
        )
        self._db().commit()
        return {
            "id": session_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
        }

    def list_sessions(self) -> list[dict[str, Any]]:
        """All sessions, most recently active first, with message counts.

        One JOINed query — the session list is the UI's home screen and must
        not become N+1 as conversations pile up.
        """
        rows = self._db().execute(
            "SELECT s.id, s.title, s.created_at, s.updated_at,"
            " COUNT(m.id) AS message_count"
            " FROM sessions AS s LEFT JOIN messages AS m ON m.session_id = s.id"
            " GROUP BY s.id"
            " ORDER BY s.updated_at DESC, s.rowid DESC"
        ).fetchall()
        return [
            dict(zip(_SESSION_FIELDS + ("message_count",), row)) for row in rows
        ]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """One session row, or None when the id is unknown."""
        row = self._db().execute(
            "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        return None if row is None else dict(zip(_SESSION_FIELDS, row))

    def rename_session(self, session_id: str, title: str) -> bool:
        """Set a new title; False when the session does not exist.

        Renaming deliberately does NOT bump ``updated_at`` — only new
        messages make a session float to the top of the list.
        """
        cursor = self._db().execute(
            "UPDATE sessions SET title = ? WHERE id = ?", (title, session_id)
        )
        self._db().commit()
        return cursor.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and (via ON DELETE CASCADE) all its messages."""
        cursor = self._db().execute(
            "DELETE FROM sessions WHERE id = ?", (session_id,)
        )
        self._db().commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------ messages

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        """All messages of one session, in conversation order."""
        rows = self._db().execute(
            "SELECT id, session_id, role, content, created_at FROM messages"
            " WHERE session_id = ? ORDER BY created_at, rowid",
            (session_id,),
        ).fetchall()
        return [dict(zip(_MESSAGE_FIELDS, row)) for row in rows]

    def append_message(self, session_id: str, role: str, content: str) -> dict[str, Any]:
        """Add one message and bump the session's ``updated_at``.

        Raises sqlite3.IntegrityError when ``session_id`` does not exist
        (the foreign key is enforced on purpose — callers validate first).
        """
        now = time.time()
        message_id = uuid4().hex
        conn = self._db()
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (message_id, session_id, role, content, now),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
        )
        conn.commit()
        return {
            "id": message_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "created_at": now,
        }

    def close(self) -> None:
        """Release the database file. Safe to call more than once."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
