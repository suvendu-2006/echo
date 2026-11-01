"""Echo settings + secrets (Phase 0.3 — the layer every later phase builds on).

Design (see DECISIONS.md):
- Settings live in a SQLite key/value table, WAL journaling from day one —
  the crash-safety habit the sacred log (Phase 3, H0-2) will extend.
- Secrets never touch tracked files or git: dev secrets may sit in a
  git-ignored ``.env``; production secrets go through the OS keychain via
  ``keyring`` (Windows Credential Manager / macOS Keychain — D-009, D-013).
- Everything here is stdlib-only so the test suite runs on any machine
  with Python 3.13+, Windows or macOS.
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Protocol

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Runtime data (databases, logs) — git-ignored, relocatable via env var.
DATA_DIR = Path(os.environ.get("ECHO_DATA_DIR", str(PROJECT_ROOT / "data")))
DB_PATH = DATA_DIR / "echo.db"

#: Optional dev-secrets file — git-ignored, never shipped (D-013).
ENV_PATH = PROJECT_ROOT / ".env"

#: Namespace for keys stored in the OS keychain.
KEYRING_SERVICE = "Echo"

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS settings ("
    " key TEXT PRIMARY KEY,"
    " value TEXT NOT NULL,"
    " updated_at REAL NOT NULL)"
)


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    """Open the Echo database with WAL journaling enabled (crash-safe).

    ``check_same_thread=False``: ASGI handlers run on worker threads, and
    WAL + our own transaction discipline make cross-thread use safe (the
    same choice app.store.SessionStore makes).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL").fetchone()
    conn.execute(_SCHEMA)
    return conn


class SettingsStore:
    """Key/value settings persisted in SQLite with WAL journaling.

    Usable as a context manager. On Windows an open connection keeps an
    exclusive lock on the database file, so the connection MUST be closed
    (or the store used via ``with``) before the file can be moved/deleted.
    """

    def __init__(self, path: Path = DB_PATH) -> None:
        self._conn: sqlite3.Connection | None = connect(path)

    def __enter__(self) -> "SettingsStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            # A real exception, not an assert: survives `python -O` and bandit B101.
            raise RuntimeError("SettingsStore used after close()")
        return self._conn

    @property
    def journal_mode(self) -> str:
        row = self._db().execute("PRAGMA journal_mode").fetchone()
        return str(row[0]).lower()

    def get(self, key: str, default: str | None = None) -> str | None:
        row = self._db().execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        return default if row is None else row[0]

    def set(self, key: str, value: str) -> None:
        self._db().execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value,"
            " updated_at = excluded.updated_at",
            (key, value, time.time()),
        )
        self._db().commit()

    def delete(self, key: str) -> None:
        self._db().execute("DELETE FROM settings WHERE key = ?", (key,))
        self._db().commit()

    def all(self) -> dict[str, str]:
        rows = self._db().execute("SELECT key, value FROM settings").fetchall()
        return dict(rows)

    def close(self) -> None:
        """Release the database file. Safe to call more than once."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class SecretStore(Protocol):
    """Where API keys live. Implementations: memory (dev/test), keyring (prod)."""

    def set_key(self, provider: str, key: str) -> None: ...

    def get_key(self, provider: str) -> str | None: ...

    def delete_key(self, provider: str) -> None: ...


class InMemorySecretStore:
    """Secrets held in process memory only — for tests and throwaway runs."""

    def __init__(self) -> None:
        self._secrets: dict[str, str] = {}

    def set_key(self, provider: str, key: str) -> None:
        self._secrets[provider] = key

    def get_key(self, provider: str) -> str | None:
        return self._secrets.get(provider)

    def delete_key(self, provider: str) -> None:
        self._secrets.pop(provider, None)


class KeyringSecretStore:
    """Secrets in the OS keychain via keyring (lazy import — optional dep).

    Cloud mode (Hugging Face/containers): a headless box has no keychain, so
    ``get_key`` falls back to environment variables — ``ECHO_KEY_<NAME>`` or
    the conventional name (``GROQ_API_KEY``, ``NVIDIA_API_KEY``). Set them as
    Space secrets; nothing is written to disk. set/delete stay keyring-only
    (a cloud copy never pretends to persist what it can't).
    """

    def __init__(self) -> None:
        try:
            import keyring  # imported lazily: tests stay hermetic without it

            self._keyring = keyring
        except Exception:  # no keyring backend at all (headless container)
            self._keyring = None

    @staticmethod
    def _env_name(provider: str) -> str:
        conventional = {
            "groq": "GROQ_API_KEY",
            "nvidia": "NVIDIA_API_KEY",
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
        }
        return conventional.get(provider.lower()) or (
            "ECHO_KEY_" + provider.upper().replace(".", "_").replace("-", "_")
        )

    def set_key(self, provider: str, key: str) -> None:
        if self._keyring is None:
            raise RuntimeError("no OS keychain in this environment — use ECHO_KEY_* secrets")
        self._keyring.set_password(KEYRING_SERVICE, provider, key)

    def get_key(self, provider: str) -> str | None:
        if self._keyring is not None:
            try:
                found = self._keyring.get_password(KEYRING_SERVICE, provider)
                if found:
                    return found
            except Exception:  # no backend — fall through to the environment
                pass
        return os.environ.get(self._env_name(provider))

    def delete_key(self, provider: str) -> None:
        if self._keyring is None:
            return
        self._keyring.delete_password(KEYRING_SERVICE, provider)


def load_env(path: Path = ENV_PATH) -> dict[str, str]:
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Real environment variables always win over the file. Returns the pairs
    that were loaded. No dependencies — same behaviour on Windows and macOS.
    """
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded
