"""MemoryStore (Phase 6.1) — facts stored once, recalled forever, locally, $0.

One SQLite database (WAL, context manager, idempotent close, RuntimeError
guards — the exam-bug hygiene every Echo store shares), one schema, and TWO
search backends behind one interface::

    store = MemoryStore()                      # default: FTS5, zero deps
    store.add("Sister's birthday is March 3")
    store.search("when is the birthday?")      # bm25-ranked hits

Backends (same interface, two engines, $0):

- **FTS5 (default)** — a stdlib virtual table mirrors every memory's text;
  ``search`` builds a tolerant MATCH expression (each token quoted, junk-safe)
  and orders by bm25 rank. An FTS syntax error degrades to a LIKE scan, and an
  empty query returns the most recent memories — search never crashes on
  whatever a model or user throws at it.
- **Chroma (optional)** — inject an ``embedder`` (an object exposing
  ``async embed(texts) -> list[list[float]]``) and the store mirrors every
  write into a Chroma collection persisted under ``<db dir>/chroma``; search
  then uses cosine similarity. ``chromadb`` is imported lazily: missing, it
  raises a RuntimeError with the pip hint ONLY when explicitly requested (an
  embedder injected) — the default constructor stays pure FTS5. SQLite is the
  source of truth either way: vector writes are a best-effort mirror, vector
  read failures fall back to FTS5.

Superseding is the soft-delete flag the distiller (Phase 6.3) uses for
contradictions: superseded rows stay in the table (audit trail) but leave
every search, list, count and wallet view.

The H5-2 seam: :func:`as_relevant_adapter` wraps a store into exactly the
object :class:`~app.harness.ctx.processors.MemoryInjectStage` expects —
``async relevant(query, k) -> list[str]`` of ``"kind: text"`` strings.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from app import config as _config

logger = logging.getLogger("echo.memory")

#: Where the optional Chroma collection persists, relative to the db file.
CHROMA_DIRNAME = "chroma"

#: The kind that pins a memory into the wallet (Phase 6.2).
CORE_KIND = "core"

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS memories ("
    " id TEXT PRIMARY KEY,"
    " text TEXT NOT NULL,"
    " kind TEXT NOT NULL DEFAULT 'fact',"  # fact|preference|decision|person (+core)
    " importance REAL NOT NULL DEFAULT 1.0,"
    " source_session TEXT,"
    " created_at REAL NOT NULL,"
    " updated_at REAL NOT NULL,"
    " superseded INTEGER NOT NULL DEFAULT 0)"
)

# The FTS5 mirror of memory text (stdlib sqlite3). `id UNINDEXED` maps every
# hit back to its row; the index is kept in sync by the CRUD methods below.
_FTS_SCHEMA = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(id UNINDEXED, text)"
)

# Fully literal statements (no string building): parameters carry every value,
# and bandit's B608 has nothing dynamic to look at.
_SQL_GET_ONE = (
    "SELECT id, text, kind, importance, source_session, created_at, updated_at, superseded"
    " FROM memories WHERE id = ?"
)
_SQL_LIST_LIVE = (
    "SELECT id, text, kind, importance, source_session, created_at, updated_at, superseded"
    " FROM memories WHERE superseded = 0 ORDER BY created_at DESC, rowid DESC LIMIT ?"
)
_SQL_LIST_EVERYTHING = (
    "SELECT id, text, kind, importance, source_session, created_at, updated_at, superseded"
    " FROM memories ORDER BY created_at DESC, rowid DESC LIMIT ?"
)
_SQL_LIKE_SCAN = (
    "SELECT id, text, kind, importance, source_session, created_at, updated_at, superseded"
    " FROM memories WHERE superseded = 0 AND text LIKE ? ESCAPE '\\'"
)

# "Tolerant query" tokenizer: FTS5 sees only word characters, each double
# quoted, so user/model junk like `query AND OR (` can never form syntax.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

#: Cap on tokens a single LIKE fallback scan walks (a pathological query
#: cannot turn into a pathological table scan).
_MAX_LIKE_TOKENS = 8


def _fts_expr(query: str) -> str:
    """Build a junk-proof FTS5 MATCH expression: quoted tokens joined by OR.

    Tokens come from ``\\w+`` so no operator characters survive; embedded
    double quotes are escaped anyway (belt and braces). OR semantics keep
    search tolerant — a document matching one of several query words still
    surfaces, and bm25 ranks documents matching MORE words first. Empty
    input -> empty expression (the caller treats that as "no query").
    """
    return " OR ".join('"%s"' % token.replace('"', '""') for token in _TOKEN_RE.findall(query))


def _like_escape(token: str) -> str:
    """Escape LIKE wildcards for a token scanned with ``ESCAPE '\\'``."""
    return token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _row_to_dict(row: sqlite3.Row) -> dict:
    """One memories row as a plain dict (``superseded`` stays a 0/1 int)."""
    return {key: row[key] for key in row.keys()}


class ChromaBackend:
    """The optional vector backend — same shape, cosine similarity instead of bm25.

    Constructed ONLY when an embedder is explicitly injected into
    :class:`MemoryStore`. ``chromadb`` is imported lazily right here, so a
    machine without it pays nothing until the day it opts in; when it does
    and the package is absent, the RuntimeError carries the pip hint.
    """

    def __init__(self, path: Path, embedder: Any) -> None:
        try:
            import chromadb  # lazy: the one heavy dep, paid for only on request
        except ImportError as exc:
            raise RuntimeError(
                "the Chroma vector backend was explicitly requested but chromadb "
                "is not installed — install it with: pip install chromadb"
            ) from exc
        self._embedder = embedder
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            "memories", metadata={"hnsw:space": "cosine"}
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Run the async embedder from sync code (the sync CRUD API's bridge).

        No running loop -> ``asyncio.run``. Already inside one (a tool call on
        the loop) -> hop to a worker thread so the loop is never blocked.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._embedder.embed(texts))
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self._embedder.embed(texts)).result()

    def index(self, memory_id: str, text: str) -> None:
        """Upsert one memory's vector (cosine space)."""
        self._collection.upsert(
            ids=[memory_id], documents=[text], embeddings=self._embed([text])
        )

    def remove(self, memory_id: str) -> None:
        """Drop one memory's vector (hard deletes only — superseding is flag-only)."""
        self._collection.delete(ids=[memory_id])

    def search(self, query: str, k: int) -> list[str]:
        """Nearest neighbours first, as memory ids (cosine distance ascending)."""
        total = int(self._collection.count())
        if total <= 0:
            return []
        result = self._collection.query(
            query_embeddings=self._embed([query]), n_results=max(1, min(k, total))
        )
        ids = result.get("ids") or []
        return [str(memory_id) for memory_id in (ids[0] if ids else [])]


class RelevantAdapter:
    """The H5-2 memory seam object: ``async relevant(query, k) -> list[str]``.

    Exactly what ``MemoryInjectStage`` calls. Fetches more candidates than
    asked for, re-orders them importance-first (stable, so bm25 order breaks
    ties), then formats each as ``"kind: text"`` — the shape the pipeline
    injects next to the latest user turn.
    """

    def __init__(self, store: "MemoryStore") -> None:
        self._store = store

    async def relevant(self, query: str, k: int = 5) -> list[str]:
        """Top memories for ``query``, importance-weighted, formatted."""
        try:
            wanted = max(1, int(k))
        except (TypeError, ValueError):
            wanted = 5
        hits = self._store.search(query, k=wanted * 3)  # over-fetch, then weight
        ranked = sorted(hits, key=lambda row: -float(row.get("importance", 1.0) or 0.0))
        return [f"{row['kind']}: {row['text']}" for row in ranked[:wanted]]


def as_relevant_adapter(store: "MemoryStore") -> RelevantAdapter:
    """Wrap a store for the H5-2 seam (``MemoryInjectStage(memory=...)``)."""
    return RelevantAdapter(store)


class MemoryStore:
    """Durable long-term memory — SQLite at heart, FTS5 by default, Chroma optional.

    Usable as a context manager (releases the Windows file lock on exit);
    ``close()`` is idempotent and every method raises RuntimeError (never an
    assert — bandit B101) after it. All writes commit before returning.
    """

    def __init__(self, path: str | Path | None = None, embedder: Any | None = None) -> None:
        # Lazy-resolved at call time: tests patch app.config.DATA_DIR after import
        # (the H0 session-log lesson — never bind runtime paths at import).
        self._path = Path(path) if path is not None else _config.DATA_DIR / "memory.db"
        if embedder is not None and not callable(getattr(embedder, "embed", None)):
            raise ValueError(
                "embedder must expose 'async def embed(texts) -> list[list[float]]'"
            )
        # Build the optional vector backend BEFORE opening SQLite, so a missing
        # chromadb cannot leak a connection on the error path.
        self._vector: ChromaBackend | None = None
        if embedder is not None:
            self._vector = ChromaBackend(self._path.parent / CHROMA_DIRNAME, embedder)
        self._conn: sqlite3.Connection | None = self._connect()

    # ------------------------------------------------------------- lifecycle

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL").fetchone()
        conn.execute(_SCHEMA)
        conn.execute(_FTS_SCHEMA)
        conn.commit()
        return conn

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            # A real exception, not an assert: survives `python -O` and bandit B101.
            raise RuntimeError("MemoryStore used after close()")
        return self._conn

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Release the database file. Safe to call more than once."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def journal_mode(self) -> str:
        """The SQLite journal mode (always ``wal`` — crash-safe from day one)."""
        row = self._db().execute("PRAGMA journal_mode").fetchone()
        return str(row[0]).lower()

    # ------------------------------------------------------------------ CRUD

    def add(
        self,
        text: str,
        kind: str = "fact",
        importance: float = 1.0,
        source_session: str | None = None,
    ) -> dict:
        """Store one memory; returns the full record (``id`` included)."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("memory text must be a non-empty string")
        now = time.time()
        memory_id = uuid.uuid4().hex
        record_kind = str(kind) or "fact"
        session = str(source_session) if source_session is not None else None
        db = self._db()
        db.execute(
            "INSERT INTO memories (id, text, kind, importance, source_session,"
            " created_at, updated_at, superseded) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
            (memory_id, text, record_kind, float(importance), session, now, now),
        )
        db.execute("INSERT INTO memories_fts (id, text) VALUES (?, ?)", (memory_id, text))
        db.commit()
        self._mirror("index", memory_id, text)
        return {
            "id": memory_id,
            "text": text,
            "kind": record_kind,
            "importance": float(importance),
            "source_session": session,
            "created_at": now,
            "updated_at": now,
            "superseded": 0,
        }

    def get(self, memory_id: str) -> dict | None:
        """One memory by id (superseded rows included — exact lookups are raw)."""
        row = self._db().execute(_SQL_GET_ONE, (memory_id,)).fetchone()
        return _row_to_dict(row) if row is not None else None

    def update(
        self,
        memory_id: str,
        text: str | None = None,
        kind: str | None = None,
        importance: float | None = None,
    ) -> dict | None:
        """Partial update; returns the fresh record, or None for an unknown id."""
        db = self._db()
        now = time.time()
        if text is not None:
            if not str(text).strip():
                raise ValueError("memory text must be a non-empty string")
            db.execute(
                "UPDATE memories SET text = ?, updated_at = ? WHERE id = ?",
                (str(text), now, memory_id),
            )
            # DELETE+INSERT keeps the FTS5 mirror in sync without relying on
            # virtual-table UPDATE semantics.
            db.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))
            db.execute("INSERT INTO memories_fts (id, text) VALUES (?, ?)", (memory_id, str(text)))
        if kind is not None:
            db.execute(
                "UPDATE memories SET kind = ?, updated_at = ? WHERE id = ?",
                (str(kind), now, memory_id),
            )
        if importance is not None:
            db.execute(
                "UPDATE memories SET importance = ?, updated_at = ? WHERE id = ?",
                (float(importance), now, memory_id),
            )
        db.commit()
        if text is not None:
            self._mirror("index", memory_id, text)
        return self.get(memory_id)

    def delete(self, memory_id: str) -> bool:
        """Hard delete (the manager screen's delete button). True if a row went."""
        db = self._db()
        cursor = db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        db.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))
        db.commit()
        self._mirror("remove", memory_id)
        return cursor.rowcount > 0

    def supersede(self, memory_id: str) -> bool:
        """Soft-flag a memory as replaced (the distiller's contradiction move).

        The row stays (audit trail) but leaves search, list, count and wallet
        views. Idempotent; False for an unknown id.
        """
        cursor = self._db().execute(
            "UPDATE memories SET superseded = 1, updated_at = ? WHERE id = ?",
            (time.time(), memory_id),
        )
        self._db().commit()
        return cursor.rowcount > 0

    def list_all(self, limit: int = 200, include_superseded: bool = False) -> list[dict]:
        """Recent-first listing (newest created wins; rowid breaks same-instant ties)."""
        wanted = max(0, int(limit))
        if wanted == 0:
            return []
        statement = _SQL_LIST_EVERYTHING if include_superseded else _SQL_LIST_LIVE
        rows = self._db().execute(statement, (wanted,)).fetchall()
        return [_row_to_dict(row) for row in rows]

    def count(self, include_superseded: bool = False) -> int:
        """How many live memories (superseded ones only when asked for)."""
        if include_superseded:
            row = self._db().execute("SELECT COUNT(*) FROM memories").fetchone()
        else:
            row = self._db().execute(
                "SELECT COUNT(*) FROM memories WHERE superseded = 0"
            ).fetchone()
        return int(row[0])

    # ---------------------------------------------------------------- search

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Top-``k`` memories for ``query`` — never raises on weird input.

        Empty/blank query -> the most recent memories. Chroma backend present
        -> cosine similarity (FTS5 on vector failure). Otherwise FTS5 bm25
        rank, with a LIKE scan as the tolerant fallback. Superseded rows are
        always excluded.
        """
        self._db()  # closed-guard (also the only touch needed for validation)
        try:
            wanted = int(k)
        except (TypeError, ValueError):
            wanted = 5
        if wanted <= 0:
            return []
        text = str(query or "").strip()
        if not text:
            return self.list_all(limit=wanted)
        if self._vector is not None:
            return self._vector_search(text, wanted)
        try:
            return self._fts_search(text, wanted)
        except sqlite3.OperationalError:
            logger.warning("FTS5 rejected query %r; LIKE fallback", text, exc_info=True)
            return self._like_search(text, wanted)

    def _fts_search(self, query: str, k: int) -> list[dict]:
        """bm25-ranked FTS5 match (ascending bm25 = best first)."""
        expression = _fts_expr(query)
        if not expression:
            return self.list_all(limit=k)
        rows = self._db().execute(
            "SELECT m.id, m.text, m.kind, m.importance, m.source_session,"
            " m.created_at, m.updated_at, m.superseded"
            " FROM memories_fts JOIN memories AS m ON m.id = memories_fts.id"
            " WHERE memories_fts MATCH ? AND m.superseded = 0"
            " ORDER BY bm25(memories_fts)",
            (expression,),
        ).fetchall()
        return [_row_to_dict(row) for row in rows][:k]

    def _like_search(self, query: str, k: int) -> list[dict]:
        """The tolerant fallback: per-token LIKE scan, any token may match."""
        tokens = _TOKEN_RE.findall(query)[:_MAX_LIKE_TOKENS] or [query[:64]]
        found: dict[str, dict] = {}
        for token in tokens:
            rows = self._db().execute(
                _SQL_LIKE_SCAN,
                (f"%{_like_escape(token)}%",),
            ).fetchall()
            for row in rows:
                found[row["id"]] = _row_to_dict(row)
        hits = sorted(found.values(), key=lambda item: -float(item["created_at"]))
        return hits[:k]

    def _vector_search(self, query: str, k: int) -> list[dict]:
        """Cosine-similarity search via the Chroma mirror; FTS5 on any failure."""
        try:
            ids = self._vector.search(query, k) if self._vector is not None else []
        except Exception:  # noqa: BLE001 — memory must never break the loop
            logger.warning("vector search failed; falling back to FTS5", exc_info=True)
            try:
                return self._fts_search(query, k)
            except sqlite3.OperationalError:
                return self._like_search(query, k)
        hits: list[dict] = []
        for memory_id in ids:
            row = self.get(memory_id)
            if row is not None and not row["superseded"]:
                hits.append(row)
        return hits[:k]

    # ----------------------------------------------------------- vector mirror

    def _mirror(self, op: str, memory_id: str, text: str | None = None) -> None:
        """Best-effort write mirror into the vector index (when present).

        SQLite is the source of truth: a mirror failure is logged and swallowed
        so the durable write always stands.
        """
        if self._vector is None:
            return
        try:
            if op == "index":
                self._vector.index(memory_id, text or "")
            elif op == "remove":
                self._vector.remove(memory_id)
        except Exception:  # noqa: BLE001 — derived cache; never break the write
            logger.warning(
                "vector mirror failed for %s; SQLite remains source of truth",
                memory_id,
                exc_info=True,
            )
