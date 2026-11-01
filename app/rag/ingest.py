"""Document ingestion (Phase 6.5 Part A) — parse, chunk, refmap, persist.

Two entry points over one pipeline (:func:`ingest_file`, :func:`ingest_folder`)
and the persistent multi-document library the query half searches:
:class:`RagLibrary`. Design: docs/phase6-rag-contract.md, DECISIONS.md D-023.

:class:`RagLibrary` follows the exam-bug hygiene every Echo store shares
(:class:`~app.memory.store.MemoryStore` is the reference): one SQLite file
(``rag.db``, WAL journaling from day one), a context manager that releases
the Windows file lock, an idempotent ``close()``, and RuntimeError — never
assert — guards after close. Three tables: ``documents``, ``chunks``,
``refmap``. VECTORS ARE NOT STORED HERE — the query half owns the FAISS/npy
side; this library holds text, hashes and provenance, and rehydrates chunks
in stable index order.

Embedding is deliberately absent: ingestion stores text + content hashes
only. (``ingest_file`` accepts an ``embedder`` argument for call-site
compatibility with the contract's wider signature but does not use it — the
query half embeds ``embed_text`` against its own hash cache.)
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app import config as _config
from app.rag.chunker import Chunk, chunk_blocks
from app.rag.parsers import SUPPORTED_SUFFIXES, parse_any
from app.rag.refmap import RefMap

logger = logging.getLogger("echo.rag.ingest")

#: Where the library database lives, relative to its directory.
DB_FILENAME = "rag.db"

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS documents ("
    " doc_id TEXT PRIMARY KEY,"
    " name TEXT NOT NULL,"
    " path TEXT NOT NULL,"
    " n_chunks INTEGER NOT NULL DEFAULT 0,"
    " ingested_at REAL NOT NULL)",
    "CREATE TABLE IF NOT EXISTS chunks ("
    " chunk_id TEXT PRIMARY KEY,"
    " doc_id TEXT NOT NULL,"
    " idx INTEGER NOT NULL,"
    " text TEXT NOT NULL,"
    " embed_text TEXT NOT NULL,"
    " content_hash TEXT NOT NULL,"
    " parent_section_id TEXT NOT NULL DEFAULT '',"
    " page INTEGER,"
    " is_ocr INTEGER NOT NULL DEFAULT 0)",
    "CREATE TABLE IF NOT EXISTS refmap ("
    " ref_id TEXT NOT NULL,"
    " chunk_id TEXT NOT NULL,"
    " PRIMARY KEY (ref_id, chunk_id))",
    "CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks (doc_id, idx)",
)

# Fully literal statements (parameters carry every value) — the bandit-B608
# discipline the memory store set.
_SQL_INSERT_DOCUMENT = (
    "INSERT OR REPLACE INTO documents (doc_id, name, path, n_chunks, ingested_at)"
    " VALUES (?, ?, ?, ?, ?)"
)
_SQL_INSERT_CHUNK = (
    "INSERT OR REPLACE INTO chunks (chunk_id, doc_id, idx, text, embed_text,"
    " content_hash, parent_section_id, page, is_ocr) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
_SQL_INSERT_REF = "INSERT OR IGNORE INTO refmap (ref_id, chunk_id) VALUES (?, ?)"
_SQL_SELECT_DOCUMENTS = (
    "SELECT doc_id, name, path, n_chunks, ingested_at FROM documents"
    " ORDER BY ingested_at, rowid"
)
_SQL_SELECT_CHUNKS = (
    "SELECT chunk_id, doc_id, idx, text, embed_text, content_hash,"
    " parent_section_id, page, is_ocr FROM chunks"
)
_SQL_DELETE_REFMAP_FOR_DOC = (
    "DELETE FROM refmap WHERE chunk_id IN"
    " (SELECT chunk_id FROM chunks WHERE doc_id = ?)"
)


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    """Rehydrate one chunks row into a :class:`~app.rag.chunker.Chunk`."""
    parent = str(row["parent_section_id"] or "")
    return Chunk(
        id=str(row["chunk_id"]),
        doc_id=str(row["doc_id"]),
        index=int(row["idx"]),
        text=str(row["text"] or ""),
        embed_text=str(row["embed_text"] or ""),
        content_hash=str(row["content_hash"] or ""),
        parent_section_id=parent,
        page=int(row["page"]) if row["page"] is not None else None,
        is_ocr=bool(row["is_ocr"]),
        # "" (no headings) round-trips to an empty path, " > " joins back apart.
        section_path=parent.split(" > ") if parent else [],
    )


def _doc_id_for(file_path: Path) -> str:
    """Stable document id from the file's resolved path — re-ingesting the
    same file replaces the same document instead of duplicating it."""
    digest = hashlib.sha256(str(file_path.resolve()).encode("utf-8")).hexdigest()
    return f"doc_{digest[:16]}"


def _looks_like_library(candidate: Any) -> bool:
    """True when the object exposes the RagLibrary write/read surface."""
    return callable(getattr(candidate, "add_chunk", None)) and callable(
        getattr(candidate, "load_refmap", None)
    )


def _normalize_library_args(first: Any, second: Any) -> tuple[Any, Path]:
    """Accept both ``(library, path)`` and the contract table's ``(path, library)``.

    The frozen contract writes ``ingest_file(path, library)``; Echo's desk
    tools may call it that way. The canonical Echo order is
    ``(library, path)``. Whichever argument quacks like a RagLibrary is the
    library; when neither does, the arguments pass through untouched so the
    natural TypeError surfaces.
    """
    if _looks_like_library(first):
        return first, Path(second)
    if _looks_like_library(second):
        return second, Path(first)
    return first, Path(second)


class RagLibrary:
    """The persistent document library — SQLite at heart, vectors elsewhere.

    Usable as a context manager (releases the Windows file lock on exit);
    ``close()`` is idempotent and every method raises RuntimeError (never an
    assert — bandit B101) after it. All writes commit before returning.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        # ``path`` is the library DIRECTORY (rag.db lives inside). Resolved at
        # call time so tests can patch app.config.DATA_DIR after import — the
        # H0 session-log lesson: never bind runtime paths at import.
        directory = Path(path) if path is not None else _config.DATA_DIR / "rag"
        self._dir = directory
        self._db_path = directory / DB_FILENAME
        self._conn: sqlite3.Connection | None = self._connect()

    # ------------------------------------------------------------- lifecycle

    def _connect(self) -> sqlite3.Connection:
        self._dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL").fetchone()
        for statement in _SCHEMA:
            conn.execute(statement)
        conn.commit()
        return conn

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            # A real exception, not an assert: survives `python -O` and bandit B101.
            raise RuntimeError("RagLibrary used after close()")
        return self._conn

    def __enter__(self) -> "RagLibrary":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Release the database file. Safe to call more than once."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def path(self) -> Path:
        """The library directory (``rag.db`` lives inside it)."""
        return self._dir

    @property
    def journal_mode(self) -> str:
        """The SQLite journal mode (always ``wal`` — crash-safe from day one)."""
        row = self._db().execute("PRAGMA journal_mode").fetchone()
        return str(row[0]).lower()

    # ------------------------------------------------------------------ CRUD

    def add_document(self, doc_id: str, name: str, path: str, n_chunks: int) -> dict:
        """Register (or replace) a document row; returns the full record."""
        ingested_at = time.time()
        self._db().execute(
            _SQL_INSERT_DOCUMENT,
            (str(doc_id), str(name), str(path), int(n_chunks), ingested_at),
        )
        self._db().commit()
        return {
            "doc_id": str(doc_id),
            "name": str(name),
            "path": str(path),
            "n_chunks": int(n_chunks),
            "ingested_at": ingested_at,
        }

    def add_chunk(self, chunk: Chunk) -> None:
        """Persist one chunk (vectors deliberately not stored here)."""
        self._db().execute(
            _SQL_INSERT_CHUNK,
            (
                chunk.id,
                chunk.doc_id,
                int(chunk.index),
                chunk.text,
                chunk.embed_text,
                chunk.content_hash,
                chunk.parent_section_id,
                chunk.page,
                1 if chunk.is_ocr else 0,
            ),
        )
        self._db().commit()

    def add_ref(self, ref_id: str, chunk_id: str) -> None:
        """Record one ref-id -> chunk-id edge (duplicate edges are ignored)."""
        if not str(ref_id) or not str(chunk_id):
            return
        self._db().execute(_SQL_INSERT_REF, (str(ref_id), str(chunk_id)))
        self._db().commit()

    def list_documents(self) -> list[dict]:
        """All documents, oldest ingestion first (stable, insertion-ordered)."""
        rows = self._db().execute(_SQL_SELECT_DOCUMENTS).fetchall()
        return [
            {
                "doc_id": row["doc_id"],
                "name": row["name"],
                "path": row["path"],
                "n_chunks": int(row["n_chunks"]),
                "ingested_at": float(row["ingested_at"]),
            }
            for row in rows
        ]

    def get_chunks(self, doc_id: str | None = None) -> list[Chunk]:
        """Rehydrate chunks — one document's, or the whole library's — always
        in stable (doc_id, index) order so the query half sees determinism."""
        if doc_id is None:
            rows = self._db().execute(
                _SQL_SELECT_CHUNKS + " ORDER BY doc_id, idx"
            ).fetchall()
        else:
            rows = self._db().execute(
                _SQL_SELECT_CHUNKS + " WHERE doc_id = ? ORDER BY idx", (str(doc_id),)
            ).fetchall()
        return [_row_to_chunk(row) for row in rows]

    def get_chunks_by_ids(self, ids: list[str]) -> list[Chunk]:
        """Chunks in the CALLER'S order (retrieval rank), unknown ids skipped."""
        wanted = [str(chunk_id) for chunk_id in ids or []]
        found: dict[str, Chunk] = {}
        ordered: list[Chunk] = []
        db = self._db()
        for chunk_id in wanted:
            if chunk_id in found:
                ordered.append(found[chunk_id])
                continue
            row = db.execute(
                _SQL_SELECT_CHUNKS + " WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
            if row is not None:
                found[chunk_id] = _row_to_chunk(row)
                ordered.append(found[chunk_id])
        return ordered

    def chunk_count(self, doc_id: str | None = None) -> int:
        """Number of chunks — the whole library, or one document's."""
        if doc_id is None:
            row = self._db().execute("SELECT COUNT(*) FROM chunks").fetchone()
        else:
            row = self._db().execute(
                "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (str(doc_id),)
            ).fetchone()
        return int(row[0])

    def load_refmap(self) -> RefMap:
        """Rebuild the reference graph from the persisted ``refmap`` table."""
        refmap = RefMap()
        rows = self._db().execute(
            "SELECT ref_id, chunk_id FROM refmap ORDER BY rowid"
        ).fetchall()
        for row in rows:
            known = refmap.definitions.setdefault(str(row["ref_id"]), [])
            chunk_id = str(row["chunk_id"])
            if chunk_id not in known:
                known.append(chunk_id)
        return refmap

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document with its chunks and ref edges. True if it went."""
        db = self._db()
        # Ref edges first (they point at chunk ids via the subquery), then the
        # chunks they belonged to, then the document row itself.
        db.execute(_SQL_DELETE_REFMAP_FOR_DOC, (str(doc_id),))
        db.execute("DELETE FROM chunks WHERE doc_id = ?", (str(doc_id),))
        cursor = db.execute("DELETE FROM documents WHERE doc_id = ?", (str(doc_id),))
        db.commit()
        return cursor.rowcount > 0


# ----------------------------------------------------------------- pipeline


def ingest_file(
    library: RagLibrary,
    path: str | Path,
    embedder: Any | None = None,
    ocr_provider: Callable[[bytes], str] | None = None,
) -> dict:
    """Ingest one file end to end: parse -> chunk -> refmap -> persist.

    Returns ``{"doc_id", "n_chunks", "pages"}`` (``pages`` counts the
    distinct non-None pages seen while parsing — 0 for page-less formats).
    Re-ingesting the same path replaces the same document in place.
    Embedding is the query half's business; ``embedder`` is accepted for
    signature compatibility and unused here.

    Argument order: ``(library, path)`` is canonical; the contract table's
    ``(path, library)`` is accepted too (see :func:`_normalize_library_args`).
    """
    del embedder  # the query half embeds embed_text against its own hash cache
    library, path = _normalize_library_args(library, path)
    file_path = Path(path)
    doc_id = _doc_id_for(file_path)
    blocks = parse_any(file_path, ocr_provider=ocr_provider)
    chunks = chunk_blocks(blocks, doc_id)
    refmap = RefMap()
    refmap.register_definitions(chunks)

    # Replace-in-place keeps re-ingests idempotent (same path -> same doc_id).
    library.delete_document(doc_id)
    library.add_document(
        doc_id=doc_id, name=file_path.name, path=str(file_path), n_chunks=len(chunks)
    )
    for chunk in chunks:
        library.add_chunk(chunk)
    for ref_id, chunk_ids in refmap.definitions.items():
        for chunk_id in chunk_ids:
            library.add_ref(ref_id, chunk_id)
    pages = len({block.page for block in blocks if block.page is not None})
    logger.info("ingested %s: %d chunks, %d pages", file_path.name, len(chunks), pages)
    return {"doc_id": doc_id, "n_chunks": len(chunks), "pages": pages}


def ingest_folder(
    library: RagLibrary,
    path: str | Path,
    ocr_provider: Callable[[bytes], str] | None = None,
) -> dict:
    """Ingest every supported file under ``path`` (recursively, sorted order).

    Files with unsupported suffixes are counted in ``skipped`` and left
    alone. A single-file path is tolerated and delegated to
    :func:`ingest_file`. Returns ``{"documents", "chunks", "skipped"}``.
    Accepts ``(library, path)`` (canonical) or ``(path, library)``.
    """
    library, path = _normalize_library_args(library, path)
    target = Path(path)
    if target.is_file():
        result = ingest_file(library, target, ocr_provider=ocr_provider)
        return {"documents": 1, "chunks": result["n_chunks"], "skipped": 0}
    if not target.exists():
        raise FileNotFoundError(f"no such file or folder: {target}")
    documents = 0
    chunks = 0
    skipped = 0
    for file_path in sorted(p for p in target.rglob("*") if p.is_file()):
        if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            skipped += 1
            continue
        result = ingest_file(library, file_path, ocr_provider=ocr_provider)
        documents += 1
        chunks += result["n_chunks"]
    logger.info(
        "ingested folder %s: %d documents, %d chunks, %d skipped",
        target.name, documents, chunks, skipped,
    )
    return {"documents": documents, "chunks": chunks, "skipped": skipped}
