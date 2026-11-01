"""Embedders (Phase 6.5) — text to unit vectors, deterministic fake first.

The contract's embedder seam is one async method::

    embedder.embed(["some text"])  ->  [[0.0] * 384]   # L2-normalized, always

Two implementations behind that one shape:

- **HashEmbedder** — the deterministic stdlib test fake. ``sha256(text)`` seeds
  a tiny xorshift64* PRNG that emits ``dim`` floats in [-1, 1), then the vector
  is L2-normalized. Same text in, same unit vector out, forever, on every
  machine — which is exactly what offline tests need, and why the query half
  can re-embed a corpus for free. It is NOT semantic: it knows nothing about
  meaning, only identity.
- **MiniLMEmbedder** — the real one (``sentence-transformers/all-MiniLM-L6-v2``,
  384-dim, ``normalize_embeddings=True``, batch 64). ``sentence_transformers``
  is imported lazily inside :meth:`MiniLMEmbedder.embed`: missing, it raises a
  RuntimeError with the ``pip install -r requirements-rag.txt`` hint ONLY when
  a caller explicitly asked for neural embeddings — the default suite pays
  nothing. Every computed vector is persisted in a **sha256(embed_text) →
  vector cache** (a ``.npy`` rows file + a pickle key→row map under the
  caller-supplied cache dir), so a re-ingest or a fresh process re-embeds
  nothing. :meth:`MiniLMEmbedder.stats` reports ``{"total", "reused", "new"}``.

The cache is written with the stdlib only (a hand-rolled ``.npy`` writer —
float64 rows, version 1.0 format), so cache round-trips are testable on hosts
without numpy. ``embed_sync`` is the shared async→sync bridge (the same
loop-hopping move :class:`app.memory.store.ChromaBackend` uses) for callers
that live outside the event loop.
"""

from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import hashlib
import logging
import math
import os
import struct
import json
from array import array
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("echo.rag.embedder")

#: Default dimensionality (all-MiniLM-L6-v2 and the HashEmbedder's default).
DEFAULT_DIM = 384

#: MiniLM encode batch size (contract: batch 64).
MINILM_BATCH = 64

#: The default neural model (the contract's 384-dim MiniLM).
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#: The one-line fix for every missing heavy dep in this package.
RAG_REQUIREMENTS_HINT = "pip install -r requirements-rag.txt"

_NPY_MAGIC = b"\x93NUMPY"
_MASK64 = (1 << 64) - 1
#: xorshift64* seed fallback (the golden ratio constant — never zero).
_PRNG_SEED_FALLBACK = 0x9E3779B97F4A7C15
#: xorshift64* output multiplier (the classic constant).
_PRNG_MULTIPLIER = 0x2545F4914F6CDD1D


@runtime_checkable
class Embedder(Protocol):
    """The embedder seam every index/cache/consumer codes against."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` into L2-normalized unit vectors (one per input)."""
        ...


def _prng_unit_floats(seed: bytes, dim: int) -> list[float]:
    """``dim`` floats in [-1.0, 1.0) from an xorshift64* PRNG seeded by ``seed``.

    Deliberately tiny and dependency-free: the seed is the first 8 bytes of a
    sha256 digest, and each step xorshifts a 64-bit state and multiplies by
    the xorshift64* constant. Deterministic across platforms and Python
    versions (pure integer math), which is the HashEmbedder's whole contract.
    """
    state = int.from_bytes(seed[:8], "big") or _PRNG_SEED_FALLBACK
    values: list[float] = []
    for _ in range(dim):
        state ^= state >> 12
        state ^= (state << 25) & _MASK64
        state ^= state >> 27
        product = (state * _PRNG_MULTIPLIER) & _MASK64
        values.append((product / (1 << 64)) * 2.0 - 1.0)
    return values


def hash_vector(text: str, dim: int = DEFAULT_DIM) -> list[float]:
    """Deterministic L2-normalized unit vector for ``text`` (HashEmbedder math).

    sha256 -> PRNG seed -> ``dim`` floats -> unit norm. Not semantic — a
    fingerprint — but stable, fast and free, which is what the offline suite
    and the two-hop/dedup paths need from a fake.
    """
    digest = hashlib.sha256(str(text).encode("utf-8")).digest()
    values = _prng_unit_floats(digest, dim)
    norm = math.sqrt(math.fsum(value * value for value in values))
    if norm == 0.0:  # unreachable for dim >= 1 (xorshift64* never emits all-zeros)
        return [0.0] * dim
    return [value / norm for value in values]


def normalized(vector: list[float]) -> list[float]:
    """Return a unit-norm copy of ``vector`` (zero vector stays zero)."""
    values = [float(x) for x in vector]
    norm = math.sqrt(math.fsum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]


def embed_sync(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    """Run an async embedder from sync code (the ChromaBackend bridge, shared).

    No running loop -> ``asyncio.run``. Already inside one (a tool call on the
    loop) -> hop to a worker thread so the loop is never blocked.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(embedder.embed(texts))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, embedder.embed(texts)).result()


# --------------------------------------------------------------- .npy (stdlib)


def _write_npy(rows: list[list[float]], dim: int) -> bytes:
    """Serialize float64 rows as a version-1.0 ``.npy`` image (stdlib only).

    Returns the bytes (the caller writes them atomically) so tests can
    round-trip without touching the filesystem.
    """
    flat = array("d", [float(value) for row in rows for value in row])
    body = flat.tobytes()
    header = "{'descr': '<f8', 'fortran_order': False, 'shape': (%d, %d), }" % (
        len(rows),
        dim,
    )
    # npy alignment: magic(6) + version(2) + header-len(2) + header + "\n" must
    # be a multiple of 64 bytes; pad with spaces, terminate with "\n".
    pad = 64 - (len(_NPY_MAGIC) + 4 + len(header) + 1) % 64
    header = header + " " * pad + "\n"
    return _NPY_MAGIC + b"\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin-1") + body


def _read_npy(path: Path) -> tuple[list[list[float]], int]:
    """Parse a float64 ``.npy`` file written by :func:`_write_npy`.

    Returns ``(rows, dim)``; raises ValueError on anything unexpected (wrong
    magic, dtype, shape mismatch) so the caller can start a fresh cache
    instead of crashing on corrupt bytes.
    """
    raw = path.read_bytes()
    if raw[: len(_NPY_MAGIC)] != _NPY_MAGIC:
        raise ValueError(f"not a .npy file: {path}")
    major = raw[6]
    if major == 1:
        header_len = struct.unpack("<H", raw[8:10])[0]
        offset = 10
    else:  # v2/v3 use a 4-byte header length
        header_len = struct.unpack("<I", raw[8:12])[0]
        offset = 12
    header = ast_literal_header(raw[offset : offset + header_len])
    if header.get("descr") != "<f8":
        raise ValueError(f"unsupported dtype {header.get('descr')!r} in {path}")
    shape = tuple(header.get("shape") or ())
    if len(shape) != 2:
        raise ValueError(f"expected a 2-D array in {path}, got shape {shape}")
    rows, dim = int(shape[0]), int(shape[1])
    values = array("d")
    values.frombytes(raw[offset + header_len :])
    if len(values) != rows * dim:
        raise ValueError(f"npy body/shape mismatch in {path}: {len(values)} != {rows * dim}")
    return [list(values[row * dim : (row + 1) * dim]) for row in range(rows)], dim


def ast_literal_header(raw_header: bytes) -> dict:
    """Parse an npy header dict safely (``ast.literal_eval`` — never ``eval``)."""
    parsed = ast.literal_eval(raw_header.decode("latin-1").strip())
    if not isinstance(parsed, dict):
        raise ValueError(f"npy header is not a dict: {type(parsed).__name__}")
    return parsed


def _atomic_write(path: Path, data: bytes) -> None:
    """Write bytes via a sibling temp file + ``os.replace`` (crash-atomic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


class _VectorFileCache:
    """sha256(embed_text) → vector map, persisted as a ``.npy`` + JSON pair.

    Layout under the caller-supplied cache dir:

    - ``minilm-vectors.npy`` — the vectors themselves, one float64 row per
      unique embed_text (written with the stdlib writer above);
    - ``minilm-map.json`` — ``{sha256_hex: row_index}``.

    A corrupt/unreadable pair (or a dim mismatch after a model change) is
    logged and discarded — the cache rebuilds itself, it never blocks work.
    The map is JSON, not pickle (SEC-U3: no pickle anywhere — a pickle file
    in a cache directory is executable code waiting to be loaded).
    """

    VECTORS_FILENAME = "minilm-vectors.npy"
    MAP_FILENAME = "minilm-map.json"

    def __init__(self, directory: str | Path, dim: int) -> None:
        self._dir = Path(directory)
        self._dim = int(dim)
        self._rows: list[list[float]] = []
        self._map: dict[str, int] = {}
        self._loaded = False

    @staticmethod
    def key(text: str) -> str:
        """The cache key: sha256 of the embed_text (hex)."""
        return hashlib.sha256(str(text).encode("utf-8")).hexdigest()

    def load(self) -> None:
        """Read the pair once (lazy, idempotent); corrupt caches start fresh."""
        if self._loaded:
            return
        self._loaded = True
        npy_path, map_path = self._dir / self.VECTORS_FILENAME, self._dir / self.MAP_FILENAME
        if not (npy_path.exists() and map_path.exists()):
            return
        try:
            rows, dim = _read_npy(npy_path)
            if dim != self._dim:
                raise ValueError(f"cache dim {dim} != model dim {self._dim}")
            raw = map_path.read_text(encoding="utf-8")
            mapping = json.loads(raw)
            if not isinstance(mapping, dict):
                raise ValueError("cache map is not a dict")
            self._rows = rows
            self._map = {str(key): int(index) for key, index in mapping.items()}
        except (ValueError, json.JSONDecodeError) as exc:
            # An expected condition (dim mismatch after a model change) — no traceback.
            logger.warning("vector cache %s discarded (%s); starting fresh", self._dir, exc)
            self._rows, self._map = [], {}
        except Exception:  # noqa: BLE001 — a corrupt cache must never break embedding
            logger.warning("vector cache %s unreadable; starting fresh", self._dir, exc_info=True)
            self._rows, self._map = [], {}

    def lookup(self, text: str) -> list[float] | None:
        """The cached vector for ``text`` (a copy), or None."""
        self.load()
        index = self._map.get(self.key(text))
        if index is None or not 0 <= index < len(self._rows):
            return None
        return list(self._rows[index])

    def store(self, items: list[tuple[str, list[float]]]) -> None:
        """Append new (text, vector) pairs and persist the pair atomically."""
        self.load()
        fresh_rows: list[list[float]] = []
        for text, vector in items:
            key = self.key(text)
            if key in self._map:
                continue
            self._map[key] = len(self._rows) + len(fresh_rows)
            fresh_rows.append([float(value) for value in vector])
        if not fresh_rows:
            return
        self._rows.extend(fresh_rows)
        _atomic_write(self._dir / self.VECTORS_FILENAME, _write_npy(self._rows, self._dim))
        _atomic_write(
            self._dir / self.MAP_FILENAME,
            json.dumps(self._map).encode("utf-8"),
        )


class HashEmbedder:
    """Deterministic stdlib embedder — sha256-seeded PRNG, unit norm, zero deps.

    The suite's embedder everywhere: same text -> same unit vector, different
    text -> (with overwhelming probability) a different one, all offline. Satisfies
    the :class:`Embedder` protocol; pass it wherever an ``embedder`` is expected.
    """

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
            raise ValueError(f"dim must be a positive int, got {dim!r}")
        self._dim = dim

    @property
    def dim(self) -> int:
        """The vector dimensionality every embed() row carries."""
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` into deterministic L2-normalized unit vectors."""
        return [hash_vector(text, self._dim) for text in texts]


class MiniLMEmbedder:
    """The real embedder — lazy sentence-transformers + a sha256 disk cache.

    Construct freely: nothing heavy happens until the first :meth:`embed`.
    Computed vectors persist under ``cache`` (``.npy`` + pickle pair), so a
    second process (or a delta re-ingest) re-embeds nothing. ``stats()``
    reports ``{"total", "reused", "new"}`` — texts requested, served from the
    cache, computed fresh.
    """

    def __init__(self, model_name: str = MINILM_MODEL, cache: str | Path | None = None) -> None:
        self._model_name = model_name
        self._cache = _VectorFileCache(cache, DEFAULT_DIM) if cache is not None else None
        self._model: Any = None
        self._total = 0
        self._reused = 0
        self._new = 0

    def _load_model(self) -> Any:
        """Lazy ``SentenceTransformer`` — the one heavy import, paid on request."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    f"MiniLM embedding was requested but sentence-transformers is not "
                    f"installed — install it with: {RAG_REQUIREMENTS_HINT}"
                ) from exc
            self._model = SentenceTransformer(self._model_name)
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` (batch 64, normalized), via the sha256 cache when present.

        Cache hits answer instantly; misses encode on the model in a worker
        thread (the event loop never blocks) and persist into the cache.
        """
        clean = [str(text) for text in texts]
        self._total += len(clean)
        results: list[list[float] | None] = [None] * len(clean)
        misses: dict[str, list[int]] = {}
        for index, text in enumerate(clean):
            cached = self._cache.lookup(text) if self._cache is not None else None
            if cached is not None:
                results[index] = cached
                self._reused += 1
            else:
                misses.setdefault(text, []).append(index)
        self._new = self._total - self._reused
        if misses:
            model = self._load_model()  # lazy import: RuntimeError hint when absent
            fresh: list[tuple[str, list[float]]] = []
            unique = list(misses)
            for start in range(0, len(unique), MINILM_BATCH):
                batch = unique[start : start + MINILM_BATCH]
                encoded = await asyncio.to_thread(
                    model.encode, batch, batch_size=MINILM_BATCH, normalize_embeddings=True
                )
                for text, row in zip(batch, encoded):
                    vector = [float(value) for value in row]
                    for index in misses[text]:
                        results[index] = vector
                    fresh.append((text, vector))
            if self._cache is not None and fresh:
                self._cache.store(fresh)
        return [list(vector) if vector is not None else [0.0] * DEFAULT_DIM for vector in results]

    def stats(self) -> dict[str, int]:
        """``{"total", "reused", "new"}`` — cache economics since construction."""
        return {"total": self._total, "reused": self._reused, "new": self._new}
