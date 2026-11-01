"""HybridIndex (Phase 6.5) — BM25 + cosine vectors behind one tiny class.

One object holds a corpus (chunk ids, their embed_texts, their unit vectors)
and answers two ranked questions:

    index.search_bm25("rule 63.7.4", k=25)        -> [("c5", 1), ...]   # (id, 1-based rank)
    index.search_vector(query_vector, k=25)       -> [("c5", 1), ...]

**Two backends per question, both automatic:**

- BM25 prefers ``rank_bm25.BM25Okapi`` (lazy import); when it is not
  installed, :class:`_SimpleBM25` — a stdlib Okapi BM25 (k1=1.5, b=0.75, idf
  ``log((N-df+0.5)/(df+0.5)+1)``) with the same API — takes over silently.
  Same results class, zero deps, so the suite never needs rank-bm25.
- Vector search prefers FAISS ``IndexFlatIP`` (lazy import — faiss-cpu drags
  numpy in with it); when it is absent, a brute-force pure-Python dot-product
  scan runs instead. The stored vectors are L2-normalized by the embedder
  (and defensively re-normalized here), so dot product == cosine and both
  paths produce identical rankings.

Tokenization (contract): lowercase, split on whitespace, strip punctuation
from token edges (``"63.7.4."`` -> ``"63.7.4"``), drop tokens shorter than 2
characters. Ties break by corpus position everywhere — deterministic, testable.

``build`` fills the index; ``add_chunks`` appends a delta (the ingestion
half's re-ingest path) and invalidates the lazily-built engines.
"""

from __future__ import annotations

import logging
import math
import string
from collections import Counter
from typing import Any, Iterable

logger = logging.getLogger("echo.rag.indexes")

#: Okapi BM25 parameters (the contract's, and rank_bm25's defaults).
BM25_K1 = 1.5
BM25_B = 0.75

#: Default candidate depth for both searches (contract: 25 + 25).
SEARCH_K = 25

_PUNCT = string.punctuation

#: Minimum kept token length (contract: drop tokens < 2 chars).
_MIN_TOKEN_LEN = 2


def tokenize(text: str) -> list[str]:
    """Lowercase, whitespace-split, punctuation-strip edges, drop < 2 chars.

    Internal punctuation survives (``"63.7.4"`` stays one token) — identifiers
    are exactly what BM25 must find verbatim.
    """
    tokens: list[str] = []
    for raw in str(text or "").lower().split():
        token = raw.strip(_PUNCT)
        if len(token) >= _MIN_TOKEN_LEN:
            tokens.append(token)
    return tokens


def _load_rank_bm25() -> Any:
    """The lazy rank_bm25 import, isolated for tests (patch to force fallback)."""
    from rank_bm25 import BM25Okapi

    return BM25Okapi


def _load_faiss() -> Any:
    """The lazy faiss import, isolated for tests (patch to force fallback)."""
    import faiss

    return faiss


class _SimpleBM25:
    """Stdlib-only Okapi BM25 — the automatic fallback when rank_bm25 is absent.

    Exact contract math per (query term, document)::

        idf = log((N - df + 0.5) / (df + 0.5) + 1)          # always positive
        score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |D| / avgdl))

    with k1=1.5, b=0.75. Same ``get_scores(query_tokens) -> list[float]``
    surface as ``rank_bm25.BM25Okapi``, so :class:`HybridIndex` treats them
    interchangeably.
    """

    def __init__(self, corpus: list[list[str]], k1: float = BM25_K1, b: float = BM25_B) -> None:
        self._k1 = float(k1)
        self._b = float(b)
        self._total_docs = len(corpus)
        self._doc_lengths = [len(doc) for doc in corpus]
        self._avgdl = (sum(self._doc_lengths) / self._total_docs) if self._total_docs else 0.0
        self._doc_freqs: Counter[str] = Counter()
        self._term_freqs: list[Counter[str]] = []
        for doc in corpus:
            counts = Counter(doc)
            self._term_freqs.append(counts)
            for term in counts:
                self._doc_freqs[term] += 1

    def get_scores(self, query: list[str]) -> list[float]:
        """One Okapi BM25 score per document, in corpus order."""
        scores: list[float] = []
        for term_freqs, doc_len in zip(self._term_freqs, self._doc_lengths):
            score = 0.0
            for term in query:  # duplicates in the query count per occurrence
                doc_freq = self._doc_freqs.get(term, 0)
                tf = term_freqs.get(term, 0)
                if doc_freq == 0 or tf == 0:
                    continue
                idf = math.log((self._total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
                norm = self._k1 * (1.0 - self._b + self._b * doc_len / self._avgdl)
                score += idf * (tf * (self._k1 + 1.0)) / (tf + norm)
            scores.append(score)
        return scores


class HybridIndex:
    """The corpus + both search engines, engines built lazily, fallbacks automatic."""

    def __init__(self, dim: int = 384) -> None:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
            raise ValueError(f"dim must be a positive int, got {dim!r}")
        self._dim = dim
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._vectors: list[list[float]] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25: Any = None
        self._bm25_backend = ""  # "" | "rank_bm25" | "simple"
        self._faiss_index: Any = None
        self._faiss_state = "untried"  # untried | ready | unavailable

    # ------------------------------------------------------------- population

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def backends(self) -> dict[str, str]:
        """Which engine served each question: ``{"bm25": ..., "vector": ...}``.

        ``bm25`` is ``"rank_bm25"`` or ``"simple"``; ``vector`` is ``"faiss"``/
        ``"bruteforce"`` (or ``"untried"`` before the first vector search).
        stats material for the pipeline's result dict.
        """
        vector_backend = {
            "ready": "faiss",
            "unavailable": "bruteforce",
            "untried": "untried",
        }.get(self._faiss_state, "untried")
        return {"bm25": self._bm25_backend, "vector": vector_backend}


    @staticmethod
    def _chunk_text(chunk: Any) -> str:
        """The text BM25 indexes: ``embed_text`` when present, else ``text``."""
        return str(getattr(chunk, "embed_text", None) or getattr(chunk, "text", "") or "")

    def build(self, chunks: Iterable[Any], vectors: list[list[float]]) -> None:
        """Fill the index from scratch (one unit vector per chunk, same order)."""
        self._ids, self._texts, self._vectors, self._corpus_tokens = [], [], [], []
        self._invalidate_engines()
        self.add_chunks(chunks, vectors)

    def add_chunks(self, chunks: Iterable[Any], vectors: list[list[float]]) -> None:
        """Append a delta (chunks + their vectors); lazy engines rebuild next use."""
        chunk_list = list(chunks)
        if len(chunk_list) != len(vectors):
            raise RuntimeError(
                f"chunks/vectors length mismatch: {len(chunk_list)} chunks vs {len(vectors)} vectors"
            )
        for chunk, vector in zip(chunk_list, vectors):
            if len(vector) != self._dim:
                raise RuntimeError(
                    f"vector dim {len(vector)} != index dim {self._dim} (chunk {getattr(chunk, 'id', '?')})"
                )
            self._ids.append(str(getattr(chunk, "id", "")))
            self._texts.append(self._chunk_text(chunk))
            self._vectors.append(normalized_vector(vector))
            self._corpus_tokens.append(tokenize(self._texts[-1]))
        if chunk_list:
            self._invalidate_engines()

    def _invalidate_engines(self) -> None:
        self._bm25 = None
        self._bm25_backend = ""
        self._faiss_index = None

    # ------------------------------------------------------------------ BM25

    def search_bm25(self, query: str, k: int = SEARCH_K) -> list[tuple[str, int]]:
        """Top-k ``(chunk_id, rank)`` by Okapi BM25 over tokenize(embed_text).

        rank_bm25 when installed, :class:`_SimpleBM25` otherwise. Chunks with
        no positive score (no query token present) are omitted — RRF should
        only fuse actual matches.
        """
        if not self._ids:
            return []
        if self._bm25 is None:
            self._build_bm25()
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        scored = [
            (position, score) for position, score in enumerate(scores) if score > 0.0
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return [(self._ids[position], rank + 1) for rank, (position, _) in enumerate(scored[:k])]

    def _build_bm25(self) -> None:
        """Build whichever BM25 engine is available (rank_bm25 first, fallback second)."""
        try:
            bm25_cls = _load_rank_bm25()
        except ImportError:
            self._bm25 = _SimpleBM25([list(doc) for doc in self._corpus_tokens])
            self._bm25_backend = "simple"
            logger.debug("rank_bm25 absent; using stdlib _SimpleBM25 (%d docs)", len(self._ids))
        else:
            self._bm25 = bm25_cls([list(doc) for doc in self._corpus_tokens])
            self._bm25_backend = "rank_bm25"

    # ---------------------------------------------------------------- vector

    def search_vector(self, query_vector: list[float], k: int = SEARCH_K) -> list[tuple[str, int]]:
        """Top-k ``(chunk_id, rank)`` by cosine (dot — everything is unit norm).

        FAISS IndexFlatIP when faiss is installed; otherwise a brute-force
        pure-Python scan with identical ordering. Non-positive similarities
        are omitted (no cosine evidence), ties break by corpus position.
        """
        if not self._ids:
            return []
        if len(query_vector) != self._dim:
            raise RuntimeError(
                f"query vector dim {len(query_vector)} != index dim {self._dim}"
            )
        query = normalized_vector(query_vector)
        if self._faiss_state == "untried":
            try:
                _load_faiss()
                self._faiss_state = "ready"
            except ImportError:
                self._faiss_state = "unavailable"
                logger.debug("faiss absent; using brute-force cosine (%d docs)", len(self._ids))
        if self._faiss_state == "ready":
            return self._search_faiss(query, k)
        return self._search_bruteforce(query, k)

    def _search_bruteforce(self, query: list[float], k: int) -> list[tuple[str, int]]:
        """The numpy-free fallback: a dot product per stored vector."""
        scored = [
            (position, math.fsum(a * b for a, b in zip(query, vector)))
            for position, vector in enumerate(self._vectors)
        ]
        scored = [pair for pair in scored if pair[1] > 0.0]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return [(self._ids[position], rank + 1) for rank, (position, _) in enumerate(scored[:k])]

    def _search_faiss(self, query: list[float], k: int) -> list[tuple[str, int]]:
        """The FAISS path: IndexFlatIP over the stored matrix (same ordering)."""
        import numpy  # faiss-cpu depends on numpy — present whenever faiss is

        if self._faiss_index is None:
            self._faiss_index = self._load_faiss().IndexFlatIP(self._dim)
            self._faiss_index.add(numpy.asarray(self._vectors, dtype="float32"))
        limit = max(1, min(int(k), len(self._ids)))
        scores, positions = self._faiss_index.search(
            numpy.asarray([query], dtype="float32"), limit
        )
        results: list[tuple[str, int]] = []
        for rank, (position, score) in enumerate(zip(positions[0], scores[0])):
            if int(position) < 0 or float(score) <= 0.0:
                continue
            results.append((self._ids[int(position)], rank + 1))
        return results

    def _load_faiss(self) -> Any:
        return _load_faiss()


def normalized_vector(vector: list[float]) -> list[float]:
    """Unit-norm copy of ``vector`` (defensive: the embedder already normalizes).

    Kept local to this module so the index never imports the embedder half.
    """
    values = [float(value) for value in vector]
    norm = math.sqrt(math.fsum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]
