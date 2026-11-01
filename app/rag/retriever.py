"""The query pipeline (Phase 6.5 Part B) — hybrid retrieval to answer-pack.

One async entry point::

    result = await rag_query(library, question, embedder=HashEmbedder())
    result.refused  /  result.sources  /  result.stats

and the stages it composes, each exported for its own tests:

- :func:`reciprocal_rank_fusion` — the contract's EXACT math: every chunk
  accumulates ``score += 1/(k + rank)`` (k=60, 1-based ranks) from each list
  it appears in. Rank-3 in both lists = 1/63 + 1/63.
- :func:`apply_parent_expansion` — the top-2 hits drag in their section
  siblings at score x0.85 (≤2 additions), so a 400-token chunk never answers
  alone.
- :class:`CrossEncoderReranker` — ms-marco-MiniLM cross-encoder (lazy import)
  or an injected ``scorer(query, text) -> float``. The WHOLE rerank block is
  wrapped: any failure (missing lib, OOM, injected exception) logs a warning
  and returns the RRF order unchanged — degradation, never death.
- :class:`SemanticCache` — LRU 500, cosine ≥ 0.96 on query vectors via the
  embedder; no embedder -> always miss.
- :func:`two_hop` — references (rule/clause/§/form...) found in the QUESTION
  and the top candidates pull their defining chunks out of the RefMap
  (definitions[ref_id] -> chunk_ids), capped at 2, promotion not duplication.

**The relevance floor** (:data:`RELEVANCE_FLOOR` = 0.40): after rerank, if the
best score is below the floor the pipeline refuses — ``RagResult(refused=True,
chunks=[], sources=[])`` — "zero tokens" beats a confident hallucination. The
floor reads the cross-encoder's sigmoid confidence scale (0..1); without a
reranker the pipeline carries raw RRF scores (top out at 2/61 ≈ 0.033), so a
reranker is the production wiring and RRF-only runs refuse unless a scorer
confirms relevance — refusal is the safe direction.

Generation belongs to the harness: :func:`rag_query` returns the pack (chunks
+ [S#]-ready sources); ``llm`` is RESERVED for the future rewrite/HyDE stages
(Phase 6.6 provider seam) and is skipped entirely when None. The ref-id
regexes are implemented HERE (mirroring the ingestion half's families, not
importing them) so the two halves stay independently testable.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass

from app.rag.embedder import RAG_REQUIREMENTS_HINT, embed_sync
from app.rag.indexes import SEARCH_K, HybridIndex

logger = logging.getLogger("echo.rag.retriever")

#: Below this best-score the pipeline refuses (contract: relevance floor 0.40).
RELEVANCE_FLOOR = 0.40

#: Reciprocal-rank-fusion smoothing constant (the contract's k=60).
RRF_K = 60

#: Parent-expansion defaults (contract: top-2, x0.85, ≤2 additions).
PARENT_TOP_N = 2
PARENT_DECAY = 0.85
PARENT_MAX_ADDITIONS = 2

#: Two-hop definition pull cap (contract: max 2).
TWO_HOP_MAX = 2

#: Semantic cache defaults (contract: LRU 500, cosine ≥ 0.96).
CACHE_MAX_ENTRIES = 500
CACHE_THRESHOLD = 0.96


# ----------------------------------------------------------------- ScoredChunk


@dataclass
class ScoredChunk:
    """One chunk's flight record through the pipeline (ranks None when absent).

    ``rrf_score`` carries the stage's current belief: the RRF score until a
    rerank replaces it with the cross-encoder confidence, and the parent/
    two-hop stages mark their additions (``expanded_from``/``two_hop_reason``)
    so provenance survives into the answer-pack.
    """

    chunk_id: str
    rrf_score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None
    expanded_from: str | None = None
    two_hop_reason: str | None = None


def _as_ranked(items: list) -> list[tuple[str, int]]:
    """Normalize a ranked list into ``(chunk_id, 1-based rank)`` pairs.

    Accepts the index engines' ``[(chunk_id, rank), ...]`` tuples as-is, or a
    plain ordered list of ids (position + 1 becomes the rank).
    """
    ranked: list[tuple[str, int]] = []
    for position, item in enumerate(items):
        if (
            not isinstance(item, str)
            and isinstance(item, (tuple, list))
            and len(item) == 2
            and isinstance(item[1], (int, float))
        ):
            ranked.append((str(item[0]), int(item[1])))
        else:
            ranked.append((str(item), position + 1))
    return ranked


def reciprocal_rank_fusion(list_a: list, list_b: list, k: int = RRF_K) -> list[ScoredChunk]:
    """Fuse two ranked lists with EXACT contract math: score += 1/(k + rank).

    1-based ranks; a chunk ranked 3rd in both lists scores 1/63 + 1/63; 1st in
    one list only scores 1/61. Result: descending score, ties keep first-list
    order (stable sort over first-seen order). ``bm25_rank``/``vector_rank``
    record where each chunk came from (None when absent from that list).
    """
    fused: dict[str, ScoredChunk] = {}
    for items, rank_attr in (
        (_as_ranked(list_a), "bm25_rank"),
        (_as_ranked(list_b), "vector_rank"),
    ):
        for chunk_id, rank in items:
            entry = fused.get(chunk_id)
            if entry is None:
                entry = ScoredChunk(chunk_id=chunk_id, rrf_score=0.0)
                fused[chunk_id] = entry
            entry.rrf_score += 1.0 / (k + rank)
            if getattr(entry, rank_attr) is None:
                setattr(entry, rank_attr, rank)
    return sorted(fused.values(), key=lambda scored: -scored.rrf_score)


def apply_parent_expansion(
    scored: list[ScoredChunk],
    all_chunks: list,
    top_n: int = PARENT_TOP_N,
    max_additions: int = PARENT_MAX_ADDITIONS,
    decay: float = PARENT_DECAY,
) -> list[ScoredChunk]:
    """Append section siblings of the top hits (score x decay, provenance tagged).

    For each of the top ``top_n`` scored chunks, chunks sharing its
    ``parent_section_id`` are appended as ``ScoredChunk(rrf_score=hit x decay,
    expanded_from=hit.chunk_id)`` — never duplicating an id already present,
    never exceeding ``max_additions`` total. The input list is not mutated.
    """
    chunk_by_id = {
        str(getattr(chunk, "id", "")): chunk for chunk in all_chunks
    }
    present = {entry.chunk_id for entry in scored}
    result = list(scored)
    additions = 0
    for hit in scored[:top_n]:
        if additions >= max_additions:
            break
        hit_chunk = chunk_by_id.get(hit.chunk_id)
        parent = getattr(hit_chunk, "parent_section_id", None) if hit_chunk is not None else None
        if not parent:
            continue
        for sibling in all_chunks:
            if additions >= max_additions:
                break
            sibling_id = str(getattr(sibling, "id", ""))
            if sibling_id in present or sibling_id == hit.chunk_id:
                continue
            if getattr(sibling, "parent_section_id", None) != parent:
                continue
            result.append(
                ScoredChunk(
                    chunk_id=sibling_id,
                    rrf_score=hit.rrf_score * decay,
                    expanded_from=hit.chunk_id,
                )
            )
            present.add(sibling_id)
            additions += 1
    return result


# ------------------------------------------------------------------- reranker


def _sigmoid(value: float) -> float:
    """Overflow-safe logistic squash (cross-encoder logits -> [0, 1])."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def chunk_text(chunk: object | None) -> str:
    """The text a scorer sees: ``text`` when present, else ``embed_text``."""
    if chunk is None:
        return ""
    return str(getattr(chunk, "text", None) or getattr(chunk, "embed_text", "") or "")


class CrossEncoderReranker:
    """ms-marco cross-encoder rerank — or an injected scorer — that never dies.

    ``scorer`` (tests/fakes): ``def scorer(query, text) -> float`` in [0, 1]
    (already a probability; clamped, not sigmoided). Without one, the real
    ``sentence_transformers.CrossEncoder`` is imported lazily and its logits
    are sigmoid-squashed into [0, 1] (a confidence, comparable to
    :data:`RELEVANCE_FLOOR`).

    ANY failure inside :meth:`rerank` — missing lib, OOM, a scorer that
    raises — is logged as a warning and answered with the input (RRF) order
    UNCHANGED: reranking is an upgrade, never a dependency.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        scorer: object | None = None,
    ) -> None:
        self._model_name = model_name
        self._scorer = scorer
        self._model: object | None = None

    def _model_scorer(self) -> object:
        """The lazy CrossEncoder scoring callable (RuntimeError hint when absent)."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    f"cross-encoder reranking was requested but sentence-transformers "
                    f"is not installed — install it with: {RAG_REQUIREMENTS_HINT}"
                ) from exc
            self._model = CrossEncoder(self._model_name)

        def score(query: str, text: str) -> float:
            prediction = self._model.predict([(query, text)])
            return float(prediction[0])

        return score

    def rerank(
        self,
        query: str,
        scored: list[ScoredChunk],
        chunks_by_id: dict[str, object],
        top_k: int = 4,
    ) -> list[ScoredChunk]:
        """Re-score ``scored`` against ``query``; top_k by score, descending.

        On ANY failure the input list is returned unchanged (RRF order) after
        a logged warning — the caller cannot tell, and does not need to.
        """
        try:
            if not scored:
                return []
            scorer = self._scorer if self._scorer is not None else self._model_scorer()
            rescored: list[ScoredChunk] = []
            for entry in scored:
                raw = float(scorer(str(query), chunk_text(chunks_by_id.get(entry.chunk_id))))
                if self._scorer is not None:
                    confidence = max(0.0, min(1.0, raw))  # injected scorer: already [0..1]
                else:
                    confidence = _sigmoid(raw)  # model logits -> probability
                rescored.append(
                    ScoredChunk(
                        chunk_id=entry.chunk_id,
                        rrf_score=confidence,
                        bm25_rank=entry.bm25_rank,
                        vector_rank=entry.vector_rank,
                        expanded_from=entry.expanded_from,
                        two_hop_reason=entry.two_hop_reason,
                    )
                )
            rescored.sort(key=lambda entry: -entry.rrf_score)  # stable: ties keep RRF order
            return rescored[: max(0, int(top_k))]
        except Exception:  # noqa: BLE001 — degradation, not death (contract)
            logger.warning(
                "rerank failed; returning RRF order unchanged (query=%r)",
                query,
                exc_info=True,
            )
            return list(scored)


# ----------------------------------------------------------------- cache


class SemanticCache:
    """LRU answer-pack cache keyed by question MEANING (cosine ≥ threshold).

    ``get`` embeds the question (via the shared async→sync bridge) and sweeps
    the cached vectors; the best cosine ≥ ``threshold`` (0.96) wins and is
    returned with its similarity injected as a ``"similarity"`` key on a COPY
    (the stored pack is never mutated; the LRU entry is touched). ``put``
    stores the answer-pack dict with the question's unit vector. No embedder
    -> always miss, put is a no-op (the contract: the cache is skipped).
    """

    def __init__(
        self,
        max_entries: int = CACHE_MAX_ENTRIES,
        threshold: float = CACHE_THRESHOLD,
        embedder: object | None = None,
    ) -> None:
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries!r}")
        self._max_entries = int(max_entries)
        self._threshold = float(threshold)
        self._embedder = embedder
        self._entries: OrderedDict[str, dict] = OrderedDict()  # key -> {"vector", "pack"}

    def _embed(self, question: str) -> list[float]:
        vectors = embed_sync(self._embedder, [str(question)])
        return [float(value) for value in vectors[0]]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        """Dot product of two vectors (both stored unit-normalized)."""
        return math.fsum(a * b for a, b in zip(left, right))

    def get(self, question: str) -> dict | None:
        """The best-matching cached pack (copy, ``"similarity"`` added), or None."""
        if self._embedder is None or not self._entries:
            return None
        query_vector = _unit(self._embed(question))
        best_key: str | None = None
        best_similarity = -1.0
        for key, entry in self._entries.items():
            similarity = self._cosine(query_vector, entry["vector"])
            if similarity > best_similarity:
                best_key, best_similarity = key, similarity
        if best_key is None or best_similarity < self._threshold:
            return None
        self._entries.move_to_end(best_key)  # LRU touch
        pack = dict(self._entries[best_key]["pack"])
        pack["similarity"] = round(best_similarity, 4)
        return pack

    def put(self, question: str, answer_pack: dict) -> None:
        """Cache ``answer_pack`` under ``question``'s meaning; LRU-evict the oldest."""
        if self._embedder is None:
            return
        key = str(question)
        self._entries[key] = {
            "vector": _unit(self._embed(question)),
            "pack": dict(answer_pack),
        }
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)


def _unit(vector: list[float]) -> list[float]:
    """Defensive unit-normalization (embedders already ship unit vectors)."""
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if norm == 0.0:
        return list(vector)
    return [value / norm for value in vector]


# ------------------------------------------------------------------ two-hop

_REF_KEYWORDS = {
    "rule": "rule",
    "regulation": "regulation",
    "reg.": "regulation",
    "reg": "regulation",
    "clause": "clause",
    "section": "section",
    "sec.": "section",
    "sec": "section",
    "form": "form",
    "schedule": "schedule",
    "procedure": "procedure",
    "standard": "standard",
}

#: The four ref-id families (contract): rule/regulation, clause/section/§/sec.,
#: form/schedule/procedure/standard CODE, connective phrases. Mirrored locally
#: from the ingestion half's families — do NOT import app.rag.refmap here.
#:
#: Each entry: (compiled pattern, keyword group index or None-for-implied,
#: number group index, implied keyword when the pattern has none — § means
#: "section"; bare connective numbers become "ref").
_REF_PATTERNS: tuple[tuple[re.Pattern[str], int | None, int, str | None], ...] = (
    (re.compile(r"\b(rule|regulation|reg\.?)\s+(\d+(?:\.\d+)*)", re.IGNORECASE), 1, 2, None),
    (re.compile(r"\b(clause|section|sec\.?)\s+(\d+(?:\.\d+)*)", re.IGNORECASE), 1, 2, None),
    (re.compile(r"§\s*(\d+(?:\.\d+)*)", re.IGNORECASE), None, 1, "section"),
    (
        re.compile(
            r"\b(form|schedule|procedure|standard)\s+([A-Za-z\-]{0,5}\d[\w\-]*(?:/[A-Za-z0-9\-]+)*)",
            re.IGNORECASE,
        ),
        1,
        2,
        None,
    ),
    (
        re.compile(
            r"\b(?:as\s+per|as\s+defined\s+in|as\s+described\s+in|as\s+set\s+out\s+in"
            r"|pursuant\s+to|in\s+accordance\s+with)\s+(?:the\s+)?"
            r"(?:(rule|regulation|reg\.?|clause|section|sec\.?|form|schedule|procedure|standard)\s+)?"
            r"(\d+(?:\.\d+)*[A-Za-z0-9\-]*|[A-Za-z\-]{0,5}\d[\w\-]*(?:/[A-Za-z0-9\-]+)*)",
            re.IGNORECASE,
        ),
        1,
        2,
        None,
    ),
)


def extract_ref_ids(text: str) -> list[str]:
    """Reference ids ("rule:63.7.4", "section:12.3", "form:sr-11") in ``text``.

    Family order, deduplicated. ``§`` maps to ``section``; bare numbers after
    connective phrases become ``ref:<n>`` (no keyword claimed them).
    """
    found: list[str] = []
    seen: set[str] = set()
    value = str(text or "")
    for pattern, keyword_group, number_group, implied in _REF_PATTERNS:
        for match in pattern.finditer(value):
            keyword = match.group(keyword_group) if keyword_group is not None else None
            number = match.group(number_group)
            if not number:
                continue
            canonical = (
                _REF_KEYWORDS.get(str(keyword).lower().rstrip("."), "ref")
                if keyword
                else (implied or "ref")
            )
            ref_id = f"{canonical}:{number.lower()}"
            if ref_id not in seen:
                seen.add(ref_id)
                found.append(ref_id)
    return found


def two_hop(
    scored: list[ScoredChunk],
    refmap: object | None,
    chunks_by_id: dict[str, object],
    max_expansions: int = TWO_HOP_MAX,
    query: str = "",
) -> list[ScoredChunk]:
    """Pull defining chunks for references found in the query and top candidates.

    Ref ids are extracted from the question FIRST, then from each scored
    candidate's text; ``refmap.definitions[ref_id] -> [chunk_ids]`` resolves
    them to defining chunks. Chunks already in ``scored`` are PROMOTED (their
    ``two_hop_reason`` set, never duplicated); new ones are appended — at most
    ``max_expansions`` — as ``ScoredChunk(rrf_score=referrer's score,
    two_hop_reason="Defined in rule:63.7.4 (referenced by ...)")``. The input
    list entries may gain a reason in place; the returned list is a new list.
    """
    if refmap is None or not scored:
        return list(scored)
    definitions = getattr(refmap, "definitions", None) or {}
    if not definitions:
        return list(scored)

    # Requests in priority order: the query's refs first, then candidates'.
    # Each request: (referrer, referrer score, ref id).
    requests: list[tuple[str, float, str]] = []
    for ref_id in extract_ref_ids(query):
        requests.append(("the query", 0.0, ref_id))
    for entry in scored:
        entry_text = chunk_text(chunks_by_id.get(entry.chunk_id))
        for ref_id in extract_ref_ids(entry_text):
            requests.append((entry.chunk_id, entry.rrf_score, ref_id))

    present = {entry.chunk_id for entry in scored}
    result = list(scored)
    added = 0
    for referrer, referrer_score, ref_id in requests:
        for definition_id in definitions.get(ref_id, []):
            definition_id = str(definition_id)
            reason = f"Defined in {ref_id} (referenced by {referrer})"
            if definition_id in present:
                for existing in result:
                    if existing.chunk_id == definition_id and existing.two_hop_reason is None:
                        existing.two_hop_reason = reason
                continue  # promotion, not duplication
            if added >= max_expansions:
                continue
            result.append(
                ScoredChunk(chunk_id=definition_id, rrf_score=referrer_score, two_hop_reason=reason)
            )
            present.add(definition_id)
            added += 1
    return result


# ------------------------------------------------------------------- pipeline


@dataclass
class RagResult:
    """The answer-pack the harness composes its grounded answer from.

    ``sources`` rows are ``{"id", "text", "section", "page"}`` — the [S#]
    labels the tool text carries are just their 1-based positions. ``stats``
    is the per-stage flight recorder (cache verdict, hit counts, best score,
    refusal reason when refused).
    """

    refused: bool
    chunks: list[ScoredChunk]
    sources: list[dict]  # [{"id", "text", "section", "page"}]
    stats: dict


async def _maybe_await(value: object) -> object:
    """``await`` library calls that turn out to be async (defensive, both halves)."""
    if inspect.isawaitable(value):
        return await value
    return value


async def rag_query(
    library: object,
    question: str,
    *,
    embedder: object,
    reranker: object | None = None,
    cache: SemanticCache | None = None,
    llm: object | None = None,
    refmap: object | None = None,
    k_final: int = 4,
) -> RagResult:
    """The full query pipeline: cache -> hybrid retrieval -> RRF -> expansion
    -> rerank -> floor -> two-hop -> answer-pack. Never generates; ``llm`` is
    reserved for the future rewrite/HyDE stages (skipped while None — the
    contract's own skip-rule also skips them when the question carries an
    explicit identifier). Every stage lands in ``result.stats``.
    """
    clean_question = str(question or "").strip()
    explicit_ids = extract_ref_ids(clean_question)
    base_stats: dict = {
        "question_chars": len(clean_question),
        "explicit_ids": explicit_ids,
        "rewrite": "skipped (explicit id present)" if explicit_ids else "skipped (no llm)",
        "k_final": int(k_final),
        "floor": RELEVANCE_FLOOR,
    }
    if not clean_question:
        return RagResult(refused=True, chunks=[], sources=[], stats={**base_stats, "refused": True, "reason": "empty question"})

    # 1. Semantic cache (needs an embedder; skipped without — always miss).
    if cache is not None:
        pack = cache.get(clean_question)
        if isinstance(pack, dict):
            similarity = float(pack.pop("similarity", 1.0))
            chunks = [ScoredChunk(**row) for row in pack.get("chunks", [])]
            return RagResult(
                refused=bool(pack.get("refused", False)),
                chunks=chunks,
                sources=list(pack.get("sources", [])),
                stats={"cache": "hit", "similarity": similarity},
            )

    # 2. Corpus + hybrid retrieval (the embedder is the vector source of truth:
    #    deterministic for HashEmbedder, sha256-cached for MiniLM).
    chunks = list(await _maybe_await(library.get_chunks()))
    if not chunks:
        return RagResult(
            refused=True,
            chunks=[],
            sources=[],
            stats={**base_stats, "refused": True, "reason": "library is empty"},
        )
    chunks_by_id: dict[str, object] = {str(getattr(chunk, "id", "")): chunk for chunk in chunks}
    embed_texts = [
        str(getattr(chunk, "embed_text", None) or getattr(chunk, "text", "") or "")
        for chunk in chunks
    ]
    vectors = await embedder.embed(embed_texts)
    index = HybridIndex(dim=len(vectors[0]))
    index.build(chunks, vectors)
    bm25_hits = index.search_bm25(clean_question, k=SEARCH_K)
    query_vector = (await embedder.embed([clean_question]))[0]
    vector_hits = index.search_vector(query_vector, k=SEARCH_K)
    stats = {
        **base_stats,
        "cache": "miss",
        "corpus_chunks": len(chunks),
        "backends": index.backends,
        "bm25_hits": len(bm25_hits),
        "vector_hits": len(vector_hits),
    }

    # 3. RRF fusion -> parent expansion.
    fused = reciprocal_rank_fusion(bm25_hits, vector_hits, k=RRF_K)
    stats["rrf_candidates"] = len(fused)
    expanded = apply_parent_expansion(fused, chunks)
    stats["parent_expansions"] = len(expanded) - len(fused)

    # 4. Rerank (any failure inside a CrossEncoderReranker degrades to RRF
    #    order; a broken injected reranker is caught here too — same contract).
    candidates = expanded
    if reranker is not None:
        try:
            candidates = reranker.rerank(clean_question, expanded, chunks_by_id, top_k=k_final)
            stats["reranked"] = True
        except Exception:  # noqa: BLE001 — a broken reranker must not break retrieval
            logger.warning("reranker raised outside rerank(); using RRF order", exc_info=True)
            stats["reranked"] = False
        candidates = candidates[:k_final]
    else:
        candidates = expanded[:k_final]
        stats["reranked"] = False
    stats["rerank_candidates"] = len(candidates)

    # 5. Relevance floor — only meaningful on the cross-encoder's [0,1]
    #    confidence scale. Without a reranker the scores are RRF ranks
    #    (~0.03 at best), which can never clear 0.40: applying the floor
    #    there refused EVERY offline query (caught live — the HashEmbedder
    #    path). No reranker -> no floor; citations let the user verify.
    best_score = max((entry.rrf_score for entry in candidates), default=0.0)
    stats["best_score"] = round(best_score, 6)
    if stats.get("reranked") and best_score < RELEVANCE_FLOOR:
        return RagResult(
            refused=True,
            chunks=[],
            sources=[],
            stats={**stats, "refused": True, "reason": "below relevance floor"},
        )

    # 6. Two-hop definitions (query refs first, then candidates'; cap 2).
    pre_two_hop = len(candidates)
    pre_reasons = [entry.two_hop_reason for entry in candidates]
    candidates = two_hop(candidates, refmap, chunks_by_id, max_expansions=TWO_HOP_MAX, query=clean_question)
    stats["two_hop_added"] = len(candidates) - pre_two_hop
    stats["two_hop_promoted"] = sum(
        1 for entry, reason in zip(candidates, pre_reasons) if reason is None and entry.two_hop_reason is not None
    )

    # Definition chunks from outside the (possibly filtered) view still need text.
    missing_ids = [entry.chunk_id for entry in candidates if entry.chunk_id not in chunks_by_id]
    if missing_ids and callable(getattr(library, "get_chunks_by_ids", None)):
        for extra in await _maybe_await(library.get_chunks_by_ids(missing_ids)) or []:
            chunks_by_id[str(getattr(extra, "id", ""))] = extra

    sources = [
        {
            "id": entry.chunk_id,
            "text": chunk_text(chunks_by_id.get(entry.chunk_id)),
            "section": str(getattr(chunks_by_id.get(entry.chunk_id), "parent_section_id", "") or ""),
            "page": getattr(chunks_by_id.get(entry.chunk_id), "page", None),
        }
        for entry in candidates
    ]
    result = RagResult(refused=False, chunks=candidates, sources=sources, stats=stats)

    # 7. Cache the successful pack (refusals recompute — new documents may land).
    if cache is not None:
        cache.put(
            clean_question,
            {
                "refused": False,
                "chunks": [asdict(entry) for entry in candidates],
                "sources": [dict(source) for source in sources],
                "stats": dict(stats),
            },
        )
    return result
