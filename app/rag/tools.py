"""The desk's RAG tools (Phase 6.5) — search, ask, ingest, list.

``build_rag_tools`` is the async factory (the
:func:`app.memory.wallet.memory_search_tool_factory` shape: await it once,
register what it returns)::

    tools = await build_rag_tools(library, embedder, reranker=..., cache=...)
    for fn in tools:
        registry.register(fn)

Four tools (contract):

- ``rag_search(query, doc_filter="")`` — **SAFE** — the retrieval pipeline ->
  an answer-pack text: numbered ``[S1]..[Sn]`` sources with section/page and
  chunk excerpts, ready for the model to cite. Below the relevance floor it
  returns the exact refusal sentence — the model never sees weak evidence.
- ``rag_ask(question)`` — **SAFE** — the same pack plus a trailing
  grounded-answer instruction block (answer ONLY from the sources, cite [S#]).
- ``rag_ingest(path)`` — **CONFIRM** — ingests a file/folder via the ingestion
  half (imported lazily INSIDE the tool: the popup shows the path before any
  of that machinery runs). Returns the ingest stats dict.
- ``rag_list_documents()`` — **SAFE** — the library's documents + chunk counts.

``doc_filter`` narrows the query to one document/section: a case-insensitive
substring match on the chunk's doc_id / parent_section_id / id, applied by a
read-only view wrapping the library (the cache is bypassed for filtered
queries — a filtered answer must never be served to an unfiltered question).
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any

from app.harness.interfaces import Danger
from app.harness.tools.base import tool
from app.rag.retriever import RagResult, rag_query

logger = logging.getLogger("echo.rag.tools")

#: The contract's exact refusal sentence — the model's only escape hatch.
REFUSAL_TEXT = "Not covered in the ingested documents."

#: The trailing block rag_ask appends (the grounded-answer contract).
GROUNDED_INSTRUCTIONS = (
    "Answer ONLY from the sources above — do not use outside knowledge or "
    "memory. Cite every claim with its [S#] label (e.g. [S1]); if the sources "
    "do not cover the question, say so plainly instead of guessing."
)

#: Per-source excerpt cap (the model needs provenance, not whole pages).
_EXCERPT_CHARS = 400


async def _resolve(value: Any) -> Any:
    """Await library calls that turn out to be async (both halves, defensively)."""
    if inspect.isawaitable(value):
        return await value
    return value


def _format_answer_pack(result: RagResult) -> str:
    """One RagResult -> the [S#]-labeled answer-pack text (or the refusal)."""
    if result.refused or not result.sources:
        return REFUSAL_TEXT
    lines: list[str] = []
    for index, source in enumerate(result.sources, start=1):
        page = source.get("page")
        section = str(source.get("section") or "-")
        lines.append(f"[S{index}] section={section} page={page if page is not None else '-'}")
        text = str(source.get("text") or "").strip()
        if len(text) > _EXCERPT_CHARS:
            text = text[:_EXCERPT_CHARS].rstrip() + "…"
        lines.append(f"    {text}")
    return "\n".join(lines)


class _FilteredLibrary:
    """Read-only library view narrowed by ``doc_filter`` (substring, CI).

    ``get_chunks`` keeps only chunks whose doc_id / parent_section_id / id
    contains the filter; ``get_chunks_by_ids`` and ``load_refmap`` pass
    through UNfiltered on purpose — a two-hop definition chunk from another
    document is still valid evidence. Everything else delegates.
    """

    def __init__(self, library: object, doc_filter: str) -> None:
        self._library = library
        self._needle = str(doc_filter or "").strip().lower()

    def _match(self, chunk: object) -> bool:
        haystack = " ".join(
            part
            for part in (
                str(getattr(chunk, "doc_id", "") or ""),
                str(getattr(chunk, "parent_section_id", "") or ""),
                str(getattr(chunk, "id", "") or ""),
            )
            if part
        ).lower()
        return self._needle in haystack

    async def get_chunks(self, doc_id: str | None = None) -> list:
        chunks = await _resolve(self._library.get_chunks(doc_id))
        return [chunk for chunk in chunks or [] if self._match(chunk)]

    async def get_chunks_by_ids(self, ids: list[str]) -> list:
        return await _resolve(self._library.get_chunks_by_ids(list(ids)))

    async def load_refmap(self) -> Any:
        return await _resolve(self._library.load_refmap())

    async def list_documents(self) -> list:
        docs = await _resolve(self._library.list_documents())
        return [
            doc
            for doc in docs or []
            if self._needle in str(doc).lower()
        ]

    async def chunk_count(self) -> int:
        return len(await self.get_chunks())


async def _load_refmap(view: object) -> Any:
    """The library's RefMap, or None — two-hop is additive, never load-bearing."""
    loader = getattr(view, "load_refmap", None)
    if not callable(loader):
        return None
    try:
        loaded = loader()
        if inspect.isawaitable(loaded):
            loaded = await loaded
        return loaded
    except Exception:  # noqa: BLE001 — a broken refmap must not break search
        logger.warning("load_refmap failed; two-hop disabled for this query", exc_info=True)
        return None


async def build_rag_tools(
    library: object,
    embedder: object,
    reranker: object | None = None,
    cache: object | None = None,
) -> list:
    """Build the four RAG tools bound to ``library`` (async per contract).

    Awaits nothing heavy — wiring only — and returns
    ``[rag_search, rag_ask, rag_ingest, rag_list_documents]``, each carrying
    the ``@tool`` metadata the registry demands. ``rag_ingest`` imports the
    ingestion half lazily inside the call, so this module imports none of it.
    """

    @tool(danger=Danger.SAFE, cost_hint=0.8)
    async def rag_search(query: str, doc_filter: str = "") -> str:
        """Search the ingested documents; returns [S#]-labeled sources with excerpts.

        Runs hybrid retrieval (BM25 + vectors -> RRF -> rerank) over the
        library and formats the surviving chunks as numbered [S1]..[Sn]
        sources with section/page and excerpts. Weak matches are refused.
        """
        filter_text = str(doc_filter or "").strip()
        view = _FilteredLibrary(library, filter_text) if filter_text else library
        refmap = await _load_refmap(view)
        result = await rag_query(
            view,
            str(query),
            embedder=embedder,
            reranker=reranker,
            cache=cache if not filter_text else None,  # filtered answers never cached
            refmap=refmap,
        )
        return _format_answer_pack(result)

    @tool(danger=Danger.SAFE, cost_hint=0.8)
    async def rag_ask(question: str) -> str:
        """Ask the ingested documents; sources plus grounded-answer instructions.

        Same retrieval as rag_search, with a trailing instruction block:
        answer ONLY from the sources and cite them as [S#].
        """
        refmap = await _load_refmap(library)
        result = await rag_query(
            library,
            str(question),
            embedder=embedder,
            reranker=reranker,
            cache=cache,
            refmap=refmap,
        )
        pack = _format_answer_pack(result)
        if result.refused or not result.sources:
            return pack  # the refusal sentence needs no grounding instructions
        return f"{pack}\n\n{GROUNDED_INSTRUCTIONS}"

    @tool(danger=Danger.CONFIRM, cost_hint=2.0)
    async def rag_ingest(path: str) -> dict:
        """Ingest a file or folder of documents into the RAG library (asks first).

        Parses/chunks/embeds via the ingestion half and persists into the
        library; returns the ingest stats dict. CONFIRM: it mutates the
        library and reads the given path, so the permission engine asks first.
        """
        try:
            from app.rag import ingest as ingest_module  # lazy: the ingestion half
        except ImportError as exc:
            raise RuntimeError(f"rag ingestion is unavailable: {exc}") from exc
        target = Path(str(path)).expanduser()
        if target.is_dir():
            stats = ingest_module.ingest_folder(str(target), library)
        else:
            stats = ingest_module.ingest_file(str(target), library)
        if inspect.isawaitable(stats):
            stats = await stats
        return stats

    @tool(danger=Danger.SAFE, cost_hint=0.3)
    async def rag_list_documents() -> str:
        """List the documents ingested into the RAG library with chunk counts."""
        docs = await _resolve(library.list_documents())
        if not docs:
            return "(no documents ingested)"
        lines: list[str] = []
        for doc in docs:
            if isinstance(doc, dict):
                name = doc.get("doc_id") or doc.get("id") or doc.get("name") or "?"
                count = doc.get("chunk_count", doc.get("chunks", "?"))
                lines.append(f"{name} ({count} chunks)")
            else:
                lines.append(str(doc))
        return "\n".join(lines)

    return [rag_search, rag_ask, rag_ingest, rag_list_documents]
