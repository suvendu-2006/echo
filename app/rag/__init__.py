"""Echo's document RAG (Phase 6.5) — the user's Allcited pipeline, two halves.

This package root exports the INGESTION half (Part A — docs/phase6-rag-contract.md,
DECISIONS.md D-023):

- :mod:`app.rag.parsers` — :class:`ParsedBlock` plus TXT/MD/HTML (stdlib),
  PDF (lazy PyMuPDF: font-size headings + optional OCR) and DOCX (lazy)
  parsers behind :func:`parse_any`;
- :mod:`app.rag.chunker` — :func:`chunk_blocks` / :class:`Chunk`: the exact
  Allcited algorithm (abbreviation + decimal masking, 400-token buffer,
  15% backward overlap, 600-token hard split, section-aware early flush);
- :mod:`app.rag.refmap` — :class:`RefMap` + :func:`references_in`: the four
  reference families and their line-start/colon definition positions;
- :mod:`app.rag.ingest` — :class:`RagLibrary` (SQLite/WAL persistence) and
  :func:`ingest_file` / :func:`ingest_folder`.

THE QUERY HALF LIVES IN ITS OWN MODULES — ``app.rag.embedder``,
``app.rag.indexes``, ``app.rag.retriever``, ``app.rag.rerank``,
``app.rag.pipeline`` and ``app.rag.tools`` — and is deliberately NOT imported
here: ingestion works standalone on the stdlib, the heavy deps stay lazy
behind RuntimeError hints, and the test suite never needs the RAG extras to
exercise this half.
"""

from app.rag.chunker import Chunk, chunk_blocks
from app.rag.ingest import RagLibrary, ingest_file, ingest_folder
from app.rag.parsers import (
    ParsedBlock,
    parse_any,
    parse_docx,
    parse_html,
    parse_md,
    parse_pdf,
    parse_txt,
)
from app.rag.refmap import (
    CLAUSE_RE,
    CONNECTIVE_RE,
    FORM_RE,
    RULE_RE,
    RefMap,
    references_in,
)

__all__ = [
    "CLAUSE_RE",
    "CONNECTIVE_RE",
    "FORM_RE",
    "Chunk",
    "ParsedBlock",
    "RULE_RE",
    "RagLibrary",
    "RefMap",
    "chunk_blocks",
    "ingest_file",
    "ingest_folder",
    "parse_any",
    "parse_docx",
    "parse_html",
    "parse_md",
    "parse_pdf",
    "parse_txt",
    "references_in",
]
