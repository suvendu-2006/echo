"""Echo's memory (Phase 6.1–6.2) — facts stored once, recalled forever, $0.

Exports:

- :class:`MemoryStore` — durable SQLite memory, FTS5 search by default,
  optional Chroma vector backend when an embedder is injected;
- :class:`ChromaBackend` — that optional backend (lazy ``chromadb`` import);
- :func:`as_relevant_adapter` / :class:`RelevantAdapter` — the H5-2 seam
  (``async relevant(query, k) -> list[str]``) MemoryInjectStage consumes;
- :class:`Wallet` — the ≤50 always-injected core facts (Phase 6.2);
- :func:`memory_search_tool_factory` — the SAFE ``memory_search`` tool.

The distiller (Phase 6.3) lives beside this package as ``app.memory.distiller``
and is deliberately NOT imported here — it consumes the store, never the
reverse, so memory works standalone before/without distillation.
"""

from app.memory.store import (
    CHROMA_DIRNAME,
    CORE_KIND,
    ChromaBackend,
    MemoryStore,
    RelevantAdapter,
    as_relevant_adapter,
)
from app.memory.wallet import WALLET_HEADER, Wallet, memory_search_tool_factory

__all__ = [
    "CHROMA_DIRNAME",
    "CORE_KIND",
    "ChromaBackend",
    "MemoryStore",
    "RelevantAdapter",
    "WALLET_HEADER",
    "Wallet",
    "as_relevant_adapter",
    "memory_search_tool_factory",
]
