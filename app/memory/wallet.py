"""The Wallet (Phase 6.2) — the ≤50 facts Echo permanently knows.

Two-tier memory:

- **Wallet** — the facts always in the system prompt. Membership rule: a
  memory is in the wallet when its ``importance >= wallet_threshold``
  (default 5.0) OR it is explicitly pinned (``kind == 'core'``). Pinned
  facts are never auto-evicted. The ``max_facts`` cap (default 50) is
  enforced on write: a promotion that would exceed it demotes the
  least-important non-pinned member back to archival (importance below the
  threshold; ties go to the oldest). ``facts()`` caps on read too, so a
  wallet stuffed past the cap by pinned members still injects at most
  ``max_facts`` lines.
- **Archival** — everything else; searched on demand via the
  ``memory_search`` tool (:func:`memory_search_tool_factory`, danger SAFE —
  reading your own memory needs no permission) or the H5-2 relevance seam
  (:func:`app.memory.store.as_relevant_adapter`).

Rendering: :meth:`Wallet.render_block` produces the system-prompt block —
``"Facts you permanently know about the user:\\n- ..."`` — empty string when
the wallet is empty. Superseded (contradicted) memories never appear in any
wallet view.

Note on external promotions: the distiller raising a memory's importance via
``store.update`` bypasses the Wallet, so bulk upserts should call
:meth:`Wallet.enforce_cap` afterwards — the write-side cap fires on Wallet
actions (``set_core``) by design.
"""

from __future__ import annotations

import logging
import typing

from app.harness.interfaces import Danger
from app.harness.tools.base import tool
from app.memory.store import CORE_KIND, MemoryStore

logger = logging.getLogger("echo.memory")

#: Header of the always-injected system-prompt block (Phase 6.2 contract).
WALLET_HEADER = "Facts you permanently know about the user:"


class Wallet:
    """The always-injected tier over a :class:`MemoryStore` (≤50 core facts)."""

    def __init__(
        self,
        store: MemoryStore,
        max_facts: int = 50,
        wallet_threshold: float = 5.0,
    ) -> None:
        self._store = store
        self._max_facts = max(1, int(max_facts))
        self._threshold = float(wallet_threshold)

    # ------------------------------------------------------------- membership

    def _members(self) -> list[dict]:
        """Every live wallet member, importance first (oldest breaks ties)."""
        rows = self._store.list_all(limit=1_000_000, include_superseded=False)
        members = [
            row
            for row in rows
            if float(row["importance"]) >= self._threshold or row["kind"] == CORE_KIND
        ]
        members.sort(key=lambda row: (-float(row["importance"]), row["created_at"]))
        return members

    def facts(self) -> list[dict]:
        """The wallet as it would be injected — importance order, capped."""
        return self._members()[: self._max_facts]

    # ---------------------------------------------------------------- pinning

    def set_core(self, memory_id: str) -> dict:
        """Pin a memory into the wallet (``kind='core'``); enforces the cap.

        Pinning is a promotion: if it pushes the wallet over ``max_facts``,
        the least-important non-pinned member is demoted to archival (pinned
        members are never evicted). Raises ValueError for an unknown id.
        """
        updated = self._store.update(memory_id, kind=CORE_KIND)
        if updated is None:
            raise ValueError(f"unknown memory id: {memory_id!r}")
        self.enforce_cap()
        return self._store.get(memory_id) or updated

    def unset_core(self, memory_id: str) -> dict | None:
        """Unpin (``kind`` back to the default ``'fact'``).

        The memory stays in the wallet only while its own importance clears
        the threshold. Raises ValueError for an unknown id.
        """
        updated = self._store.update(memory_id, kind="fact")
        if updated is None:
            raise ValueError(f"unknown memory id: {memory_id!r}")
        return updated

    def demote_to_archival(self, memory_id: str) -> dict | None:
        """Drop a memory out of the wallet: importance below the threshold.

        A pinned memory's ``kind`` resets to ``'fact'`` (the pre-pin kind is
        not recorded); other kinds are preserved. Returns the updated record,
        or None for an unknown id.
        """
        current = self._store.get(memory_id)
        if current is None:
            return None
        below_threshold = min(float(current["importance"]), self._threshold - 1.0)
        kind = "fact" if current["kind"] == CORE_KIND else current["kind"]
        return self._store.update(memory_id, kind=kind, importance=below_threshold)

    def enforce_cap(self) -> list[str]:
        """Demote members until the wallet fits ``max_facts``; returns evicted ids.

        The least-important non-pinned member goes first (ties: the oldest).
        A wallet where every member is pinned cannot shrink — the warning is
        logged and ``facts()`` caps on read instead.
        """
        evicted: list[str] = []
        while True:
            members = self._members()
            if len(members) <= self._max_facts:
                return evicted
            candidates = [member for member in members if member["kind"] != CORE_KIND]
            if not candidates:
                logger.warning(
                    "wallet cap (%d) exceeded by pinned facts; read-time cap applies",
                    self._max_facts,
                )
                return evicted
            victim = min(candidates, key=lambda member: (member["importance"], member["created_at"]))
            self.demote_to_archival(victim["id"])
            evicted.append(victim["id"])

    # -------------------------------------------------------------- rendering

    def render_block(self) -> str:
        """The system-prompt wallet block; exactly ``""`` when empty."""
        members = self.facts()
        if not members:
            return ""
        lines = "\n".join(f"- {member['kind']}: {member['text']}" for member in members)
        return f"{WALLET_HEADER}\n{lines}"


async def memory_search_tool_factory(store: MemoryStore) -> typing.Callable:
    """Build the ``memory_search`` tool bound to ``store`` (async per contract).

    Await once, then ``registry.register(tool)``. The tool is
    ``@tool(danger=Danger.SAFE)`` — a SAFE read of the user's own memory, so
    the permission engine's default mode allows it without a popup. Returns
    the top 5 hits as ``"- kind: text"`` lines, most relevant first.
    """

    @tool(danger=Danger.SAFE, cost_hint=0.5)
    async def memory_search(query: str) -> str:
        """Search Echo's long-term memory for facts about the user.

        Returns the stored memories matching the query, most relevant first.
        Reading your own memory needs no permission.
        """
        hits = store.search(str(query), k=5)
        if not hits:
            return "No memories matched that query."
        return "\n".join(f"- {hit['kind']}: {hit['text']}" for hit in hits)

    return memory_search
