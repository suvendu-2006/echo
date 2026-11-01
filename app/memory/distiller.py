"""The distiller (Phase 6.3) — what Echo should remember, filed after a session.

PLAN step 6.3: "Post-session small-model job extracts facts/decisions/
preferences → upserts with timestamps, replaces contradictions. Done when:
sister's birthday told in session 1 → known in session 2."

The pieces and their contracts
------------------------------
**The memory store (duck-typed, SYNC — SQLite under the real store).** The
real store lives in sibling modules built in parallel, so this module depends
only on this exact shape; tests inject a fake with the same four methods::

    store.add(text, kind, importance, source_session=...) -> dict
        Records one memory. The STORE stamps its own timestamps
        (created_at / updated_at) — the distiller passes source_session
        and nothing else. ``source_session`` is passed as a keyword.
    store.search(query, k) -> list[dict]
        Memories similar to ``query``; each dict carries at least
        ``"id"`` and ``"text"``.
    store.supersede(id) -> None
        Marks an old memory replaced (kept for audit, no longer active).
    store.list_all() -> list[dict]
        Every active memory — the duplicate-suppression snapshot.

**The extractor (injectable, async).** :class:`FactExtractor` is one method:
``async extract(transcript) -> list[dict]`` with each dict shaped
``{"text": str, "kind": "fact"|"preference"|"decision"|"person",
"importance": float 1..10}``. :class:`DefaultHeuristicExtractor` is the
zero-model offline fallback — regex/keyword heuristics over the transcript.
It is EXPLICITLY CRUDE: it misses phrasings, over-captures, and reads user
and assistant text alike. A real small-model extractor plugs in via the same
protocol later; the door (chatservice) wires it through settings when a
provider is configured.

**The flow.** :meth:`Distiller.distill_session` extracts facts, then per
fact: duplicate-suppress (exact text, case-insensitive, against the store's
active memories or the caller's ``existing`` snapshot) → contradiction-check
(:meth:`Distiller` searches the store for similar memories; the injectable
``contradiction_checker(new_text, old_text) -> bool`` decides "same subject,
conflicting value"; the default is a conservative heuristic that only fires
on name/birthday/preference markers for the same subject) → supersede the
conflicting olds and add the new fact, or plain add. Counters come back as
``{"added": n, "superseded": m, "facts": [...], "duplicates": d}``.

**Fire-and-forget contract (for the coordinator).** The door calls
:meth:`Distiller.distill_from_events` after each chat turn's final event —
``asyncio.create_task(distiller.distill_from_events(events, session_id))`` —
the "post-session" moment. It builds the transcript from the user/assistant
events (capped at :data:`MAX_TRANSCRIPT_CHARS`), then runs
:meth:`Distiller.distill_session`. NEITHER method ever raises: every failure
returns ``{"added": 0, "superseded": 0, "facts": [], "duplicates": 0,
"error": "..."}`` and is logged under the ``echo.memory`` logger, so a
background task can never produce an unhandled-exception storm. A failure
mid-batch may leave part of the batch written — the store is the source of
truth, and a retry deduplicates via exact-text matching.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from typing import Any, NamedTuple, Protocol, runtime_checkable

__all__ = [
    "FACT_KINDS",
    "MAX_TRANSCRIPT_CHARS",
    "SEARCH_K",
    "DefaultHeuristicExtractor",
    "Distiller",
    "FactExtractor",
    "default_contradiction",
    "transcript_from_events",
]

logger = logging.getLogger("echo.memory")

#: Cap on the transcript built from HarnessEvents before extraction
#: (the events path bounds the post-turn job; ``distill_session`` itself
#: accepts whatever transcript the caller hands it).
MAX_TRANSCRIPT_CHARS = 8000

#: How many similar memories the contradiction check consults per fact.
SEARCH_K = 5

#: The kinds a fact may carry (the :class:`FactExtractor` contract).
FACT_KINDS: tuple[str, ...] = ("fact", "preference", "decision", "person")

#: Confidence weights the heuristic fallback assigns per kind.
IMPORTANCE_NAME = 9.0
IMPORTANCE_BIRTHDAY = 8.0
IMPORTANCE_DECISION = 7.0
IMPORTANCE_PREFERENCE = 6.0
IMPORTANCE_DEFAULT = 5.0


# ----------------------------------------------------------------- protocol


@runtime_checkable
class FactExtractor(Protocol):
    """The extraction seam (Phase 6.3): transcript → worth-remembering facts.

    Each returned dict is ``{"text": str, "kind": one of :data:`FACT_KINDS`,
    "importance": float 1..10}``. Implementations: the offline
    :class:`DefaultHeuristicExtractor` today; a real small-model extractor
    via the same protocol once a provider is configured.
    """

    async def extract(self, transcript: str) -> list[dict]:
        """Pull the facts out of one session transcript. Never blocks on I/O
        beyond the extractor's own model call, if it makes one."""
        ...


# -------------------------------------------------- heuristic scan helpers
#
# ONE scanner (_scan) feeds both the default extractor (fact text/kind/
# importance) and the default contradiction checker (marker/value) — zero
# drift between what gets stored and what gets compared.


class _Match(NamedTuple):
    """One heuristic hit inside a sentence."""

    text: str                 # canonical fact text — what gets stored
    kind: str
    importance: float
    marker: tuple[str, ...]   # personalized subject: ("name",) / ("birthday", rel) / ("preference", object)
    value: str                # casefolded comparable value behind that marker
    contradictable: bool      # only name/birthday/preference auto-contradict; decisions never do


_NAME_RE = re.compile(
    r"\b(?:my name(?:'s| is)|call me)\s+"
    r"(?P<value>[a-z][\w'’-]*(?:\s+[a-z][\w'’-]+){0,3})",
    re.IGNORECASE,
)
_BIRTHDAY_RE = re.compile(
    r"\bmy\s+(?P<rel>[a-z]+?)\s*(?:'s|\bs)\s+birthday\s+is\s+"
    r"(?P<value>[^.!?\n]{2,})",
    re.IGNORECASE,
)
_PREF_RE = re.compile(
    r"\bi\s+(?:really\s+|kind of\s+|sort of\s+)?"
    r"(?P<verb>prefer|like|love|hate|enjoy|dislike)\s+"
    r"(?P<object>[^.!?\n]{2,})",
    re.IGNORECASE,
)
_DECISION_RE = re.compile(
    r"\b(?P<subject>i|we)\s+(?:have\s+|'ve\s+)?decided\s+"
    r"(?P<value>[^.!?\n]{2,})",
    re.IGNORECASE,
)

_POSITIVE_PREFS = frozenset({"prefer", "like", "love", "enjoy"})
_NEGATIVE_PREFS = frozenset({"hate", "dislike"})

#: Words that end a captured value without being part of it.
_TRAILING_STOPWORDS = frozenset({
    "and", "but", "or", "so", "because", "also", "then", "when", "while",
    "please", "that", "which", "who", "though", "although",
})

#: First words that mark a "call me X" capture as NOT a name (crude guard).
_NON_NAMES = frozenset({
    "later", "now", "back", "again", "soon", "today", "tomorrow", "tonight",
    "first", "next", "here", "there", "away", "up", "down", "off", "on",
    "in", "out", "the", "a", "an", "it",
})

#: Tokens dropped from the contradiction-search query (store search terms).
_QUERY_STOPWORDS = frozenset({
    "my", "i", "we", "me", "you", "your", "is", "are", "was", "were", "be",
    "been", "the", "a", "an", "to", "of", "on", "in", "at", "it", "that",
    "this", "and", "or", "but", "s",
})

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _canon(text: str) -> str:
    """Case-insensitive, whitespace-collapsed form used for exact-text keys."""
    return " ".join(str(text).split()).casefold()


def _trim_value(value: str) -> str:
    """Drop trailing connector words and punctuation from a captured value."""
    words = value.split()
    while words and words[-1].casefold().strip(".,;:!?") in _TRAILING_STOPWORDS:
        words.pop()
    return " ".join(words).strip(" .,;:!?-")


def _clean_date(value: str) -> str:
    """Normalize a captured date value ("on May 3" → "May 3")."""
    cleaned = value.strip()
    lowered = cleaned.casefold()
    for lead in ("on ", "in "):
        if lowered.startswith(lead):
            cleaned = cleaned[len(lead):].strip()
            break
    return cleaned.strip(" .,;:!?-")


def _scan(text: str) -> list[_Match]:
    """Run every heuristic pattern over one stretch of text.

    This is the single source of truth for both extraction (the fact texts)
    and the default contradiction check (markers + values) — one scanner,
    zero drift.
    """
    matches: list[_Match] = []
    for found in _NAME_RE.finditer(text):
        value = _trim_value(found.group("value"))
        if not value or value.split()[0].casefold() in _NON_NAMES:
            continue
        matches.append(
            _Match(
                f"My name is {value}",
                "person",
                IMPORTANCE_NAME,
                ("name",),
                value.casefold(),
                True,
            )
        )
    for found in _BIRTHDAY_RE.finditer(text):
        relative = found.group("rel").strip()
        value = _clean_date(found.group("value"))
        if not relative or not value:
            continue
        matches.append(
            _Match(
                f"My {relative}'s birthday is {value}",
                "person",
                IMPORTANCE_BIRTHDAY,
                ("birthday", relative.casefold()),
                value.casefold(),
                True,
            )
        )
    for found in _PREF_RE.finditer(text):
        verb = found.group("verb").casefold()
        obj = _trim_value(found.group("object"))
        if not obj:
            continue
        polarity = "positive" if verb in _POSITIVE_PREFS else "negative"
        matches.append(
            _Match(
                f"I {verb} {obj}",
                "preference",
                IMPORTANCE_PREFERENCE,
                ("preference", obj.casefold()),
                polarity,
                True,
            )
        )
    for found in _DECISION_RE.finditer(text):
        value = _trim_value(found.group("value"))
        if not value:
            continue
        subject = found.group("subject").capitalize()
        # Decisions are recorded but never auto-contradict: restating a
        # decision with more detail is not a conflict.
        matches.append(
            _Match(
                f"{subject} decided {value}",
                "decision",
                IMPORTANCE_DECISION,
                ("decision",),
                value.casefold(),
                False,
            )
        )
    return matches


def _sentences(transcript: str) -> list[str]:
    """Split a transcript into sentence-ish pieces (lines, then .!? breaks)."""
    pieces: list[str] = []
    for line in transcript.splitlines():
        for piece in _SENTENCE_SPLIT_RE.split(line):
            piece = piece.strip()
            if piece:
                pieces.append(piece)
    return pieces


def _search_query(text: str) -> str:
    """The keyword query handed to ``store.search`` for one fact's text."""
    tokens = [
        token
        for token in re.findall(r"[\w'’-]+", text.casefold())
        if token not in _QUERY_STOPWORDS
    ]
    return " ".join(tokens)[:120]


def _normalize_fact(raw: object) -> dict | None:
    """Coerce one extractor output into a safe fact dict (None → skip).

    Defensive against a scripted/custom extractor: missing or empty text is
    dropped, unknown kinds fall back to ``"fact"``, importance is clamped to
    the protocol's 1..10 range — the store never sees garbage.
    """
    if not isinstance(raw, dict):
        return None
    text = raw.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    kind = raw.get("kind")
    if kind not in FACT_KINDS:
        kind = "fact"
    try:
        importance = float(raw.get("importance", IMPORTANCE_DEFAULT))
    except (TypeError, ValueError):
        importance = IMPORTANCE_DEFAULT
    importance = max(1.0, min(10.0, importance))
    return {"text": text.strip(), "kind": kind, "importance": importance}


# --------------------------------------------------- default implementations


class DefaultHeuristicExtractor:
    """The zero-model offline fallback (Phase 6.3).

    Regex/keyword heuristics over the transcript: "my name is X" / "call me
    X" → person fact (importance 9); "my <relative>'s birthday is X" →
    person (8); "I/we decided X" → decision (7); "I prefer/like/love/hate X"
    → preference (6). Duplicate statements within one extraction collapse to
    one fact. Fact texts are CANONICAL single sentences (original casing of
    the captured value kept) so exact-text dedup catches restatements.

    This is deliberately crude — a stopgap that keeps memory working at $0
    with no model. A real small-model extractor plugs in through the same
    :class:`FactExtractor` protocol; the door wires it through settings when
    a provider is configured.
    """

    async def extract(self, transcript: str) -> list[dict]:
        """Heuristically pull facts out of one transcript (never raises on
        ordinary input; the Distiller additionally guards this call)."""
        facts: list[dict] = []
        seen: set[str] = set()
        for sentence in _sentences(str(transcript)):
            for match in _scan(sentence):
                key = _canon(match.text)
                if key in seen:  # deduplicate within one extraction
                    continue
                seen.add(key)
                facts.append(
                    {"text": match.text, "kind": match.kind, "importance": match.importance}
                )
        return facts


def default_contradiction(new_text: str, old_text: str) -> bool:
    """The conservative built-in contradiction decision (injectable seam).

    True only when both texts parse to the same personalized marker — the
    same name, the same relative's birthday, or a preference about the exact
    same object — AND the value behind the marker differs (for preferences,
    a polarity flip: like vs hate). Anything the scanner cannot parse, and
    every decision, is never a contradiction (fail-closed: keep both).
    """
    try:
        new_matches = _scan(str(new_text))
        old_matches = _scan(str(old_text))
    except Exception:  # a scan bug must never crash the distiller
        logger.exception("contradiction scan failed — treating as non-contradiction")
        return False
    for new in new_matches:
        if not new.contradictable:
            continue
        for old in old_matches:
            if not old.contradictable:
                continue
            if new.marker == old.marker and new.value != old.value:
                return True
    return False


def transcript_from_events(events: list, max_chars: int = MAX_TRANSCRIPT_CHARS) -> str:
    """Build the distiller transcript from HarnessEvents.

    User + assistant ``content`` only (tool traffic is noise for fact
    extraction), newline-joined, hard-capped at ``max_chars`` — the cap
    bounds the post-turn job no matter how long the session grew.
    """
    lines: list[str] = []
    for event in events or []:
        if getattr(event, "type", "") not in ("user", "assistant"):
            continue
        content = getattr(event, "content", "")
        if isinstance(content, str) and content.strip():
            lines.append(content.strip())
    return "\n".join(lines)[:max_chars]


# ---------------------------------------------------------------- distiller


class Distiller:
    """Extracts facts from a finished session and files them into the store.

    Args:
        store: the SYNC memory store (duck-typed shape documented in the
            module docstring: ``add`` / ``search`` / ``supersede`` /
            ``list_all``).
        extractor: an object satisfying :class:`FactExtractor`; ``None``
            selects :class:`DefaultHeuristicExtractor`.
        contradiction_checker: ``(new_text, old_text) -> bool`` deciding
            "same subject, conflicting value"; ``None`` selects
            :func:`default_contradiction`. A checker that raises is treated
            as "no contradiction" (fail-closed: keep both) and logged.
        clock: monotonic time source, kept for symmetry with the harness
            seams (used to time the distill pass for the debug log).

    The distiller never raises: every failure collapses to an error result
    dict logged under ``echo.memory``.
    """

    def __init__(
        self,
        store: Any,
        extractor: FactExtractor | None = None,
        contradiction_checker: Callable[[str, str], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.store = store
        self.extractor: FactExtractor = (
            extractor if extractor is not None else DefaultHeuristicExtractor()
        )
        self._fallback = DefaultHeuristicExtractor()
        self.contradiction_checker: Callable[[str, str], bool] = (
            contradiction_checker if contradiction_checker is not None else default_contradiction
        )
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic

    # ---------------------------------------------------------- public API

    async def distill_session(
        self, session_id: str, transcript: str, existing: list[dict] | None = None
    ) -> dict:
        """Distill one transcript into the memory store (never raises).

        ``existing`` optionally supplies the caller's snapshot of active
        memories (``store.list_all()`` is consulted when omitted). Returns
        ``{"added": n, "superseded": m, "facts": [...], "duplicates": d}``;
        on any failure ``{"added": 0, "superseded": 0, "facts": [],
        "duplicates": 0, "error": "..."}``.
        """
        try:
            return await self._distill(session_id, transcript, existing)
        except Exception as exc:
            logger.exception("memory distillation failed for session %s", session_id)
            return {
                "added": 0,
                "superseded": 0,
                "facts": [],
                "duplicates": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }

    async def distill_from_events(self, events: list, session_id: str) -> dict:
        """The "post-session" moment: distill straight from HarnessEvents.

        The coordinator calls this FIRE-AND-FORGET after each chat turn's
        final event (``asyncio.create_task(...)``) — it never raises, so the
        background task can never produce an unhandled-exception storm.
        Builds the transcript (user + assistant content, capped at
        :data:`MAX_TRANSCRIPT_CHARS`), then runs :meth:`distill_session`.
        """
        try:
            transcript = transcript_from_events(events)
        except Exception as exc:
            logger.exception("could not build a transcript for session %s", session_id)
            return {
                "added": 0,
                "superseded": 0,
                "facts": [],
                "duplicates": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return await self.distill_session(session_id, transcript)

    # ------------------------------------------------------------- internals

    async def _distill(
        self, session_id: str, transcript: str, existing: list[dict] | None
    ) -> dict:
        started = self._clock()
        if not isinstance(transcript, str) or not transcript.strip():
            return {"added": 0, "superseded": 0, "facts": [], "duplicates": 0}

        try:
            facts = await self.extractor.extract(transcript)
        except Exception as exc:
            logger.warning(
                "fact extractor %s raised (%s) — falling back to the heuristic extractor",
                type(self.extractor).__name__,
                exc,
            )
            facts = await self._fallback.extract(transcript)
        if not isinstance(facts, (list, tuple)):
            facts = []

        if existing is not None:
            known = list(existing)
        else:
            known = self.store.list_all()
        known_texts = {
            _canon(memory.get("text", "")) for memory in known if isinstance(memory, dict)
        }

        added = 0
        superseded = 0
        duplicates = 0
        kept: list[dict] = []
        for raw in facts:
            fact = _normalize_fact(raw)
            if fact is None:
                continue
            key = _canon(fact["text"])
            if not key or key in known_texts:
                duplicates += 1
                continue
            replaced = self._replace_contradictions(fact["text"])
            self.store.add(
                fact["text"], fact["kind"], fact["importance"], source_session=session_id
            )
            known_texts.add(key)
            added += 1
            superseded += len(replaced)
            kept.append({**fact, "superseded": replaced})

        logger.debug(
            "distilled session %s: added=%d superseded=%d duplicates=%d in %.3fs",
            session_id,
            added,
            superseded,
            duplicates,
            self._clock() - started,
        )
        return {"added": added, "superseded": superseded, "facts": kept, "duplicates": duplicates}

    def _replace_contradictions(self, new_text: str) -> list[Any]:
        """Supersede every stored memory the checker flags against ``new_text``.

        Store failures (search/supersede) propagate to the never-raises
        wrapper; a misbehaving CHECKER is fail-closed (no contradiction).
        Returns the superseded memory ids for observability.
        """
        candidates = self.store.search(_search_query(new_text), SEARCH_K)
        if not isinstance(candidates, (list, tuple)):
            return []
        replaced: list[Any] = []
        for old in candidates:
            if not isinstance(old, dict):
                continue
            old_id = old.get("id")
            if old_id is None:
                continue
            try:
                conflicts = bool(self.contradiction_checker(new_text, str(old.get("text", ""))))
            except Exception:
                logger.exception(
                    "contradiction checker raised — keeping the old memory (fail-closed)"
                )
                continue
            if conflicts:
                self.store.supersede(old_id)
                replaced.append(old_id)
        return replaced
