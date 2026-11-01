"""The RefMap (Phase 6.5 Part A) — references and definitions, all offline.

Regulations cross-reference each other constantly ("pursuant to rule 63.7.4",
"on Form HR-702"). The query half's two-hop stage uses this module to pull a
referenced section's DEFINING chunk into the answer pack. Exactly four
reference families are recognized (docs/phase6-rag-contract.md, D-023):

===========================  ========================  =======================
family                       regex shape               ref id
===========================  ========================  =======================
rule / regulation            ``rule 63.7.4``           ``rule:63.7.4``
clause / section / § / sec.  ``clause 4.2``            ``clause:4.2``
form / schedule / procedure  ``Form HR-702``           ``form:HR-702``
  / standard                 (``[A-Z]{2,4}-[A-Z0-9]{2,5}`` codes)
connective phrases           ``as per HR-702``         ``form:HR-702`` (code
  (as defined in / pursuant                            shaped) or ``ref:63.7.4``
  to / as per)                                         (bare number)
===========================  ========================  =======================

A chunk DEFINES a ref id when the id match sits at a line start or is
immediately followed by ``:`` — "Rule 63.7.4: Annual leave" defines
``rule:63.7.4``; "employees follow rule 63.7.4 during leave" merely
references it. :meth:`RefMap.register_definitions` fills
``definitions[ref_id] -> [chunk_ids]`` (dedup, first-seen order); the query
half walks that dict for two-hop expansion.
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator

#: Family 1 — "rule 63.7.4", "Regulation 12.3.1" (multi-part numbers only).
RULE_RE = re.compile(r"\b(?:rule|regulation)\s+(\d+(?:\.\d+)+)\b", re.IGNORECASE)

#: Family 2 — "clause 4.2", "section 7.1.2", "§9.1", "sec. 2.3" (the § needs
#: no word boundary — it is not a word character itself).
CLAUSE_RE = re.compile(
    r"(?:\b(?:clause|section|sec\.)\s*|§\s*)(\d+(?:\.\d+)+)\b", re.IGNORECASE
)

#: Family 3 — "Form HR-702", "schedule AB-12": keyword case-insensitive, the
#: code itself uppercase per the contract's ``[A-Z]{2,4}-[A-Z0-9]{2,5}``.
FORM_RE = re.compile(
    r"\b(?i:form|schedule|procedure|standard)\s+([A-Z]{2,4}-[A-Z0-9]{2,5})\b"
)

#: Family 4 — connective phrases capture the following id-ish token (an
#: optional determiner is skipped: "as per the ISO-88").
CONNECTIVE_RE = re.compile(
    r"\b(?:as defined in|pursuant to|as per)\s+(?:the\s+|this\s+|said\s+|its\s+)?"
    r"([A-Za-z0-9§][\w§.\-/]*)",
    re.IGNORECASE,
)

#: Trailing sentence punctuation stripped from a connective capture ("as per
#: the ISO-88." must not swallow the final dot into the id).
_TRAILING_PUNCT = ".,;:!?'\""

#: A connective capture shaped like a family-3 code normalizes into that
#: family's vocabulary so two-hop can actually find its definition.
_CODE_SHAPE_RE = re.compile(r"[A-Z]{2,4}-[A-Z0-9]{2,5}")

#: Family keywords: when the token right after a connective IS one of these,
#: the family regex owns the id ("as defined in section 4.2" -> clause:4.2);
#: the connective must not emit a second, vaguer id for the same target.
_FAMILY_KEYWORD_RE = re.compile(
    r"(?i)(?:rule|regulation|clause|section|sec|form|schedule|procedure|standard|§)\.?"
)


def _connective_ref_id(token: str) -> str | None:
    """Normalize a connective capture into a ref id (None = not id-ish)."""
    token = token.strip(_TRAILING_PUNCT)
    if token.startswith("§"):
        number = token[1:].strip(_TRAILING_PUNCT)
        return f"clause:{number}" if number else None
    if _FAMILY_KEYWORD_RE.fullmatch(token):
        return None  # a family keyword follows — its regex catches the id
    if _CODE_SHAPE_RE.fullmatch(token):
        return f"form:{token.upper()}"
    if any(char.isdigit() for char in token):
        return f"ref:{token}"
    return None


def _iter_ref_matches(text: str) -> Iterator[tuple[str, int, int]]:
    """All reference matches in ``text`` as ``(ref_id, start, end)``, in
    document order (ties by longest match), without overlapping duplicates."""
    found: list[tuple[int, int, str]] = []
    for match in RULE_RE.finditer(text):
        found.append((match.start(), match.end(), f"rule:{match.group(1)}"))
    for match in CLAUSE_RE.finditer(text):
        found.append((match.start(), match.end(), f"clause:{match.group(1)}"))
    for match in FORM_RE.finditer(text):
        found.append((match.start(), match.end(), f"form:{match.group(1).upper()}"))
    for match in CONNECTIVE_RE.finditer(text):
        ref_id = _connective_ref_id(match.group(1))
        if ref_id is not None:
            found.append((match.start(), match.end(), ref_id))
    found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    last_end = -1
    for start, end, ref_id in found:
        if start < last_end:
            continue  # inside an already-accepted (longer/earlier) match
        last_end = end
        yield ref_id, start, end


def references_in(text: str) -> list[str]:
    """Ref ids referenced by ``text``, first-appearance order, deduplicated."""
    seen: set[str] = set()
    ordered: list[str] = []
    for ref_id, _start, _end in _iter_ref_matches(text):
        if ref_id not in seen:
            seen.add(ref_id)
            ordered.append(ref_id)
    return ordered


def _is_definition_position(text: str, start: int, end: int) -> bool:
    """True when a match DEFINES its id: line start, or id followed by ':'.

    Line start allows leading whitespace only ("Rule 63.7.4 governs..." or a
    heading line); the colon form is the STRICT "immediately followed" one
    ("clause 4.2: Probation"). Mid-sentence mentions never define.
    """
    line_start = text.rfind("\n", 0, start) + 1
    if not text[line_start:start].strip():
        return True
    return end < len(text) and text[end] == ":"


class RefMap:
    """Reference graph over chunks: ``definitions[ref_id] -> [chunk_ids]``.

    Built once per query by :meth:`register_definitions` (or rehydrated from
    the library's ``refmap`` table); read by the query half's two-hop stage.
    """

    def __init__(self) -> None:
        self.definitions: dict[str, list[str]] = {}

    def references_in(self, text: str) -> list[str]:
        """Ref ids referenced by ``text`` (module-level :func:`references_in`)."""
        return references_in(text)

    def register_definitions(self, chunks: Iterable) -> None:
        """Record which chunks DEFINE which ref ids (dedup, first-seen order).

        A chunk defines an id when one of the four family matches inside its
        text sits at a line start or is immediately followed by ``:``.
        """
        for chunk in chunks:
            text = str(getattr(chunk, "text", "") or "")
            for ref_id, start, end in _iter_ref_matches(text):
                if not _is_definition_position(text, start, end):
                    continue
                known = self.definitions.setdefault(ref_id, [])
                chunk_id = str(getattr(chunk, "id", "") or "")
                if chunk_id and chunk_id not in known:
                    known.append(chunk_id)
