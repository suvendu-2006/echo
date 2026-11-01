"""The Allcited chunker (Phase 6.5 Part A) — deterministic, section-aware.

The exact ingestion algorithm from the user's design (docs/phase6-rag-contract.md,
DECISIONS.md D-023), run over :class:`~app.rag.parsers.ParsedBlock` lists:

1. **Mask** decimal dots and abbreviation dots (``@@DOT@@``) so ``rule 63.7.4``
   and ``Dr. Smith`` can never look like sentence ends.
2. **Sentence-split** on ``[.!?]`` + whitespace, then restore the dots.
3. **Hard-split** any single sentence over ``max_tokens`` (600) into
   whitespace-packed pieces of at most that many tokens.
4. **400-token buffer**: flush into a chunk when ADDING the next sentence
   would push the estimate past ``target_tokens``. Token counts are
   estimates — ``ceil(len(chars) / 4)`` — no tokenizer dependency.
5. **Overlap**: on a budget flush, walk the buffer BACKWARDS and carry
   sentences until they sum to ``int(target * overlap_ratio)`` (60 tokens at
   the defaults) into the NEXT buffer, so any sentence near a boundary exists
   fully in two chunks. Sentence granularity is atomic: the boundary sentence
   is always carried, even when it alone passes the overlap target.
6. **Section-aware early flush**: when the incoming sentence's top-level
   section differs from the buffer's, flush early once the buffer holds at
   least :data:`SECTION_FLUSH_MIN_TOKENS` (150). No overlap is carried across
   a section boundary — the section header in ``embed_text`` is the context.

Everything is deterministic: same blocks in, same chunks out, every run (no
randomness, no model calls, no dict-order dependence).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

#: Early-flush threshold for a section change (tokens already in the buffer).
SECTION_FLUSH_MIN_TOKENS = 150

#: The mask substituted for protected dots before sentence splitting.
DOT_MASK = "@@DOT@@"

#: Dots BETWEEN digits ("63.7.4", "2.0") are masked wholesale.
_DECIMAL_DOT_RE = re.compile(r"(?<=\d)\.(?=\d)")

#: Single-letter initials ("Harry J. Truman") keep their dot mid-sentence.
_INITIAL_RE = re.compile(r"\b([A-Za-z])\.(?=\s)")

#: Abbreviations whose trailing dot never ends a sentence. Longer forms are
#: listed first so masking is unambiguous; the word-boundary lookbehind stops
#: "p." from matching inside "Stop.".
_ABBREVIATIONS = (
    "i.e.",
    "e.g.",
    "approx.",
    "figs.",
    "nos.",
    "No.",
    "pp.",
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "St.",
    "Fig.",
    "vs.",
    "etc.",
    "sec.",
    "Sec.",
    "Dept.",
    "Univ.",
    "Prof.",
    "Jr.",
    "Sr.",
    "Inc.",
    "Ltd.",
    "Est.",
    "est.",
    "ca.",
    "cf.",
    "p.",
    "v.",
)

#: The sentence boundary: punctuation, then whitespace.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def token_estimate(text: str) -> int:
    """Estimate a text's token count as ``ceil(len(chars) / 4)`` — $0, exact
    enough for buffer math, and identical on every machine."""
    if not text:
        return 0
    return -(-len(text) // 4)


def _mask_decimals(text: str) -> str:
    """Protect decimal and abbreviation dots from the sentence splitter.

    Digit-dot-digit runs ("63.7.4") are fully masked; each known
    abbreviation's dot is masked when it stands alone before whitespace or
    end-of-text; single-letter initials keep their dot. Callers MUST restore
    the dots (:data:`DOT_MASK` -> ``.``) after splitting.
    """
    masked = _DECIMAL_DOT_RE.sub(DOT_MASK, text)
    for abbrev in _ABBREVIATIONS:
        stem, dot = abbrev[:-1], abbrev[-1]
        masked = re.sub(
            r"(?<!\w)" + re.escape(stem) + re.escape(dot) + r"(?=\s|$)",
            stem + DOT_MASK,
            masked,
        )
    return _INITIAL_RE.sub(r"\1" + DOT_MASK, masked)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences with abbreviation/decimal dots protected.

    ``"Pursuant to rule 63.7.4 leave accrues. It is paid."`` is TWO sentences
    — the identifier never becomes a boundary.
    """
    masked = _mask_decimals(text)
    parts = _SENTENCE_SPLIT_RE.split(masked)
    return [part.replace(DOT_MASK, ".").strip() for part in parts if part.strip()]


def _hard_split(sentence: str, max_tokens: int) -> list[str]:
    """Word-split one over-long sentence into pieces of at most max_tokens.

    Greedy left-to-right packing; a single word longer than ``max_tokens``
    becomes its own (oversized) piece rather than being broken mid-word.
    """
    if token_estimate(sentence) <= max_tokens:
        return [sentence]
    pieces: list[str] = []
    current = ""
    for word in sentence.split():
        candidate = f"{current} {word}" if current else word
        if not current or token_estimate(candidate) <= max_tokens:
            current = candidate
            continue
        pieces.append(current)
        current = word
    if current:
        pieces.append(current)
    return pieces


@dataclass
class _Sentence:
    """One buffered sentence plus the provenance it carries into a chunk."""

    text: str
    section_path: tuple[str, ...]
    page: int | None
    is_ocr: bool


@dataclass
class Chunk:
    """One retrievable piece of one document (the Phase 6.5 chunk shape).

    ``embed_text`` is the contextual header (the joined section path, when
    the block has headings) plus the chunk text — that is what the query
    half embeds and what BM25 tokenizes. ``content_hash`` is the sha256 of
    ``text`` and drives delta re-ingest hash reuse. ``parent_section_id`` is
    the same joined path (the two-hop/parent-expansion seam). Vectors are NOT
    here — the query half owns them.
    """

    id: str
    doc_id: str
    index: int
    text: str
    embed_text: str
    content_hash: str
    parent_section_id: str
    page: int | None
    is_ocr: bool
    section_path: list[str]


def chunk_blocks(
    blocks: list,
    doc_id: str,
    target_tokens: int = 400,
    max_tokens: int = 600,
    overlap_ratio: float = 0.15,
    section_aware: bool = True,
) -> list[Chunk]:
    """Turn parsed blocks into deterministic, overlapping, section-aware chunks.

    Args:
        blocks: :class:`~app.rag.parsers.ParsedBlock` objects (any format).
        doc_id: Stable document id — chunk ids are ``{doc_id}_chunk_{i:04d}``.
        target_tokens: Buffer size; a chunk flushes when the NEXT sentence
            would push the estimate past it (default 400).
        max_tokens: Hard ceiling per SENTENCE — longer sentences are
            word-split into pieces of at most this many tokens (default 600).
        overlap_ratio: Fraction of ``target_tokens`` the backward-carried
            overlap aims for (default 0.15 -> 60 tokens, sentence-granular).
        section_aware: When True, a new top-level section flushes a buffer
            already holding >= :data:`SECTION_FLUSH_MIN_TOKENS` (no overlap
            across the boundary). When False, only the token budget flushes.

    Returns:
        Chunks in document order, ids/hashes/headers fully derived — the
        same input always yields byte-identical chunks.
    """
    target_tokens = max(1, int(target_tokens))
    max_tokens = max(target_tokens, int(max_tokens))
    overlap_target = max(0, int(target_tokens * overlap_ratio))

    # Flatten blocks -> sentences (hard-split pieces included), keeping each
    # sentence's provenance so mixed blocks never lose page/OCR metadata.
    sentences: list[_Sentence] = []
    for block in blocks:
        text = str(getattr(block, "text", "") or "").strip()
        if not text:
            continue  # near-empty OCR-disabled PDF pages: no text, no chunk
        section_path = tuple(getattr(block, "section_path", None) or ())
        page = getattr(block, "page", None)
        is_ocr = bool(getattr(block, "is_ocr", False))
        for sentence in split_sentences(text):
            for piece in _hard_split(sentence, max_tokens):
                sentences.append(_Sentence(piece, section_path, page, is_ocr))

    chunks: list[Chunk] = []

    def build_chunk(buf: list[_Sentence]) -> Chunk:
        """Materialize the buffer as the next chunk (ids/hashes derived)."""
        index = len(chunks)
        text = " ".join(item.text for item in buf)
        section_path = list(buf[0].section_path)
        header = " > ".join(section_path)
        return Chunk(
            id=f"{doc_id}_chunk_{index:04d}",
            doc_id=doc_id,
            index=index,
            text=text,
            embed_text=f"{header}\n\n{text}" if header else text,
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            parent_section_id=header,
            page=buf[0].page,
            is_ocr=buf[0].is_ocr,
            section_path=section_path,
        )

    buffer: list[_Sentence] = []
    buffer_tokens = 0
    for sentence in sentences:
        if section_aware and buffer:
            buffer_top = buffer[0].section_path[0] if buffer[0].section_path else ""
            incoming_top = sentence.section_path[0] if sentence.section_path else ""
            if incoming_top != buffer_top and buffer_tokens >= SECTION_FLUSH_MIN_TOKENS:
                chunks.append(build_chunk(buffer))  # section boundary: no overlap
                buffer, buffer_tokens = [], 0
        cost = token_estimate(sentence.text)
        if buffer and buffer_tokens + cost > target_tokens:
            # Budget flush: carry the tail backwards so the boundary sentence
            # exists fully in this chunk AND the next one (sentence-atomic).
            carry: list[_Sentence] = []
            carried = 0
            for held in reversed(buffer):
                carry.insert(0, held)
                carried += token_estimate(held.text)
                if carried >= overlap_target:
                    break
            chunks.append(build_chunk(buffer))
            buffer = list(carry)
            buffer_tokens = sum(token_estimate(item.text) for item in buffer)
        buffer.append(sentence)
        buffer_tokens += cost
    if buffer:
        chunks.append(build_chunk(buffer))
    return chunks
