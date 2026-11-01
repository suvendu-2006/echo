"""Document parsers (Phase 6.5 Part A) — one contract, many formats.

Every format — TXT, Markdown, HTML, PDF, DOCX — parses into the SAME shape: a
list of :class:`ParsedBlock`, each carrying its text, the heading-derived
``section_path`` that led to it, the page it came from (PDF only), a source
format tag and an ``is_ocr`` flag. The chunker (:mod:`app.rag.chunker`) turns
blocks into retrievable chunks; nothing downstream ever sees format-specific
detail. That is the "one contract, many formats" line of the Phase 6.5
contract (docs/phase6-rag-contract.md, DECISIONS.md D-023).

Formats:

- **TXT** — the whole document as one block (stdlib).
- **MD** — ATX headings (``#``..``######``) build the section tree; an H1
  gives ``[h1]``, an H2 ``[h1, h2]``, and so on (stdlib). ``#`` lines inside
  fenced code blocks are never headings.
- **HTML** — ``h1``-``h6`` headings build the same tree, ``p``/``li`` text
  accumulates, ``script``/``style`` content is dropped entirely (stdlib
  ``html.parser``).
- **PDF** — PyMuPDF (**lazy import**): lines set in the largest font size on
  a page start headings, and the section tree tracks font-size tiers (a new
  size ranks between the tiers already seen). Pages with fewer than 50
  characters of text go to the optional ``ocr_provider`` (rasterized PNG in,
  text out) and come back with ``is_ocr=True``; without a provider (the
  default — OCR is disabled) they are kept as near-empty blocks so page
  numbering survives.
- **DOCX** — python-docx (**lazy import**), ``Heading 1..6`` styles.

The heavy imports pay nothing until a real PDF/DOCX actually arrives: a
missing dependency raises RuntimeError (never an assert) with the
``pip install -r requirements-rag.txt`` hint, so the offline test suite never
needs the RAG extras.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

#: The one pip hint every lazy RAG import raises (requirements-rag.txt is an
#: OPTIONAL extra — the suite itself never needs it).
RAG_DEPS_HINT = "pip install -r requirements-rag.txt"

#: A page with fewer characters of extractable text than this is considered
#: image-only and routed to the OCR provider (when one is configured).
PDF_OCR_PAGE_MIN_CHARS = 50

#: A heading line must be at most this long — a page set in one uniform font
#: has no size hierarchy, and its long lines must never become sections.
_PDF_MAX_HEADING_CHARS = 200

#: Font sizes closer than this are the same tier (fitz sizes are floats).
_PDF_SIZE_EPSILON = 0.2

#: ATX heading: 1-6 hashes, then whitespace, then the heading text (closing
#: hashes trimmed). ``#hashtag`` is not a heading.
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

#: DOCX built-in heading styles ("Heading 1".."Heading 6").
_DOCX_HEADING_RE = re.compile(r"^(?:Heading|heading)\s+([1-6])$")


def _missing_dep(dep: str) -> RuntimeError:
    """The RuntimeError a missing optional RAG dependency raises (with hint)."""
    return RuntimeError(
        f"{dep} is required to parse this document format but is not installed — "
        f"install the optional RAG extras with: {RAG_DEPS_HINT}"
    )


# --------------------------------------------------------------- data model


@dataclass
class ParsedBlock:
    """One parsed piece of one document — the universal parser output.

    ``section_path`` is the heading trail leading to the block (markdown and
    HTML headings, PDF font-size tiers); ``page`` is set only for PDFs;
    ``source`` tags the format (``txt``/``md``/``html``/``pdf``/``pdf-ocr``/
    ``docx``); ``is_ocr`` marks text that came from the OCR provider.
    """

    text: str
    section_path: list[str] = field(default_factory=list)
    page: int | None = None
    source: str = ""
    is_ocr: bool = False


# ------------------------------------------------------------- stdlib formats


def parse_txt(data: bytes) -> list[ParsedBlock]:
    """Parse plain text: one block, one synthetic ``Document`` section."""
    text = data.decode("utf-8", errors="replace").strip()
    return [
        ParsedBlock(text=text, section_path=["Document"], page=None, source="txt")
    ]


def parse_md(data: bytes) -> list[ParsedBlock]:
    """Parse Markdown: ATX headings build the section tree, one block per run.

    An H1 starts ``[h1]``, an H2 under it ``[h1, h2]``; a later H2 under the
    same H1 replaces the deeper tail. Text before the first heading belongs
    to the synthetic ``["Document"]`` section. ``#`` lines inside ```/``~~~``
    fenced code blocks are body text, never headings.
    """
    blocks: list[ParsedBlock] = []
    section_path: list[str] = ["Document"]
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        """Emit the accumulated text as one block (skipping empty runs)."""
        body = re.sub(r"\n{3,}", "\n\n", "\n".join(buf)).strip()
        if body:
            blocks.append(
                ParsedBlock(
                    text=body, section_path=list(section_path), page=None, source="md"
                )
            )
        buf.clear()

    for line in data.decode("utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            buf.append(line)
            continue
        match = None if in_fence else _MD_HEADING_RE.match(stripped)
        if match:
            flush()
            level = len(match.group(1))
            section_path = section_path[: level - 1] + [match.group(2).strip()]
            continue
        buf.append(line)
    flush()
    return blocks


def _normalize_html_text(raw: str) -> str:
    """Collapse inline whitespace per line; drop the empty lines."""
    lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in raw.split("\n"))
    return "\n".join(line for line in lines if line)


class _HTMLTextExtractor(HTMLParser):
    """html.parser subclass: headings -> section tree, p/li text -> blocks.

    ``script``/``style`` content (and anything nested in them) is dropped —
    a ``"<h1>"`` inside a script string must never become a section.
    """

    _HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
    _BLOCK_TAGS = {"p", "li", "br", "div", "tr", "ul", "ol", "table", "blockquote"}
    _DROP_TAGS = {"script", "style", "title"}  # metadata, never body text

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[ParsedBlock] = []
        self.section_path: list[str] = ["Document"]
        self._buf: list[str] = []
        self._skip_depth = 0
        self._heading_level: int | None = None

    # -------------------------------------------------------------- internals

    def _take_text(self) -> str:
        text = _normalize_html_text("".join(self._buf))
        self._buf = []
        return text

    def _flush_block(self) -> None:
        text = self._take_text()
        if text:
            self.blocks.append(
                ParsedBlock(
                    text=text,
                    section_path=list(self.section_path),
                    page=None,
                    source="html",
                )
            )

    # ----------------------------------------------------------- parser hooks

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        tag = tag.lower()
        if tag in self._DROP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in self._HEADING_TAGS:
            self._flush_block()  # the block before the heading ends here
            self._heading_level = int(tag[1])
            return
        if tag in self._BLOCK_TAGS:
            self._buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._DROP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in self._HEADING_TAGS and self._heading_level is not None:
            name = self._take_text()
            if name:
                level = self._heading_level
                self.section_path = self.section_path[: level - 1] + [name]
            self._heading_level = None
            return
        if tag in self._BLOCK_TAGS:
            self._buf.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._buf.append(data)

    def close(self) -> None:
        super().close()
        self._flush_block()  # trailing text without a closing block tag


def parse_html(data: bytes) -> list[ParsedBlock]:
    """Parse HTML via stdlib html.parser: h1-h6 tree, p/li text, no scripts."""
    extractor = _HTMLTextExtractor()
    extractor.feed(data.decode("utf-8", errors="replace"))
    extractor.close()
    return extractor.blocks


# ------------------------------------------------------------ heavy formats


def _pdf_heading_level(sizes: dict[float, int], size: float) -> int:
    """Rank a heading font size among the tiers already seen (1 = topmost).

    A size new to the document slots between the tiers larger and smaller
    than it, so "20pt title, 16pt section, 13pt subsection" levels 1, 2, 3
    no matter which page they first appear on.
    """
    if size not in sizes:
        sizes[size] = 1 + sum(1 for known in sizes if known > size)
    return sizes[size]


def parse_pdf(data: bytes, ocr_provider: Callable[[bytes], str] | None = None) -> list[ParsedBlock]:
    """Parse a PDF with PyMuPDF (lazy): font-size headings + optional OCR.

    Per page: spans and their font sizes are read from ``get_text("dict")``;
    a line set in the page's largest size (when the page really has a size
    hierarchy, and the line is heading-length) starts a heading and the
    section tree tracks size tiers. Pages with under
    :data:`PDF_OCR_PAGE_MIN_CHARS` characters of text are rasterized and
    handed to ``ocr_provider(image_bytes) -> str`` when one is configured
    (``is_ocr=True``, source ``pdf-ocr``); otherwise a near-empty block is
    kept so page numbering survives. OCR is DISABLED by default (pass
    ``None``) — Tesseract is an optional user install.
    """
    try:
        import fitz  # PyMuPDF — the one heavy PDF dep, paid for only on real PDFs
    except ImportError as exc:
        raise _missing_dep("pymupdf") from exc

    blocks: list[ParsedBlock] = []
    section_path: list[str] = []
    size_levels: dict[float, int] = {}
    document = fitz.open(stream=data, filetype="pdf")
    try:
        for page in document:
            page_no = int(page.number) + 1
            lines: list[tuple[str, float]] = []
            try:
                payload = page.get_text("dict") or {}
            except Exception:  # noqa: BLE001 — a broken page must not kill the doc
                payload = {}
            for raw_block in payload.get("blocks") or []:
                if raw_block.get("type") != 0:  # image block, no text spans
                    continue
                for line in raw_block.get("lines") or []:
                    spans = line.get("spans") or []
                    text = "".join(str(span.get("text") or "") for span in spans).strip()
                    if not text:
                        continue
                    size = max(
                        (float(span.get("size") or 0.0) for span in spans), default=0.0
                    )
                    lines.append((text, size))

            total_chars = sum(len(text) for text, _ in lines)
            if total_chars < PDF_OCR_PAGE_MIN_CHARS:
                # Image-only (or near-empty) page: OCR when enabled, else keep
                # a near-empty block so the page number survives into chunks.
                ocr_text = ""
                if ocr_provider is not None:
                    pixmap = page.get_pixmap()
                    ocr_text = str(ocr_provider(pixmap.tobytes("png")) or "").strip()
                if ocr_text:
                    blocks.append(
                        ParsedBlock(
                            text=ocr_text,
                            section_path=list(section_path),
                            page=page_no,
                            source="pdf-ocr",
                            is_ocr=True,
                        )
                    )
                else:
                    blocks.append(
                        ParsedBlock(
                            text="",
                            section_path=list(section_path),
                            page=page_no,
                            source="pdf-ocr" if ocr_provider is not None else "pdf",
                            is_ocr=ocr_provider is not None,
                        )
                    )
                continue

            page_max = max(size for _, size in lines)
            page_min = min(size for _, size in lines)
            has_hierarchy = page_max > page_min + _PDF_SIZE_EPSILON
            body: list[str] = []
            for text, size in lines:
                is_heading = (
                    has_hierarchy
                    and size >= page_max - _PDF_SIZE_EPSILON
                    and len(text) <= _PDF_MAX_HEADING_CHARS
                )
                if not is_heading:
                    body.append(text)
                    continue
                if body:
                    blocks.append(
                        ParsedBlock(
                            text=" ".join(body),
                            section_path=list(section_path),
                            page=page_no,
                            source="pdf",
                        )
                    )
                    body = []
                level = _pdf_heading_level(size_levels, round(size, 2))
                section_path = section_path[: level - 1] + [text]
            if body:
                blocks.append(
                    ParsedBlock(
                        text=" ".join(body),
                        section_path=list(section_path),
                        page=page_no,
                        source="pdf",
                    )
                )
    finally:
        document.close()
    return blocks


def parse_docx(data: bytes) -> list[ParsedBlock]:
    """Parse DOCX with python-docx (lazy): Heading 1-6 styles build the tree."""
    try:
        import docx  # python-docx — lazy like every heavy dep in this package
    except ImportError as exc:
        raise _missing_dep("python-docx") from exc

    blocks: list[ParsedBlock] = []
    section_path: list[str] = ["Document"]
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            blocks.append(
                ParsedBlock(
                    text=body, section_path=list(section_path), page=None, source="docx"
                )
            )
        buf.clear()

    for paragraph in docx.Document(io.BytesIO(data)).paragraphs:
        text = str(paragraph.text or "").strip()
        style = str(getattr(paragraph.style, "name", "") or "")
        heading = _DOCX_HEADING_RE.match(style)
        if not text:
            continue
        if heading:
            flush()
            level = int(heading.group(1))
            section_path = section_path[: level - 1] + [text]
            continue
        if style.lower() == "title":
            flush()
            section_path = [text]
            continue
        buf.append(text)
    flush()
    return blocks


# ------------------------------------------------------------------ dispatch

#: Suffixes :func:`parse_any` knows (the same set ingest_folder walks).
SUPPORTED_SUFFIXES = frozenset(
    {".txt", ".md", ".markdown", ".html", ".htm", ".pdf", ".docx"}
)


def parse_any(
    path: str | Path, ocr_provider: Callable[[bytes], str] | None = None
) -> list[ParsedBlock]:
    """Parse any supported file by suffix, reading its bytes from disk.

    Dispatch: ``.txt`` / ``.md`` / ``.markdown`` / ``.html`` / ``.htm`` /
    ``.pdf`` / ``.docx``. An unknown suffix raises ValueError; a PDF or DOCX
    without its optional dependency raises RuntimeError with the pip hint.
    """
    file_path = Path(path)
    data = file_path.read_bytes()
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return parse_txt(data)
    if suffix in {".md", ".markdown"}:
        return parse_md(data)
    if suffix in {".html", ".htm"}:
        return parse_html(data)
    if suffix == ".pdf":
        return parse_pdf(data, ocr_provider=ocr_provider)
    if suffix == ".docx":
        return parse_docx(data)
    raise ValueError(
        f"unsupported document suffix {suffix!r} for {file_path.name!r} — "
        f"supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
    )
