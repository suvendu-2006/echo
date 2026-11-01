"""Read-only + atomic-write file tools (H3-5).

``list_files``/``glob_files``/``read_file`` are SAFE (read-only — they may
run in parallel with other SAFE tools); ``write_file`` is CONFIRM and ATOMIC:
content lands in a unique sibling temp file in the same directory (same
volume, so the rename is atomic), is flushed + fsynced, then
``os.replace``d over the target. A crash — or a failed rename — therefore
leaves either the old file or the new one, never a half-written mix, and the
temp file is cleaned up so failed writes leave no debris (H4-5's
"denied/interrupted write leaves the filesystem untouched" starts here).

All paths go through ``Path.expanduser``; everything is UTF-8 with
``errors="replace"`` so a stray binary byte degrades instead of crashing.
"""

from __future__ import annotations

import glob as glob_module
import os
import tempfile
from pathlib import Path

from app.harness.interfaces import Danger
from app.harness.tools.base import tool

#: read_file's char cap for the numbered body (H3-5: "char-capped").
READ_CHAR_CAP = 8_000


@tool(danger=Danger.SAFE)
def list_files(path: str, pattern: str = "*") -> str:
    """List a directory: sorted entries, directories marked with a trailing slash, files with sizes."""
    directory = Path(path).expanduser()
    if not directory.is_dir():
        raise RuntimeError(f"not a directory: {path}")
    entries = sorted(directory.glob(pattern), key=lambda p: (p.is_file(), p.name.casefold()))
    if not entries:
        return "(no entries)"
    lines: list[str] = []
    for entry in entries:
        if entry.is_dir():
            lines.append(f"{entry.name}/")
        else:
            try:
                size = entry.stat().st_size
            except OSError:
                size = -1
            lines.append(f"{entry.name} ({size} bytes)" if size >= 0 else f"{entry.name} (size unknown)")
    return "\n".join(lines)


@tool(danger=Danger.SAFE)
def glob_files(pattern: str) -> str:
    """Expand a glob pattern (** crosses directories) into a sorted list of matching paths."""
    matches = sorted(glob_module.glob(pattern, recursive=True))
    if not matches:
        return "(no matches)"
    return "\n".join(matches)


@tool(danger=Danger.SAFE)
def read_file(path: str, offset_line: int = 1, max_lines: int = 400) -> str:
    """Read part of a file as line-numbered text ("  12| text"); use offset_line/max_lines to page."""
    start = max(1, int(offset_line))
    limit = max(1, int(max_lines))
    width = max(4, len(str(start + limit - 1)))
    target = Path(path).expanduser()
    lines: list[str] = []
    used = 0
    char_capped = False
    shown = 0
    with open(target, "r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if line_no < start:
                continue
            if shown >= limit:
                more_exist = True  # we only needed to know the range continued
                break
            formatted = f"{line_no:>{width}}| {raw.rstrip('\n').rstrip('\r')}"
            if len(formatted) > READ_CHAR_CAP:  # one huge line is capped alone
                formatted = formatted[:READ_CHAR_CAP] + "…[line truncated]"
            if used + len(formatted) + 1 > READ_CHAR_CAP and lines:
                char_capped = True
                more_exist = True
                break
            lines.append(formatted)
            used += len(formatted) + 1
            shown += 1
        else:
            more_exist = False
    if not lines:
        return f"(no lines in range {start}..{start + limit - 1})"
    if more_exist:
        next_offset = start + shown
        reason = "char cap" if char_capped else "max_lines cap"
        lines.append(
            f"[truncated at {reason}: showing {shown} line(s); "
            f"call again with offset_line={next_offset} to continue]"
        )
    return "\n".join(lines)


@tool(danger=Danger.CONFIRM)
def write_file(path: str, content: str) -> str:
    """Write text to a file atomically (temp file + fsync + rename); never leaves a half-written file."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    # Sibling temp file: same directory => same volume => os.replace is atomic.
    fd, temp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except BaseException:
        # The target was never touched; remove our temp so no debris remains.
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return f"wrote {len(content.encode('utf-8'))} bytes to {path}"
