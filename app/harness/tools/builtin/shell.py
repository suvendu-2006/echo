"""run_shell (H3-5) — Echo's strongest hand, gated the hardest (D-012).

``run_shell`` is **CONFIRM ALWAYS**: the dispatcher's permission engine gates
every single invocation, and the ONLY thing that can auto-allow it is an
explicit allowlist rule (e.g. ``run_shell(npm run:*)``). There is no mode, no
danger downgrade, and no default that runs shell without a decision.

``split_compound`` splits a command on top-level ``&&``/``||``/``;``/``|``/
newlines, quote-aware — a MATCHING AID for writing those allowlist rules,
NEVER a security boundary (PLAN v4 fix #4): quotes, ``$()``/backticks,
unbalanced quotes and encodings defeat naive splitting, so this module
refuses to guess — anything it cannot confidently parse comes back UNSPLIT
(one segment), and opaque commands match no concrete rule in the permission
engine, falling through to the always-ask boundary. Input that reaches the
shell is whatever the caller passed; the splitter touches nothing.
"""

from __future__ import annotations

import asyncio

from app.harness.interfaces import Danger
from app.harness.tools.base import tool

#: Output cap (chars) — the dispatcher may spill, but raw process output is
#: itself capped first: a runaway command cannot flood even the spill path.
SHELL_OUTPUT_CAP = 20_000

#: Hard wall-clock per command (seconds, H3-5).
SHELL_TIMEOUT_SECONDS = 30.0

# bandit: shell-interpreter execution is deliberate here. Justification: every
# run_shell invocation passes the permission engine first (D-012) — CONFIRM
# by default, auto-allowed only by an explicit allowlist rule. Compound
# splitting is a matching aid, never the boundary (module docstring).


def split_compound(command: str) -> list[str]:
    """Split a shell command on top-level ``&&``/``||``/``;``/``|``/newlines.

    Quote-aware: separators inside single/double quotes, inside
    ``$(...)`` substitutions and inside backticks do NOT split. Empty
    segments (from ``a && && b`` or a trailing separator) are dropped.

    MATCHING AID ONLY — never a security boundary. Anything this function
    cannot confidently parse (unbalanced quotes, unterminated substitution)
    is returned UNSPLIT as one segment rather than guessed at; permission
    rules that rely on segment matching must treat such input with suspicion
    and let the always-CONFIRM boundary decide.
    """
    segments: list[str] = []
    buf: list[str] = []
    quote = ""
    substitution_depth = 0  # inside $( ... )
    backtick = False
    i, n = 0, len(command)
    while i < n:
        c = command[i]
        if quote:
            buf.append(c)
            if quote == '"' and c == "\\" and i + 1 < n:
                buf.append(command[i + 1])  # keep \" style escapes glued
                i += 2
                continue
            if c == quote:
                quote = ""
            i += 1
            continue
        if backtick:
            buf.append(c)
            if c == "`":
                backtick = False
            i += 1
            continue
        if c in "'\"":
            quote = c
            buf.append(c)
            i += 1
            continue
        if c == "`":
            backtick = True
            buf.append(c)
            i += 1
            continue
        if c == "$" and command[i : i + 2] == "$(":
            substitution_depth += 1
            buf.append("$(")
            i += 2
            continue
        if substitution_depth and c == ")":
            substitution_depth -= 1
            buf.append(c)
            i += 1
            continue
        if substitution_depth == 0 and command[i : i + 2] in ("&&", "||"):
            segments.append("".join(buf).strip())
            buf = []
            i += 2
            continue
        if substitution_depth == 0 and c in ";|\n":
            segments.append("".join(buf).strip())
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    if quote or backtick or substitution_depth:
        # Cannot confidently parse: hand it back UNSPLIT (never guess).
        stripped = command.strip()
        return [stripped] if stripped else []
    segments.append("".join(buf).strip())
    return [segment for segment in segments if segment]


@tool(danger=Danger.CONFIRM)
async def run_shell(command: str, workdir: str = ".") -> str:
    """Run a shell command and return its exit code plus combined output.

    CONFIRM EVERY TIME (D-012): the permission engine gates every invocation;
    only an explicit allowlist rule can run it without asking. Compound
    splitting (split_compound) is a MATCHING AID for rules, never a security
    boundary. Times out after 30 seconds (command is killed); output capped
    at 20000 characters.
    """
    # Bandit B602 note (justification): shell execution is this tool's whole
    # purpose; the permission engine gates EVERY invocation (D-012 — CONFIRM
    # always, an explicit allowlist rule is the only auto-allow), so the risk
    # lives behind the H4 gate, not in this call site.
    process = await asyncio.create_subprocess_shell(  # nosec B602
        command,
        cwd=workdir or ".",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=SHELL_TIMEOUT_SECONDS)
        timed_out = False
    except asyncio.TimeoutError:
        process.kill()
        try:
            stdout, _ = await process.communicate()
        except Exception:  # already-dead process races on some platforms
            stdout = b""
        timed_out = True

    text = stdout.decode("utf-8", errors="replace").replace("\r\n", "\n")
    if len(text) > SHELL_OUTPUT_CAP:
        text = text[:SHELL_OUTPUT_CAP] + f"\n[output truncated at {SHELL_OUTPUT_CAP} characters]"
    header = (
        f"exit code {process.returncode} (killed after {SHELL_TIMEOUT_SECONDS:g}s timeout)"
        if timed_out
        else f"exit code {process.returncode}"
    )
    return f"{header}\n{text if text else '(no output)'}"
