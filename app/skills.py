"""Tool-maker + skills (Phase 7.4) — Echo forges its own hands, safely.

Three pieces behind one door:

- :data:`run_script` — the CONFIRM-always forge. Runs a model-written Python
  script in a sandboxed subprocess: ``asyncio.create_subprocess_exec(
  sys.executable, "-I", "-c", code)`` — ISOLATED mode, LIST arguments, NO
  shell (Windows-safe: nothing is ever parsed by cmd.exe). 60s timeout kills
  a runaway script; stdout/stderr are decoded (utf-8, ``replace``) and capped
  at 20k characters each. The tool NEVER raises — every failure comes back as
  readable text the model can recover from. On success its docstring tells the
  model the next step: offer to save the script as a reusable skill.
- :class:`SkillsStore` — the audit-friendly shelf (Phase 7.4 contract):
  "Scripts live in code rows, not loose files — nothing executes that isn't
  in the store." Same SQLite hygiene family as every Echo store (WAL,
  context manager, idempotent close, RuntimeError-not-assert guards).
- :func:`build_skill_tools` — every ENABLED skill row becomes a ``@tool``
  named exactly like the skill, its docstring the description, its body the
  SAME sandboxed execution path; each run increments the row's ``runs`` and
  ``last_status``. The returned list registers into the :class:`ToolRegistry`
  like builtins (``fn._echo_tool`` via ``@tool``); registration duplicates are
  the coordinator's to suppress (the subagent/RAG wiring pattern).

The permission engine is the real guard (D-012 family): ``run_script`` is
``Danger.CONFIRM`` ALWAYS, and the popup shows the FULL code before anything
runs. Running ``python -c <code>`` is not eval/exec of files — the code string
IS the feature, and it never executes without a human-visible CONFIRM round
(an explicit allowlist rule is the only auto-allow, same seam as run_shell).
The sandbox (isolation, timeout, output cap) is comfort and containment, not
the security boundary.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from app import config as _config
from app.harness.interfaces import Danger
from app.harness.tools.base import tool
from app.harness.tools.context import ToolContext

logger = logging.getLogger("echo.skills")

#: Hard wall-clock per script (contract 7.4: 60s -> kill). The module-level
#: default; :func:`run_script_tool_factory` can inject a different one.
SCRIPT_TIMEOUT_SECONDS = 60.0

#: Output cap per stream (chars) — a runaway script cannot flood even the
#: dispatcher's spill path with one stream's output.
SCRIPT_OUTPUT_CAP = 20_000

#: ``ctx.state`` key under which the door publishes the SkillsStore singleton
#: (the ``wire_state`` pattern from app/harness/subagents.py): the dispatcher
#: shares ctx.state with the injected ToolContext by reference, so this is the
#: only bridge save_skill needs to reach the store.
SKILLS_STORE_STATE_KEY = "echo.skills_store"

_SKILLS_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS skills ("
    " id TEXT PRIMARY KEY,"
    " name TEXT UNIQUE NOT NULL,"
    " code TEXT NOT NULL,"
    " description TEXT,"
    " enabled INTEGER NOT NULL DEFAULT 1,"
    " danger TEXT NOT NULL DEFAULT 'confirm',"
    " created_at REAL,"
    " runs INTEGER NOT NULL DEFAULT 0,"
    " last_status TEXT)"
)

# Fully literal statements (no string building): parameters carry every value,
# so bandit's B608 has nothing dynamic to look at. The save's ON CONFLICT
# clause IS the replace-on-duplicate rule: code/description/danger update,
# while id/created_at/enabled/runs/last_status keep their values.
_SQL_SAVE = (
    "INSERT INTO skills"
    " (id, name, code, description, enabled, danger, created_at, runs, last_status)"
    " VALUES (?, ?, ?, ?, 1, ?, ?, 0, NULL)"
    " ON CONFLICT(name) DO UPDATE SET"
    " code = excluded.code, description = excluded.description,"
    " danger = excluded.danger"
)
_SQL_GET = (
    "SELECT id, name, code, description, enabled, danger, created_at, runs, last_status"
    " FROM skills WHERE name = ?"
)
_SQL_LIST_ALL = (
    "SELECT id, name, code, description, enabled, danger, created_at, runs, last_status"
    " FROM skills ORDER BY created_at, name"
)
_SQL_LIST_ENABLED = (
    "SELECT id, name, code, description, enabled, danger, created_at, runs, last_status"
    " FROM skills WHERE enabled = 1 ORDER BY created_at, name"
)
_SQL_SET_ENABLED = "UPDATE skills SET enabled = ? WHERE name = ?"
_SQL_DELETE = "DELETE FROM skills WHERE name = ?"
_SQL_RECORD_RUN = "UPDATE skills SET runs = runs + 1, last_status = ? WHERE name = ?"

#: Run statuses recorded in ``last_status`` (derived from the sandbox report).
_STATUS_OK = "ok"
_STATUS_TIMEOUT = "timeout"
_STATUS_ERROR = "error"


# ------------------------------------------------------------- the sandbox


def _cap(text: str) -> str:
    """One stream's output, elided with a visible marker past the cap."""
    if len(text) > SCRIPT_OUTPUT_CAP:
        return text[:SCRIPT_OUTPUT_CAP] + (
            f"\n[output truncated at {SCRIPT_OUTPUT_CAP} characters]"
        )
    return text


def _run_status(report: str) -> str:
    """Derive the ``last_status`` value from a sandbox report's header."""
    if "timed out" in report:
        return _STATUS_TIMEOUT
    if report.startswith("exit code 0"):
        return _STATUS_OK
    return _STATUS_ERROR


async def _run_sandboxed(code: str, timeout_seconds: float | None = None) -> str:
    """Run ``code`` on an isolated interpreter; return the formatted report.

    ``python -I -c <code>`` — LIST arguments all the way (no shell to parse,
    Windows-safe), isolated mode (no user site, no ``PYTHON*`` env influence,
    no script-dir on sys.path). On timeout the process is killed and the
    report says so. Output sections are capped at :data:`SCRIPT_OUTPUT_CAP`
    each, so neither stream can drown the other.
    """
    timeout = SCRIPT_TIMEOUT_SECONDS if timeout_seconds is None else float(timeout_seconds)
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-c",
        code,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        timed_out = False
    except asyncio.TimeoutError:
        process.kill()
        try:
            stdout, stderr = await process.communicate()
        except Exception:  # already-dead process races on some platforms
            stdout, stderr = b"", b""
        timed_out = True

    out = _cap(stdout.decode("utf-8", errors="replace").replace("\r\n", "\n"))
    err = _cap(stderr.decode("utf-8", errors="replace").replace("\r\n", "\n"))
    header = (
        f"exit code {process.returncode} (timed out after {timeout:g}s; killed)"
        if timed_out
        else f"exit code {process.returncode}"
    )
    return "\n".join(
        [
            header,
            "--- stdout ---",
            out if out else "(no output)",
            "--- stderr ---",
            err if err else "(no output)",
        ]
    )


# ------------------------------------------------------- the forge (run_script)


def run_script_tool_factory(timeout_seconds: float | None = None) -> Callable:
    """Build the ``run_script`` tool with an injectable timeout (tests).

    ``timeout_seconds=None`` keeps the module-level default, resolved at CALL
    time (so patching ``app.skills.SCRIPT_TIMEOUT_SECONDS`` also works).
    """

    @tool(danger=Danger.CONFIRM, cost_hint=2.0)
    async def run_script(code: str, ctx: ToolContext) -> str:
        """Run a Python script in a sandbox; report exit code, stdout, stderr.
        CONFIRM EVERY TIME — the popup shows the full code (the permission gate
        is the real guard, D-012). Isolated interpreter, no shell, 60s timeout,
        output capped. If this solved the task, offer to save it as a reusable
        skill with save_skill.

        Sandbox internals (human-facing, not the model menu): the script runs
        via ``asyncio.create_subprocess_exec(sys.executable, "-I", "-c", code)``
        — list arguments, never a shell (Windows-safe); a runaway script is
        killed at the timeout; stdout/stderr are decoded utf-8 (``replace``)
        and capped at 20000 characters each. The tool never raises: every
        failure returns as readable text (H2-3). ``ctx`` is the H3-2 seam,
        accepted for dispatcher symmetry, unused today.
        """
        # Never raise into the loop (H2-3): every failure is readable text.
        try:
            return await _run_sandboxed(code, timeout_seconds=timeout_seconds)
        except Exception as exc:
            logger.warning("run_script could not start: %s: %s", type(exc).__name__, exc)
            return f"script failed to run: {type(exc).__name__}: {exc}"

    return run_script


#: The built-in forge tool (module-level, registered like the other builtins).
run_script = run_script_tool_factory()


# ------------------------------------------------------------ the shelf (store)


def _row_to_dict(row: sqlite3.Row) -> dict:
    """One skills row as a plain dict (``enabled``/``runs`` stay 0/1 ints)."""
    return {key: row[key] for key in row.keys()}


class SkillsStore:
    """Named, proven scripts — SQLite at heart, code rows never loose files.

    Usable as a context manager (releases the Windows file lock on exit);
    ``close()`` is idempotent and every method raises RuntimeError (never an
    assert — bandit B101) after it. All writes commit before returning.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        # Lazy-resolved at call time: tests patch app.config.DATA_DIR after
        # import (the H0 session-log lesson — never bind runtime paths at import).
        self._path = Path(path) if path is not None else _config.DATA_DIR / "skills.db"
        self._conn: sqlite3.Connection | None = self._connect()

    # ------------------------------------------------------------- lifecycle

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL").fetchone()
        conn.execute(_SKILLS_SCHEMA)
        conn.commit()
        return conn

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            # A real exception, not an assert: survives `python -O` and bandit B101.
            raise RuntimeError("SkillsStore used after close()")
        return self._conn

    def __enter__(self) -> "SkillsStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Release the database file. Safe to call more than once."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def journal_mode(self) -> str:
        """The SQLite journal mode (always ``wal`` — crash-safe from day one)."""
        row = self._db().execute("PRAGMA journal_mode").fetchone()
        return str(row[0]).lower()

    # ------------------------------------------------------------------ CRUD

    def save(
        self,
        name: str,
        code: str,
        description: str = "",
        danger: str = "confirm",
    ) -> dict:
        """Insert a skill, or REPLACE an existing name's script in place.

        A duplicate name updates code, description and danger while KEEPING
        the row's id (and its created_at, enabled flag and run counters) —
        re-saving a skill is an edit of the same tool, not a new identity.
        """
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("skill name must not be empty")
        if not str(code).strip():
            raise ValueError("skill code must not be empty")
        self._db().execute(
            _SQL_SAVE,
            (
                uuid.uuid4().hex,
                clean_name,
                str(code),
                str(description or ""),
                str(danger or "confirm").strip().lower() or "confirm",
                time.time(),
            ),
        )
        self._db().commit()
        record = self.get(clean_name)
        if record is None:  # cannot happen after a committed upsert
            raise RuntimeError(f"skill {clean_name!r} vanished right after save")
        return record

    def get(self, name: str) -> dict | None:
        """One skill row by name, or None."""
        row = self._db().execute(_SQL_GET, (str(name),)).fetchone()
        return _row_to_dict(row) if row is not None else None

    def list(self, include_disabled: bool = True) -> list[dict]:
        """All skill rows (oldest first); only enabled ones when asked."""
        sql = _SQL_LIST_ALL if include_disabled else _SQL_LIST_ENABLED
        rows = self._db().execute(sql).fetchall()
        return [_row_to_dict(row) for row in rows]

    def set_enabled(self, name: str, enabled: bool) -> bool:
        """Enable/disable a skill; True when a row was actually updated."""
        cursor = self._db().execute(_SQL_SET_ENABLED, (1 if enabled else 0, str(name)))
        self._db().commit()
        return cursor.rowcount > 0

    def delete(self, name: str) -> bool:
        """Forget a skill; True when something was deleted."""
        cursor = self._db().execute(_SQL_DELETE, (str(name),))
        self._db().commit()
        return cursor.rowcount > 0

    def record_run(self, name: str, status: str) -> None:
        """Count one execution and remember how it ended (unknown name: no-op)."""
        self._db().execute(_SQL_RECORD_RUN, (str(status), str(name)))
        self._db().commit()


# -------------------------------------------------- the save_skill tool + wiring


def _store_from_state(ctx: ToolContext | None) -> SkillsStore | None:
    """The store the door wired into ``ctx.state``, if it is one of ours."""
    if ctx is None or not isinstance(ctx.state, dict):
        return None
    candidate = ctx.state.get(SKILLS_STORE_STATE_KEY)
    return candidate if isinstance(candidate, SkillsStore) else None


def wire_skills_state(ctx: Any, store: SkillsStore) -> None:
    """Publish ``store`` in ``ctx.state`` so the save_skill tool can find it.

    One wiring point for the door (the dispatcher shares ``ctx.state`` with
    the injected :class:`ToolContext` by reference — same bridge the task
    tool uses). Without it, save_skill falls back to a brief connection on
    the default ``skills.db``.
    """
    state = getattr(ctx, "state", None)
    if isinstance(state, dict):
        state[SKILLS_STORE_STATE_KEY] = store


@tool(danger=Danger.CONFIRM)
def save_skill(name: str, code: str, ctx: ToolContext, description: str = "") -> str:
    """Save a proven script as a named, reusable skill (a tool Echo keeps).

    Use after run_script succeeded: give a short snake_case name, the exact
    code that worked, and a one-line description. The skill becomes a tool
    named after itself (still CONFIRM-gated). Saving is CONFIRM too.
    """
    store = _store_from_state(ctx)
    if store is not None:
        record = store.save(name, code, description)
    else:
        # No door-wired store: a brief connection on the default shelf —
        # opened and closed inside the call (Windows keeps no lock behind).
        with SkillsStore() as fallback:
            record = fallback.save(name, code, description)
    return f"skill saved: {record['name']} — it now runs as a tool named {record['name']!r}"


def _danger_from_row(row: dict) -> Danger:
    """Parse a row's danger class; anything unreadable fails safe to CONFIRM."""
    try:
        return Danger(str(row.get("danger") or "").strip().lower())
    except ValueError:
        return Danger.CONFIRM


def _skill_tool_from_row(row: dict, store: SkillsStore) -> Callable | None:
    """One skill row -> one ``@tool`` (or None for a broken row — never raise).

    The tool is named exactly the skill's name; its model-facing description
    is the row's description (with a fallback when that is blank). The body
    runs the row's code through the SAME sandboxed path as run_script and
    books the attempt (``runs`` + ``last_status``) before returning.
    """
    name = str(row.get("name") or "").strip()
    code = str(row.get("code") or "")
    if not name or not code.strip():
        logger.warning("skipping malformed skill row %r", row.get("name"))
        return None
    danger = _danger_from_row(row)
    description = str(row.get("description") or "").strip()

    async def run_skill(ctx: ToolContext) -> str:
        try:
            report = await _run_sandboxed(code)
            status = _run_status(report)
        except Exception as exc:  # sandbox itself failed (never the script)
            logger.warning("skill %r could not run: %s", name, exc)
            report = f"script failed to run: {type(exc).__name__}: {exc}"
            status = _STATUS_ERROR
        with contextlib.suppress(Exception):
            store.record_run(name, status)
        return report

    run_skill.__name__ = name
    run_skill.__doc__ = description or (
        f"Run the saved skill {name!r} (no description was provided)."
    )
    return tool(danger=danger, cost_hint=2.0)(run_skill)


def build_skill_tools(store: SkillsStore) -> list:
    """One registered-ready ``@tool`` per ENABLED skill; never raises.

    Disabled rows are skipped (that IS the enable/disable switch); a broken
    row (blank name/code) is skipped with a log line instead of poisoning the
    build. Danger comes from the row, fail-safe to CONFIRM when unreadable.
    Registration into a registry is the caller's job — duplicate names (e.g.
    rebuilding onto a live registry) raise there and are suppressed by the
    coordinator, exactly like the Gmail/MCP/RAG wiring.
    """
    tools: list[Callable] = []
    for row in store.list(include_disabled=False):
        fn = _skill_tool_from_row(row, store)
        if fn is not None:
            tools.append(fn)
    return tools


__all__ = [
    "SKILLS_STORE_STATE_KEY",
    "SCRIPT_OUTPUT_CAP",
    "SCRIPT_TIMEOUT_SECONDS",
    "SkillsStore",
    "build_skill_tools",
    "run_script",
    "run_script_tool_factory",
    "save_skill",
    "wire_skills_state",
]
