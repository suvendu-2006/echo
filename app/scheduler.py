"""Scheduler + notifications (Phase 7.3) — Echo's night shift.

The part of Echo that works while nobody is typing: the 8 AM briefing and
the price watchers fire on schedule and enter the harness through the same
door chat does. PLAN 7.3: jobs run headless "entering through the
``run_turn`` door", so a scheduled briefing is a FULL agent turn — tools,
memory, voice — not a canned message.

Wiring (the coordinator owns the door; this module never imports it)::

    # The runner seam: an async callable(task, job_name) -> final text that
    # runs ONE full agent turn through the harness door (ChatService /
    # AgentLoop.run_turn) and returns the final assistant text. The
    # scheduler treats it as an opaque async function — tests inject fakes.
    async def harness_runner(task: str, job_name: str) -> str: ...

    # Build the notifier where winotify is known to be optional:
    try:
        notifier: Notifier | None = ToastNotifier()   # lazy winotify import
    except RuntimeError:                              # winotify not installed
        notifier = None                               # log-only fallback

    scheduler = Scheduler(JobStore(), harness_runner, notifier=notifier)
    asyncio.create_task(scheduler.run_forever())      # + scheduler.stop() at shutdown

Pieces:

- :func:`parse_schedule` — pure text -> dict schedule parser. Exactly three
  shapes: ``"daily 08:00"``, ``"weekly SAT 09:30"`` (MON..SUN,
  case-insensitive), ``"every 30 minutes"``; anything else raises
  RuntimeError (an Echo error the UI can show, never an assert).
- :func:`compute_next` — pure schedule -> next epoch, local time via the
  stdlib ``time`` module only (no deps — D-009). Daily/weekly snap to
  wall-clock local time, DST-safe via ``time.mktime`` field normalization;
  intervals add minutes. A job whose ``next_run`` is already in the past at
  startup (missed while Echo was down) is simply ``next_run <= now`` — it
  fires once on the first poll and its ``next_run`` is recomputed forward
  from NOW: the missed briefing still arrives, and no backlog time-travels.
- :class:`JobStore` — SQLite with the exam-bug hygiene every Echo store
  shares (WAL from day one, context manager, idempotent close,
  RuntimeError-not-assert use-after-close guard, lazy ``DATA_DIR`` resolve
  at construction — bug #6 never again).
- :class:`Scheduler` — the poll loop. ``run_forever`` = ``tick`` + sleep
  forever; ``tick`` is the single poll exposed for tests: take due jobs
  ONE at a time, toast, run the runner, record last_run/last_result,
  recompute next_run. Nothing a job, a notifier or the store does can
  raise out of the loop — failures land in the job row's ``last_result``
  as ``failed: ...`` and the loop moves on.
- :class:`ToastNotifier` — Windows toast via lazy ``winotify``; missing
  dependency raises RuntimeError at construction (the caller falls back to
  ``notifier=None``), while ``notify()`` itself never raises.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from app import config as _config

logger = logging.getLogger("echo.scheduler")

#: The runner seam: ``async (task, job_name) -> final text``. The coordinator
#: wires it to the harness door (``run_turn``); tests inject fakes.
Runner = Callable[[str, str], Awaitable[str]]

#: Stable toast-title prefix (the Automations UI keys off it).
NOTIFY_TITLE_PREFIX = "Echo: "

#: Toast body cap — the popup clips anyway; keep it readable.
NOTIFY_BODY_CAP = 120

#: ``last_result`` cap — a briefing answer is long; the job row keeps the head.
RESULT_CAP = 500

#: Where a job with an unparsable schedule is pushed, so a corrupt row
#: cannot hot-loop the scheduler every poll (re-armed an hour out).
_BROKEN_SCHEDULE_BACKOFF_SECONDS = 3600.0

_WEEKDAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_WEEKDAYS = {name: index for index, name in enumerate(_WEEKDAY_NAMES)}


def parse_schedule(text: str) -> dict[str, Any]:
    """Parse a schedule string into its dict form (pure — no I/O, no clock).

    Exactly three shapes are valid (case-insensitive, surrounding whitespace
    ignored):

    - ``"daily HH:MM"``     -> ``{"kind": "daily", "hour": H, "minute": M}``
    - ``"weekly DAY HH:MM"``-> ``{"kind": "weekly", "weekday": 0..6, "hour": H,
      "minute": M}`` with MON=0 .. SUN=6
    - ``"every N minutes"`` -> ``{"kind": "interval", "minutes": N}`` (N >= 1)

    Anything else — unknown shape, an impossible clock time (24:00, 08:60),
    an unknown day name, or N < 1 / not a whole number — raises RuntimeError:
    schedule text comes from users and from the model, so a bad one is an
    expected event, never a crash (and never an assert, which ``python -O``
    would erase).
    """
    parts = (text or "").strip().casefold().split()
    if len(parts) == 2 and parts[0] == "daily":
        hour, minute = _parse_hhmm(parts[1])
        return {"kind": "daily", "hour": hour, "minute": minute}
    if len(parts) == 3 and parts[0] == "weekly":
        weekday = _WEEKDAYS.get(parts[1])
        hour, minute = _parse_hhmm(parts[2])
        if weekday is None:
            raise RuntimeError(
                f"invalid schedule {text!r}: day must be one of "
                f"{'|'.join(_WEEKDAY_NAMES).upper()}"
            )
        return {"kind": "weekly", "weekday": weekday, "hour": hour, "minute": minute}
    if len(parts) == 3 and parts[0] == "every" and parts[2] == "minutes":
        try:
            minutes = int(parts[1])
        except ValueError as exc:
            raise RuntimeError(
                f"invalid schedule {text!r}: {parts[1]!r} is not a whole number "
                "of minutes"
            ) from exc
        if minutes < 1:
            raise RuntimeError(
                f"invalid schedule {text!r}: interval must be at least 1 minute"
            )
        return {"kind": "interval", "minutes": minutes}
    raise RuntimeError(
        f"invalid schedule {text!r}: expected 'daily HH:MM', "
        "'weekly DAY HH:MM' (MON..SUN) or 'every N minutes'"
    )


def _parse_hhmm(text: str) -> tuple[int, int]:
    """Parse ``HH:MM`` into (hour, minute) with real 24 h bounds.

    Raises RuntimeError (not ValueError) so every bad schedule surfaces as
    the same Echo error type callers already handle.
    """
    pieces = text.split(":")
    if len(pieces) != 2 or not pieces[0].isdigit() or not pieces[1].isdigit():
        raise RuntimeError(f"invalid time {text!r}: expected HH:MM")
    hour, minute = int(pieces[0]), int(pieces[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise RuntimeError(f"invalid time {text!r}: hour 0-23, minute 0-59")
    return hour, minute


def compute_next(schedule: dict[str, Any], now_epoch: float) -> float:
    """Next epoch (seconds) a parsed schedule should fire, strictly after
    ``now_epoch`` — pure, deterministic, inject a clock for tests.

    Local wall-clock time via the stdlib ``time`` module only (no deps,
    D-009):

    - daily: today at HH:MM while that is still ahead, else tomorrow.
    - weekly: the next occurrence of ``weekday`` (0=MON..6=SUN) at HH:MM —
      later this week, or next week when this week's slot already passed.
    - interval: ``now + minutes * 60``.

    Date arithmetic goes through ``time.mktime`` field normalization
    (``tm_mday`` 32 means the 1st of the next month), which stays correct
    across daylight-saving shifts where adding 86400 seconds would drift.
    The strictly-after rule also means a job evaluated exactly AT its slot
    re-arms to the next one — a fire can never double-book itself.
    """
    kind = schedule.get("kind")
    if kind == "interval":
        minutes = int(schedule["minutes"])
        if minutes < 1:
            raise RuntimeError(f"invalid interval schedule {schedule!r}: minutes >= 1")
        return float(now_epoch) + minutes * 60
    if kind in ("daily", "weekly"):
        hour = int(schedule["hour"])
        minute = int(schedule["minute"])
        local = time.localtime(now_epoch)
        day_offset = 0
        if kind == "weekly":
            day_offset = (int(schedule["weekday"]) - local.tm_wday) % 7
        candidate = time.mktime(
            (
                local.tm_year,
                local.tm_mon,
                local.tm_mday + day_offset,
                hour,
                minute,
                0,
                0,
                0,
                -1,  # tm_isdst=-1: let the platform resolve DST
            )
        )
        if candidate <= now_epoch:
            day_offset += 1 if kind == "daily" else 7
            candidate = time.mktime(
                (
                    local.tm_year,
                    local.tm_mon,
                    local.tm_mday + day_offset,
                    hour,
                    minute,
                    0,
                    0,
                    0,
                    -1,
                )
            )
        return candidate
    raise RuntimeError(f"unknown schedule kind in {schedule!r}")


def _connect(path: Path) -> sqlite3.Connection:
    """Open a jobs database with app.config.connect's hygiene."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL").fetchone()
    conn.execute(_SCHEMA)
    return conn


_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS jobs ("
    " id TEXT PRIMARY KEY,"
    " name TEXT UNIQUE,"
    " schedule TEXT NOT NULL,"
    " task TEXT NOT NULL,"
    " enabled INTEGER DEFAULT 1,"
    " last_run REAL,"
    " next_run REAL,"
    " last_result TEXT,"
    " created_at REAL)"
)

_JOB_FIELDS = (
    "id",
    "name",
    "schedule",
    "task",
    "enabled",
    "last_run",
    "next_run",
    "last_result",
    "created_at",
)

# Fully literal statements (no string building): parameters carry every
# value, and bandit's B608 has nothing dynamic to look at.
_SQL_LIST_ENABLED = (
    "SELECT id, name, schedule, task, enabled, last_run, next_run,"
    " last_result, created_at FROM jobs WHERE enabled = 1"
    " ORDER BY created_at, rowid"
)
_SQL_LIST_ALL = (
    "SELECT id, name, schedule, task, enabled, last_run, next_run,"
    " last_result, created_at FROM jobs ORDER BY created_at, rowid"
)
_SQL_DUE = (
    "SELECT id, name, schedule, task, enabled, last_run, next_run,"
    " last_result, created_at FROM jobs"
    " WHERE enabled = 1 AND next_run IS NOT NULL AND next_run <= ?"
    " ORDER BY next_run, rowid"
)


class JobStore:
    """Scheduled jobs, persisted in SQLite with WAL journaling.

    Usable as a context manager. On Windows an open connection keeps an
    exclusive lock on the database file, so the connection MUST be closed
    (or the store used via ``with``) before the file can be moved/deleted.

    The default path resolves at construction time, never at import time
    (exam bug #6: an import-time DATA_DIR binding made a store immune to
    ``ECHO_DATA_DIR`` set after import — the data dir must follow the env
    var, not the import order).
    """

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            path = _config.DATA_DIR / "jobs.db"
        self._conn: sqlite3.Connection | None = _connect(Path(path))

    def __enter__(self) -> "JobStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            # A real exception, not an assert: survives `python -O` and bandit B101.
            raise RuntimeError("JobStore used after close()")
        return self._conn

    @property
    def journal_mode(self) -> str:
        row = self._db().execute("PRAGMA journal_mode").fetchone()
        return str(row[0]).lower()

    # ------------------------------------------------------------ job CRUD

    def add(self, name: str, schedule: str, task: str) -> dict[str, Any]:
        """Create a job and return its row.

        ``schedule`` must parse (:func:`parse_schedule`) — an invalid one
        raises RuntimeError before the database is touched. Names are
        UNIQUE: a duplicate raises RuntimeError (an Echo error the
        Automations UI can show, not a raw sqlite3.IntegrityError).

        ``next_run`` is computed at creation from the real wall clock, so a
        job added to a running scheduler becomes due at its first natural
        slot; the scheduler re-anchors it from its own (injectable) clock
        after every run.
        """
        parsed = parse_schedule(schedule)
        now = time.time()
        job_id = uuid.uuid4().hex
        try:
            self._db().execute(
                "INSERT INTO jobs (id, name, schedule, task, enabled,"
                " last_run, next_run, last_result, created_at)"
                " VALUES (?, ?, ?, ?, 1, NULL, ?, NULL, ?)",
                (job_id, name, schedule, task, compute_next(parsed, now), now),
            )
            self._db().commit()
        except sqlite3.IntegrityError as exc:
            raise RuntimeError(f"a job named {name!r} already exists") from exc
        job = self.get(name)
        if job is None:  # pragma: no cover — the row was just inserted
            raise RuntimeError(f"job {name!r} vanished immediately after insert")
        return job

    def get(self, name: str) -> dict[str, Any] | None:
        """One job row by its (unique) name, or None when unknown."""
        row = self._db().execute(
            "SELECT id, name, schedule, task, enabled, last_run, next_run,"
            " last_result, created_at FROM jobs WHERE name = ?",
            (name,),
        ).fetchone()
        return None if row is None else dict(zip(_JOB_FIELDS, row))

    def list(self, include_disabled: bool = False) -> list[dict[str, Any]]:
        """All jobs, creation order — the Automations screen's list.

        Disabled (paused) jobs are hidden until ``include_disabled``:
        they are paused, not gone.
        """
        sql = _SQL_LIST_ALL if include_disabled else _SQL_LIST_ENABLED
        rows = self._db().execute(sql).fetchall()
        return [dict(zip(_JOB_FIELDS, row)) for row in rows]

    def set_enabled(self, name: str, enabled: bool) -> bool:
        """Pause (False) or resume (True) a job; False when name is unknown."""
        cursor = self._db().execute(
            "UPDATE jobs SET enabled = ? WHERE name = ?",
            (1 if enabled else 0, name),
        )
        self._db().commit()
        return cursor.rowcount > 0

    def delete(self, name: str) -> bool:
        """Remove a job outright; True when it existed."""
        cursor = self._db().execute("DELETE FROM jobs WHERE name = ?", (name,))
        self._db().commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------ run bookkeeping

    def mark_run(self, job_id: str, when: float, result_text: str | None) -> None:
        """Record that job ``job_id`` ran at ``when`` with ``result_text``
        (the caller caps it at :data:`RESULT_CAP`)."""
        self._db().execute(
            "UPDATE jobs SET last_run = ?, last_result = ? WHERE id = ?",
            (when, result_text, job_id),
        )
        self._db().commit()

    def set_next_run(self, job_id: str, when: float) -> None:
        """Store the next scheduled fire time for job ``job_id``."""
        self._db().execute(
            "UPDATE jobs SET next_run = ? WHERE id = ?", (when, job_id)
        )
        self._db().commit()

    def due(self, now: float) -> list[dict[str, Any]]:
        """Jobs that should run at or before epoch ``now``: enabled, with a
        ``next_run`` in the past (or exactly now), oldest first — the
        scheduler takes them ONE at a time in this order. Jobs missed while
        Echo was down surface here naturally (their persisted ``next_run``
        is already in the past) and fire once on the first poll."""
        rows = self._db().execute(_SQL_DUE, (now,)).fetchall()
        return [dict(zip(_JOB_FIELDS, row)) for row in rows]

    def close(self) -> None:
        """Release the database file. Safe to call more than once."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class Notifier(Protocol):
    """Anything with ``notify(title, body) -> None``.

    Implementations must swallow their own failures (see ToastNotifier) —
    the scheduler's night shift outlives a dead toast backend.
    """

    def notify(self, title: str, body: str) -> None: ...


class ToastNotifier:
    """Windows toast notifications via winotify (lazy import — optional dep).

    Construction is the only place winotify can fail: without the package
    installed this raises RuntimeError and the caller passes
    ``notifier=None`` to the Scheduler, which then logs instead of toasting
    — the night shift degrades to log-only, never breaks. ``notify()``
    itself never raises: a toast that cannot show (no desktop session,
    macOS after the move — D-009) is logged and the job carries on.
    """

    def __init__(self, app_id: str = "Echo") -> None:
        try:
            from winotify import Notification  # imported lazily: optional dep
        except ImportError as exc:
            raise RuntimeError(
                "winotify is not installed — toast notifications disabled "
                "(pip install winotify); the scheduler falls back to log-only"
            ) from exc
        self._notification_cls = Notification
        self._app_id = app_id

    def notify(self, title: str, body: str) -> None:
        try:
            toast = self._notification_cls(app_id=self._app_id, title=title, msg=body)
            toast.show()
        except Exception:
            logger.exception("scheduler: toast %r could not be shown", title)


class Scheduler:
    """The poll loop — due jobs fire one at a time, come rain or come crashing.

    Args:
        job_store: the :class:`JobStore` (or duck-typed stand-in) holding
            the job rows — the source of truth for what is due.
        runner: the async harness-door callable ``(task, job_name) -> text``
            the COORDINATOR wires to ``run_turn``; tests inject fakes.
        clock: epoch-seconds provider, injectable for tests. Epoch-based
            (``time.time``, not monotonic) because ``next_run`` values are
            persisted wall-clock epochs that must survive restarts.
        poll_seconds: how long :meth:`run_forever` sleeps between polls.
        notifier: a :class:`Notifier` or None — None means log-only.

    Never-raises guarantee: a poll can do no worse than skip work. Runner,
    notifier and store failures are caught, logged, and recorded in the job
    row (``last_result = "failed: ..."``); the loop always continues.
    """

    def __init__(
        self,
        job_store: Any,
        runner: Runner,
        clock: Callable[[], float] = time.time,
        poll_seconds: float = 20.0,
        notifier: Notifier | None = None,
    ) -> None:
        self._store = job_store
        self._runner = runner
        self._clock = clock
        self.poll_seconds = poll_seconds
        self._notifier = notifier
        self._stopping = False

    # ------------------------------------------------------------ the loop

    async def run_forever(self) -> None:
        """Poll forever: one :meth:`tick`, then sleep ``poll_seconds``.

        Never raises out of the loop — job failures are recorded in the job
        row and logged. Ends when :meth:`stop` is called (clean return) or
        the surrounding task is cancelled (CancelledError propagates — that
        is a shutdown, not a failure). Missed-while-down jobs need no
        special casing: their persisted ``next_run`` is already in the
        past, so the first poll is due them, and :func:`compute_next`
        re-anchors from *now* — one run, no backlog replay.
        """
        self._stopping = False
        while not self._stopping:
            await self.tick()
            await asyncio.sleep(self.poll_seconds)

    def stop(self) -> None:
        """Ask :meth:`run_forever` to finish after the current poll
        (the clean way for the API layer to shut the night shift down)."""
        self._stopping = True

    async def tick(self) -> list[str]:
        """One poll: run every due job, ONE at a time. Returns fired names.

        The unit ``run_forever`` awaits — tests drive it directly with an
        injected clock. Every failure inside (store, notifier, runner,
        schedule math) is caught and logged here: a poll can never take
        the loop down.
        """
        now = self._clock()
        try:
            due_jobs = self._store.due(now)
        except Exception:
            logger.exception("scheduler: job store unreadable — poll skipped")
            return []
        fired: list[str] = []
        for job in due_jobs:
            name = str(job.get("name", job.get("id", "?")))
            try:
                await self._execute(job)
                fired.append(name)
            except Exception:
                # _execute already guards its own steps; this is the last
                # line of defence — the loop must outlive anything.
                logger.exception("scheduler: job %r crashed the executor", name)
        return fired

    # ------------------------------------------------------------- actions

    async def fire_now(self, name: str) -> None:
        """Run a job immediately, whatever its schedule says (the
        Automations "run-now" button), then re-arm it forward.

        Looks up by name (names are UNIQUE); an unknown name raises
        RuntimeError — this is an API-triggered call, so the caller gets a
        real error. Explicit human intent overrides the pause switch: a
        disabled job fired now still runs, exactly like a scheduled run
        (toast, runner, mark_run, next_run forward). Job-internal failures
        never propagate — same never-raises path as the poll.
        """
        job = self._store.get(name)
        if job is None:
            raise RuntimeError(f"no job named {name!r}")
        await self._execute(job)

    # --------------------------------- thin delegates (frozen contract's
    # pause/resume/delete/add live here for the API layer; the JobStore
    # stays the source of truth).

    def add_job(self, name: str, schedule: str, task: str) -> dict[str, Any]:
        """Create a job (schedule validated first); returns the job row."""
        return self._store.add(name, schedule, task)

    def pause(self, name: str) -> bool:
        """Stop a job firing on schedule (run-now still works)."""
        return self._store.set_enabled(name, False)

    def resume(self, name: str) -> bool:
        """Re-arm a paused job."""
        return self._store.set_enabled(name, True)

    def delete(self, name: str) -> bool:
        """Remove a job outright."""
        return self._store.delete(name)

    # ------------------------------------------------------------- internals

    async def _execute(self, job: dict[str, Any]) -> None:
        """Run one job end-to-end: toast, runner, mark, re-arm. Never raises.

        Shared by the poll and :meth:`fire_now`, so a manual run looks
        exactly like a scheduled run in the job row and in the toasts.
        """
        job_id = str(job["id"])
        name = str(job["name"])
        task = str(job["task"])
        self._notify(NOTIFY_TITLE_PREFIX + name, task[:NOTIFY_BODY_CAP])
        try:
            result = await self._runner(task, name)
            result_text = "done" if result is None else str(result)
        except Exception as exc:
            logger.exception("scheduler: job %r runner failed", name)
            result_text = f"failed: {exc}"
        try:
            self._store.mark_run(job_id, self._clock(), result_text[:RESULT_CAP])
        except Exception:
            logger.exception("scheduler: could not record last run of %r", name)
        self._rearm(job)

    def _rearm(self, job: dict[str, Any]) -> None:
        """Recompute next_run from the schedule, anchored at *now*. Never raises.

        A schedule that no longer parses (a row edited underneath us) is
        pushed an hour out instead of hot-looping every poll, and logged.
        """
        name = str(job.get("name", "?"))
        try:
            schedule = parse_schedule(str(job["schedule"]))
            when = compute_next(schedule, self._clock())
        except Exception:
            logger.exception(
                "scheduler: job %r has a broken schedule %r — pushed an hour out",
                name,
                job.get("schedule"),
            )
            when = self._clock() + _BROKEN_SCHEDULE_BACKOFF_SECONDS
        try:
            self._store.set_next_run(str(job["id"]), when)
        except Exception:
            logger.exception("scheduler: could not store next run of %r", name)

    def _notify(self, title: str, body: str) -> None:
        """Fire the notifier; with none, log-only. Never raises."""
        if self._notifier is None:
            logger.info("scheduler notification: %s — %s", title, body)
            return
        try:
            self._notifier.notify(title, body)
        except Exception:
            logger.exception("scheduler: notifier failed for %r (job continues)", title)
