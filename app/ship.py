"""``python -m app.ship`` — the Phase 8.3 ship CLI (startup, backup, soak).

The one front door for putting Echo on a machine and keeping it safe::

    python -m app.ship install-startup        # launch Echo at login (Windows)
    python -m app.ship uninstall-startup      # remove the login entry
    python -m app.ship backup                 # zip the whole data folder
    python -m app.ship restore <zip> [--yes]  # STOP THE SERVER FIRST
    python -m app.ship soak [--turns N]       # memory-leak exam, offline

Every command prints human-readable output and exits 0 (done) or 1 (not
done) — argparse's usual exit code 2 is folded to 1 so scripts only ever
see two answers. No network anywhere: the soak drives the REAL harness
door (ChatService with ``use_harness=True`` — sacred log, tools, memory)
over a MockProvider, so it proves the loop, not the provider.

``restore`` is two-step by design (reversible-first): without ``--yes`` it
validates the zip, prints exactly what WOULD be replaced, and exits 1
having changed nothing. The soak runs in a throwaway data folder — five
hundred fake turns must never pollute the user's real sessions.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: Sample memory every N turns (Phase 8.3 spec: every 100).
SOAK_SAMPLE_EVERY = 100

#: Warmup turns before the FIRST sample: stores, registries and caches pay
#: their one-time allocation cost here, so the growth check measures
#: per-turn leaks, not setup. Capped by the turn count itself.
SOAK_WARMUP_TURNS = 5

#: PASS iff the last sample is <= 1.5x the first (bounded growth).
SOAK_GROWTH_LIMIT = 1.5


# ------------------------------------------------------------------ soak


def _fmt(num_bytes: int) -> str:
    """Human-readable byte count (ASCII only — ship output must survive cp1252)."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _make_sampler() -> tuple[str, Callable[[], int], bool]:
    """Pick the memory backend once: psutil RSS, else stdlib, else tracemalloc.

    Order: ``psutil`` (true process RSS — the ideal), then ``resource``
    (Unix high-water mark; KB on Linux, bytes on macOS), then
    ``tracemalloc`` current (Python-heap only, works everywhere including
    a stock Windows box with no extra installs). The third element says
    whether WE started tracemalloc (and must stop it afterwards).
    """
    try:
        import psutil
    except ImportError:
        pass
    else:
        process = psutil.Process()
        return "process RSS (psutil)", lambda: int(process.memory_info().rss), False
    try:
        import resource
    except ImportError:
        pass
    else:
        factor = 1 if sys.platform == "darwin" else 1024  # ru_maxrss units differ
        return (
            "max RSS high-water (resource)",
            lambda: int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * factor,
            False,
        )
    import tracemalloc

    started = not tracemalloc.is_tracing()
    if started:
        tracemalloc.start()
    return (
        "traced Python heap (tracemalloc)",
        lambda: int(tracemalloc.get_traced_memory()[0]),
        started,
    )


async def _drive_turn(service: Any, turn: int) -> None:
    """One full turn through the door; RuntimeError if it never completes."""
    final_seen = False
    async for payload in service.stream_chat(f"soak probe {turn}: status check?"):
        kind = payload.get("type")
        if kind == "final":
            final_seen = True
        elif kind == "error":
            raise RuntimeError(f"turn {turn} failed: {payload.get('message', '')}")
    if not final_seen:
        raise RuntimeError(f"turn {turn} never reached final")


async def _settle_and_sample(
    samples: list[tuple[int, int]], turn: int, sample: Callable[[], int]
) -> None:
    """Take one steady-state sample: settle the turn, then measure.

    The fire-and-forget distillers (Phase 6.3) of recent turns are drained
    and a full ``gc.collect()`` runs first, so every sample sees the same
    state — cyclic asyncio garbage waiting for its next GC pass must never
    masquerade as a leak (nor hide one). Only TRULY retained memory counts.
    """
    pending = [
        task for task in asyncio.all_tasks() if task is not asyncio.current_task()
    ]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    gc.collect()
    samples.append((turn, sample()))


async def _soak_turns(
    turns: int, sampler: tuple[str, Callable[[], int], bool] | None = None
) -> list[tuple[int, int]]:
    """Drive the door for ``turns`` turns in a throwaway data folder.

    Returns the (turn, memory_bytes) samples: one after the warmup, one
    every :data:`SOAK_SAMPLE_EVERY` turns, and a final one. The Echo data
    folder is pointed at a temp dir for the duration and restored after;
    every SQLite handle is closed before that dir is deleted (Windows
    locks open files — the exam's bug #1, respected everywhere).
    """
    import app.config
    from app.chatservice import ChatService
    from app.config import InMemorySecretStore, SettingsStore
    from app.providers.base import MockProvider, StreamEvent
    from app.store import SessionStore

    label, sample, _started = sampler if sampler is not None else _make_sampler()
    scratch = Path(tempfile.mkdtemp(prefix="echo-soak-")) / "data"
    original_data_dir = app.config.DATA_DIR
    app.config.DATA_DIR = scratch
    samples: list[tuple[int, int]] = []
    try:
        print(f"memory source: {label}", flush=True)
        print(f"turns: {turns} (samples every {SOAK_SAMPLE_EVERY}, "
              f"first sample after {min(SOAK_WARMUP_TURNS, turns)} warmup turns)")
        print("   turn  memory", flush=True)
        db = scratch / "echo.db"
        settings = SettingsStore(db)
        store = SessionStore(db)
        provider = MockProvider(
            script=[
                StreamEvent(type="text", text="All systems nominal."),
                StreamEvent(type="end"),
            ]
        )
        service = ChatService(
            settings,
            InMemorySecretStore(),
            store,
            chain_factory=lambda: provider,
            use_harness=True,  # the REAL door: sacred log, tools, memory
        )
        try:
            warmup = min(SOAK_WARMUP_TURNS, turns)
            for turn in range(1, turns + 1):
                await _drive_turn(service, turn)
                if turn == warmup or turn % SOAK_SAMPLE_EVERY == 0:
                    await _settle_and_sample(samples, turn, sample)
                    print(f"  {turn:>6}  {_fmt(samples[-1][1])}", flush=True)
            if not samples or samples[-1][0] != turns:
                # Final sample with the stores still open (teardown frees
                # memory, which would flatter the growth check).
                await _settle_and_sample(samples, turns, sample)
        finally:
            # Fire-and-forget distillers (Phase 6.3) must land before we
            # close the stores under them.
            pending = [
                task
                for task in asyncio.all_tasks()
                if task is not asyncio.current_task()
            ]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            service.close()
            settings.close()
            store.close()
        return samples
    finally:
        app.config.DATA_DIR = original_data_dir
        shutil.rmtree(scratch.parent, ignore_errors=True)


def _soak(turns: int) -> int:
    """The soak command: print the table, judge bounded growth, exit 0/1."""
    if turns < 1:
        print("error: --turns must be at least 1")
        return 1
    print(
        f"Echo soak - {turns} turns through the harness door "
        "(MockProvider, offline, throwaway data folder)"
    )
    sampler = _make_sampler()
    try:
        samples = asyncio.run(_soak_turns(turns, sampler))
    except Exception as exc:
        print(f"error: soak failed: {exc}")
        return 1
    finally:
        if sampler[2]:  # we started tracemalloc; we stop it
            with contextlib.suppress(Exception):
                import tracemalloc

                tracemalloc.stop()
    first_turn, first = samples[0]
    last_turn, last = samples[-1]
    limit = max(first, 1) * SOAK_GROWTH_LIMIT
    if last <= limit:
        print(
            f"PASS - memory bounded: {_fmt(last)} at turn {last_turn} <= "
            f"{SOAK_GROWTH_LIMIT:g}x the {_fmt(first)} first sample (turn {first_turn})"
        )
        return 0
    print(
        f"FAIL - memory grew unbounded: {_fmt(last)} at turn {last_turn} > "
        f"{SOAK_GROWTH_LIMIT:g}x the {_fmt(first)} first sample (turn {first_turn})"
    )
    return 1


# ----------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.ship",
        description="Echo ship: login auto-start, backup/restore, memory soak.",
    )
    commands = parser.add_subparsers(dest="command", metavar="command")
    commands.add_parser(
        "install-startup", help="launch Echo at login (writes the Startup-folder entry)"
    )
    commands.add_parser("uninstall-startup", help="remove the login auto-start entry")
    commands.add_parser("backup", help="zip the whole data folder into backups/")
    restore = commands.add_parser(
        "restore", help="restore a backup into data/ (STOP THE SERVER FIRST)"
    )
    restore.add_argument("zip", nargs="?", default="", help="path to an echo-backup-*.zip")
    restore.add_argument(
        "--yes",
        action="store_true",
        help="actually restore (without it: dry run, nothing changes)",
    )
    soak = commands.add_parser("soak", help="N-turn memory-leak exam through the door")
    soak.add_argument(
        "--turns", type=int, default=500, help="turns to drive (default 500)"
    )
    return parser


def _install_startup() -> int:
    from app.backup import write_startup_entry

    path = write_startup_entry()
    if path is None:
        # The manual instruction was already printed (non-Windows v1).
        print("no automatic entry written - see the instruction above")
        return 0
    print(f"Echo will start at login: {path}")
    print("remove it anytime with: python -m app.ship uninstall-startup")
    return 0


def _uninstall_startup() -> int:
    from app.backup import remove_startup_entry

    if remove_startup_entry():
        print("login auto-start entry removed")
    else:
        print("no login auto-start entry found (nothing to remove)")
    return 0


def _backup() -> int:
    from app.backup import create_backup

    path = create_backup()
    print(f"backup written: {path}")
    print("restore it with: python -m app.ship restore <zip> --yes (stop the server first)")
    return 0


def _restore(zip_name: str, assume_yes: bool) -> int:
    import app.config
    from app.backup import read_manifest, restore_backup

    if not zip_name:
        print("usage: python -m app.ship restore <echo-backup-*.zip> [--yes]")
        return 1
    zip_path = Path(zip_name)
    data_dir = Path(app.config.DATA_DIR)
    if not assume_yes:
        # Dry run: run the FULL trust gate, change nothing, exit 1.
        manifest = read_manifest(zip_path)
        print("DRY RUN - nothing has been changed.")
        print(
            f"would replace {manifest['file_count']} files in {data_dir} "
            f"from {zip_path.name} (created {manifest['created_at']})"
        )
        print("re-run with --yes to actually restore - and STOP THE ECHO SERVER FIRST")
        return 1
    print(f"restoring {zip_path.name} into {data_dir} ...")
    restored = restore_backup(zip_path)["restored"]
    print(f"restored {restored} files into {data_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run one ship command. Exit codes are 0 (done) / 1 (not done) only."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse usage errors: fold 2 -> 1
        return 1 if exc.code else 0
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    handlers = {
        "install-startup": lambda: _install_startup(),
        "uninstall-startup": lambda: _uninstall_startup(),
        "backup": lambda: _backup(),
        "restore": lambda: _restore(args.zip, args.yes),
        "soak": lambda: _soak(args.turns),
    }
    try:
        return int(handlers[args.command]())
    except Exception as exc:  # a CLI never tracebacks at the user
        print(f"error: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
