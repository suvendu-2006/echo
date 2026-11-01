"""Backup + restore of the whole data folder, and the auto-start entry (8.3).

The ship phase's safety net. ``create_backup`` zips EVERYTHING under
:data:`app.config.DATA_DIR` (the sacred log, sessions, memory, skills, jobs,
RAG library — Echo's whole soul) into a timestamped zip next to the data
folder, with a manifest INSIDE the zip so ``restore_backup`` can refuse
foreign archives. Volatile caches are excluded: ``traces/`` and ``spills/``
are regenerable, and SQLite ``-wal``/``-shm`` sidecar files are mid-write
journals that would corrupt a restored database.

STOP-THE-SERVER RULE (read this before restoring): this module closes
NOTHING. On Windows an open SQLite connection locks its database file, so
the caller MUST stop the Echo server before ``restore_backup`` — the ship
CLI prints the same warning. ``create_backup`` while the server runs may
capture a mid-write snapshot; prefer backing up with the server stopped.

Cross-platform by construction: ``zipfile`` + ``pathlib`` + ``os.replace``
only — identical behaviour on Windows today and macOS after the move
(DECISIONS.md D-009). The startup entry is Windows-only in v1 (a .bat in
the user's Startup FOLDER — never a registry hack); other platforms get a
printed manual instruction and ``None``, never a platform-based raise.
"""

from __future__ import annotations

import json
import os
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

#: The manifest file every Echo backup carries INSIDE the zip. Its presence
#: (plus a matching name and file count) is what makes an archive ours.
MANIFEST_NAME = "echo-backup-manifest.json"

#: Filename prefix of every backup zip in the backups/ folder.
ZIP_PREFIX = "echo-backup-"

#: Name of the Windows Startup-folder entry this module writes.
STARTUP_ENTRY_NAME = "echo.bat"

#: Regenerable caches inside data/ that never belong in a backup: tool
#: output spills (H3-4) and local trace taps (H7-1). The sacred log holds
#: the durable record of both.
VOLATILE_DIRS = ("traces", "spills")

#: Manifest keys that must be present for an archive to be trusted.
_MANIFEST_KEYS = ("name", "created_at", "file_count")


# ------------------------------------------------------------------ backup


def _excluded(rel_path: Path) -> bool:
    """True when a data/ sub-path must never enter a backup zip."""
    if rel_path.parts and rel_path.parts[0] in VOLATILE_DIRS:
        return True
    return rel_path.name.endswith(("-wal", "-shm"))  # SQLite write-ahead journals


def create_backup(backup_dir: Path | None = None) -> Path:
    """Zip the whole data directory into ``backups/`` next to it (atomically).

    Everything under :data:`app.config.DATA_DIR` is included except the
    volatile caches (:data:`VOLATILE_DIRS`) and SQLite ``-wal``/``-shm``
    sidecar journals (see module docstring for the stop-the-server rule).
    A manifest (name, created_at, file_count) is written INSIDE the zip —
    restore refuses archives without one.

    Atomic write: the zip is built at a temp path and ``os.replace``d into
    place, so a crash mid-backup never leaves a half-written backup (and
    never damages an existing one). Concurrent backups within the same
    second get a ``-2``/``-3`` suffix instead of overwriting each other.

    Args:
        backup_dir: where to write the zip; ``None`` means
            ``DATA_DIR.parent / "backups"`` (the repo's ``backups/``).

    Returns:
        The path of the freshly written backup zip.
    """
    from app.config import DATA_DIR  # call-time read: tests patch the attr

    data_dir = Path(DATA_DIR)
    target_dir = (
        Path(backup_dir) if backup_dir is not None else data_dir.parent / "backups"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = target_dir / f"{ZIP_PREFIX}{stamp}.zip"
    suffix = 1
    while target.exists():  # two backups inside one second: keep both
        suffix += 1
        target = target_dir / f"{ZIP_PREFIX}{stamp}-{suffix}.zip"

    temp_path = target_dir / f".{target.name}.{os.getpid()}.tmp"
    try:
        with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            file_count = 0
            for path in sorted(data_dir.rglob("*")):
                if not path.is_file():
                    continue
                rel = path.relative_to(data_dir)
                if _excluded(rel):
                    continue
                zf.write(path, rel.as_posix())
                file_count += 1
            manifest = {
                "name": target.name,
                "created_at": datetime.now().astimezone().isoformat(
                    timespec="seconds"
                ),
                "file_count": file_count,
            }
            zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=True))
        os.replace(temp_path, target)  # atomic on the same volume
    finally:
        if temp_path.exists():
            temp_path.unlink()  # never leave temp litter behind
    return target


def read_manifest(zip_path: Path) -> dict:
    """Read and validate a backup's manifest WITHOUT extracting anything.

    The trust gate: an archive is an Echo backup only when it carries
    :data:`MANIFEST_NAME`, the manifest holds all of :data:`_MANIFEST_KEYS`,
    its ``name`` matches this zip's own filename (renamed backups are
    refused — the manifest is the identity), and its ``file_count`` matches
    the number of data files actually inside (tamper check).

    Raises:
        ValueError: for a missing file, a foreign zip (no manifest), a
            manifest missing keys, a renamed backup, or a count mismatch.
    """
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise ValueError(f"backup not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        if MANIFEST_NAME not in zf.namelist():
            raise ValueError(
                f"refusing to restore: {zip_path.name} carries no "
                f"{MANIFEST_NAME} — not an Echo backup"
            )
        try:
            manifest = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"refusing to restore: {zip_path.name} has a corrupt manifest ({exc})"
            ) from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"refusing to restore: {zip_path.name} manifest is not an object")
    for key in _MANIFEST_KEYS:
        if key not in manifest:
            raise ValueError(
                f"refusing to restore: {zip_path.name} manifest is missing {key!r}"
            )
    if str(manifest["name"]) != zip_path.name:
        raise ValueError(
            f"refusing to restore: manifest says {manifest['name']!r} but this "
            f"file is {zip_path.name!r} (was the backup renamed?)"
        )
    with zipfile.ZipFile(zip_path) as zf:
        actual = sum(
            1
            for info in zf.infolist()
            if not info.is_dir() and info.filename != MANIFEST_NAME
        )
    if actual != int(manifest["file_count"]):
        raise ValueError(
            f"refusing to restore: {zip_path.name} manifest counts "
            f"{manifest['file_count']} files but the archive holds {actual}"
        )
    return manifest


def _safe_target(data_dir: Path, arcname: str) -> Path:
    """Join an archive member onto data_dir, refusing zip-slip escapes.

    A malicious or foreign zip could carry ``../`` climbs, absolute paths
    or drive letters; every restored member must resolve INSIDE data_dir.
    """
    name = arcname.replace("\\", "/").rstrip("/")
    parts = [part for part in name.split("/") if part not in ("", ".")]
    drive_like = len(parts[0]) == 2 and parts[0][1] == ":" if parts else False
    if not parts or drive_like or any(part == ".." for part in parts):
        raise ValueError(f"refusing to restore unsafe path from backup: {arcname!r}")
    target = data_dir.joinpath(*parts)
    if not target.resolve().is_relative_to(data_dir.resolve()):
        raise ValueError(f"refusing to restore unsafe path from backup: {arcname!r}")
    return target


def _restore_mtime(target: Path, info: zipfile.ZipInfo) -> None:
    """Best-effort: give the restored file the mtime recorded in the zip."""
    try:
        stamp = time.mktime((*info.date_time, 0, 0, -1))
        os.utime(target, (stamp, stamp))
    except (OSError, OverflowError, ValueError):
        pass  # cosmetic only — restore must not fail over timestamps


def restore_backup(zip_path: Path) -> dict:
    """Restore a backup INTO the data directory, replacing existing files.

    **STOP THE SERVER FIRST — THIS FUNCTION CLOSES NOTHING.** The caller
    must have shut Echo down (``python -m app.serve`` holds open SQLite
    connections that Windows locks; restoring under it fails mid-way).
    Files present in data/ but absent from the backup are left untouched —
    restore REPLACES, it never wipes.

    Safety gates (all :class:`ValueError`, never silent): foreign zips are
    refused via :func:`read_manifest`, and every archive member passes the
    zip-slip guard (:func:`_safe_target`). Each file is written to a temp
    file in its final folder then ``os.replace``d into place, so a crash
    mid-restore leaves no half-written files.

    Args:
        zip_path: path to an Echo backup created by :func:`create_backup`.

    Returns:
        ``{"restored": n}`` — how many files were placed back into data/.

    Raises:
        ValueError: foreign/corrupt/tampered zip, or an unsafe member path.
        OSError: e.g. a database file is still locked by a running server.
    """
    from app.config import DATA_DIR  # call-time read: tests patch the attr

    data_dir = Path(DATA_DIR)
    zip_path = Path(zip_path)
    read_manifest(zip_path)  # foreign zips never get this far
    restored = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                with_dir = _safe_target(data_dir, info.filename)
                with_dir.mkdir(parents=True, exist_ok=True)
                continue
            if info.filename == MANIFEST_NAME:
                continue
            target = _safe_target(data_dir, info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.parent / f".echo-restore-{os.getpid()}-{restored}.tmp"
            with zf.open(info) as src, open(temp, "wb") as dst:
                for chunk in iter(lambda: src.read(1 << 20), b""):
                    dst.write(chunk)
            os.replace(temp, target)
            _restore_mtime(target, info)
            restored += 1
    return {"restored": restored}


def list_backups(backup_dir: Path | None = None) -> list[dict]:
    """Every backup zip in ``backups/`` — newest first, validity flagged.

    Unreadable/corrupt zips are still listed (with ``valid: False``) so the
    user can see and delete them; listing never raises for a bad file.

    Args:
        backup_dir: folder to scan; ``None`` means
            ``DATA_DIR.parent / "backups"``.

    Returns:
        dicts of ``path`` (Path), ``name``, ``size_bytes``, ``created_at``
        (from the manifest, empty when unreadable), ``file_count`` and
        ``valid`` (manifest present and well-formed).
    """
    from app.config import DATA_DIR  # call-time read: tests patch the attr

    folder = (
        Path(backup_dir) if backup_dir is not None else Path(DATA_DIR).parent / "backups"
    )
    if not folder.is_dir():
        return []
    entries: list[dict] = []
    for zip_path in sorted(folder.glob(f"{ZIP_PREFIX}*.zip"), reverse=True):
        entry: dict = {
            "path": zip_path,
            "name": zip_path.name,
            "size_bytes": zip_path.stat().st_size,
            "created_at": "",
            "file_count": None,
            "valid": False,
        }
        try:
            manifest = read_manifest(zip_path)
        except (OSError, ValueError, zipfile.BadZipFile):
            entries.append(entry)  # corrupt zip: listed, flagged, not hidden
            continue
        entry["created_at"] = str(manifest["created_at"])
        entry["file_count"] = int(manifest["file_count"])
        entry["valid"] = True
        entries.append(entry)
    return entries


# ----------------------------------------------------------- startup entry


def _startup_dir() -> Path:
    """The OS folder login auto-start entries live in.

    Windows: the user's Startup FOLDER (``%APPDATA%\\...\\Startup``) — a
    plain file, never a registry hack. macOS/Linux paths are documented for
    the manual instructions (v1 writes nothing there). Tests patch this
    function to redirect the entry into a temp dir.
    """
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return (
            Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
        )
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    return Path.home() / ".config" / "autostart"


def _manual_instruction() -> str:
    """The per-platform how-to for auto-start without our installer."""
    if sys.platform == "darwin":
        return (
            "Echo v1 has no automatic startup entry on macOS — add it to Login "
            "Items manually (System Settings -> General -> Login Items) and "
            "point it at start.sh inside the Echo folder."
        )
    return (
        "Echo v1 has no automatic startup entry on this platform — create an "
        "autostart .desktop file in ~/.config/autostart that runs start.sh "
        "inside the Echo folder."
    )


def write_startup_entry() -> Path | None:
    """Install the login auto-start entry. Returns its path (Windows-only).

    Windows: writes ``echo.bat`` into the user's Startup FOLDER running
    ``.venv\\Scripts\\python.exe -m app.serve`` from the project root —
    the same thing start.bat does, headless at login. Idempotent: a second
    call overwrites the previous entry atomically (temp + ``os.replace``).
    A user-set ``ECHO_DATA_DIR`` is baked in so a relocated data folder
    survives the reboot.

    macOS / Linux: v1 writes nothing (no .desktop/.plist generation) —
    prints the manual instruction and returns ``None``. Never raises for
    the platform.

    Returns:
        The written ``echo.bat`` path on Windows, otherwise ``None``.
    """
    if sys.platform != "win32":
        print(_manual_instruction())
        return None
    from app.config import PROJECT_ROOT  # call-time: the repo root

    entry = _startup_dir() / STARTUP_ENTRY_NAME
    lines = [
        "@echo off",
        "rem Echo auto-start entry (Phase 8.3) - written by app.backup.write_startup_entry.",
        "rem Remove with: python -m app.ship uninstall-startup",
        f'cd /d "{PROJECT_ROOT}"',
    ]
    data_dir = os.environ.get("ECHO_DATA_DIR")
    if data_dir:
        lines.append(f'set "ECHO_DATA_DIR={data_dir}"')
    lines.append(f'"{PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"}" -m app.serve')
    entry.parent.mkdir(parents=True, exist_ok=True)
    temp = entry.with_name(f".echo-startup-{os.getpid()}.tmp")
    with open(temp, "w", encoding="utf-8", newline="") as handle:
        handle.write("\r\n".join(lines) + "\r\n")
    os.replace(temp, entry)
    return entry


def remove_startup_entry() -> bool:
    """Remove the login auto-start entry. True when something was removed.

    Windows: deletes ``echo.bat`` from the Startup FOLDER; absent entry ->
    ``False`` (removing twice is safe). Other platforms: v1 wrote nothing,
    so ``False`` plus the manual instruction.
    """
    if sys.platform != "win32":
        print(_manual_instruction())
        return False
    entry = _startup_dir() / STARTUP_ENTRY_NAME
    if not entry.exists():
        return False
    entry.unlink()
    return True
