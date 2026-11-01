"""Echo's built-in tools (H3-5) — files (read SAFE, write CONFIRM-atomic) + shell (CONFIRM always).

Registered through the same ToolRegistry as everything else (Phase 4's MCP
tools will join them); danger classes: list_files/glob_files/read_file SAFE,
write_file/run_shell CONFIRM.
"""

from __future__ import annotations

import typing

from app.harness.tools.builtin.files import glob_files, list_files, read_file, write_file
from app.harness.tools.builtin.shell import (
    SHELL_OUTPUT_CAP,
    SHELL_TIMEOUT_SECONDS,
    run_shell,
    split_compound,
)
from app.harness.tools.registry import ToolRegistry

#: The v1 built-in catalog, in menu-display order (SAFE first).
BUILTIN_TOOLS: tuple[typing.Callable, ...] = (
    list_files,
    glob_files,
    read_file,
    write_file,
    run_shell,
)


def register_builtins(registry: ToolRegistry) -> None:
    """Convenience for wiring: register every built-in into ``registry``."""
    for fn in BUILTIN_TOOLS:
        registry.register(fn)


__all__ = [
    "BUILTIN_TOOLS",
    "SHELL_OUTPUT_CAP",
    "SHELL_TIMEOUT_SECONDS",
    "glob_files",
    "list_files",
    "read_file",
    "register_builtins",
    "run_shell",
    "split_compound",
    "write_file",
]
