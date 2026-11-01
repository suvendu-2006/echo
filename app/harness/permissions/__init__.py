"""The permission engine package (H4) — the guard: rules, hooks, modes, ask gate.

Implements the frozen ``interfaces.PermissionEngine`` protocol. Imports nothing
from the other harness stages (session_log/loop/tools are sibling stages).
"""

from app.harness.permissions.callbacks import (
    AfterHook,
    BeforeHook,
    BlockResult,
    allow_result,
)
from app.harness.permissions.engine import (
    FILE_KIND,
    SHELL_KIND,
    TOOL_KINDS,
    UNKNOWN_KIND,
    PermissionEngine,
    Rule,
    default_expand,
    default_rules,
    normalize_path,
    parse_rule,
    tool_kind,
)

__all__ = [
    "AfterHook",
    "BeforeHook",
    "BlockResult",
    "FILE_KIND",
    "PermissionEngine",
    "Rule",
    "SHELL_KIND",
    "TOOL_KINDS",
    "UNKNOWN_KIND",
    "allow_result",
    "default_expand",
    "default_rules",
    "normalize_path",
    "parse_rule",
    "tool_kind",
]
