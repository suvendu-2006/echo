"""The permission engine (H4) — Echo's guard: deny always wins, fail closed.

Implements the FROZEN ``interfaces.PermissionEngine`` protocol (never edits it);
spec: docs/phase3-contract.md H4 + PLAN.md stage H4.

Rule syntax (H4-1) — ``Tool(specifier)`` strings::

    Bash(npm run:*)            shell tools: trailing ":*" = prefix match
    run_shell(echo hello)      shell tools without ":*" = exact full match
    Edit(~/docs/**)            file tools: glob; `**` crosses separators
    Delete(**)                 `*` stays within one path segment
    mcp__files__read_file(*)   MCP tools match by their mcp__server__tool name
    mcp__files__*              tool-name wildcards (`*` only, never `?`/`[...]`)

A bare ``Tool`` (no parens) means ``Tool(*)``. A specifier of exactly ``*``
always means "any arguments" for that tool — including tools of unknown kind.

Tool kinds come from ``TOOL_KINDS`` (extendable per engine via ``kind_map``),
plus one argument heuristic: any unlisted tool carrying a ``"path"`` argument is
treated as a file tool. Shell rules match ``arguments["command"]``; file rules
match ``arguments["path"]`` with Windows-first normalization (D-009): ``\\`` ==
``/``, case-insensitive, drive-letter case folded, ``~`` expanded on BOTH the
rule and the argument (inject a fake home via ``expand_root=`` or by patching
``Path.home`` — tests must never depend on the real one).

Shell compound awareness — a MATCHING AID ONLY, never a security boundary.
Commands split on top-level ``&&``/``||``/``;``/``|``/newlines (quote-aware);
an ALLOW rule matches a compound only when EVERY segment matches, while DENY
and ASK rules match when ANY segment matches. Anything opaque — ``$(...)``/
backticks anywhere, separators hidden inside quotes, unbalanced quotes —
matches no concrete specifier, so it falls through to the mode default where
CONFIRM shell always asks (D-012: the always-ask boundary).

Evaluation order (H4-2): before-hooks -> DENY rules -> ASK rules -> ALLOW rules
-> mode default -> after-hooks. Deny always wins regardless of declaration
order. Modes: "ask" (default: SAFE allow, CONFIRM ask, BLOCK deny), "yolo"
(tests only — the constructor refuses it without ``allow_yolo=True``; it
relaxes only the mode default — explicit deny/ask rules still apply),
"paranoid" (everything asks; BLOCK still denies). ``Danger.BLOCK`` denies in
every mode: interfaces.py defines BLOCK as "never dispatched", so no mode may
resurrect it. A call whose danger is not a ``Danger`` instance is denied
(fail closed) unless an explicit rule decided first.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.harness.interfaces import Danger, Decision, HarnessContext
from app.harness.permissions.callbacks import BlockResult
from app.providers.base import ToolCall

log = logging.getLogger(__name__)

SHELL_KIND = "shell"
FILE_KIND = "file"
UNKNOWN_KIND = "unknown"

# The kind map seam (H4): tool name -> kind. Extend/override per engine via
# ``PermissionEngine(kind_map={...})``; unlisted tools fall to the "path"
# argument heuristic, then to UNKNOWN_KIND (only a `*` specifier matches).
TOOL_KINDS: dict[str, str] = {
    "run_shell": SHELL_KIND,
    "Bash": SHELL_KIND,
    "read_file": FILE_KIND,
    "write_file": FILE_KIND,
    "list_files": FILE_KIND,
    "glob_files": FILE_KIND,
    "Edit": FILE_KIND,
    "Delete": FILE_KIND,
}


def tool_kind(call: ToolCall, kind_map: Mapping[str, str] | None = None) -> str:
    """Classify a call as "shell", "file" or "unknown" (decides specifier semantics)."""
    kinds = TOOL_KINDS if not kind_map else {**TOOL_KINDS, **kind_map}
    kind = kinds.get(call.name)
    if kind:
        return kind
    arguments = call.arguments
    if isinstance(arguments, dict) and "path" in arguments:
        return FILE_KIND
    return UNKNOWN_KIND


def default_expand(path: str) -> str:
    """Expand a leading ``~`` via ``Path.home()`` (monkeypatch-friendly).

    Tests either patch ``Path.home`` or inject a wholly custom ``expand_root``;
    nothing here depends on the real home directory.
    """
    if path == "~":
        return str(Path.home())
    if path.startswith("~/") or path.startswith("~\\"):
        return str(Path.home() / path[2:])
    return path


def normalize_path(path: str, expand: Callable[[str], str] | None = None) -> str:
    """Windows-first path normalization (D-009): ``~`` out, ``\\`` -> ``/``,
    lowercased (drive letters included), duplicate slashes collapsed, trailing
    slash trimmed (a drive root like ``c:/`` survives)."""
    expanded = (expand or default_expand)(path).replace("\\", "/").lower()
    unc = expanded.startswith("//")
    normalized = re.sub(r"/{2,}", "/", expanded)
    if unc:
        normalized = "/" + normalized
    while len(normalized) > 3 and normalized.endswith("/"):
        normalized = normalized[:-1]
    return normalized


@lru_cache(maxsize=512)
def _glob_regex(pattern: str) -> "re.Pattern[str]":
    """Compile a normalized glob: `**` crosses separators, `*` stays in one."""
    parts: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        if pattern[i] == "*":
            if pattern[i : i + 3] == "**/":  # zero-or-more directories: a/**/b ~ a/b
                parts.append("(?:.*/)?")
                i += 3
            elif pattern[i : i + 2] == "**":
                parts.append(".*")
                i += 2
            else:
                parts.append("[^/]*")
                i += 1
        else:
            parts.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(parts))


@lru_cache(maxsize=512)
def _tool_regex(tool_pattern: str) -> "re.Pattern[str]":
    """Compile a tool-name pattern where `*` is the only wildcard."""
    return re.compile(".*".join(re.escape(part) for part in tool_pattern.split("*")))


def _split_compound(command: str) -> list[str] | None:
    """Split a shell command into top-level segments — a matching AID only.

    Splits on ``&&``/``||``/``;``/``|``/newlines outside quotes. Returns None
    for anything opaque (command substitution, backticks, separators hidden
    inside quotes, unbalanced quotes): opaque commands match no concrete
    specifier, so they always reach the mode default and CONFIRM shell asks.
    """
    segments: list[str] = []
    buf: list[str] = []
    quote = ""
    hidden_separator = False
    i, n = 0, len(command)
    while i < n:
        c = command[i]
        if quote:
            if quote == '"' and c == "\\" and i + 1 < n and command[i + 1] in ('"', "\\", "$", "`"):
                buf.append(command[i : i + 2])
                i += 2
                continue
            if c == quote:
                quote = ""
            elif c in ";|&\n":
                hidden_separator = True  # separator smuggled inside quotes
            elif c == "$" and command[i : i + 2] == "$(":
                return None  # command substitution, even hidden in quotes
            elif c == "`":
                return None
            buf.append(c)
            i += 1
            continue
        if c in "'\"":
            quote = c
            buf.append(c)
            i += 1
            continue
        if c == "$" and command[i : i + 2] == "$(":
            return None
        if c == "`":
            return None
        if c == "\\":  # keep escapes glued to their target (paths, \; ...)
            buf.append(command[i : i + 2])
            i += 2
            continue
        if command[i : i + 2] in ("&&", "||"):
            segments.append("".join(buf).strip())
            buf = []
            i += 2
            continue
        if c in ";|&\n":
            segments.append("".join(buf).strip())
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    if quote or hidden_separator:
        return None  # unbalanced quotes / smuggled separators -> opaque
    segments.append("".join(buf).strip())
    return [segment for segment in segments if segment]


@dataclass(repr=False)
class Rule:
    """One parsed ``Tool(specifier)`` permission rule (H4-1).

    ``matches(call, kind)`` is the direct test API; ``compound`` picks the
    shell-compound semantics ("all" segments for ALLOW, "any" for DENY/ASK —
    the engine passes the right one per effect list). Matching is fail-closed:
    unknown kinds, missing path/command arguments and opaque commands all
    decline to match.
    """

    tool_pattern: str
    specifier: str
    expand_root: Callable[[str], str] = default_expand

    def __repr__(self) -> str:
        return f"Rule({self.tool_pattern}({self.specifier}))"

    def matches(
        self, call: ToolCall, kind: str | None = None, *, compound: str = "all"
    ) -> bool:
        if _tool_regex(self.tool_pattern).fullmatch(call.name) is None:
            return False
        if self.specifier == "*":
            return True  # any arguments, any kind
        resolved = kind if kind is not None else tool_kind(call)
        if resolved == SHELL_KIND:
            return self._matches_shell(call, compound)
        if resolved == FILE_KIND:
            return self._matches_path(call)
        return False  # unknown kind: only a `*` specifier matches (fail closed)

    # ------------------------------------------------------------ shell kind

    def _matches_shell(self, call: ToolCall, compound: str) -> bool:
        arguments = call.arguments if isinstance(call.arguments, dict) else {}
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return False
        prefix = self.specifier[:-2] if self.specifier.endswith(":*") else None
        segments = _split_compound(command.strip())
        if not segments:  # None (opaque) or empty
            return False
        if len(segments) == 1:
            return self._matches_text(segments[0], prefix)
        if compound == "any":  # DENY/ASK: one dangerous segment is enough
            return any(self._matches_text(segment, prefix) for segment in segments)
        return all(self._matches_text(segment, prefix) for segment in segments)

    def _matches_text(self, text: str, prefix: str | None) -> bool:
        if prefix is not None:
            return text.startswith(prefix)
        return text == self.specifier  # exact full command, no substring games

    # ------------------------------------------------------------- file kind

    def _matches_path(self, call: ToolCall) -> bool:
        arguments = call.arguments if isinstance(call.arguments, dict) else {}
        path = arguments.get("path")
        if not isinstance(path, str) or not path.strip():
            return False
        pattern = normalize_path(self.specifier, self.expand_root)
        target = normalize_path(path, self.expand_root)
        return _glob_regex(pattern).fullmatch(target) is not None


def parse_rule(rule: str, expand_root: Callable[[str], str] | None = None) -> Rule:
    """Parse a ``Tool(specifier)`` rule string (H4-1); bare ``Tool`` = ``Tool(*)``.

    Malformed rules raise ValueError immediately: a config typo must surface at
    startup, never silently match nothing (fail closed).
    """
    text = rule.strip()
    expand = expand_root or default_expand
    if "(" not in text:
        if not text or "(" in text or ")" in text:
            raise ValueError(f"malformed permission rule: {rule!r}")
        return Rule(tool_pattern=text, specifier="*", expand_root=expand)
    match = re.fullmatch(r"(?P<tool>[^()]+)\((?P<spec>.*)\)", text)
    if match is None:
        raise ValueError(f"malformed permission rule: {rule!r}")
    tool_pattern = match.group("tool").strip()
    specifier = match.group("spec")
    if not tool_pattern or not specifier.strip():
        raise ValueError(f"malformed permission rule: {rule!r}")
    return Rule(tool_pattern=tool_pattern, specifier=specifier, expand_root=expand)


def default_rules() -> list[str]:
    """Starter ASK rules (H4 / D-012): writes and shell always ask unless an
    explicit allow rule fires first; everything else falls to the mode."""
    return ["write_file(**)", "run_shell(*)"]


def _coerce_rules(
    rules: Iterable[str | Rule] | None, expand: Callable[[str], str]
) -> tuple[Rule, ...]:
    if not rules:
        return ()
    return tuple(item if isinstance(item, Rule) else parse_rule(item, expand) for item in rules)


def _hook_label(hook: Callable[..., object], index: int) -> str:
    return getattr(hook, "__name__", None) or f"{type(hook).__name__}[{index}]"


class PermissionEngine:
    """The guard (H4-2/H4-3/H4-4) — pattern rules, hooks, modes, the ask gate.

    Satisfies the frozen ``interfaces.PermissionEngine`` protocol. ``decide``
    contains no awaits (hooks are synchronous), so each decision is atomic on
    the event loop.

    Scratch seams, reset by every ``decide()`` (single-decision flow; a caller
    running decisions concurrently should snapshot them immediately):

    - ``last_rewritten_arguments``: arguments after a before-hook rewrite
      (None when no hook rewrote) — how the dispatcher/loop sees what a hook
      changed, since ``decide`` itself returns only a ``Decision``.
    - ``last_block_result``: the fake result recorded when a before-hook blocks
      via ``allow_result(...)`` — the dispatcher returns it WITHOUT running the
      tool (decision_context seam, kept simple by design).
    - ``last_reason``: human-readable reason; fodder for DENIED result hints.
    """

    def __init__(
        self,
        *,
        deny_rules: Iterable[str | Rule] | None = None,
        ask_rules: Iterable[str | Rule] | None = None,
        allow_rules: Iterable[str | Rule] | None = None,
        mode: str = "ask",
        allow_yolo: bool = False,
        before_hooks: Iterable[Callable[..., object]] = (),
        after_hooks: Iterable[Callable[..., object]] = (),
        ask_callback: Callable[..., object] | None = None,
        ask_timeout_seconds: float = 30.0,
        kind_map: Mapping[str, str] | None = None,
        expand_root: Callable[[str], str] | None = None,
    ) -> None:
        if mode not in ("ask", "yolo", "paranoid"):
            raise ValueError(f"unknown mode: {mode!r}")
        if mode == "yolo" and not allow_yolo:
            raise ValueError("mode='yolo' is tests-only; construct with allow_yolo=True")
        if isinstance(ask_timeout_seconds, bool) or not isinstance(ask_timeout_seconds, (int, float)):
            raise ValueError(f"ask_timeout_seconds must be a number, got {ask_timeout_seconds!r}")
        if ask_timeout_seconds <= 0:
            raise ValueError(f"ask_timeout_seconds must be positive, got {ask_timeout_seconds!r}")
        expand = expand_root or default_expand
        self._deny = _coerce_rules(deny_rules, expand)
        # ask_rules=None picks up the safe starter set; pass [] for rule-free engines.
        self._ask = _coerce_rules(default_rules() if ask_rules is None else ask_rules, expand)
        self._allow = _coerce_rules(allow_rules, expand)
        self._mode = mode
        self._before_hooks = tuple(before_hooks or ())
        self._after_hooks = tuple(after_hooks or ())
        self._ask_callback = ask_callback
        self._ask_timeout_seconds = float(ask_timeout_seconds)
        self._kind_map = dict(kind_map) if kind_map else None
        self.last_block_result: dict | None = None
        self.last_rewritten_arguments: dict | None = None
        self.last_reason: str = ""

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def ask_timeout_seconds(self) -> float:
        return self._ask_timeout_seconds

    # --------------------------------------------------------------- decide

    async def decide(self, call: ToolCall, danger: Danger, ctx: HarnessContext) -> Decision:
        """H4-2 evaluation order; deny always wins; fail closed everywhere."""
        self.last_block_result = None
        self.last_rewritten_arguments = None
        self.last_reason = ""
        decision, reason = self._evaluate(call, danger, ctx)
        self.last_reason = reason
        self._observe(call, danger, ctx, {"decision": decision.value, "reason": reason, "tool": call.name})
        return decision

    def _evaluate(self, call: ToolCall, danger: Danger, ctx: HarnessContext) -> tuple[Decision, str]:
        kind = tool_kind(call, self._kind_map)

        # 1) before-hooks (H4-3): rewrite arguments, block, or fail closed.
        for index, hook in enumerate(self._before_hooks):
            label = _hook_label(hook, index)
            try:
                outcome = hook(call, danger, ctx)
            except Exception as exc:
                log.warning("before-hook %s raised; fail-closed deny: %s", label, exc, exc_info=True)
                return Decision.DENY, f"before-hook {label} raised: {exc}"
            if outcome is None:
                continue
            if isinstance(outcome, BlockResult):
                self.last_block_result = dict(outcome.result)
                return Decision.DENY, f"before-hook {label} blocked (fake result recorded)"
            if isinstance(outcome, dict):
                call.arguments = outcome  # rewrite in place on the ToolCall (the seam)
                self.last_rewritten_arguments = call.arguments
                continue
            if not outcome:
                return Decision.DENY, f"before-hook {label} blocked"
            log.warning("before-hook %s misbehaved (returned %r); fail-closed deny", label, outcome)
            return Decision.DENY, f"before-hook {label} misbehaved"

        # 2) DENY -> 3) ASK -> 4) ALLOW. Deny always wins by evaluation order,
        #    no matter which list declared what.
        for rule in self._deny:
            if rule.matches(call, kind, compound="any"):
                return Decision.DENY, f"deny rule {rule}"
        for rule in self._ask:
            if rule.matches(call, kind, compound="any"):
                return Decision.ASK, f"ask rule {rule}"
        for rule in self._allow:
            if rule.matches(call, kind, compound="all"):
                return Decision.ALLOW, f"allow rule {rule}"

        # 5) mode default. BLOCK never passes, whatever the mode.
        if not isinstance(danger, Danger):
            return Decision.DENY, "unclassified danger (fail closed)"
        if danger is Danger.BLOCK:
            return Decision.DENY, "BLOCK danger is never dispatched"
        if self._mode == "paranoid":
            return Decision.ASK, "paranoid mode asks"
        if self._mode == "yolo":
            return Decision.ALLOW, "yolo mode allows"
        if danger is Danger.SAFE:
            return Decision.ALLOW, "ask mode: SAFE defaults to allow"
        return Decision.ASK, "ask mode: CONFIRM defaults to ask"

    # ----------------------------------------------------------- after-hooks

    def _observe(self, call: ToolCall, danger: Danger, ctx: HarnessContext, result: dict) -> None:
        """H4-3 after-hooks: observe the decision, never change it.

        Each hook gets its own copy of the observation, so one hook mutating it
        can neither influence the next observer nor the decision.
        """
        for index, hook in enumerate(self._after_hooks):
            label = _hook_label(hook, index)
            try:
                hook(call, danger, ctx, dict(result))
            except Exception as exc:
                log.warning("after-hook %s raised (ignored): %s", label, exc, exc_info=True)

    # ------------------------------------------------------------------ ask

    async def ask(self, call: ToolCall, danger: Danger, ctx: HarnessContext) -> Decision:
        """H4-4: delegate to the injectable async ``ask_callback`` -> bool.

        No callback, a timeout, or a callback exception all DENY (fail closed).
        Returns Decision.ALLOW/DENY only — never ASK.
        """
        if self._ask_callback is None:
            log.info("ask(%s): no callback configured; fail-closed deny", call.name)
            return Decision.DENY
        try:
            answer = await asyncio.wait_for(
                self._ask_callback(call, danger, ctx), timeout=self._ask_timeout_seconds
            )
        except Exception as exc:
            log.warning("ask(%s) failed (%s); fail-closed deny", call.name, exc, exc_info=True)
            return Decision.DENY
        return Decision.ALLOW if answer else Decision.DENY
