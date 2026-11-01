"""ToolRegistry (H3-3) — the catalog + the pruned menu for small windows.

Stores ``@tool``-decorated functions by their spec name and answers the
model-facing question: *which tools does THIS model get to see?*

``specs(context_window)`` prunes per the contract — SAFE tools first, then
CONFIRM (BLOCK tools are registered for visibility but never advertised,
never dispatched), cheapest-first (``cost_hint`` ascending, name ascending
as the deterministic tie-break) within each class, and a hard cap when the
window is small: <= 8192 tokens -> at most 8 tools, <= 32768 -> at most 20,
otherwise the whole menu (``None`` -> everything).

Registration is strict: non-``@tool`` objects raise RuntimeError (untagged
tools cannot exist), duplicate names raise RuntimeError (a silent overwrite
would swap a tool's meaning under a running session).
"""

from __future__ import annotations

import json
import typing

from app.harness.interfaces import Danger
from app.harness.tools.base import ToolMeta
from app.providers.base import ToolSpec

#: Small-window caps (H3-3): a local 8k model gets 8 tools, not 30.
_CAPS: tuple[tuple[int, int], ...] = ((8192, 8), (32768, 20))


def _window_cap(context_window: int | None) -> int | None:
    """Max menu size for a context window (None = unbounded)."""
    if context_window is None:
        return None
    for threshold, cap in _CAPS:
        if context_window <= threshold:
            return cap
    return None


def _copy_spec(spec: ToolSpec) -> ToolSpec:
    """A deep-enough copy: callers may mutate what we hand them, never ours.

    The parameters are JSON-native by construction, so a JSON round-trip is
    a bulletproof copy for the small dicts involved.
    """
    return ToolSpec(
        name=spec.name,
        description=spec.description,
        parameters=json.loads(json.dumps(spec.parameters)),
    )


class ToolRegistry:
    """Name -> tool function; danger/cost live on each fn's ``_echo_tool``."""

    def __init__(self) -> None:
        self._tools: dict[str, typing.Callable] = {}

    # ------------------------------------------------------------ registry

    def register(self, fn: typing.Callable) -> None:
        """Add one ``@tool``-decorated function; rejects anything else."""
        meta = getattr(fn, "_echo_tool", None)
        if not isinstance(meta, ToolMeta):
            raise RuntimeError(
                f"{fn!r} is not a @tool-decorated function (missing _echo_tool) "
                "— untagged tools cannot be registered"
            )
        name = meta.spec.name
        if name in self._tools:
            raise RuntimeError(f"duplicate tool name: {name!r} is already registered")
        self._tools[name] = fn

    def get(self, name: str) -> typing.Callable:
        """The tool function for ``name``; RuntimeError when unknown."""
        try:
            return self._tools[name]
        except KeyError:
            raise RuntimeError(f"unknown tool: {name!r}") from None

    def danger_of(self, name: str) -> Danger:
        """The registered danger class of ``name``; RuntimeError when unknown."""
        return self.get(name)._echo_tool.danger

    def all(self) -> list[typing.Callable]:
        """Every registered function (all danger classes), registration order."""
        return list(self._tools.values())

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    # ---------------------------------------------------------------- menu

    def specs(self, context_window: int | None = None) -> list[ToolSpec]:
        """The pruned tool menu (H3-3): SAFE first, then CONFIRM, never BLOCK.

        Cheapest-first within each danger class; capped for small windows
        (<=8192 -> 8 tools, <=32768 -> 20, otherwise all). Spec copies are
        returned — the menu a caller mutates is theirs, not the registry's.
        """
        safe: list[ToolMeta] = []
        confirm: list[ToolMeta] = []
        for fn in self._tools.values():
            meta: ToolMeta = fn._echo_tool
            if meta.danger is Danger.SAFE:
                safe.append(meta)
            elif meta.danger is Danger.CONFIRM:
                confirm.append(meta)
            # Danger.BLOCK: registered for visibility, never advertised.
        key = lambda meta: (meta.cost_hint, meta.spec.name)  # noqa: E731
        ordered = sorted(safe, key=key) + sorted(confirm, key=key)
        cap = _window_cap(context_window)
        if cap is not None:
            ordered = ordered[:cap]
        return [_copy_spec(meta.spec) for meta in ordered]
