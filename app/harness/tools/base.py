"""The @tool decorator (H3-1) — type hints + docstring in, ToolSpec out.

Every Echo tool is a plain Python function wearing exactly one tag::

    @tool(danger=Danger.SAFE, cost_hint=1.0)
    def read_file(path: str, offset_line: int = 1, max_lines: int = 400) -> str:
        '''Read part of a file, line-numbered.

        Longer help the model never sees.
        '''

The decorator builds the JSON schema automatically (the ADK trick — zero
boilerplate): ``inspect.signature`` says which parameters exist,
``typing.get_type_hints`` says what they are (str/int/float/bool/list/dict,
plus ``Optional[X]`` -> X + ``"nullable": true``), and a parameter without a
default becomes ``required``. The description is the docstring's FIRST
paragraph only — the model menu stays small.

``danger`` is a REQUIRED keyword (H3 contract): decorating without it raises
``RuntimeError`` immediately — untagged tools cannot exist, per H2-5's rule
lived at the seams. ``cost_hint`` (default 1.0) is the menu-pruning weight:
cheapest tools survive a small context window first (H3-3).

Opt-in context (H3-2): a final parameter annotated ``ctx: ToolContext`` marks
the tool context-aware — ``ctx`` never appears in the schema; the dispatcher
injects it at call time.

The metadata lives on the function object itself as ``fn._echo_tool`` (a
:class:`ToolMeta`); the registry refuses anything without it. No ``eval``/
``exec`` anywhere — schema building is pure inspection.
"""

from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass

from app.harness.interfaces import Danger
from app.harness.tools.context import ToolContext
from app.providers.base import ToolSpec

_UNSET = object()

#: JSON-schema type per Python annotation (checked top-down: bool before int,
#: since bool IS an int in Python).
_SIMPLE_TYPES: tuple[tuple[object, str], ...] = (
    (str, "string"),
    (bool, "boolean"),
    (int, "integer"),
    (float, "number"),
)


@dataclass(frozen=True)
class ToolMeta:
    """Everything the registry/dispatcher need to know about one tool."""

    spec: ToolSpec        # name + model-facing description + JSON schema
    danger: Danger        # SAFE / CONFIRM / BLOCK — the permission seam
    cost_hint: float      # menu-pruning weight (cheapest-first, H3-3)
    takes_ctx: bool       # the dispatcher injects ToolContext when True


def _first_paragraph(docstring: str | None) -> str:
    """The model-facing description: the docstring's first paragraph."""
    if not docstring:
        return ""
    paragraph = docstring.strip().split("\n\n", 1)[0]
    return " ".join(paragraph.split())  # collapse the indent-induced whitespace


def _schema_for(hint: object) -> dict:
    """JSON-schema fragment for one type hint (best effort, never raises).

    Unrecognized/complex hints yield ``{}`` — accept anything — because the
    dispatcher validates defensively anyway; an over-permissive schema beats
    a crash at decoration time.
    """
    if hint is inspect.Parameter.empty:
        return {}
    if hint is type(None):
        return {"type": "null"}
    origin = typing.get_origin(hint)
    if origin is typing.Union:  # covers both Optional[X] and X | None
        args = typing.get_args(hint)
        rest = [a for a in args if a is not type(None)]
        nullable = len(rest) < len(args)
        if len(rest) == 1:
            schema = _schema_for(rest[0])
        else:
            schema = {}
        if nullable and schema.get("type") != "null":
            schema = {**schema, "nullable": True}
        return schema
    if origin is list:
        args = typing.get_args(hint)
        schema: dict = {"type": "array"}
        if len(args) == 1:
            items = _schema_for(args[0])
            if items:
                schema["items"] = items
        return schema
    if origin is dict:
        return {"type": "object"}
    for python_type, json_type in _SIMPLE_TYPES:
        if hint is python_type:
            return {"type": json_type}
    if hint in (list, dict):  # bare `list`/`dict` annotations
        return {"type": "array" if hint is list else "object"}
    return {}


def build_schema(fn: object) -> tuple[dict, bool]:
    """Build ``ToolSpec.parameters`` for ``fn`` + whether it takes a ToolContext.

    Returns ``(schema, takes_ctx)``. Params WITHOUT defaults are required;
    ``ctx: ToolContext`` is excluded from the schema entirely (H3-2).
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError) as exc:  # not a plain function
        raise RuntimeError(f"@tool cannot inspect {fn!r}: {exc}") from exc
    try:
        hints = typing.get_type_hints(fn)  # resolves `from __future__` strings
    except Exception:
        hints = dict(getattr(fn, "__annotations__", {}) or {})  # degrade gracefully

    properties: dict[str, dict] = {}
    required: list[str] = []
    takes_ctx = False
    for name, param in signature.parameters.items():
        hint = hints.get(name, inspect.Parameter.empty)
        if hint is ToolContext:
            if param.default is not inspect.Parameter.empty or param.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                raise RuntimeError(
                    f"tool {getattr(fn, '__name__', fn)!r}: ctx must be a plain, "
                    "defaultless keyword-able parameter"
                )
            takes_ctx = True
            continue  # injected by the dispatcher, never part of the menu
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise RuntimeError(
                f"tool {getattr(fn, '__name__', fn)!r}: *args/**kwargs cannot be "
                "modeled in a JSON schema"
            )
        properties[name] = _schema_for(hint)
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema, takes_ctx


def tool(
    danger: Danger = _UNSET,  # type: ignore[assignment]
    *,
    cost_hint: float = 1.0,
) -> typing.Callable:
    """Tag a function as an Echo tool and generate its ToolSpec.

    ``danger`` is required (``@tool`` with no danger raises RuntimeError —
    untagged tools cannot exist). ``cost_hint`` orders the pruned menu (H3-3).
    Metadata lands on ``fn._echo_tool``; the function itself stays callable.
    """
    if danger is _UNSET or danger is None:
        raise RuntimeError(
            "@tool requires a danger classification, e.g. @tool(danger=Danger.SAFE) "
            "— untagged tools cannot exist"
        )
    if not isinstance(danger, Danger):
        raise RuntimeError(f"@tool danger must be a Danger, got {danger!r}")
    if isinstance(cost_hint, bool) or not isinstance(cost_hint, (int, float)):
        raise RuntimeError(f"@tool cost_hint must be a number, got {cost_hint!r}")
    if cost_hint < 0:
        raise RuntimeError(f"@tool cost_hint must be >= 0, got {cost_hint!r}")

    def decorate(fn: typing.Callable) -> typing.Callable:
        if not callable(fn):
            raise RuntimeError(f"@tool can only decorate callables, got {fn!r}")
        name = getattr(fn, "__name__", "")
        if not name or name == "<lambda>":
            raise RuntimeError(f"@tool requires a named function, got {fn!r}")
        schema, takes_ctx = build_schema(fn)
        spec = ToolSpec(
            name=name,
            description=_first_paragraph(inspect.getdoc(fn)),
            parameters=schema,
        )
        fn._echo_tool = ToolMeta(  # noqa: B010 — the contract's metadata slot
            spec=spec,
            danger=danger,
            cost_hint=float(cost_hint),
            takes_ctx=takes_ctx,
        )
        return fn

    return decorate
