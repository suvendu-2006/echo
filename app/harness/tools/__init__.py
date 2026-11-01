"""The tool system (H3): @tool decorator, ToolContext, registry, dispatcher, builtins.

Stage modules import from here; the frozen seam stays ``app.harness.interfaces``
(:class:`~app.harness.interfaces.ToolDispatcher` protocol). The dispatcher
additionally ships the optional danger capability the H2 loop consumes::

    dispatcher.danger_of(call: ToolCall) -> Danger    # sync, total
"""

from app.harness.interfaces import Danger
from app.harness.tools.base import ToolMeta, build_schema, tool
from app.harness.tools.context import ToolContext
from app.harness.tools.dispatcher import SPILL_PREVIEW_CHARS, ToolDispatcher, validate_arguments
from app.harness.tools.registry import ToolRegistry

__all__ = [
    "SPILL_PREVIEW_CHARS",
    "Danger",
    "ToolContext",
    "ToolDispatcher",
    "ToolMeta",
    "ToolRegistry",
    "build_schema",
    "tool",
    "validate_arguments",
]
