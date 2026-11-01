"""ModelRouter (Phase 7.2) — cheap-first escalation, decided by dumb rules.

One question answered BEFORE a turn builds its chain: *which brain handles
this?* A fast (cheap) model rides everything small; a strong model takes
anything that looks like work. No extra LLM calls decide it — DUMB RULES only::

    router = ModelRouter(settings, factory_fast=..., factory_strong=...,
                         factory_default=plain_chain_factory)
    role = router.pick("what is 2+2")        # -> "fast"  (~$0 turn)
    role = router.pick(huge_organize_task)   # -> "strong"
    chain = await router.chain_for(text)     # the chain for the chosen role

``pick`` rules (all case-insensitive, in order — anything that trips one goes
strong, else fast):

1. empty (after strip) -> fast (nothing to be wrong about);
2. longer than :data:`FAST_MAX_CHARS` (240) stripped characters -> strong;
3. markdown present (a ``` fence or a line starting with ``#``) -> strong —
   formatted input is work input;
4. more than one ``?`` -> strong (a multi-question message is a task list);
5. a task verb anywhere — "organize", "delete", "send", "create", "ingest",
   "search all", "summarize all" -> strong. Imperatives are work whether they
   arrive as one line or as a multi-line list.

``chain_for(text)`` returns the chain object for ``pick(text)``'s role by
calling the matching injected factory (sync or async — an awaitable product is
awaited). A DISABLED router (``router.enabled`` in 0/false/no/off) or a missing
role factory falls back to the single plain ``factory_default()`` chain — the
same one :class:`~app.chatservice.ChatService` builds from the active provider
— so the router is pure optimization: removing it can never remove capability.
Both roles are meant to build from the same provider registry + keyring
(docs/phase7-contract.md 7.2); the factories keep this module provider-neutral.

``should_escalate(last_payloads)`` is the fast-brain retry hook (D-024): True
when the turn saw a ``system`` payload reporting status ``aborted`` — a guard
trip. The DOOR (coordinator) uses it to retry the turn ONCE on the strong
brain; this module only reports, it never re-runs anything itself.
"""

from __future__ import annotations

import inspect
import re
from typing import Any

from app.config import SettingsStore

#: Settings key that turns the whole router off ("1"/on by default).
ROUTER_ENABLED_KEY = "router.enabled"

#: Off-values for ``router.enabled`` (same family as harness/memory toggles).
_DISABLED_VALUES = ("0", "false", "no", "off")

#: The stripped-length ceiling for the fast brain (contract 7.2: <= 240 chars).
FAST_MAX_CHARS = 240

#: Task verbs that mark a turn as work (single words match on word
#: boundaries; the multi-word phrases match as substrings).
_TASK_VERB_WORDS = ("organize", "delete", "send", "create", "ingest")
_TASK_VERB_PHRASES = ("search all", "summarize all")
_TASK_VERB_RE = re.compile(
    r"\b(" + "|".join(_TASK_VERB_WORDS) + r")\b", re.IGNORECASE
)

#: The role labels ``pick`` returns.
FAST = "fast"
STRONG = "strong"


class ModelRouter:
    """Picks fast-or-strong per turn and hands back the matching chain."""

    def __init__(
        self,
        settings: SettingsStore,
        factory_fast: Any | None = None,
        factory_strong: Any | None = None,
        factory_default: Any | None = None,
    ) -> None:
        self._settings = settings
        self._factory_fast = factory_fast
        self._factory_strong = factory_strong
        self._factory_default = factory_default

    # ------------------------------------------------------------- the pick

    def pick(self, text: str) -> str:
        """``"fast"`` or ``"strong"`` — dumb rules only, never a model call.

        Empty (or non-string) input never escalates work that isn't there:
        empty -> fast; a non-string is nonsense and nonsense rides the strong
        brain (cheap-first is an optimization, strong is the safe default).
        """
        if not isinstance(text, str):
            return STRONG
        stripped = text.strip()
        if not stripped:
            return FAST
        if len(stripped) > FAST_MAX_CHARS:
            return STRONG
        if "```" in text or self._has_markdown_header(text):
            return STRONG
        if text.count("?") > 1:
            return STRONG
        if self._has_task_verb(text):
            return STRONG
        return FAST

    @staticmethod
    def _has_markdown_header(text: str) -> bool:
        """Any line that starts (after indent) with ``#`` — an ATX header."""
        return any(line.lstrip().startswith("#") for line in text.splitlines())

    @staticmethod
    def _has_task_verb(text: str) -> bool:
        lowered = text.lower()
        return bool(_TASK_VERB_RE.search(text)) or any(
            phrase in lowered for phrase in _TASK_VERB_PHRASES
        )

    # -------------------------------------------------------------- chains

    def _enabled(self) -> bool:
        value = self._settings.get(ROUTER_ENABLED_KEY)
        if value is None:
            return True  # default "1": on until the user says otherwise
        return str(value).strip().lower() not in _DISABLED_VALUES

    async def strong_chain(self) -> Any:
        """The strong role's chain, explicitly — the escalation path.

        Atticus pattern: a fast-brain turn that hits a guard retries ONCE on
        the strong brain with prior context. Echo's door calls this when the
        router is enabled and a turn aborted.
        """
        if self._factory_strong is not None:
            return await self._build(self._factory_strong)
        if self._factory_default is not None:
            return await self._build(self._factory_default)
        raise RuntimeError("no strong or default chain factory configured")

    async def chain_for(self, text: str) -> Any:
        """The chain for this turn: the picked role's factory, or the default.

        Falls back to ``factory_default()`` when the router is disabled OR the
        picked role has no factory — the plain active-provider chain, so a
        half-configured router degrades to pre-router behaviour instead of
        breaking chat. No factory at all is a wiring error: RuntimeError.
        """
        if self._enabled():
            role = self.pick(text)
            factory = self._factory_fast if role == FAST else self._factory_strong
            if factory is not None:
                return await self._build(factory)
        if self._factory_default is None:
            raise RuntimeError(
                "ModelRouter has no usable chain factory (role factory missing "
                "and no factory_default configured)"
            )
        return await self._build(self._factory_default)

    @staticmethod
    async def _build(factory: Any) -> Any:
        """Call a chain factory; await the product if the factory is async."""
        result = factory()
        if inspect.isawaitable(result):
            result = await result
        return result

    # ------------------------------------------------- the retry hook (D-024)

    def should_escalate(self, last_payloads: list[dict]) -> bool:
        """True when the turn saw a ``system`` aborted payload (a guard trip).

        Accepts both payload shapes the door sees: the flat
        ``{"type": "system", "status": "aborted"}`` dict and the loop-event
        shape ``{"type": "system", "result": {"status": "aborted", ...}}``.
        Anything else (no payloads, other types, clean finishes) -> False.
        """
        for payload in last_payloads or []:
            if not isinstance(payload, dict) or payload.get("type") != "system":
                continue
            status = payload.get("status")
            if status is None:
                result = payload.get("result")
                if isinstance(result, dict):
                    status = result.get("status")
            if status == "aborted":
                return True
        return False
