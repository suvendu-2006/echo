"""Chat orchestration (PLAN Phase 2.5 + Phase 3 H7-3) — the door to Echo.

One method does all of the talking::

    service.stream_chat(text, session_id)  ->  async iterator of payload dicts

yielding exactly the WebSocket payload shapes from docs/phase2-contract.md:
``status: thinking`` -> (session + user message persisted) -> ``status:
streaming`` on the first text delta -> one ``delta`` per chunk -> the
assistant message persisted -> ``final``. Provider failures of every kind
collapse to a SINGLE generic ``error`` dict — the details go to the
``echo.server`` log, never to the client (SEC-2).

**The harness door (Phase 3, H7-3):** when ``use_harness`` is on (the
``harness.enabled`` setting — the server default), the same method routes
through :class:`app.harness.loop.AgentLoop` — the sacred log, the tool
desk, the permission engine, compaction, tracing. Payloads stay identical
for tool-free chats; two additive kinds appear when tools are involved:
``tool`` / ``tool_done`` (activity lines) and ``permission`` (the H4-4
ask popup, answered by the server routing ``permission_answer`` frames to
:meth:`ChatService.answer_permission`). Voice and the scheduler will enter
through this same door. Dual-write note (D-020): user/assistant messages
are persisted to BOTH the Phase 2 SessionStore (UI sessions API) and the
sacred log; the log is authoritative.

``interrupt()`` asks the running stream to stop; the partial reply is kept
(never thrown away) and the stream finishes with ``final`` + ``status:
idle``.

The service is provider-neutral: it speaks only ``adapter.stream(request)``
(app/providers/base.py, DECISIONS D-002) and never imports a concrete
adapter at module level. The default chain factory is built lazily per chat
so a settings change takes effect on the very next message; tests inject a
MockProvider-based factory and touch no network.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any, AsyncIterator, Callable

from app.config import SecretStore, SettingsStore
from app.providers.base import (
    Message,
    ProviderError,
    ProviderUnavailable,
    Request,
)
from app.store import SessionStore

LOGGER = logging.getLogger("echo.server")

#: Contract cap: the largest chat text a client may send.
MAX_TEXT_CHARS = 32000

#: Settings keys (same SQLite key/value store as Phase 0).
ACTIVE_PROVIDER_KEY = "active.provider"
ACTIVE_MODEL_KEY = "active.model"
FAILOVER_ORDER_KEY = "failover.order"
HARNESS_ENABLED_KEY = "harness.enabled"

#: The ONLY provider-failure text a client ever sees. Every concrete reason
#: (dead endpoint, bad key, bad model) stays in the server log.
PROVIDER_FAILED_MESSAGE = "the provider failed — try again or switch model"

#: The only turn-aborted text a client ever sees (non-provider aborts).
TURN_ABORTED_MESSAGE = "the turn was stopped by a safety guard"

#: Length cap for chat-derived session titles.
_TITLE_CHARS = 60


class PermissionBroker:
    """Routes the permission engine's ask-events to the UI and back (H4-4).

    ``open`` registers a pending request and returns its WS payload;
    ``answer`` resolves it from the server's WS receive loop; ``wait`` is
    the engine-side await — fail-closed on timeout or interrupt, exactly
    like the engine's own policy.
    """

    def __init__(self, interrupt: asyncio.Event) -> None:
        self._interrupt = interrupt
        self._pending: dict[str, asyncio.Future[bool]] = {}

    def open(self, tool_name: str, arguments: dict) -> tuple[str, dict]:
        import uuid

        request_id = uuid.uuid4().hex
        self._pending[request_id] = asyncio.get_running_loop().create_future()
        return request_id, {
            "type": "permission",
            "id": request_id,
            "tool": tool_name,
            "arguments": arguments,
        }

    def answer(self, request_id: str, allow: bool) -> None:
        future = self._pending.pop(request_id, None)
        if future is not None and not future.done():
            future.set_result(bool(allow))

    async def wait(self, request_id: str, timeout: float) -> bool:
        """Await the user's answer — fail closed on timeout or interrupt.

        RT-1 (red-team finding): cancelling our await wrapper also cancels the
        future itself, and ``future.result()`` on a cancelled future raises
        CancelledError — a BaseException that sailed past every fail-closed
        net and killed the turn mid-popup. The fix: never touch a cancelled
        future's result; pop first, cancel only when unfinished, return False.
        """
        future = self._pending.get(request_id)
        if future is None:
            return False
        answer_task = asyncio.ensure_future(future)
        interrupt_task = asyncio.ensure_future(self._interrupt.wait())
        done, pending = await asyncio.wait(
            {answer_task, interrupt_task},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        if answer_task in done and answer_task.exception() is None:
            self._pending.pop(request_id, None)
            return bool(answer_task.result())
        # Timeout or interrupt: fail closed (H4's policy, mirrored here).
        self._pending.pop(request_id, None)
        if not future.done():
            future.cancel()
        return False


class _LoopEngineView:
    """The loop's engine view: ASK defers to the tool runner (single popup).

    The H2 loop escalates ASK→engine.ask before dispatching, and the H3
    dispatcher does the same inside dispatch() — both owning escalation
    means every popup fires twice (caught live in door testing). The door's
    loop gets this view instead: DENY rules still deny, but an ASK verdict
    is passed through as ALLOW so the DISPATCHER's single ask owns the
    popup. Standalone dispatcher users keep the real engine.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    async def decide(self, call, danger, ctx):
        decision = await self._engine.decide(call, danger, ctx)
        from app.harness.interfaces import Decision as HarnessDecision

        return HarnessDecision.ALLOW if decision is HarnessDecision.ASK else decision

    async def ask(self, call, danger, ctx):
        return await self._engine.ask(call, danger, ctx)


class ChatService:
    """Turns one user utterance into a persisted, streamed conversation turn."""

    def __init__(
        self,
        settings: SettingsStore,
        secrets: SecretStore,
        store: SessionStore,
        chain_factory: Callable[[], Any] | None = None,
        use_harness: bool = False,
    ) -> None:
        self._settings = settings
        self._secrets = secrets
        self._store = store
        self._chain_factory = (
            chain_factory if chain_factory is not None else self._default_chain_factory
        )
        self._interrupt = asyncio.Event()
        # Phase 3 H7-3: when True, chats flow through the harness door
        # (AgentLoop.run_turn — the single door voice and the scheduler use
        # too). Default False keeps the Phase 2 pinned behavior; the server
        # flips it on from the harness.enabled setting.
        self._use_harness = bool(use_harness)
        self._broker: PermissionBroker | None = None
        self._sacred_log: Any = None
        self._trace_sink: Any = None
        self._registry: Any = None
        self._memory_store: Any = None
        self._rag_library_instance: Any = None

    # ------------------------------------------------------------- control

    def interrupt(self) -> None:
        """Ask the running stream (if any) to stop after its current event."""
        self._interrupt.set()

    def close(self) -> None:
        """Release the service's long-lived resources (sacred + memory logs).

        Windows locks open SQLite files (exam bugs #1–#3), so anything that
        owns a ChatService must close it on shutdown — create_app's lifespan
        does; tests do it in cleanup.
        """
        for resource in (
            self._sacred_log,
            self._trace_sink,
            self._memory_store,
            getattr(self, "_rag_library_instance", None),
            getattr(self, "_skills_store", None),
        ):
            if resource is None:
                continue
            closer = getattr(resource, "close", None)
            if closer is not None:
                with contextlib.suppress(Exception):
                    closer()
        self._sacred_log = None
        self._trace_sink = None
        self._memory_store = None

    def answer_permission(self, request_id: str, allow: bool) -> None:
        """Public WS entry: resolve a pending permission popup (H4-4)."""
        broker = self._broker
        if broker is not None:
            broker.answer(request_id, allow)

    # -------------------------------------------------------------- stream

    async def stream_chat(
        self, text: str, session_id: str | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream one chat turn as contract payload dicts (see module doc).

        Validation failures yield exactly one ``error`` dict and touch
        nothing. Provider failures yield exactly one generic ``error`` dict;
        the details are logged. An interrupted stream saves whatever content
        arrived and still ends with ``final`` + ``status: idle``.
        """
        stripped = text.strip() if isinstance(text, str) else ""
        if not stripped:
            yield {"type": "error", "message": "text must not be empty"}
            return
        if len(text) > MAX_TEXT_CHARS:
            yield {
                "type": "error",
                "message": f"text is too long (max {MAX_TEXT_CHARS} characters)",
            }
            return

        if self._use_harness:
            # Phase 3 H7-3: the single door — same WS contract payloads,
            # powered by the harness (sacred log, tools, permissions).
            async for payload in self._stream_turn_harness(text, session_id):
                yield payload
            return

        self._interrupt.clear()
        yield {"type": "status", "state": "thinking"}

        if session_id is None:
            session_id = self._store.create_session(title=stripped[:_TITLE_CHARS])["id"]
        elif self._store.get_session(session_id) is None:
            yield {"type": "error", "message": "unknown session"}
            return
        self._store.append_message(session_id, "user", text)

        request = Request(
            messages=[
                Message(role=message["role"], content=message["content"])
                for message in self._store.get_messages(session_id)
            ]
        )

        try:
            chain = self._chain_factory()
        except Exception:
            LOGGER.exception("failed to build the provider chain")
            yield {"type": "error", "message": PROVIDER_FAILED_MESSAGE}
            return

        parts: list[str] = []
        streaming = False
        interrupted = False
        gen = chain.stream(request)
        try:
            async for event in gen:
                if self._interrupt.is_set():
                    interrupted = True
                    break
                if event.type == "text":
                    if not streaming:
                        yield {"type": "status", "state": "streaming"}
                        streaming = True
                    parts.append(event.text)
                    yield {"type": "delta", "text": event.text}
                elif event.type == "error":
                    # The adapter's own failure report (base.py contract):
                    # terminal, never a failover signal — one generic dict out.
                    LOGGER.warning("provider stream failed: %s", event.error)
                    yield {"type": "error", "message": PROVIDER_FAILED_MESSAGE}
                    return
                elif event.type == "end":
                    break
                # "tool_call"/"usage" — the legacy path is text chat; ignored.
        except (ProviderUnavailable, ProviderError) as exc:
            LOGGER.warning(
                "provider raised mid-stream: %s: %s", type(exc).__name__, exc
            )
            yield {"type": "error", "message": PROVIDER_FAILED_MESSAGE}
            return
        finally:
            closer = getattr(gen, "aclose", None)
            if closer is not None:
                with contextlib.suppress(Exception):
                    await closer()

        assistant = self._store.append_message(session_id, "assistant", "".join(parts))
        yield {"type": "final", "session_id": session_id, "message": assistant}
        if interrupted:
            yield {"type": "status", "state": "idle"}

    # ------------------------------------------------------- default chain

    def _default_chain_factory(self) -> Any:
        """Build the real chain from settings + the OS keychain (D-013).

        Lazy imports everywhere: this runs only when an un-injected chat
        starts, never at module import, and concrete adapters stay out of
        this module's import graph. A missing provider/model config raises
        ProviderConfigError, which stream_chat folds into the generic error.
        """
        from app.providers.failover import ProviderChain
        from app.providers.ollama import ollama_compat_provider
        from app.providers.openai_compat import OpenAICompatProvider

        active = self._settings.get(ACTIVE_PROVIDER_KEY) or "ollama"
        model = self._settings.get(ACTIVE_MODEL_KEY) or ""
        try:
            order = json.loads(self._settings.get(FAILOVER_ORDER_KEY) or "[]")
        except ValueError:
            order = []
        if not isinstance(order, list):
            order = []

        from app.server import KNOWN_PROVIDERS  # lazy: avoids the import cycle

        names: list[str] = []
        for name in [active, *(n for n in order if isinstance(n, str))]:
            if name in KNOWN_PROVIDERS and name not in names:
                names.append(name)

        adapters: list[Any] = []
        for name in names:
            if name == "ollama":
                if not model:
                    continue
                adapters.append(ollama_compat_provider(model))
                continue
            base_url = (
                self._settings.get(f"providers.{name}.base_url")
                or KNOWN_PROVIDERS[name]
            )
            if not base_url or not model:
                continue
            adapters.append(
                OpenAICompatProvider(
                    name=name,
                    model=model,
                    base_url=base_url,
                    api_key=self._secrets.get_key(name) or "",
                )
            )
        if not adapters:
            raise ProviderError(
                "no usable provider/model configured — pick one in settings"
            )
        return ProviderChain(adapters)

    def _memory(self) -> tuple[Any, Any, Any] | None:
        """(store, wallet, relevant-adapter) — None when memory is disabled.

        Singleton per service; the wallet's render block rides into every
        parent request via the pipeline's memory seam (H5-2), and the
        memory_search tool joins the tool desk.
        """
        if (self._settings.get("memory.enabled") or "1").strip().lower() in (
            "0",
            "false",
            "no",
            "off",
        ):
            return None
        cached = getattr(self, "_memory_tuple", None)
        if cached is not None:
            return cached
        with contextlib.suppress(Exception):
            from app.memory import MemoryStore, Wallet, as_relevant_adapter

            store = getattr(self, "_memory_store", None)
            if store is None:
                store = MemoryStore()
                self._memory_store = store
            wallet = Wallet(store)
            self._memory_tuple = (store, wallet, as_relevant_adapter(store))
            return self._memory_tuple
        return None

    async def _distill_turn(self, session_id: str, user_text: str, reply: str) -> None:
        """Fire-and-forget Phase 6.3: file what this turn taught Echo."""
        with contextlib.suppress(Exception):
            from app.memory.distiller import Distiller

            memory_tuple = self._memory()
            if memory_tuple is None:
                return
            transcript = f"User: {user_text}\nEcho: {reply}"
            distiller = Distiller(memory_tuple[0])
            await distiller.distill_session(session_id, transcript)

    def _router(self) -> Any | None:
        """Phase 7.2 cheap-first router (None when disabled or unconfigured)."""
        cached = getattr(self, "_router_instance", None)
        if cached is not None:
            return cached
        with contextlib.suppress(Exception):
            from app.router import ROUTER_ENABLED_KEY, ModelRouter

            flag = (self._settings.get(ROUTER_ENABLED_KEY) or "1").strip().lower()
            if flag in ("0", "false", "no", "off"):
                return None

            def build_role(provider_key: str, model_key: str):
                def factory() -> Any:
                    from app.providers.openai_compat import OpenAICompatProvider
                    from app.server import KNOWN_PROVIDERS

                    provider = self._settings.get(provider_key) or ""
                    model = self._settings.get(model_key) or ""
                    base_url = (
                        self._settings.get(f"providers.{provider}.base_url")
                        or KNOWN_PROVIDERS.get(provider, "")
                    )
                    if not provider or not model or not base_url:
                        raise ProviderError("router role not configured")
                    return OpenAICompatProvider(
                        name=provider,
                        model=model,
                        base_url=base_url,
                        api_key=self._secrets.get_key(provider) or "",
                    )

                return factory

            router = ModelRouter(
                self._settings,
                factory_fast=build_role("router.fast_provider", "router.fast_model"),
                factory_strong=build_role(
                    "router.strong_provider", "router.strong_model"
                ),
                factory_default=self._chain_factory,
            )
            self._router_instance = router
            return router
        return None

    async def run_task(self, task: str, job_name: str = "") -> str:
        """One full agent turn for the scheduler (Phase 7.3) — the door.

        Task text in, final assistant text out. Never raises: failures come
        back as "failed: ..." strings the job row can store.
        """
        content = ""
        previous = self._use_harness
        self._use_harness = True  # scheduled tasks always take the full door
        try:
            async for payload in self.stream_chat(task, None):
                kind = payload.get("type")
                if kind == "final":
                    content = (payload.get("message") or {}).get("content", "")
                elif kind == "error":
                    return f"failed: {payload.get('message', '')}"
        except Exception as exc:
            return f"failed: {exc!r}"
        finally:
            self._use_harness = previous
        return content or "done"

    def memory_access(self) -> tuple[Any, Any] | None:
        """(store, wallet) for the memory-manager API — None when disabled."""
        memory_tuple = self._memory()
        if memory_tuple is None:
            return None
        return memory_tuple[0], memory_tuple[1]

    # ------------------------------------------------- integrations (Phase 4)

    async def _wire_integrations(self, registry: Any) -> None:
        """Gmail tools + MCP servers join the tool desk (PLAN 4.2/4.3).

        Best-effort by design: a dead MCP server or unconnected Gmail is
        normal life — failures log and move on, the chat never blocks.
        """
        with contextlib.suppress(Exception):
            from app.mcp.gmail import GmailAccount, build_gmail_tools

            account = GmailAccount(self._secrets)
            if account.is_connected() and not getattr(self, "_gmail_wired", False):
                for gmail_tool in build_gmail_tools(account):
                    with contextlib.suppress(Exception):
                        registry.register(gmail_tool)
                self._gmail_wired = True
        manager = await self._ensure_mcp_manager()
        if manager is not None:
            with contextlib.suppress(Exception):
                await manager.connect_all()
                await manager.refresh_tools(registry)
        # Phase 6.5: the RAG library's tools (search/ask/ingest/documents)
        with contextlib.suppress(Exception):
            from app.rag.embedder import HashEmbedder, MiniLMEmbedder
            from app.rag.tools import build_rag_tools

            # MiniLM fails LAZILY (at embed time), so probe availability now
            # and pick the embedder that will actually work in this process.
            try:
                import sentence_transformers  # noqa: F401 — availability probe

                embedder: Any = MiniLMEmbedder()  # real neural embeddings
            except Exception:
                embedder = HashEmbedder()  # graceful: lexical-only retrieval
            rag_tools = await build_rag_tools(self._rag_library(), embedder)
            for rag_tool in rag_tools:
                with contextlib.suppress(Exception):
                    registry.register(rag_tool)
        # Phase 7.4: run_script + save_skill + every enabled learned skill
        with contextlib.suppress(Exception):
            from app.skills import (
                SkillsStore,
                build_skill_tools,
                run_script,
                save_skill,
            )

            skills_store = getattr(self, "_skills_store", None)
            if skills_store is None:
                skills_store = SkillsStore()
                self._skills_store = skills_store
            registry.register(run_script)
            registry.register(save_skill)
            for skill_tool in build_skill_tools(skills_store):
                with contextlib.suppress(Exception):
                    registry.register(skill_tool)

    def _rag_library(self) -> Any:
        """The RAG library singleton (its directory is created lazily)."""
        library = getattr(self, "_rag_library_instance", None)
        if library is None:
            from app.rag.ingest import RagLibrary

            library = RagLibrary()
            self._rag_library_instance = library
        return library

    async def _ensure_mcp_manager(self) -> Any | None:
        manager = getattr(self, "_mcp_manager", None)
        if manager is not None:
            return manager
        with contextlib.suppress(Exception):
            from app.mcp.manager import McpManager, load_server_configs

            configs = load_server_configs()
            if configs:
                manager = McpManager(configs)
                self._mcp_manager = manager
        return manager

    def _efficient(self, dispatcher: Any) -> Any:
        """Wrap the dispatcher with the Phase 4.4 cache + squeezer ($0 budget)."""
        from app.harness.interfaces import Danger
        from app.mcp.efficiency import Squeezer, ToolResultCache

        cache = getattr(self, "_result_cache", None)
        if cache is None:
            cache = ToolResultCache()
            self._result_cache = cache
        squeezer = getattr(self, "_squeezer", None)
        if squeezer is None:
            squeezer = Squeezer()
            self._squeezer = squeezer

        class _EfficientDispatcher:
            def __init__(self, inner: Any) -> None:
                self._inner = inner

            def specs(self, context_window: int | None = None) -> list:
                return self._inner.specs(context_window)

            def danger_of(self, call):
                helper = getattr(self._inner, "danger_of", None)
                if helper is None:
                    return Danger.CONFIRM
                try:
                    return Danger(helper(call))
                except Exception:
                    return Danger.CONFIRM

            async def dispatch(self, call, ctx) -> dict:
                danger = self.danger_of(call)
                key = None
                if danger is Danger.SAFE:
                    try:
                        key = (
                            call.name,
                            json.dumps(call.arguments, sort_keys=True, default=str),
                        )
                        hit = cache.get(key)
                        if hit is not None:
                            return {
                                "status": "OK",
                                "cached": True,
                                "result": hit,
                            }
                    except Exception:
                        key = None  # cache must never break a dispatch
                result = await self._inner.dispatch(call, ctx)
                if (
                    key is not None
                    and danger is Danger.SAFE
                    and result.get("status") == "OK"
                ):
                    squeezed = squeezer.squeeze(str(result.get("result", "")))
                    result["result"] = squeezed
                    cache.put(key, squeezed)
                return result

            async def dispatch_many(self, calls, ctx) -> list:
                inner_many = getattr(self._inner, "dispatch_many", None)
                if inner_many is None:
                    return [await self.dispatch(call, ctx) for call in calls]
                return await inner_many(calls, ctx)

        return _EfficientDispatcher(dispatcher)

    def integrations_status(self) -> dict[str, Any]:
        """UI snapshot: Gmail connection + MCP server states (never secrets)."""
        gmail: dict[str, Any] = {"connected": False}
        with contextlib.suppress(Exception):
            from app.mcp.gmail import GmailAccount

            account = GmailAccount(self._secrets)
            creds = account.credentials() if hasattr(account, "credentials") else None
            gmail = {
                "connected": account.is_connected(),
                "email": (creds or {}).get("email", ""),
            }
        mcp: list[dict[str, Any]] = []
        manager = getattr(self, "_mcp_manager", None)
        if manager is not None:
            with contextlib.suppress(Exception):
                mcp = manager.status()
        return {"gmail": gmail, "mcp": mcp}

    def disconnect_gmail(self) -> None:
        with contextlib.suppress(Exception):
            from app.mcp.gmail import GmailAccount

            GmailAccount(self._secrets).disconnect()

    async def connect_gmail(self) -> dict[str, Any]:
        """Run the OAuth loopback flow using the user's own OAuth client."""
        from app.mcp.gmail import GmailAccount, GoogleOAuthFlow

        client_id = self._secrets.get_key("gmail.client_id")
        client_secret = self._secrets.get_key("gmail.client_secret")
        if not client_id or not client_secret:
            return {
                "ok": False,
                "error": "set gmail.client_id and gmail.client_secret first "
                "(your own Google Cloud OAuth client — see app/mcp/gmail.py)",
            }
        flow = GoogleOAuthFlow(client_id, client_secret)
        result = await flow.connect()
        if not result.get("ok", True) and result.get("error"):
            return {"ok": False, "error": result["error"]}
        GmailAccount(self._secrets).save_credentials(result)
        self._gmail_wired = False  # re-register tools on the next turn
        return {"ok": True, "email": result.get("email", "")}

    async def refresh_mcp(self) -> dict[str, Any]:
        manager = await self._ensure_mcp_manager()
        if manager is None:
            return {"mcp": []}
        with contextlib.suppress(Exception):
            await manager.connect_all()
            if self._registry is not None:
                await manager.refresh_tools(self._registry)
        return {"mcp": manager.status()}

    # ================================================ the harness door (H7-3)

    async def _stream_turn_harness(
        self, text: str, session_id: str | None
    ) -> AsyncIterator[dict[str, Any]]:
        """One chat turn through the harness (sacred log → loop → tools).

        WS payloads are IDENTICAL to the Phase 2 contract for tool-free
        chats, plus additive kinds: ``tool`` / ``tool_done`` (activity) and
        ``permission`` (the ask popup, answered via ``answer_permission``).
        Interrupt works; partial replies are kept and still end in ``final``.
        """
        from app.harness.context import make_ctx
        from app.harness.ctx.budget import EstimateCounter, WindowRegistry
        from app.harness.ctx.compact import CompactEngine, ExtractiveSummarizer
        from app.harness.ctx.processors import RequestPipeline
        from app.harness.loop import AgentLoop
        from app.harness.permissions.engine import PermissionEngine, default_rules
        from app.harness.session_log import SessionLog
        from app.harness.subagents import SubagentRunner, task as task_tool, wire_state
        from app.harness.tools.builtin import register_builtins
        from app.harness.tools.dispatcher import ToolDispatcher
        from app.harness.tools.registry import ToolRegistry
        from app.harness.tracing import tracer_from_settings

        self._interrupt.clear()
        yield {"type": "status", "state": "thinking"}

        if session_id is None:
            session_id = self._store.create_session(title=text[:_TITLE_CHARS])["id"]
        elif self._store.get_session(session_id) is None:
            yield {"type": "error", "message": "unknown session"}
            return
        # Dual-write during the transition (D-020): SessionStore feeds the
        # UI's sessions API; the sacred log is the authoritative record.
        self._store.append_message(session_id, "user", text)

        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        broker = PermissionBroker(self._interrupt)
        self._broker = broker

        # One sacred log + one tool registry + one trace sink per service
        # (lazy singletons). The tracer MUST be cached here, not built per
        # turn: make_ctx registers it on the log (add_trace), and a fresh
        # tracer every turn would pin ~2 KB forever per turn — the leak the
        # Phase 8.3 soak caught (exam bug #7).
        log = self._sacred_log
        if log is None:
            from app.config import DATA_DIR

            log = SessionLog(DATA_DIR / "harness.db")
            self._sacred_log = log
        trace = self._trace_sink
        if trace is None:
            trace = tracer_from_settings(self._settings, self._secrets)
            self._trace_sink = trace
        registry = self._registry
        if registry is None:
            registry = ToolRegistry()
            register_builtins(registry)
            self._registry = registry
        await self._wire_integrations(registry)

        model = self._settings.get(ACTIVE_MODEL_KEY) or ""
        window = WindowRegistry().window_for(model or "default")
        counter = EstimateCounter()

        def build_pipeline() -> RequestPipeline:
            return RequestPipeline(
                window,
                counter,
                compact_engine=CompactEngine(
                    log, counter, ExtractiveSummarizer(), window
                ),
            )

        ask_timeout = float(self._settings.get("permissions.ask_timeout") or 30)

        async def ask_callback(call, danger, ctx) -> bool:
            # The engine expects a BOOL answer (it maps bool -> ALLOW/DENY).
            # Returning a Decision enum here would be truthy even when DENY —
            # a deny would silently become an allow (caught by the door exam).
            request_id, payload = broker.open(call.name, dict(call.arguments))
            await queue.put(payload)
            return await broker.wait(request_id, ask_timeout)

        engine = PermissionEngine(
            ask_rules=default_rules(),
            # RT-2 hardening (red-team finding): secrets never enter the
            # model's context — reads of credential files are DENIED outright,
            # no popup (there is no legitimate reason for the model to read
            # them; the keychain already holds the keys).
            deny_rules=[
                "read_file(**/.env)",
                "read_file(**/.env.*)",
                "write_file(**/.env)",
            ],
            mode="ask",
            ask_callback=ask_callback,
        )
        dispatcher = self._efficient(
            ToolDispatcher(registry, engine=engine, log=log, state={})
        )
        # Phase 6 memory: wallet facts ride into every parent request via the
        # H5 pipeline's memory seam; memory_search joins the tool desk. Child
        # subagent pipelines stay memory-less (context firewalls).
        memory_tuple = self._memory()
        pipeline = RequestPipeline(
            window,
            counter,
            memory=(memory_tuple[2] if memory_tuple else None),
            compact_engine=CompactEngine(
                log, counter, ExtractiveSummarizer(), window
            ),
        )
        if memory_tuple is not None:
            with contextlib.suppress(Exception):
                from app.memory.wallet import memory_search_tool_factory

                memory_tool = await memory_search_tool_factory(memory_tuple[0])
                registry.register(memory_tool)

        runner = SubagentRunner(
            log=log,
            llm_factory=self._chain_factory,
            pipeline_factory=build_pipeline,
            engine=engine,
        )
        # Phase 7.2: cheap-first routing — small talk rides the fast brain,
        # heavy tasks the strong one (falls back to the default chain when
        # the router is disabled or roles are unconfigured).
        turn_chain = self._chain_factory()
        router = self._router()
        if router is not None:
            with contextlib.suppress(Exception):
                turn_chain = await router.chain_for(text) or turn_chain
        ctx = make_ctx(
            session_id=session_id,
            llm=turn_chain,
            log=log,
            tools=dispatcher,
            perms=engine,
            trace=trace,
        )
        wire_state(ctx, runner)  # the CONFIRM "task" tool finds the runner here
        skills_store = getattr(self, "_skills_store", None)
        if skills_store is not None:
            with contextlib.suppress(Exception):
                from app.skills import wire_skills_state

                wire_skills_state(ctx, skills_store)
        with contextlib.suppress(Exception):
            registry.register(task_tool)

        def build_loop(llm: Any) -> AgentLoop:
            loop_ctx = make_ctx(
                session_id=session_id,
                llm=llm,
                log=log,
                tools=dispatcher,
                perms=engine,
                trace=self._trace_sink,
            )
            wire_state(loop_ctx, runner)
            return AgentLoop(
                loop_ctx,
                pipeline,
                dispatcher,
                _LoopEngineView(engine),  # the dispatcher owns the single ask popup
                on_delta=lambda delta: queue.put({"type": "delta", "text": delta}),
            )

        parts: list[str] = []
        state: dict[str, Any] = {"aborted": None, "terminal": ""}

        async def run_loop(escalate: bool) -> None:
            """One conversation pass — escalating ONCE on the strong brain
            when the pass aborts with nothing to show (Atticus pattern)."""
            llm: Any = turn_chain
            try:
                while True:
                    aborted = False
                    try:
                        async for event in build_loop(llm).run_turn(text):
                            if event.type == "tool_call" and event.tool_call is not None:
                                await queue.put({"type": "status", "state": "tool_running"})
                                await queue.put(
                                    {
                                        "type": "tool",
                                        "name": event.tool_call.name,
                                        "arguments": event.tool_call.arguments,
                                    }
                                )
                            elif event.type == "tool_result":
                                call = event.meta.get("for_call")
                                await queue.put(
                                    {
                                        "type": "tool_done",
                                        "name": call.name if call else "?",
                                        "status": (event.result or {}).get("status", ""),
                                    }
                                )
                            elif event.type == "assistant":
                                state["terminal"] = event.content
                            elif event.type == "system":
                                result = event.result or {}
                                if result.get("status") == "aborted":
                                    state["aborted"] = str(result.get("reason", ""))
                                    aborted = True
                    except Exception as exc:  # provider explosion = a clean abort
                        state["aborted"] = f"provider error: {exc}"
                        aborted = True
                    # Phase 9 (Atticus pattern): escalate ONCE on the strong brain
                    # when the pass aborted with nothing to show. Content, user
                    # interrupt, or a second attempt never escalate.
                    if not aborted or state["terminal"] or "".join(parts) or self._interrupt.is_set():
                        return
                    strong: Any = None
                    if escalate and router is not None:
                        with contextlib.suppress(Exception):
                            strong = await router.strong_chain()
                    if strong is None or strong is llm:
                        return
                    await queue.put({"type": "status", "state": "thinking"})
                    escalate = False
                    llm = strong
            finally:
                # The sentinel ends the drain loop — ALWAYS, even on an
                # unexpected exception (exam bug #8: the escalation rewrite
                # dropped it and the door hung forever after the first turn).
                await queue.put(None)

        loop_task = asyncio.create_task(run_loop(True))
        streaming = False
        interrupted = False
        try:
            while True:
                if self._interrupt.is_set():
                    interrupted = True
                    break
                item = await queue.get()
                if item is None:
                    break
                kind = item.get("type")
                if kind == "delta":
                    if not streaming:
                        yield {"type": "status", "state": "streaming"}
                        streaming = True
                    parts.append(item.get("text", ""))
                yield item
        finally:
            if loop_task.done() and loop_task.exception() is not None:
                LOGGER.exception("harness loop task crashed")
            if not loop_task.done():
                loop_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await loop_task
            self._broker = None

        content = "".join(parts) or state["terminal"]
        if state["aborted"] is not None and not content:
            # A clean abort with nothing to show: the failure taxonomy picks
            # the specific human message (never a stack trace, SEC-2).
            LOGGER.warning("harness turn aborted: %s", state["aborted"])
            from app.providers.errors import explain

            reason = state["aborted"]
            yield {
                "type": "error",
                "message": (
                    explain(reason)
                    if "provider" in reason.lower()
                    else TURN_ABORTED_MESSAGE
                ),
            }
            return
        assistant = self._store.append_message(session_id, "assistant", content)
        yield {"type": "final", "session_id": session_id, "message": assistant}
        # Phase 6.3: file what this turn taught Echo (fire-and-forget, never
        # raises, never blocks the UI).
        with contextlib.suppress(Exception):
            asyncio.create_task(self._distill_turn(session_id, text, content))
        if interrupted:
            yield {"type": "status", "state": "idle"}
