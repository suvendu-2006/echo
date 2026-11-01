"""The Echo local server (PLAN Phase 2) — REST + WebSocket, loopback only.

Implements docs/phase2-contract.md exactly: the settings/provider/session
REST endpoints and the ``/ws/chat`` streaming socket. Everything is built
through :func:`create_app` with injectable seams so the whole suite runs
offline — tests swap in MockProvider chains, an in-memory secret store, a
fake catalog, and temp-file stores; the defaults build the real things.

Security posture (SEC-2, TESTS.md):
- the serve entry point binds 127.0.0.1 ONLY (app/serve.py);
- the WebSocket handshake validates Origin and rejects everything foreign;
- frames are capped (64 KB per message, 32,000 chars of chat text);
- ``GET /api/settings`` reports a ``configured`` flag, never key material;
- client-facing errors are one human-readable ``{"detail": ...}`` or one
  generic WebSocket ``error`` dict — details stay in the "echo.server" log.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

from app.chatservice import (
    HARNESS_ENABLED_KEY,
    MAX_TEXT_CHARS,
    ChatService,
)
from app.config import (
    PROJECT_ROOT,
    KeyringSecretStore,
    SecretStore,
    SettingsStore,
)
from app.auth import AuthStore, LoginThrottle, SESSION_COOKIE
from app.store import SessionStore

LOGGER = logging.getLogger("echo.server")

#: The provider registry per the frozen contract. ``openai-compatible`` has
#: no fixed base URL — the user supplies one (settings key
#: ``providers.openai-compatible.base_url``).
#: Paths reachable without a session once the password gate is live (Phase 9).
PUBLIC_PATHS = frozenset({
    "/login", "/login.js", "/style.css", "/healthz",
    "/api/auth/status", "/api/auth/login", "/api/auth/setup",
})

KNOWN_PROVIDERS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini-compat": "https://generativelanguage.googleapis.com/v1beta/openai",
    "zai": "https://api.z.ai/api/paas/v4",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openai-compatible": "",
    "ollama": "http://localhost:11434/v1",
}

#: Contract caps for the WebSocket.
MAX_MESSAGE_BYTES = 64 * 1024

#: Sanity cap for an API key on the wire (real keys are far shorter).
MAX_KEY_CHARS = 4096

_WS_REJECT_CODE = 1008  # policy violation — uvicorn answers the handshake 403
_WS_OVERSIZE_CODE = 1009  # message too big

_GENERIC_ERROR = "something went wrong — see the server log"


# ----------------------------------------------------------------- origin


def check_origin(origin: str | None, host: str) -> bool:
    """SEC-2 gate for the WebSocket handshake.

    Missing origin (non-browser clients: curl, scripts, tests) is allowed;
    an origin whose host:port matches the server's Host header is allowed;
    anything else — including scheme-less junk and lookalike hosts — is
    rejected. One pure function so it carries its own unit tests.
    """
    if origin is None:
        return True
    if not host:
        return False
    parsed = urlparse(origin)
    if parsed.scheme not in ("http", "https", "ws", "wss"):
        return False
    return parsed.netloc.lower() == host.lower()


# ---------------------------------------------------------------- catalog


class ProviderCatalog:
    """Reachability + discovery for providers, behind two async methods.

    The default implementation touches the network (lazily); tests inject a
    fake with the same two methods. ``test_key`` answers "does this key
    work?" without ever echoing the key back; ``available_models`` powers
    live discovery and degrades to ``[]`` when a backend is down — never an
    exception (same law as app.providers.ollama).
    """

    async def test_key(self, name: str, key: str) -> dict[str, Any]:
        """Probe the provider's model list with ``key`` -> ok/models | ok/error."""
        import httpx  # lazy, like every wire-touching import in Echo

        base_url = KNOWN_PROVIDERS.get(name, "")
        if not base_url:
            return {"ok": False, "error": f"unknown provider: {name}"}
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{base_url}/models", headers=headers)
        except Exception as exc:  # unreachable is an answer, not a crash
            return {"ok": False, "error": f"unreachable: {type(exc).__name__}"}
        if response.status_code != 200:
            return {"ok": False, "error": f"HTTP {response.status_code}"}
        try:
            payload = response.json()
        except ValueError:
            return {"ok": False, "error": "provider sent a malformed body"}
        data = payload.get("data") if isinstance(payload, dict) else None
        models = [
            entry["id"]
            for entry in data
            if isinstance(entry, dict) and isinstance(entry.get("id"), str)
        ] if isinstance(data, list) else []
        return {"ok": True, "models": models}

    async def available_models(self, name: str) -> list[str]:
        """Live model discovery; ``[]`` by default and whenever discovery fails."""
        if name != "ollama":
            return []
        from app.providers.ollama import list_ollama_models  # lazy (patchable)

        try:
            return await list_ollama_models()
        except Exception:  # list_ollama_models never raises; belt and suspenders
            return []


# ------------------------------------------------------------------ bodies


class KeyBody(BaseModel):
    key: str


class ActiveBody(BaseModel):
    provider: str
    model: str


class FailoverBody(BaseModel):
    order: list[str]


class TitleBody(BaseModel):
    title: str | None = None


class PasswordBody(BaseModel):
    password: str


class AutomationBody(BaseModel):
    title: str
    schedule: str
    task: str | None = None


class RouterBody(BaseModel):
    enabled: bool
    fast: dict | None = None
    strong: dict | None = None


class PathBody(BaseModel):
    path: str


# ------------------------------------------------------------------- app


def create_app(
    settings: SettingsStore | None = None,
    secrets: SecretStore | None = None,
    store: SessionStore | None = None,
    chain_factory: Callable[[], Any] | None = None,
    catalog: ProviderCatalog | None = None,
    origin_check: Callable[[str | None, str], bool] | None = None,
    use_harness: bool | None = None,
) -> FastAPI:
    """Build the Echo FastAPI app with injectable seams (offline tests).

    Defaults build the real stack: SQLite settings + sessions, the OS
    keychain for secrets, a ProviderChain built from settings, live provider
    discovery, and the standard origin rule. The frontend's static files are
    mounted LAST so /api and /ws always win. ``use_harness=None`` reads the
    ``harness.enabled`` setting (default ON — Phase 3 H7-3's single door);
    tests pass an explicit bool to pin either path.
    """
    settings = settings if settings is not None else SettingsStore()
    secrets = secrets if secrets is not None else KeyringSecretStore()
    store = store if store is not None else SessionStore()
    catalog = catalog if catalog is not None else ProviderCatalog()
    check = origin_check if origin_check is not None else check_origin
    if use_harness is None:
        flag = (settings.get(HARNESS_ENABLED_KEY) or "1").strip().lower()
        use_harness = flag not in ("0", "false", "no", "off")
    service = ChatService(
        settings=settings,
        secrets=secrets,
        store=store,
        chain_factory=chain_factory,
        use_harness=use_harness,
    )

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Runs once uvicorn is up — the serve entry's banner (SEC-2: the
        # server only ever listens on 127.0.0.1, see app/serve.py).
        print("Echo UI → http://127.0.0.1:8000", flush=True)
        # Phase 7.3: the night shift — scheduled jobs enter the harness door.
        scheduler = None
        scheduler_task = None
        try:
            from app.scheduler import JobStore, Scheduler, ToastNotifier

            try:
                notifier = ToastNotifier()  # lazy winotify; optional
            except Exception:
                notifier = None
            scheduler = Scheduler(
                JobStore(), service.run_task, notifier=notifier
            )
            app.state.scheduler = scheduler
            app.state.job_store = scheduler._store
            scheduler_task = asyncio.create_task(scheduler.run_forever())
        except Exception:
            LOGGER.exception("scheduler failed to start — continuing without it")
        try:
            yield
        finally:
            if scheduler_task is not None:
                scheduler_task.cancel()
            if scheduler is not None:
                scheduler.stop()
            job_store = getattr(app.state, "job_store", None)
            if job_store is not None:
                with contextlib.suppress(Exception):
                    job_store.close()
            # Release the service's SQLite connections on shutdown
            # (Windows locks open SQLite files — exam bugs #1–#3).
            closer = getattr(app.state.echo_service, "close", None)
            if closer is not None:
                closer()

    app = FastAPI(title="Echo", lifespan=_lifespan)
    app.state.echo_service = service  # lifecycle + future scheduler access
    app.state.auth = AuthStore()
    app.state.login_throttle = LoginThrottle()
    if app.state.auth.env_password_pending():
        app.state.auth.apply_env_password()  # headless cloud: ECHO_PASSWORD seeds the hash

    # Voice engines (Phase 5) — light constructors; heavy libs lazy-import.
    # Engines persist across connections so wake-word mute state (echo guard)
    # survives between rounds.
    from app.voice.stt import WhisperSTT
    from app.voice.tts import PiperTTS
    from app.voice.vad import SileroVAD
    from app.voice.wake import WakeWordEngine

    app.state.voice_engines = {
        "stt": WhisperSTT(),
        "vad": SileroVAD(),
        "wake": WakeWordEngine(),
        "tts": PiperTTS(),
    }

    # ---- error policy: one human-readable {"detail"}, never a stack trace.

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "invalid request"})

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception(
            "unhandled error on %s %s", request.method, request.url.path
        )
        return JSONResponse(status_code=500, content={"detail": _GENERIC_ERROR})

    # ------------------------------------------------- auth gate (Phase 9)

    from fastapi.responses import RedirectResponse

    @app.middleware("http")
    async def auth_gate(request: Request, call_next):
        """Once a password exists, every route needs a session — except the
        public allowlist (login page, auth API, health, login-page assets).
        Local mode without a password stays open, exactly as before."""
        auth: AuthStore = app.state.auth
        path = request.url.path
        if auth.is_enabled() and path not in PUBLIC_PATHS and not path.startswith("/api/auth"):
            token = request.cookies.get(SESSION_COOKIE)
            if not auth.valid_session(token):
                if path.startswith("/api/"):
                    return JSONResponse(status_code=401, content={"detail": "login required"})
                return RedirectResponse("/login", status_code=302)
        return await call_next(request)

    def _ws_authorized(websocket: WebSocket) -> bool:
        """WebSocket handshake check — cookies ride the upgrade request."""
        auth: AuthStore = app.state.auth
        return (not auth.is_enabled()) or auth.valid_session(
            websocket.cookies.get(SESSION_COOKIE)
        )

    @app.get("/login")
    async def login_page() -> FileResponse:
        return FileResponse(str(PROJECT_ROOT / "web" / "static" / "login.html"))

    @app.get("/login.js")
    async def login_js() -> FileResponse:
        return FileResponse(str(PROJECT_ROOT / "web" / "static" / "login.js"))

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/api/auth/status")
    async def auth_status() -> dict[str, Any]:
        auth: AuthStore = app.state.auth
        return {"setup": not auth.is_enabled(), "enabled": auth.is_enabled()}

    @app.post("/api/auth/setup")
    async def auth_setup(body: PasswordBody, request: Request) -> dict[str, Any]:
        auth: AuthStore = app.state.auth
        if auth.is_enabled():
            raise HTTPException(status_code=403, detail="password already exists")
        try:
            auth.set_password(body.password)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        response = JSONResponse({"ok": True})
        response.set_cookie(
            SESSION_COOKIE, auth.issue_session(), max_age=30 * 24 * 3600,
            httponly=True, samesite="lax", secure=request.url.scheme == "https",
        )
        return response

    @app.post("/api/auth/login")
    async def auth_login(body: PasswordBody, request: Request) -> dict[str, Any]:
        auth: AuthStore = app.state.auth
        throttle: LoginThrottle = app.state.login_throttle
        client_ip = request.client.host if request.client else "unknown"
        if throttle.blocked(client_ip):
            raise HTTPException(status_code=429, detail="too many attempts - wait a minute")
        if not auth.verify_password(body.password):
            throttle.record_fail(client_ip)
            raise HTTPException(status_code=401, detail="wrong password")
        throttle.reset(client_ip)
        response = JSONResponse({"ok": True})
        response.set_cookie(
            SESSION_COOKIE, auth.issue_session(), max_age=30 * 24 * 3600,
            httponly=True, samesite="lax", secure=request.url.scheme == "https",
        )
        return response

    @app.post("/api/auth/logout")
    async def auth_logout() -> dict[str, Any]:
        response = JSONResponse({"ok": True})
        response.delete_cookie(SESSION_COOKIE)
        return response

    def _failover_order() -> list[str]:
        try:
            order = json.loads(settings.get("failover.order") or "[]")
        except ValueError:
            order = []
        if not isinstance(order, list):
            return []
        return [name for name in order if isinstance(name, str)]

    # ------------------------------------------------------------ settings

    @app.get("/api/settings")
    async def get_settings() -> dict[str, Any]:
        order = _failover_order()
        providers = [
            {
                "name": name,
                "base_url": base_url,
                "failover_order": order.index(name) if name in order else None,
                "configured": secrets.get_key(name) is not None,
            }
            for name, base_url in KNOWN_PROVIDERS.items()
        ]
        return {
            "active": {
                "provider": settings.get("active.provider") or "",
                "model": settings.get("active.model") or "",
            },
            "providers": providers,
        }

    @app.put("/api/settings/active")
    async def put_active(body: ActiveBody) -> dict[str, Any]:
        if body.provider not in KNOWN_PROVIDERS:
            raise HTTPException(
                status_code=422, detail=f"unknown provider: {body.provider}"
            )
        settings.set("active.provider", body.provider)
        settings.set("active.model", body.model)
        return {"active": {"provider": body.provider, "model": body.model}}

    @app.put("/api/settings/failover")
    async def put_failover(body: FailoverBody) -> dict[str, Any]:
        unknown = [name for name in body.order if name not in KNOWN_PROVIDERS]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"unknown provider(s): {', '.join(unknown)}",
            )
        settings.set("failover.order", json.dumps(body.order))
        return {"order": body.order}

    @app.put("/api/settings/providers/{name}/key")
    async def put_key(name: str, body: KeyBody) -> Response:
        if name not in KNOWN_PROVIDERS:
            raise HTTPException(
                status_code=404, detail=f"unknown provider: {name}"
            )
        if not body.key.strip():
            raise HTTPException(status_code=422, detail="key must not be empty")
        if len(body.key) > MAX_KEY_CHARS:
            raise HTTPException(
                status_code=422,
                detail=f"key too long (max {MAX_KEY_CHARS} characters)",
            )
        secrets.set_key(name, body.key)
        return Response(status_code=204)

    @app.delete("/api/settings/providers/{name}/key")
    async def delete_key(name: str) -> Response:
        if name not in KNOWN_PROVIDERS:
            raise HTTPException(
                status_code=404, detail=f"unknown provider: {name}"
            )
        if secrets.get_key(name) is not None:
            secrets.delete_key(name)
        return Response(status_code=204)  # idempotent, keyring-safe

    @app.post("/api/settings/providers/{name}/test")
    async def test_provider(name: str) -> dict[str, Any]:
        if name not in KNOWN_PROVIDERS:
            raise HTTPException(
                status_code=404, detail=f"unknown provider: {name}"
            )
        return await catalog.test_key(name, secrets.get_key(name) or "")

    @app.get("/api/providers/available")
    async def available_models(provider: str) -> dict[str, Any]:
        if provider not in KNOWN_PROVIDERS:
            raise HTTPException(
                status_code=404, detail=f"unknown provider: {provider}"
            )
        try:
            models = await catalog.available_models(provider)
        except Exception:
            LOGGER.exception("model discovery failed for %s", provider)
            models = []
        return {"models": models}

    # ------------------------------------------------------------ sessions

    @app.get("/api/sessions")
    async def list_sessions() -> list[dict[str, Any]]:
        return store.list_sessions()

    @app.post("/api/sessions", status_code=201)
    async def create_session(body: TitleBody | None = None) -> dict[str, Any]:
        title = (body.title if body is not None else "") or ""
        return store.create_session(title=title.strip() or "New chat")

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> dict[str, Any]:
        session = store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="no such session")
        session["messages"] = [
            {
                "id": message["id"],
                "role": message["role"],
                "content": message["content"],
                "created_at": message["created_at"],
            }
            for message in store.get_messages(session_id)
        ]
        return session

    @app.patch("/api/sessions/{session_id}")
    async def rename_session(session_id: str, body: TitleBody) -> dict[str, Any]:
        title = (body.title or "").strip()
        if not title:
            raise HTTPException(status_code=422, detail="title must not be empty")
        if not store.rename_session(session_id, title):
            raise HTTPException(status_code=404, detail="no such session")
        session = store.get_session(session_id)
        if session is None:  # pragma: no cover — a renamed session exists
            raise HTTPException(status_code=404, detail="no such session")
        return session

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str) -> Response:
        if not store.delete_session(session_id):
            raise HTTPException(status_code=404, detail="no such session")
        return Response(status_code=204)

    # -------------------------------------------------------- integrations

    @app.get("/api/integrations")
    async def get_integrations() -> dict[str, Any]:
        return service.integrations_status()

    @app.post("/api/integrations/gmail/connect")
    async def post_gmail_connect() -> dict[str, Any]:
        # Opens the browser + waits for the Google consent redirect. The
        # user needs their own OAuth client (never shipped, D-013).
        return await service.connect_gmail()

    @app.post("/api/integrations/gmail/disconnect")
    async def post_gmail_disconnect() -> Response:
        service.disconnect_gmail()
        return Response(status_code=204)

    @app.post("/api/integrations/mcp/refresh")
    async def post_mcp_refresh() -> dict[str, Any]:
        return await service.refresh_mcp()

    # ----------------------------------------------------------- websocket

    @app.websocket("/ws/chat")
    async def ws_chat(websocket: WebSocket) -> None:
        if not check(
            websocket.headers.get("origin"), websocket.headers.get("host", "")
        ) or not _ws_authorized(websocket):
            # Rejected BEFORE accept: uvicorn answers the handshake with
            # HTTP 403; the Starlette test client raises WebSocketDisconnect.
            await websocket.close(
                code=_WS_REJECT_CODE, reason="origin not allowed"
            )
            return
        await websocket.accept()

        recv_task: asyncio.Task[str | None] | None = None
        pump: asyncio.Task[None] | None = None
        try:
            while True:
                if recv_task is None:
                    recv_task = asyncio.create_task(_receive_text(websocket))
                waiters: set[asyncio.Future[Any]] = {recv_task}
                if pump is not None:
                    waiters.add(pump)
                done, _ = await asyncio.wait(
                    waiters, return_when=asyncio.FIRST_COMPLETED
                )
                if pump is not None and pump in done:
                    error = pump.exception()
                    pump = None
                    if error is not None:
                        break  # the stream task died (client gone / bug)
                if recv_task in done:
                    raw = recv_task.result()
                    recv_task = None
                    if raw is None:
                        break  # client hung up
                    if len(raw.encode("utf-8")) > MAX_MESSAGE_BYTES:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "message too large (max 64 KB)",
                            }
                        )
                        await websocket.close(
                            code=_WS_OVERSIZE_CODE, reason="message too large"
                        )
                        break
                    try:
                        payload = json.loads(raw)
                    except ValueError:
                        payload = None
                    if not isinstance(payload, dict):
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "malformed message — expected a JSON object",
                            }
                        )
                        continue

                    message_type = payload.get("type")
                    if message_type == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif message_type == "interrupt":
                        service.interrupt()
                    elif message_type == "permission_answer":
                        # H4-4 over WS: resolve the engine's pending ask. No
                        # reply frame — the unblocked stream keeps talking.
                        raw_id = payload.get("id")
                        if isinstance(raw_id, str) and raw_id:
                            service.answer_permission(raw_id, bool(payload.get("allow")))
                    elif message_type == "chat":
                        text = payload.get("text")
                        if not isinstance(text, str):
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "message": "chat needs a text field",
                                }
                            )
                            continue
                        if len(text) > MAX_TEXT_CHARS:
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "message": f"text too long (max {MAX_TEXT_CHARS} characters)",
                                }
                            )
                            await websocket.close(
                                code=_WS_OVERSIZE_CODE, reason="text too long"
                            )
                            break
                        if pump is not None:
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "message": "already streaming — wait for final or interrupt",
                                }
                            )
                            continue
                        raw_session = payload.get("session_id")
                        session_id = (
                            raw_session
                            if isinstance(raw_session, str) and raw_session
                            else None
                        )
                        pump = asyncio.create_task(
                            _pump_chat(websocket, service, text, session_id)
                        )
                    else:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": f"unknown message type: {message_type!r}",
                            }
                        )
        except WebSocketDisconnect:
            pass
        finally:
            for task in (recv_task, pump):
                if task is not None and not task.done():
                    task.cancel()
            for task in (recv_task, pump):
                if task is not None:
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task

    # ------------------------------------------------------ rag (Phase 6.5)

    @app.get("/api/rag/documents")
    async def rag_documents() -> dict[str, Any]:
        library = service._rag_library()  # noqa: SLF001 — the service owns it
        return {"documents": library.list_documents()}

    @app.post("/api/rag/ingest")
    async def rag_ingest(body: PathBody) -> dict[str, Any]:
        target = body.path.strip()
        if not target:
            raise HTTPException(status_code=422, detail="path must not be empty")
        from pathlib import Path as _Path

        from app.rag.ingest import ingest_file, ingest_folder

        library = service._rag_library()  # noqa: SLF001
        path = _Path(target)
        try:
            if path.is_dir():
                return await asyncio.to_thread(ingest_folder, library, path)
            return await asyncio.to_thread(ingest_file, library, path)
        except FileNotFoundError:
            raise HTTPException(status_code=422, detail="path does not exist")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.delete("/api/rag/documents/{doc_id}")
    async def rag_delete(doc_id: str) -> Response:
        library = service._rag_library()  # noqa: SLF001
        if not library.delete_document(doc_id):
            raise HTTPException(status_code=404, detail="no such document")
        return Response(status_code=204)

    # ---------------------------------------------------------- memory (6.4)

    @app.get("/api/memory")
    async def list_memory(q: str = "", limit: int = 100) -> dict[str, Any]:
        access = service.memory_access()
        if access is None:
            raise HTTPException(status_code=503, detail="memory is disabled")
        store, wallet = access
        rows = store.search(q, k=limit) if q.strip() else store.list_all(limit=limit)
        wallet_ids = {fact["id"] for fact in wallet.facts()}
        return {
            "memories": [
                {**row, "in_wallet": row["id"] in wallet_ids} for row in rows
            ],
            "wallet": wallet.render_block(),
        }

    @app.patch("/api/memory/{memory_id}")
    async def edit_memory(memory_id: str, body: TitleBody) -> dict[str, Any]:
        access = service.memory_access()
        if access is None:
            raise HTTPException(status_code=503, detail="memory is disabled")
        store, _ = access
        updated = store.update(memory_id, text=(body.title or "").strip() or None)
        if updated is None:
            raise HTTPException(status_code=404, detail="no such memory")
        return updated

    @app.delete("/api/memory/{memory_id}")
    async def delete_memory(memory_id: str) -> Response:
        access = service.memory_access()
        if access is None:
            raise HTTPException(status_code=503, detail="memory is disabled")
        store, _ = access
        if not store.delete(memory_id):
            raise HTTPException(status_code=404, detail="no such memory")
        return Response(status_code=204)

    @app.post("/api/memory/{memory_id}/core")
    async def pin_memory(memory_id: str) -> dict[str, Any]:
        access = service.memory_access()
        if access is None:
            raise HTTPException(status_code=503, detail="memory is disabled")
        _, wallet = access
        return wallet.set_core(memory_id)

    @app.post("/api/memory/{memory_id}/demote")
    async def demote_memory(memory_id: str) -> dict[str, Any]:
        access = service.memory_access()
        if access is None:
            raise HTTPException(status_code=503, detail="memory is disabled")
        _, wallet = access
        demoted = wallet.demote_to_archival(memory_id)
        if demoted is None:
            raise HTTPException(status_code=404, detail="no such memory")
        return demoted

    @app.get("/api/memory/export")
    async def export_memory() -> dict[str, Any]:
        access = service.memory_access()
        if access is None:
            raise HTTPException(status_code=503, detail="memory is disabled")
        store, wallet = access
        return {
            "wallet": wallet.render_block(),
            "memories": store.list_all(limit=1000, include_superseded=True),
        }

    # ------------------------------------------------- phase 7 automations

    @app.get("/api/automations")
    async def list_automations() -> dict[str, Any]:
        scheduler = getattr(app.state, "scheduler", None)
        if scheduler is None:
            return {"automations": [], "scheduler": "off"}
        return {"automations": scheduler._store.list(include_disabled=True)}

    @app.post("/api/automations")
    async def add_automation(body: AutomationBody) -> dict[str, Any]:
        scheduler = getattr(app.state, "scheduler", None)
        if scheduler is None:
            raise HTTPException(status_code=503, detail="scheduler unavailable")
        from app.scheduler import parse_schedule

        try:
            parse_schedule(body.schedule)  # validate before storing
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return scheduler.add_job(body.title, body.schedule, body.task or "")

    @app.post("/api/automations/{name}/run-now")
    async def run_automation(name: str) -> dict[str, Any]:
        scheduler = getattr(app.state, "scheduler", None)
        if scheduler is None:
            raise HTTPException(status_code=503, detail="scheduler unavailable")
        result = await scheduler.fire_now(name)
        return {"name": name, "result": str(result)[:500]}

    @app.post("/api/automations/{name}/pause")
    async def pause_automation(name: str) -> dict[str, Any]:
        scheduler = getattr(app.state, "scheduler", None)
        if scheduler is None:
            raise HTTPException(status_code=503, detail="scheduler unavailable")
        if not scheduler.pause(name):
            raise HTTPException(status_code=404, detail="no such automation")
        return {"name": name, "enabled": False}

    @app.post("/api/automations/{name}/resume")
    async def resume_automation(name: str) -> dict[str, Any]:
        scheduler = getattr(app.state, "scheduler", None)
        if scheduler is None:
            raise HTTPException(status_code=503, detail="scheduler unavailable")
        if not scheduler.resume(name):
            raise HTTPException(status_code=404, detail="no such automation")
        return {"name": name, "enabled": True}

    @app.delete("/api/automations/{name}")
    async def delete_automation(name: str) -> Response:
        scheduler = getattr(app.state, "scheduler", None)
        if scheduler is None:
            raise HTTPException(status_code=503, detail="scheduler unavailable")
        if not scheduler.delete(name):
            raise HTTPException(status_code=404, detail="no such automation")
        return Response(status_code=204)

    # ------------------------------------------------------ phase 7 skills

    @app.get("/api/skills")
    async def list_skills() -> dict[str, Any]:
        store = getattr(app.state.echo_service, "_skills_store", None)
        if store is None:
            return {"skills": []}
        return {"skills": store.list(include_disabled=True)}

    @app.post("/api/skills/{name}/enable")
    async def enable_skill(name: str) -> dict[str, Any]:
        store = getattr(app.state.echo_service, "_skills_store", None)
        if store is None or not store.set_enabled(name, True):
            raise HTTPException(status_code=404, detail="no such skill")
        return {"name": name, "enabled": True}

    @app.post("/api/skills/{name}/disable")
    async def disable_skill(name: str) -> dict[str, Any]:
        store = getattr(app.state.echo_service, "_skills_store", None)
        if store is None or not store.set_enabled(name, False):
            raise HTTPException(status_code=404, detail="no such skill")
        return {"name": name, "enabled": False}

    @app.delete("/api/skills/{name}")
    async def delete_skill(name: str) -> Response:
        store = getattr(app.state.echo_service, "_skills_store", None)
        if store is None or not store.delete(name):
            raise HTTPException(status_code=404, detail="no such skill")
        return Response(status_code=204)

    # ------------------------------------------------------- phase 7 router

    @app.get("/api/router")
    async def get_router() -> dict[str, Any]:
        def role(provider_key: str, model_key: str) -> dict[str, str]:
            return {
                "provider": settings.get(provider_key) or "",
                "model": settings.get(model_key) or "",
            }

        return {
            "enabled": (settings.get("router.enabled") or "1")
            .strip()
            .lower()
            not in ("0", "false", "no", "off"),
            "fast": role("router.fast_provider", "router.fast_model"),
            "strong": role("router.strong_provider", "router.strong_model"),
        }

    @app.put("/api/router")
    async def put_router(body: RouterBody) -> dict[str, Any]:
        settings.set("router.enabled", "1" if body.enabled else "0")
        if body.fast is not None:
            settings.set("router.fast_provider", body.fast.get("provider", ""))
            settings.set("router.fast_model", body.fast.get("model", ""))
        if body.strong is not None:
            settings.set("router.strong_provider", body.strong.get("provider", ""))
            settings.set("router.strong_model", body.strong.get("model", ""))
        return await get_router()

    # -------------------------------------------------------- voice (Phase 5)

    @app.websocket("/ws/voice")
    async def ws_voice(websocket: WebSocket) -> None:
        """Voice socket (plan Phase 5): mic audio in, spoken Echo out.

        Client frames: {"type":"voice_start"} begins a conversation round,
        {"type":"audio","data":"<b64 pcm16 16k>"} feeds the mic stream,
        {"type":"voice_stop"} requests a stop, {"type":"ping"} liveness.
        Server frames: the VoicePipeline's events (status listening/thinking/
        speaking, beep, transcript, tts audio, chat payloads, error). The
        pipeline runs ONE conversation round per voice_start and the server
        re-enters it for the next — engines persist per connection so the
        wake word's mute state survives between rounds (echo guard).
        """
        if not check(
            websocket.headers.get("origin"), websocket.headers.get("host", "")
        ) or not _ws_authorized(websocket):
            await websocket.close(
                code=_WS_REJECT_CODE, reason="origin not allowed"
            )
            return
        await websocket.accept()

        from app.voice.pipeline import VoicePipeline

        engines = app.state.voice_engines
        mic_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        session_box: dict[str, str | None] = {"id": None}
        round_task: asyncio.Task[None] | None = None
        pipeline: VoicePipeline | None = None

        async def mic_stream():
            while True:
                chunk = await mic_queue.get()
                if chunk is None:
                    return
                yield chunk

        async def pump_round() -> None:
            nonlocal pipeline
            pipeline = VoicePipeline(
                chat=service,
                stt=engines["stt"],
                tts=engines["tts"],
                wake=engines["wake"],
                vad=engines["vad"],
                on_event=None,  # events are consumed via run()'s yields
            )
            try:
                async for event in pipeline.run(
                    mic_stream(), session_id=session_box["id"]
                ):
                    if event.get("type") == "final":
                        session_box["id"] = (event.get("message") or {}).get(
                            "session_id", session_box["id"]
                        )
                    await websocket.send_json(event)
            except WebSocketDisconnect:
                raise
            except Exception:
                LOGGER.exception("voice round failed unexpectedly")
                with contextlib.suppress(Exception):
                    await websocket.send_json(
                        {"type": "error", "message": _GENERIC_ERROR}
                    )

        recv_task: asyncio.Task[str | None] | None = None
        try:
            while True:
                if recv_task is None:
                    recv_task = asyncio.create_task(_receive_text(websocket))
                waiters: set[asyncio.Future[Any]] = {recv_task}
                if round_task is not None:
                    waiters.add(round_task)
                done, _ = await asyncio.wait(
                    waiters, return_when=asyncio.FIRST_COMPLETED
                )
                if round_task is not None and round_task in done:
                    round_task = None  # a round finished; await the next voice_start
                if recv_task in done:
                    raw = recv_task.result()
                    recv_task = None
                    if raw is None:
                        break
                    try:
                        payload = json.loads(raw)
                    except ValueError:
                        continue
                    message_type = payload.get("type")
                    if message_type == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif message_type == "voice_stop":
                        if pipeline is not None:
                            pipeline.request_stop()
                    elif message_type == "voice_start":
                        if round_task is None or round_task.done():
                            round_task = asyncio.create_task(pump_round())
                    elif message_type == "audio":
                        import base64 as _b64

                        data = payload.get("data")
                        if isinstance(data, str) and len(data) <= 262144:
                            mic_queue.put_nowait(_b64.b64decode(data))
        except WebSocketDisconnect:
            pass
        finally:
            if pipeline is not None:
                pipeline.request_stop()
            mic_queue.put_nowait(None)
            for task in (recv_task, round_task):
                if task is not None and not task.done():
                    task.cancel()
            for task in (recv_task, round_task):
                if task is not None:
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task

    # -------------------------------------------------------------- static
    # Mounted LAST so /api and /ws always take precedence over the UI.

    app.mount(
        "/",
        StaticFiles(
            directory=str(PROJECT_ROOT / "web" / "static"),
            html=True,
            check_dir=False,  # the frontend may not exist yet
        ),
        name="static",
    )
    return app


# ------------------------------------------------------------ ws helpers


async def _receive_text(websocket: WebSocket) -> str | None:
    """Next text frame, or None once the client hangs up.

    Binary frames come back as "" (the JSON parse then reports them as
    malformed) so the main loop has exactly two outcomes to handle.
    """
    message = await websocket.receive()
    if message["type"] == "websocket.disconnect":
        return None
    text = message.get("text")
    return text if isinstance(text, str) else ""


async def _pump_chat(
    websocket: WebSocket,
    service: ChatService,
    text: str,
    session_id: str | None,
) -> None:
    """Forward one chat's payloads to the socket until the stream ends.

    Runs as its own task so the connection loop keeps reading while the
    model streams — that is what makes mid-stream interrupts possible.
    Unexpected failures become one generic error frame, never a crash.
    """
    try:
        async for payload in service.stream_chat(text, session_id):
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        raise
    except Exception:
        LOGGER.exception("chat stream failed unexpectedly")
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "message": _GENERIC_ERROR})
