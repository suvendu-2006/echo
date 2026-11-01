"""MCP client manager (Phase 4.1) — Echo's hands: stdio servers as tool sources.

The Model Context Protocol (MCP) over stdio is newline-delimited JSON-RPC 2.0
on a subprocess's stdin/stdout. This module implements the lifecycle with ZERO
third-party dependencies:

    McpServerConnection  — one server: spawn -> initialize handshake ->
                           list_tools -> call_tool -> clean stop
    StdioTransport       — the default wire transport (subprocess pipes);
                           injectable for offline tests (see McpTransport)
    McpManager           — starts every configured server (failures are
                           per-server, never fatal), discovers tools and
                           registers them into the harness ToolRegistry as
                           ``mcp__<server>__<tool>``
    load_server_configs  — tolerant loader for ``data/mcp_servers.json``

Safety posture (fail-safe everywhere):

- Every MCP tool registers as ``Danger.CONFIRM`` by DEFAULT — the permission
  engine gates every call, and H4 rules can match/allow them by name
  (``mcp__files__read_file(*)``). A server the user explicitly trusts as
  read-only may set ``"danger": "safe"`` in its config entry; anything but a
  valid Danger value falls back to CONFIRM.
- Servers are spawned via ``asyncio.create_subprocess_exec`` (NO shell, no
  list-args only (no shell interpreter) — Windows-safe) with the current environment plus
  the entry's ``env`` overrides merged in.
- Timeouts everywhere (default 10s per request, 15s for the whole start);
  every failure raises :class:`McpError` — the MANAGER catches, logs and
  marks that server ``failed``. A dead MCP server can never crash the app.
- One inflight request per connection (id counter + a single pending future);
  ``call_tool`` is serialized per connection with a lock.

Registration mechanism (documented choice): ``@tool`` cannot express a
server-provided JSON schema (it always builds one from type hints, and a
dynamic signature forgery would need eval/exec — forbidden by SEC-U3).
So MCP tools attach the exact same metadata contract by hand: a
:class:`~app.harness.tools.base.ToolMeta` (the ``fn._echo_tool`` slot the
registry reads) built from the server's own ``inputSchema``, then registered
through ``ToolRegistry.register`` — the same path every builtin takes.

Server-to-client notifications (``notifications/...``) are ignored; server
REQUESTS (sampling, roots — with an id) are dropped with a logged warning:
Phase 4 consumes tools, nothing else.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from app.config import DATA_DIR
from app.harness.interfaces import Danger
from app.harness.tools.base import ToolMeta
from app.harness.tools.registry import ToolRegistry
from app.providers.base import ToolSpec

log = logging.getLogger(__name__)

#: The stdio transport protocol version this client speaks (MCP spec 2024-11-05).
PROTOCOL_VERSION = "2024-11-05"

#: Echo's client identity sent in the ``initialize`` handshake.
CLIENT_INFO = {"name": "echo", "version": "0.1.0"}

#: Per-request timeout (seconds) — a hung server must not hang Echo.
REQUEST_TIMEOUT_SECONDS = 10.0

#: Whole-startup timeout (spawn + initialize handshake, seconds).
START_TIMEOUT_SECONDS = 15.0

#: Graceful-stop timeouts inside the stdio transport (seconds).
STOP_GRACE_SECONDS = 2.0

#: Where the user's real server config lives (docs/mcp_servers.example.json
#: documents the format; this file is git-ignored runtime data).
DEFAULT_SERVERS_PATH = DATA_DIR / "mcp_servers.json"

#: Registry-name prefix for every MCP tool: ``mcp__<server>__<tool>`` (the
#: H4 permission rules match exactly this shape).
NAME_PREFIX = "mcp__"

#: Config ``danger`` values accepted for a whole server (validated, else CONFIRM).
_VALID_DANGER = {"safe": Danger.SAFE, "confirm": Danger.CONFIRM, "block": Danger.BLOCK}


class McpError(Exception):
    """Any MCP lifecycle failure (spawn, protocol, timeout, tool error).

    The manager catches these per server — a broken server is marked ``failed``
    in ``status()`` and Echo carries on. Raising (instead of returning Nones)
    keeps the failure reason next to the code that detected it.
    """


class McpToolError(McpError):
    """The server reported ``isError`` for a tool call (a tool-level failure)."""


# ------------------------------------------------------------------- transport


@runtime_checkable
class McpTransport(Protocol):
    """The I/O seam under one MCP connection — injectable for offline tests.

    The default implementation (:class:`StdioTransport`) spawns the server
    subprocess and speaks newline-delimited JSON-RPC over its pipes. Tests
    inject an in-memory fake with the same four methods, so every protocol
    behaviour is testable without ever spawning a process:

    - ``start()``: bring the channel up (spawn for stdio; no-op for fakes);
    - ``send_receive(payload)``: one JSON-RPC request -> the matching
      response dict (raises on a dead channel; may be cancelled on timeout —
      the connection wraps every call in ``asyncio.wait_for``);
    - ``send_notification(payload)``: one JSON-RPC notification (no reply);
    - ``stop()``: shut the channel down cleanly (idempotent).
    """

    async def start(self) -> None: ...

    async def send_receive(self, payload: dict) -> dict: ...

    async def send_notification(self, payload: dict) -> None: ...

    async def stop(self) -> None: ...


class StdioTransport:
    """Default transport: the server subprocess + newline-delimited JSON-RPC.

    One inflight request at a time (a single pending future keyed by id —
    MCP servers handle a tool call while we wait, so parallelism buys
    nothing and lost-correlation bugs buy less). Notifications interleave
    freely between requests: every write is one full line under a write
    lock, and responses are correlated by id, never by order.
    """

    def __init__(self, command: list[str], env: dict | None = None) -> None:
        self._command = list(command)
        self._env = dict(env) if env else {}
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._request_lock: asyncio.Lock | None = None
        self._write_lock: asyncio.Lock | None = None
        self._next_id = 0
        self._pending: asyncio.Future | None = None
        self._pending_id: int | None = None

    # ------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """Spawn the server subprocess (NO shell) and start the read loops."""
        if self._process is not None:
            return  # idempotent
        self._request_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        # Merge the current environment with the entry's overrides: MCP
        # servers (npx, python…) need PATH and friends to even launch.
        merged_env = {**os.environ, **self._env}
        try:
            self._process = await asyncio.create_subprocess_exec(  # noqa: S603 — list-args, no shell
                *self._command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=merged_env,
            )
        except OSError as exc:
            raise McpError(f"cannot spawn MCP server {self._command!r}: {exc}") from exc
        self._reader_task = asyncio.create_task(self._read_loop())
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        log.info("MCP server spawned: %s (pid %s)", self._command, self._process.pid)

    async def stop(self) -> None:
        """Close stdin, wait briefly, escalate terminate -> kill. Idempotent."""
        process, self._process = self._process, None
        try:
            if process is not None and process.stdin is not None and not process.stdin.is_closing():
                process.stdin.close()
            if process is not None and process.returncode is None:
                try:
                    await asyncio.wait_for(process.wait(), timeout=STOP_GRACE_SECONDS)
                except asyncio.TimeoutError:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=STOP_GRACE_SECONDS)
                    except asyncio.TimeoutError:
                        process.kill()
                        try:
                            await process.wait()
                        except Exception:  # already-dead process races on some platforms
                            pass
        except Exception as exc:
            log.warning("MCP server stop %s raised (ignored): %s", self._command, exc)
        finally:
            self._fail_pending(McpError("MCP transport stopped"))
            for task in (self._reader_task, self._stderr_task):
                if task is not None and not task.done():
                    task.cancel()
            for task in (self._reader_task, self._stderr_task):
                if task is not None:
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):  # noqa: B014 — shutdown is best-effort
                        pass
            self._reader_task = None
            self._stderr_task = None

    # ----------------------------------------------------------------- wire

    async def send_receive(self, payload: dict) -> dict:
        """One JSON-RPC request -> the response with the matching id."""
        if self._request_lock is None or self._write_lock is None:
            raise McpError("MCP transport not started")
        async with self._request_lock:  # one inflight request per connection
            if self._process is None:
                raise McpError("MCP transport not started")
            self._next_id += 1
            request_id = self._next_id
            wire = {**payload, "jsonrpc": "2.0", "id": request_id}
            future: asyncio.Future = asyncio.get_running_loop().create_future()
            self._pending = future
            self._pending_id = request_id
            try:
                await self._write_line(wire)
                return await future
            finally:
                if self._pending is future:  # timed out / cancelled: stop tracking
                    self._pending = None
                    self._pending_id = None

    async def send_notification(self, payload: dict) -> None:
        """One JSON-RPC notification (no id, no reply expected)."""
        if self._write_lock is None:
            raise McpError("MCP transport not started")
        async with self._write_lock:
            if self._process is None:
                raise McpError("MCP transport not started")
            await self._write_line({**payload, "jsonrpc": "2.0"})

    # --------------------------------------------------------------- helpers

    async def _write_line(self, message: dict) -> None:
        """One message = one UTF-8 JSON line on the server's stdin."""
        process = self._process
        if process is None or process.stdin is None:
            raise McpError("MCP server process is not running")
        data = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            process.stdin.write(data)
            await process.stdin.drain()
        except (OSError, RuntimeError) as exc:  # broken pipe / closed transport
            raise McpError(f"MCP server stdin write failed: {exc}") from exc

    async def _read_loop(self) -> None:
        """Parse stdout lines forever; resolve the pending future by id."""
        process = self._process
        stdout = process.stdout if process is not None else None
        if stdout is None:
            return
        while True:
            line = await stdout.readline()
            if not line:
                break  # EOF: the server exited
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except ValueError:
                log.warning("MCP server sent non-JSON line (dropped): %.200s", text)
                continue
            if not isinstance(message, dict):
                log.warning("MCP server sent non-object JSON (dropped): %.200s", text)
                continue
            self._handle_message(message)
        self._fail_pending(McpError("MCP server exited before responding"))

    def _handle_message(self, message: dict) -> None:
        """Correlate one inbound message; drop everything we did not ask for.

        Responses (no ``method``) resolve the pending future by id; anything
        else came FROM the server — notifications are ignored, and server
        REQUESTS (sampling, roots: they carry an id) are dropped with a
        logged warning (Phase 4 consumes tools, nothing else).
        """
        if "method" in message:
            if "id" in message:
                log.warning("MCP server request %r dropped (not supported)", message.get("method"))
            elif not str(message.get("method", "")).startswith("notifications/"):
                log.debug("MCP server message without id ignored: %r", message.get("method"))
            return
        pending, pending_id = self._pending, self._pending_id
        if pending is not None and pending_id == message.get("id") and not pending.done():
            pending.set_result(message)
        else:
            log.warning("MCP response for unknown/closed id %r dropped", message.get("id"))

    def _fail_pending(self, exc: McpError) -> None:
        """Wake a waiter when the channel dies (exited / stopped)."""
        pending = self._pending
        if pending is not None and not pending.done():
            pending.set_exception(exc)
        self._pending = None
        self._pending_id = None

    async def _drain_stderr(self) -> None:
        """Keep the server's stderr pipe empty (a full pipe deadlocks servers)."""
        process = self._process
        stderr = process.stderr if process is not None else None
        if stderr is None:
            return
        while True:
            line = await stderr.readline()
            if not line:
                break
            log.debug("MCP server stderr: %.200s", line.decode("utf-8", errors="replace").strip())


# ---------------------------------------------------------------- connection


class McpServerConnection:
    """One MCP server: lifecycle + the three protocol calls Echo needs.

    State machine (the values ``McpManager.status()`` reports):
    ``stopped`` -> (``start``) -> ``connected`` | ``failed``; ``stop`` -> ``stopped``.

    The transport is injectable (see :class:`McpTransport`): production uses
    :class:`StdioTransport`; tests inject an in-memory fake so the protocol
    is exercised without a single subprocess.
    """

    def __init__(
        self,
        name: str,
        command: list[str],
        env: dict | None = None,
        transport: McpTransport | None = None,
        request_timeout: float = REQUEST_TIMEOUT_SECONDS,
        start_timeout: float = START_TIMEOUT_SECONDS,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise RuntimeError("McpServerConnection requires a non-empty name")
        if not isinstance(command, list) or not command or not all(isinstance(arg, str) for arg in command):
            raise RuntimeError(f"MCP server {name!r}: command must be a non-empty list of strings")
        self.name = name
        self.command = list(command)
        self.env = dict(env) if env else None
        self._transport: McpTransport = transport if transport is not None else StdioTransport(self.command, self.env)
        self._request_timeout = float(request_timeout)
        self._start_timeout = float(start_timeout)
        self.state = "stopped"

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        """Spawn (via the transport) and run the initialize handshake.

        Order matters and is asserted by tests: ``initialize`` REQUEST first,
        THEN the ``notifications/initialized`` notification. Any failure
        (spawn, timeout, protocol) marks this connection ``failed`` and raises
        :class:`McpError`; the half-open transport is stopped best-effort so
        no orphan process survives.
        """
        if self.state == "connected":
            return  # idempotent
        self.state = "stopped"
        try:
            await asyncio.wait_for(self._transport.start(), timeout=self._start_timeout)
            result = await self._request(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": dict(CLIENT_INFO),
                },
                timeout=self._start_timeout,
            )
            if not isinstance(result, dict):
                raise McpError(f"initialize returned a non-object result: {result!r}")
            await self._transport.send_notification({"method": "notifications/initialized"})
        except asyncio.TimeoutError as exc:
            self.state = "failed"
            await self._stop_transport_quietly()
            raise McpError(f"MCP server {self.name!r} start timed out") from exc
        except McpError:
            self.state = "failed"
            await self._stop_transport_quietly()
            raise
        except Exception as exc:
            self.state = "failed"
            await self._stop_transport_quietly()
            raise McpError(f"MCP server {self.name!r} failed to start: {exc}") from exc
        self.state = "connected"
        log.info("MCP server %r connected (protocol %s)", self.name, result.get("protocolVersion", "?"))

    async def stop(self) -> None:
        """Shut the connection down; idempotent (the transport double-stops safely), never raises."""
        await self._stop_transport_quietly()
        self.state = "stopped"

    async def _stop_transport_quietly(self) -> None:
        try:
            await self._transport.stop()
        except Exception as exc:
            log.warning("MCP server %r transport stop raised (ignored): %s", self.name, exc)

    # -------------------------------------------------------------- protocol

    async def list_tools(self) -> list[dict]:
        """``tools/list`` -> the server's tool entries (name/description/inputSchema).

        Malformed entries are skipped with a warning; a wholly unusable
        response raises :class:`McpError` (the manager marks the server failed).
        """
        result = await self._request("tools/list", {})
        raw = result.get("tools") if isinstance(result, dict) else None
        if not isinstance(raw, list):
            raise McpError(f"MCP server {self.name!r}: tools/list returned no tool list")
        tools: list[dict] = []
        for entry in raw:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str) and entry["name"].strip():
                tools.append(entry)
            else:
                log.warning("MCP server %r advertised a malformed tool %r (skipped)", self.name, entry)
        return tools

    async def call_tool(self, tool_name: str, arguments: dict) -> "str | dict":
        """``tools/call`` -> the joined text content, or an ERROR envelope.

        Success returns the ``{"type":"text"}`` content blocks joined with
        newlines. When the server flags ``isError`` (a tool-level failure,
        not a protocol one) this returns the dispatcher's error envelope
        ``{"status": "ERROR", "hint": <text>}`` verbatim — the registered
        tool function turns it into the ERROR path (see ``_make_tool_fn``).
        Protocol/transport failures raise :class:`McpError` as everywhere.
        """
        result = await self._request("tools/call", {"name": tool_name, "arguments": dict(arguments)})
        if not isinstance(result, dict):
            raise McpError(f"MCP server {self.name!r}: tools/call returned a non-object result")
        texts: list[str] = []
        content = result.get("content")
        for item in content if isinstance(content, list) else []:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                texts.append(item["text"])
        joined = "\n".join(texts)
        if result.get("isError"):
            return {"status": "ERROR", "hint": joined or f"MCP tool {tool_name!r} failed without a message"}
        return joined

    # --------------------------------------------------------------- request

    async def _request(self, method: str, params: dict, timeout: float | None = None) -> Any:
        """One JSON-RPC request -> the ``result`` value; errors raise McpError."""
        payload = {"method": method, "params": params}
        try:
            response = await asyncio.wait_for(
                self._transport.send_receive(payload), timeout=timeout or self._request_timeout
            )
        except asyncio.TimeoutError as exc:
            raise McpError(f"MCP server {self.name!r}: {method} timed out") from exc
        if not isinstance(response, dict):
            raise McpError(f"MCP server {self.name!r}: {method} got a non-object response")
        error = response.get("error")
        if isinstance(error, dict):
            raise McpError(f"MCP server {self.name!r}: {method} error {error.get('code')}: {error.get('message')}")
        if "result" not in response:
            raise McpError(f"MCP server {self.name!r}: {method} response has neither result nor error")
        return response["result"]


# ----------------------------------------------------------------- manager


class McpManager:
    """Owns every configured server and merges their tools into the desk.

    ``connect_all`` starts each server — a failure is per-server (logged,
    marked ``failed`` in ``status()``) and NEVER raises: Echo runs happily
    with zero MCP servers. ``refresh_tools`` discovers the live servers'
    tools and registers each as ``mcp__<server>__<tool>`` through the same
    ``ToolRegistry`` builtins use. ``shutdown`` stops everything.
    """

    def __init__(
        self,
        servers_config: list[dict],
        request_timeout: float = REQUEST_TIMEOUT_SECONDS,
        start_timeout: float = START_TIMEOUT_SECONDS,
        transport_factory: "Callable[[str, list[str], dict | None], McpTransport] | None" = None,
    ) -> None:
        """``transport_factory(name, command, env) -> McpTransport`` is the test
        seam: the default builds a real :class:`StdioTransport` per server;
        tests inject in-memory fakes so no subprocess is ever spawned."""
        if not isinstance(servers_config, list):
            raise RuntimeError("McpManager expects a list of server config dicts")
        self._configs: list[dict] = []
        self._connections: dict[str, McpServerConnection] = {}
        self._tool_counts: dict[str, int] = {}
        self._request_timeout = float(request_timeout)
        self._start_timeout = float(start_timeout)
        self._transport_factory = transport_factory
        for entry in servers_config:
            cleaned = _clean_entry(entry, "McpManager init")
            if cleaned is not None:
                self._configs.append(cleaned)

    # ------------------------------------------------------------ lifecycle

    async def connect_all(self) -> None:
        """Start every configured server; per-server failures never raise."""
        for config in self._configs:
            name = config["name"]
            if name in self._connections and self._connections[name].state == "connected":
                continue  # idempotent: already up
            transport = (
                self._transport_factory(name, config["command"], config.get("env"))
                if self._transport_factory is not None
                else None
            )
            connection = McpServerConnection(
                name=name,
                command=config["command"],
                env=config.get("env"),
                transport=transport,
                request_timeout=self._request_timeout,
                start_timeout=self._start_timeout,
            )
            try:
                await connection.start()
            except Exception as exc:
                log.warning("MCP server %r failed to connect: %s", name, exc)
            self._connections[name] = connection
            self._tool_counts.setdefault(name, 0)

    async def refresh_tools(self, registry: ToolRegistry) -> int:
        """Register every live server's tools into ``registry``; returns the count.

        Idempotent: tools already in the registry are skipped, so calling
        refresh repeatedly (the "no restart" goal) neither duplicates nor
        raises. A server whose ``tools/list`` fails is marked ``failed`` and
        the rest carry on.
        """
        registered = 0
        for name, connection in self._connections.items():
            if connection.state != "connected":
                continue
            try:
                tools = await connection.list_tools()
            except Exception as exc:
                log.warning("MCP server %r tool discovery failed: %s", name, exc)
                connection.state = "failed"
                continue
            danger = _danger_for(self._config_for(name))
            for entry in tools:
                full_name = f"{NAME_PREFIX}{name}__{entry['name']}"
                if full_name in registry:
                    log.debug("MCP tool %r already registered (refresh is idempotent)", full_name)
                    continue
                _register_server_tool(registry, connection, full_name, entry, danger)
                registered += 1
            self._tool_counts[name] = len(tools)
            log.info("MCP server %r offers %d tool(s)", name, len(tools))
        return registered

    async def shutdown(self) -> None:
        """Stop every connection; never raises (Echo exit must be clean)."""
        for name, connection in self._connections.items():
            try:
                await connection.stop()
            except Exception as exc:
                log.warning("MCP server %r shutdown raised (ignored): %s", name, exc)
        for name in self._tool_counts:
            connection = self._connections.get(name)
            if connection is not None:
                connection.state = "stopped"

    # ---------------------------------------------------------------- status

    def status(self) -> list[dict]:
        """Per-server state for the settings UI: name / state / tool count."""
        report: list[dict] = []
        for config in self._configs:
            name = config["name"]
            connection = self._connections.get(name)
            report.append(
                {
                    "name": name,
                    "state": connection.state if connection is not None else "stopped",
                    "tools": self._tool_counts.get(name, 0),
                }
            )
        return report

    def _config_for(self, name: str) -> dict:
        for config in self._configs:
            if config["name"] == name:
                return config
        return {}


# ------------------------------------------------------------- registration


def _register_server_tool(
    registry: ToolRegistry,
    connection: McpServerConnection,
    full_name: str,
    entry: dict,
    danger: Danger,
) -> None:
    """Register one server tool as ``full_name`` through the @tool contract.

    Mechanism (documented choice, see module docstring): build the
    ``ToolMeta`` that ``@tool`` would have produced — same dataclass, same
    ``fn._echo_tool`` slot — but with the SERVER's ``inputSchema`` as the
    spec parameters (type-hint introspection cannot express remote schemas,
    and dynamic signatures would need eval/exec, which SEC-U3 forbids).
    ``ToolRegistry.register`` then treats it exactly like any builtin.
    """
    schema = entry.get("inputSchema")
    if isinstance(schema, dict):
        parameters = json.loads(json.dumps(schema))  # deep-enough copy; JSON-native
        parameters.setdefault("type", "object")
    else:
        parameters = {"type": "object", "properties": {}}
    description = entry.get("description")
    if not isinstance(description, str) or not description.strip():
        description = f"MCP tool {entry['name']!r} from server {connection.name!r} (no description given)"

    tool_name = str(entry["name"])

    async def fn(**arguments: Any) -> str:
        """Call through to the MCP server; server-reported errors -> ERROR path."""
        outcome = await connection.call_tool(tool_name, arguments)
        if isinstance(outcome, dict):  # the server set isError: fail via the dispatcher
            raise McpToolError(str(outcome.get("hint", f"MCP tool {tool_name!r} failed")))
        return outcome

    fn.__name__ = full_name
    fn.__qualname__ = full_name
    fn.__doc__ = description
    fn._echo_tool = ToolMeta(  # noqa: B010 — the contract's metadata slot, as @tool writes it
        spec=ToolSpec(name=full_name, description=description, parameters=parameters),
        danger=danger,
        cost_hint=1.0,
        takes_ctx=False,
    )
    registry.register(fn)
    log.info("registered MCP tool %r (danger %s)", full_name, danger.value)


def _danger_for(server_config: dict) -> Danger:
    """The server's danger class: CONFIRM unless a VALID config override says so.

    Fail-safe: ``"danger": "safe"`` is opt-in trust for read-only servers;
    any missing/unknown value (``"banana"``, ``5``, ``None``) -> CONFIRM.
    """
    raw = server_config.get("danger")
    if isinstance(raw, str):
        mapped = _VALID_DANGER.get(raw.strip().lower())
        if mapped is not None:
            return mapped
        log.warning("MCP server %r has unknown danger %r — using CONFIRM", server_config.get("name"), raw)
    return Danger.CONFIRM


# ------------------------------------------------------------ config loader


def load_server_configs(path: "str | Path" = DEFAULT_SERVERS_PATH) -> list[dict]:
    """Read ``data/mcp_servers.json`` -> validated config entries; never raises.

    Missing file -> ``[]`` (MCP is opt-in). Junk is tolerated entry-by-entry:
    each malformed entry is skipped with a logged warning, duplicate names
    keep their FIRST occurrence, and a wholly unparsable file yields ``[]``.
    The top level may be a JSON list of entries OR an object carrying the
    entries under a ``"servers"`` key (the shipped
    ``docs/mcp_servers.example.json`` uses the object shape so it can hold a
    ``_README`` block); anything else is junk -> ``[]``.
    Recognized keys per entry: ``name`` (unique str), ``command`` (non-empty
    list of str), ``env`` (optional dict of str->str), ``danger`` (optional
    "safe"/"confirm"/"block" — validated again at registration).
    """
    config_path = Path(path)
    if not config_path.exists():
        return []
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("MCP config %s unreadable (%s) — starting with no MCP servers", config_path, exc)
        return []
    if isinstance(raw, dict) and isinstance(raw.get("servers"), list):
        raw = raw["servers"]  # the documented example's wrapper shape
    if not isinstance(raw, list):
        log.warning("MCP config %s holds no server list — starting with no MCP servers", config_path)
        return []
    configs: list[dict] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        cleaned = _clean_entry(entry, f"{config_path.name}[{index}]")
        if cleaned is None:
            continue
        if cleaned["name"] in seen:
            log.warning("MCP config entry %r appears twice — keeping the first", cleaned["name"])
            continue
        seen.add(cleaned["name"])
        configs.append(cleaned)
    return configs


def _clean_entry(entry: Any, label: str) -> "dict | None":
    """Validate one config entry -> the cleaned dict, or None (logged skip)."""
    if not isinstance(entry, dict):
        log.warning("MCP config %s: entry is not an object (skipped)", label)
        return None
    name = entry.get("name")
    if not isinstance(name, str) or not name.strip():
        log.warning("MCP config %s: missing/blank 'name' (skipped)", label)
        return None
    command = entry.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(arg, str) and arg.strip() for arg in command)
    ):
        log.warning("MCP config %s (%s): 'command' must be a non-empty list of strings (skipped)", label, name)
        return None
    env = entry.get("env")
    if env is not None:
        if not isinstance(env, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in env.items()
        ):
            log.warning("MCP config %s (%s): 'env' must be a dict of strings (skipped)", label, name)
            return None
    cleaned = {"name": name.strip(), "command": list(command)}
    if isinstance(env, dict):
        cleaned["env"] = dict(env)
    if entry.get("danger") is not None:
        cleaned["danger"] = entry["danger"]
    return cleaned
