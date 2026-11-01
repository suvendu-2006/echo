"""The Gmail MCP tools (PLAN 4.3) — OAuth connect + the birthday-email flow.

Settings → Email: "Connect Gmail" → Google login → one-time consent →
"Connected as you@gmail.com" → and from then on the birthday-email flow works
end to end: ``gmail_search_contact`` finds the address, the model drafts the
email, the permission popup SHOWS the draft (the tool's arguments) and only a
confirmed ALLOW reaches ``gmail_send_email`` — draft-before-send is enforced
by the permission engine's ASK gate (D-010), not by a convention.

Setup note (one-time, user-owned — tests never need it): Google requires the
app to present ITS OWN OAuth client. Create an "OAuth client ID" of type
*Desktop app* at console.cloud.google.com (enable the Gmail API first) and
store the two values through the secret store (D-013 — NEVER a file)::

    secrets.set_key(GMAIL_CLIENT_ID_KEY, "<your client id>")
    secrets.set_key(GMAIL_CLIENT_SECRET_KEY, "<your client secret>")

Echo ships no client credentials of its own: an OAuth client for a desktop
assistant must belong to the Google account that consents to it.

Mechanics (PLAN 4.3):
- ``GoogleOAuthFlow`` implements the OAuth 2.0 Authorization Code flow with a
  loopback redirect (``http://127.0.0.1:<port>``) + PKCE (RFC 7636, S256).
  The flow is split into testable steps — ``build_url()`` /
  ``wait_for_callback()`` / ``exchange(code)`` — and ONLY ``connect()`` opens
  a browser and listens; every step accepts an injected token exchanger so
  the whole thing runs offline in tests.
- ``GmailAccount`` keeps credentials under the keyring provider name
  ``"gmail"`` as one JSON blob (D-013), and talks to the Gmail API through
  the injectable ``http_send`` transport (same style as the provider seam —
  tests never touch a socket).
- ``build_gmail_tools(account)`` returns ``@tool``-decorated functions that
  register into the harness registry exactly like builtins, so every send is
  a CONFIRM the permission popup guards.

Secrets hygiene: tokens and auth codes are never logged — the redirect
listener silences the HTTP access log (its request line carries the code).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import secrets
import time
import webbrowser
from email.message import EmailMessage
from http.server import BaseHTTPRequestHandler, HTTPServer
from inspect import isawaitable
from typing import Any, Callable
from urllib.parse import parse_qs, quote, urlencode, urlsplit

import httpx

from app.config import SecretStore
from app.harness.interfaces import Danger
from app.harness.tools.base import tool

#: Seconds before a stalled Google endpoint counts as dead.
DEFAULT_TIMEOUT_SECONDS = 60.0

#: Where the user's own OAuth client credentials live in the secret store.
GMAIL_CLIENT_ID_KEY = "gmail_client_id"
GMAIL_CLIENT_SECRET_KEY = "gmail_client_secret"

#: Keyring provider name under which the account credentials blob is stored.
#: D-013: secrets go to the OS keychain, never to a file.
GMAIL_KEYRING_PROVIDER = "gmail"

#: Google endpoints (documented constants — the tests inject fakes instead).
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"
SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
SENDAS_URL = "https://gmail.googleapis.com/gmail/v1/users/me/settings/sendAs"

#: Scopes: send (the birthday flow) + read-only (finding the contact).
GMAIL_SCOPES = (
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/userinfo.email",  # consent shows which account connected
)

#: Cap on error-body snippets echoed into hints — never echo a whole page.
MAX_ERROR_BODY_CHARS = 200

_REDIRECT_PAGE = (
    "<html><body><h2>Echo is connected to Gmail.</h2>"
    "You can close this window and return to Echo.</body></html>"
)

#: The injected token exchanger: async (or sync) callable taking the token
#: request payload dict, returning the parsed token response dict.
TokenExchanger = Callable[..., dict]

#: The injected tokeninfo fetcher: callable(access_token) -> dict | None.
TokeninfoFetcher = Callable[..., "dict | None"]

#: The injected HTTP transport for GmailAccount: async callable
#: (method, url, headers, json_body) -> {"status_code": int, "body": <parsed>}.
HttpSend = Callable[..., dict]


def decode_jwt_payload(id_token: str) -> dict:
    """Decode the payload segment of a JWT id_token — WITHOUT verifying it.

    Signature verification is deliberately skipped (and this is safe here):
    the token arrives directly from Google's token endpoint over TLS, so no
    third party could have substituted it. Pure decoding, never raises.
    """
    try:
        payload_b64 = id_token.split(".")[1]
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    except (IndexError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def parse_callback_query(query: str, expected_state: str | None = None) -> str:
    """Parse the loopback redirect query into the authorization code.

    ``query`` is the raw query string (``code=...&state=...``). An OAuth
    error (``error=access_denied``), a missing code, or a state mismatch all
    raise RuntimeError — a mismatch is the CSRF signal, and restarting the
    flow is always safe.
    """
    params = parse_qs(query or "")
    error = params.get("error")
    if error and error[0]:
        raise RuntimeError(f"oauth redirect carried an error: {error[0]}")
    codes = params.get("code")
    if not codes or not codes[0]:
        raise RuntimeError("oauth redirect carried no authorization code")
    if expected_state is not None and params.get("state", [""])[0] != expected_state:
        raise RuntimeError("oauth state mismatch (possible CSRF) — restart the flow")
    return codes[0]


class GoogleOAuthFlow:
    """The OAuth 2.0 Authorization Code flow with loopback redirect + PKCE.

    Split into steps so tests drive each one with an injected ``auth_client``
    (the token exchanger) and never open a browser — ONLY :meth:`connect`
    does the full end-to-end run (browser + listener + exchange).

    Attributes after :meth:`build_url`:
        state: the one-time CSRF state token (``secrets.token_urlsafe``).
        code_verifier: the PKCE verifier (RFC 7636; 43–128 chars).
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_port: int = 8765,
        auth_client: TokenExchanger | None = None,
        *,
        tokeninfo_fetcher: TokeninfoFetcher | None = None,
    ) -> None:
        if not client_id or not client_secret:
            raise RuntimeError(
                "GoogleOAuthFlow needs the user's own OAuth client — see the "
                "module docstring (store it under gmail_client_id / "
                "gmail_client_secret in the secret store)"
            )
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_port = int(redirect_port)
        self._auth_client = auth_client
        self._tokeninfo_fetcher = tokeninfo_fetcher
        self._captured_query: str | None = None
        self.state: str | None = None
        self.code_verifier: str | None = None

    # ---------------------------------------------------------------- steps

    def build_url(self) -> str:
        """Step 1: build the consent-screen URL (fresh state + PKCE every call)."""
        self.state = secrets.token_urlsafe(16)
        self.code_verifier = secrets.token_urlsafe(64)  # 86 chars — within [43, 128]
        challenge = (
            base64.urlsafe_b64encode(
                hashlib.sha256(self.code_verifier.encode("ascii")).digest()
            ).decode("ascii").rstrip("=")
        )
        params = {
            "client_id": self._client_id,
            "redirect_uri": self._redirect_uri(),
            "response_type": "code",
            "scope": " ".join(GMAIL_SCOPES),
            "state": self.state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "access_type": "offline",  # ask for a refresh_token
            "prompt": "consent",       # the one-time consent screen
        }
        return f"{AUTH_URL}?{urlencode(params, quote_via=quote)}"

    def auth_url(self) -> str:
        """Alias of :meth:`build_url` (the PLAN's original name for step 1)."""
        return self.build_url()

    def wait_for_callback(self, timeout_seconds: float = 300.0) -> str:
        """Step 2 (BLOCKING): listen once on the loopback redirect; return the code.

        Opens a tiny HTTP server on ``127.0.0.1:<redirect_port>``, serves
        exactly one GET (the browser's redirect), and parses the query. The
        access log is silenced — the request line carries the auth code.
        """
        server = self._make_listener()
        try:
            query = self._serve_until_callback(server, timeout_seconds)
        finally:
            server.server_close()
        return parse_callback_query(query, expected_state=self.state)

    async def exchange(self, code: str) -> dict:
        """Step 3: swap the authorization code for tokens (+ the account email).

        Returns ``{"access_token", "refresh_token", "email"}``. The email
        comes from the id_token payload (unverified decode — the token came
        straight from Google over TLS), falling back to the tokeninfo
        endpoint. Any OAuth error raises RuntimeError.
        """
        if self.code_verifier is None:
            raise RuntimeError("build_url() must be called before exchange()")
        payload = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self._redirect_uri(),
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "code_verifier": self.code_verifier,
        }
        response = await self._run_exchanger(payload)
        if not isinstance(response, dict):
            raise RuntimeError("token endpoint returned a non-object response")
        if response.get("error"):
            raise RuntimeError(f"token exchange failed: {response['error']}")
        access_token = response.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise RuntimeError("token endpoint returned no access_token")
        email = await self._extract_email(response)
        refresh_token = response.get("refresh_token")
        return {
            "access_token": access_token,
            "refresh_token": refresh_token if isinstance(refresh_token, str) else "",
            "email": email,
        }

    # -------------------------------------------------------------- connect

    async def connect(self, timeout_seconds: float = 300.0) -> dict:
        """The full flow: consent URL → browser → loopback callback → tokens.

        ONLY this method opens the system browser and listens on the
        loopback port. Returns the :meth:`exchange` dict on success, or
        ``{"status": "ERROR", "hint": ...}`` on any failure — never raises.
        """
        try:
            url = self.build_url()
            server = self._make_listener()  # bind BEFORE the browser can land
            try:
                webbrowser.open(url)
                query = await asyncio.to_thread(
                    self._serve_until_callback, server, timeout_seconds
                )
            finally:
                server.server_close()
            code = parse_callback_query(query, expected_state=self.state)
            return await self.exchange(code)
        except Exception as exc:
            return {"status": "ERROR", "hint": f"gmail connect failed: {exc}"}

    # -------------------------------------------------------------- details

    def _redirect_uri(self) -> str:
        return f"http://127.0.0.1:{self._redirect_port}"

    async def _run_exchanger(self, payload: dict) -> dict:
        exchanger = self._auth_client or self._default_token_exchange
        result = exchanger(payload)
        if isawaitable(result):
            result = await result
        return result

    async def _default_token_exchange(self, payload: dict) -> dict:
        """The real token request: form-encoded POST to Google's token endpoint."""
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
                response = await client.post(TOKEN_URL, data=payload)
            try:
                return response.json()
            except ValueError:
                return {"error": f"non-JSON token response (HTTP {response.status_code})"}
        except httpx.HTTPError as exc:
            return {"error": f"cannot reach {TOKEN_URL} ({exc})"}

    async def _extract_email(self, response: dict) -> str:
        """id_token decode first, tokeninfo fallback, "" when both are silent."""
        id_token = response.get("id_token")
        if isinstance(id_token, str) and id_token:
            email = decode_jwt_payload(id_token).get("email")
            if isinstance(email, str) and email:
                return email
        fetcher = self._tokeninfo_fetcher
        access_token = response.get("access_token")
        if fetcher is not None and isinstance(access_token, str) and access_token:
            info = fetcher(access_token)
            if isawaitable(info):
                info = await info
            if isinstance(info, dict):
                email = info.get("email")
                if isinstance(email, str) and email:
                    return email
        return ""

    def _make_listener(self) -> HTTPServer:
        """The loopback catcher: one GET captured, access log silenced."""
        flow = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                flow._captured_query = urlsplit(self.path).query
                body = _REDIRECT_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return  # NEVER log: the request line carries code + state

        server = HTTPServer(("127.0.0.1", self._redirect_port), _Handler)
        server.timeout = 0.5
        return server

    def _serve_until_callback(self, server: HTTPServer, timeout_seconds: float) -> str:
        """(Blocking) handle requests until the redirect lands or time runs out."""
        self._captured_query = None
        deadline = time.monotonic() + timeout_seconds
        while self._captured_query is None:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"no oauth redirect within {timeout_seconds:.0f}s")
            server.handle_request()
        return self._captured_query


class GmailAccount:
    """One connected Gmail account — credentials in the keychain, send + search.

    Credentials live as one JSON blob under the keyring provider name
    ``"gmail"`` (D-013: the OS keychain, NEVER a file). All network traffic
    goes through the injectable ``http_send`` transport::

        async http_send(method, url, headers, json_body)
            -> {"status_code": <int>, "body": <parsed JSON or raw text>}

    (default: httpx). Every public method returns an envelope and NEVER
    raises — errors are messages, not crashes (PLAN H2-3 discipline).
    """

    def __init__(self, secrets: SecretStore, http_send: HttpSend | None = None) -> None:
        self._secrets = secrets
        self._http_send: HttpSend = http_send or self._default_http_send

    # ---------------------------------------------------------- credentials

    def save_credentials(self, creds: dict) -> None:
        """Store the OAuth credentials (JSON) in the secret store."""
        if not isinstance(creds, dict):
            raise RuntimeError("gmail credentials must be a dict")
        self._secrets.set_key(GMAIL_KEYRING_PROVIDER, json.dumps(creds))

    def is_connected(self) -> bool:
        """True when a credentials blob is stored and parses as an object."""
        return self.credentials() is not None

    def credentials(self) -> dict | None:
        """The stored credentials, or None when not connected / corrupt."""
        raw = self._secrets.get_key(GMAIL_KEYRING_PROVIDER)
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def disconnect(self) -> None:
        """Forget the account (the consent stays until revoked at Google)."""
        self._secrets.delete_key(GMAIL_KEYRING_PROVIDER)

    # ------------------------------------------------------------ the flows

    async def send_email(self, to: str, subject: str, body: str) -> dict:
        """Send one email via the Gmail API — the CONFIRMED final step.

        Returns ``{"status": "OK", "id": ...}`` or
        ``{"status": "ERROR", "hint": ...}`` — never raises. The draft
        approval happened earlier, at the permission popup (D-010).
        """
        try:
            if "\n" in to or "\r" in to or "\n" in subject or "\r" in subject:
                return {"status": "ERROR", "hint": "gmail send refused: recipient/subject contains a line break"}
            if not to.strip():
                return {"status": "ERROR", "hint": "gmail send refused: no recipient"}
            creds = self.credentials()
            if creds is None:
                return {"status": "ERROR", "hint": "gmail is not connected — connect first (Settings → Email)"}
            token = creds.get("access_token")
            if not isinstance(token, str) or not token:
                return {"status": "ERROR", "hint": "stored gmail credentials carry no access_token — reconnect"}
            raw = self._build_raw_message(to, subject, body, creds.get("email"))
            response = await self._call_http(
                "POST", SEND_URL, {"Authorization": f"Bearer {token}"}, {"raw": raw}
            )
            status_code = response.get("status_code")
            if status_code != 200:
                snippet = _snippet(response.get("body"))
                return {"status": "ERROR", "hint": f"gmail send failed: HTTP {status_code}: {snippet}"}
            body_data = response.get("body")
            message_id = body_data.get("id") if isinstance(body_data, dict) else None
            return {"status": "OK", "id": message_id if isinstance(message_id, str) else ""}
        except Exception as exc:
            return {"status": "ERROR", "hint": f"gmail send failed: {exc}"}

    async def search_contact(self, name: str) -> dict:
        """Look up an email address by (display) name — the flow's first step.

        Queries the Gmail profiles endpoint (users.settings.sendAs — the
        Gmail API's own profile list; full address-book search is a future
        People-API upgrade). Tolerant parse: bare lists, alternate key
        names and garbage bodies all degrade to fewer/no matches, never an
        error. Returns ``{"status": "OK", "email": <best|None>, "matches": [...]}``.
        """
        try:
            creds = self.credentials()
            if creds is None:
                return {"status": "ERROR", "hint": "gmail is not connected — connect first (Settings → Email)"}
            token = creds.get("access_token")
            if not isinstance(token, str) or not token:
                return {"status": "ERROR", "hint": "stored gmail credentials carry no access_token — reconnect"}
            response = await self._call_http(
                "GET", SENDAS_URL, {"Authorization": f"Bearer {token}"}, None
            )
            if response.get("status_code") != 200:
                snippet = _snippet(response.get("body"))
                return {"status": "ERROR", "hint": f"gmail contact search failed: HTTP {response.get('status_code')}: {snippet}"}
            matches = parse_sendas_matches(response.get("body"), name)
            return {
                "status": "OK",
                "matches": matches,
                "email": matches[0]["email"] if matches else None,
            }
        except Exception as exc:
            return {"status": "ERROR", "hint": f"gmail contact search failed: {exc}"}

    # -------------------------------------------------------------- details

    @staticmethod
    def _build_raw_message(to: str, subject: str, body: str, from_email: object) -> str:
        """RFC 5322 message → base64url, the Gmail send wire format."""
        message = EmailMessage()
        message["To"] = to
        message["Subject"] = subject
        if isinstance(from_email, str) and from_email:
            message["From"] = from_email
        message.set_content(body or "")
        return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")

    async def _call_http(
        self, method: str, url: str, headers: dict, json_body: dict | None
    ) -> dict:
        result = self._http_send(method, url, headers, json_body)
        if isawaitable(result):
            result = await result
        if not isinstance(result, dict):
            raise RuntimeError("gmail http transport returned a non-dict response")
        return result

    async def _default_http_send(
        self, method: str, url: str, headers: dict, json_body: dict | None
    ) -> dict:
        """The real transport: httpx request, parsed body, never raises."""
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
                response = await client.request(method, url, headers=headers, json=json_body)
            try:
                body: Any = response.json()
            except ValueError:
                body = response.text
            return {"status_code": response.status_code, "body": body}
        except httpx.HTTPError as exc:
            return {"status_code": 0, "body": f"transport failure: {exc}"}


def parse_sendas_matches(body: object, name: str) -> list[dict]:
    """Tolerantly extract ``{"email", "display_name"}`` matches from a body.

    Accepts ``{"sendAs": [...]}`` (the real Gmail shape), a bare list, or
    ``{"results"|"contacts": [...]}``; skips non-dict entries and entries
    without an email; matches case-insensitively on substring against both
    the display name and the address. Garbage in → no matches, never an error.
    """
    if not isinstance(name, str) or not name.strip():
        return []
    if isinstance(body, dict):
        entries: object = (
            body.get("sendAs")
            if body.get("sendAs") is not None
            else body.get("results") if body.get("results") is not None
            else body.get("contacts")
        )
    else:
        entries = body
    if not isinstance(entries, list):
        return []
    needle = name.strip().lower()
    matches: list[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        email = entry.get("sendAsEmail") or entry.get("email") or entry.get("address")
        display = entry.get("displayName") or entry.get("display_name") or entry.get("name") or ""
        if not isinstance(email, str) or not email:
            continue
        haystack = f"{display} {email}".lower()
        if needle in haystack:
            matches.append({"email": email, "display_name": str(display)})
    return matches


def _snippet(body: object) -> str:
    """A short, single-line error-body snippet for hints — never a whole page."""
    return " ".join(str(body).split())[:MAX_ERROR_BODY_CHARS]


def build_gmail_tools(account: GmailAccount) -> list:
    """The Gmail harness tools (PLAN 4.3), built against one account.

    Returns two ``@tool``-decorated functions ready for
    ``ToolRegistry.register`` — exactly like builtins, so the permission
    engine guards every call:

    - ``gmail_send_email(to, subject, body)`` — CONFIRM. The popup shows the
      arguments, which ARE the draft: the user's allow is the send approval
      (draft-before-send, D-010).
    - ``gmail_search_contact(name)`` — SAFE (read-only profile query).

    Register once per account: the registry rejects duplicate names, so
    building twice into one registry is (correctly) a RuntimeError.
    """

    @tool(danger=Danger.CONFIRM, cost_hint=1.0)
    async def gmail_send_email(to: str, subject: str, body: str) -> dict:
        """Send an email from the connected Gmail account.

        Runs only after the user confirmed the exact draft (to, subject,
        body) shown by the permission popup.
        """
        return await account.send_email(to, subject, body)

    @tool(danger=Danger.SAFE, cost_hint=1.0)
    async def gmail_search_contact(name: str) -> dict:
        """Look up a contact's email address by name in the connected Gmail account."""
        return await account.search_contact(name)

    return [gmail_send_email, gmail_search_contact]
