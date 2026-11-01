"""Echo authentication (Phase 9) — the gate for cloud deployments.

Design:
- **Password**: PBKDF2-HMAC-SHA256 (210k iterations, 16-byte salt) persisted in
  ``data/auth.json`` (mode 0600). First run has no password → Echo stays open
  (identical to local behavior); visiting /setup creates it. On a headless
  cloud VM, set ``ECHO_PASSWORD`` once — the hash is created at startup.
- **Sessions**: HMAC-SHA256-signed cookie ``echo_session`` (30 days) with a
  32-byte random secret kept in auth.json. Constant-time comparisons only.
- **Fail closed**: once a password exists, every route requires a valid
  session except the explicit public allowlist (login page, auth API,
  health, the static assets the login page itself needs).
- **Brute force**: in-memory per-IP counter — 5 failed logins → 60 s lockout.

Secrets and hashes never appear in logs; nothing here imports heavy deps.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

from app import config as _config

#: Where the auth state lives (lazy-resolved at call time — exam bug #6 rule).
_AUTH_PATH = None

#: PBKDF2 parameters (OWASP guidance).
_PBKDF2_ITERATIONS = 210_000
_SALT_BYTES = 16
_HASH_BYTES = 32

#: Signed session lifetime (30 days).
SESSION_TTL_SECONDS = 30 * 24 * 3600

#: Cookie name.
SESSION_COOKIE = "echo_session"

#: Paths reachable WITHOUT a session once auth is enabled.
PUBLIC_PATHS = frozenset({
    "/login",
    "/api/auth/status",
    "/api/auth/login",
    "/api/auth/setup",
    "/healthz",
    # the login page's own assets
    "/style.css",
    "/login.js",
    "/favicon.ico",
})

#: Prefixes reachable without a session (static assets for the login page).
PUBLIC_PREFIXES = ("/style.css",)

#: Brute-force guard.
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 60


def _auth_path() -> Path:
    global _AUTH_PATH
    if _AUTH_PATH is None:
        _AUTH_PATH = _config.DATA_DIR / "auth.json"
    return _AUTH_PATH


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS, dklen=_HASH_BYTES
    )


class AuthStore:
    """Password + session-secret state, persisted 0600 in ``data/auth.json``."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path is not None else _auth_path()
        self._state: dict[str, Any] | None = None

    # ------------------------------------------------------------- state io

    def _load(self) -> dict[str, Any]:
        if self._state is None:
            try:
                raw = self._path.read_text(encoding="utf-8")
                state = json.loads(raw)
                self._state = state if isinstance(state, dict) else {}
            except (OSError, ValueError):
                self._state = {}
        return self._state

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        os.chmod(tmp, 0o600)  # private before it even lands
        os.replace(tmp, self._path)

    # ----------------------------------------------------------- password

    def is_enabled(self) -> bool:
        """True once a password exists — the gate is then live (fail closed)."""
        return bool(self._load().get("password_hash"))

    def set_password(self, password: str) -> None:
        if not password or len(password) < 6:
            raise ValueError("password must be at least 6 characters")
        state = self._load()
        salt = secrets.token_bytes(_SALT_BYTES)
        state["password_hash"] = {
            "salt": salt.hex(),
            "iterations": _PBKDF2_ITERATIONS,
            "hash": _hash_password(password, salt).hex(),
        }
        state.setdefault("secret", secrets.token_hex(32))  # session signing key
        self._save()

    def verify_password(self, password: str) -> bool:
        """Constant-time password check. False when auth is not set up."""
        state = self._load()
        record = state.get("password_hash")
        if not isinstance(record, dict):
            return False
        try:
            salt = bytes.fromhex(record["salt"])
            expected = bytes.fromhex(record["hash"])
            iterations = int(record.get("iterations", _PBKDF2_ITERATIONS))
        except (KeyError, ValueError):
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, iterations, dklen=len(expected)
        )
        return hmac.compare_digest(candidate, expected)

    # ------------------------------------------------------------ sessions

    def _secret(self) -> bytes:
        state = self._load()
        secret = state.get("secret")
        if not isinstance(secret, str) or not secret:
            secret = secrets.token_hex(32)
            state["secret"] = secret
            self._save()
        return secret.encode("utf-8")

    def issue_session(self) -> str:
        """A signed session token: base64(payload) + '.' + hmac."""
        payload = json.dumps({"exp": time.time() + SESSION_TTL_SECONDS}).encode("utf-8")
        body = base64.urlsafe_b64encode(payload).decode("ascii")
        signature = hmac.new(self._secret(), body.encode("ascii"), hashlib.sha256).hexdigest()
        return f"{body}.{signature}"

    def valid_session(self, token: str | None) -> bool:
        """Constant-time token check with expiry. None/empty → False."""
        if not token or "." not in token:
            return False
        body, _, signature = token.partition(".")
        expected = hmac.new(self._secret(), body.encode("ascii"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return False
        try:
            payload = json.loads(base64.urlsafe_b64decode(body.encode("ascii")))
        except (ValueError, TypeError):
            return False
        return float(payload.get("exp", 0)) > time.time()

    def env_password_pending(self) -> bool:
        """Headless-cloud helper: ECHO_PASSWORD seeds the hash once at startup."""
        return bool(os.environ.get("ECHO_PASSWORD")) and not self.is_enabled()

    def apply_env_password(self) -> bool:
        password = os.environ.get("ECHO_PASSWORD")
        if password and not self.is_enabled():
            self.set_password(password)
            return True
        return False


class LoginThrottle:
    """Per-IP brute-force guard: MAX_FAILED_ATTEMPTS fails → LOCKOUT_SECONDS."""

    def __init__(self) -> None:
        self._fails: dict[str, list[float]] = {}

    def blocked(self, ip: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        window = [t for t in self._fails.get(ip, []) if now - t < LOCKOUT_SECONDS]
        self._fails[ip] = window
        return len(window) >= MAX_FAILED_ATTEMPTS

    def record_fail(self, ip: str, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self._fails.setdefault(ip, []).append(now)

    def reset(self, ip: str) -> None:
        self._fails.pop(ip, None)
