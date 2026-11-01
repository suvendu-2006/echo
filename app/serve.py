"""``python -m app.serve`` — the entry point.

SEC-2: the host defaults to 127.0.0.1 — Echo is a local-first assistant and
its WebSocket API must never listen on a public interface by accident. The
ONLY way to bind wider is the explicit ``ECHO_HOST`` environment variable
(used by the Hugging Face Spaces deployment, which fronts the app with its
own HTTPS and password gate). ECHO_PORT likewise (Spaces use 7860).
"""

from __future__ import annotations

import os

import uvicorn

from app.server import create_app

HOST = os.environ.get("ECHO_HOST", "127.0.0.1")
PORT = int(os.environ.get("ECHO_PORT", "8000"))


def main() -> None:
    uvicorn.run(create_app(), host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
