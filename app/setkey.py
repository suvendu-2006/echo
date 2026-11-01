"""`python -m app.setkey NAME [VALUE]` — store one secret in the OS keychain.

The user-friendly front door for credentials like gmail.client_id /
gmail.client_secret (D-013: secrets live in the keychain, never in files).
Omit VALUE to be prompted safely (input is hidden on Windows/macOS).

    python -m app.setkey gmail.client_id 1234-abcd.apps.googleusercontent.com
    python -m app.setkey gmail.client_secret        # prompts, hidden input
    python -m app.setkey --list                     # show stored names only
"""

from __future__ import annotations

import getpass
import sys

from app.config import KeyringSecretStore

_LISTABLE = (
    "gmail",
    "gmail.client_id",
    "gmail.client_secret",
    "langfuse.public_key",
    "langfuse.secret_key",
)


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--list":
        store = KeyringSecretStore()
        for name in _LISTABLE:
            print(f"{'set' if store.get_key(name) is not None else 'unset':5}  {name}")
        return 0
    if not argv or not argv[0]:
        print(__doc__)
        return 1
    name = argv[0]
    if len(argv) >= 2:
        value = argv[1]
    else:
        value = getpass.getpass(f"value for {name} (hidden input): ")
    if not value.strip():
        print("value must not be empty")
        return 1
    KeyringSecretStore().set_key(name, value)
    print(f"saved {name!r} to the OS keychain (never in files, D-013)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
