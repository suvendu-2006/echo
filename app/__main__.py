"""`python -m app` — the Phase 0 banner.

The real startup flow (web server, tray, scheduler) arrives with later phases;
this entry point is pinned by TESTS.md check T0-02.
"""

from app import __version__


def main() -> None:
    print(f"Echo v{__version__}")


if __name__ == "__main__":
    main()
