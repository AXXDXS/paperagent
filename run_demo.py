"""Repository-local one-command entry point for the bundled offline demo."""

from __future__ import annotations

import sys

from repro_agent.cli.main import main


if __name__ == "__main__":
    raise SystemExit(main(["demo", *sys.argv[1:]]))
