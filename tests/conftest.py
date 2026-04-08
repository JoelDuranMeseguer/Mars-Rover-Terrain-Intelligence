"""Pytest configuration."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# The codebase uses imports like `from src.data...`, so tests need repo root on sys.path.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep `src/` itself available too for modules imported as `mrti...`.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
