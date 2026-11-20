"""JevNavAgent: Budget-Decoupled Single-Forward Visual Navigation for Gigapixel Pathology."""
from __future__ import annotations

import sys
from pathlib import Path

_CORE_DIR = str(Path(__file__).resolve().parent / "core")
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)

__version__ = "1.0.0"

__all__ = ["__version__"]
