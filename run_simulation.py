#!/usr/bin/env python3
"""Convenience entry point for running from a source checkout without installing.

Prefer ``pip install -e .`` and the ``grid-gmm`` command; this script is equivalent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from grid_gmm.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
