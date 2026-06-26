"""Pytest-wide test setup.

Keeps the suite hermetic: the store searches the committed `world-models/` bundled dir by default,
but unit tests build their own throwaway models under a tmp root and assert on exact name lists. We
point `WMH_BUNDLED_DIR` at a path that does not exist so the default bundled discovery is disabled
during tests; the store tests that DO exercise bundled models pass `bundled_dir=` explicitly and are
unaffected by this.
"""

from __future__ import annotations

import os

from wmh.config.store import BUNDLED_DIR_ENV

# Set before any test imports/constructs a store; a nonexistent path -> no bundled models.
os.environ.setdefault(BUNDLED_DIR_ENV, os.path.join(os.devnull, "wmh-no-bundled"))
