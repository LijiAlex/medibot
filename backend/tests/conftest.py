"""Test-wide defaults.

Tracing is switched off for the suite. With a key in `.env` it would otherwise be on, and
every test run would ship its spans to a real LangSmith project — hundreds of runs that
tell you nothing, mixed in with the ones that do.

The tracing tests are unaffected: they open their own `tracing_context(enabled="local")`,
which builds the tree in memory and overrides this.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="session")
def _no_hosted_tracing() -> None:
    import os

    os.environ["LANGSMITH_TRACING"] = "false"
