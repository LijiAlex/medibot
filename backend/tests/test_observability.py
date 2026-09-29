"""Observability configuration. Step A1: naming the project, and turning nothing on.

The property worth pinning is the negative one. Installing langsmith and importing
medibot.config must not start tracing: if it did, every developer running the app would
silently ship staff questions to a hosted service without ever asking for it.

These tests reload config under a controlled environment. load_dotenv is stubbed out
during the reload, because the point is what config decides, not what the developer
happens to have in their own .env.
"""

from __future__ import annotations

import importlib
import os

import pytest

import medibot.config


LANGSMITH_VARS = ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT")


@pytest.fixture
def reload_config(monkeypatch):
    """Reload medibot.config with the LangSmith variables set to exactly what is asked.

    config sets LANGSMITH_PROJECT on os.environ directly, which monkeypatch does not
    track, so the surrounding values are saved and restored by hand and config is
    reloaded once more at the end to leave the module as the rest of the suite found it.
    """
    saved = {name: os.environ.get(name) for name in LANGSMITH_VARS}

    def _reload(**env: str | None):
        monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
        for name in LANGSMITH_VARS:
            os.environ.pop(name, None)
        for name, value in env.items():
            if value is not None:
                os.environ[name] = value
        return importlib.reload(medibot.config)

    yield _reload

    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    importlib.reload(medibot.config)


def test_project_defaults_to_medibot(reload_config):
    """Unset, traces still land somewhere named rather than in LangSmith's "default"."""
    config = reload_config()
    assert config.LANGSMITH_PROJECT == "medibot"
    assert os.environ["LANGSMITH_PROJECT"] == "medibot"


def test_explicit_project_is_left_alone(reload_config):
    """setdefault, not assignment: a project chosen by the operator wins."""
    config = reload_config(LANGSMITH_PROJECT="medibot-eval")
    assert config.LANGSMITH_PROJECT == "medibot-eval"
    assert os.environ["LANGSMITH_PROJECT"] == "medibot-eval"


def test_importing_config_does_not_enable_tracing(reload_config):
    """The negative property this step exists for.

    Naming a project is not the same as opting in. With LANGSMITH_TRACING absent it must
    stay absent, so the SDK stays dormant and no key is required.
    """
    reload_config()
    assert "LANGSMITH_TRACING" not in os.environ
    assert "LANGSMITH_API_KEY" not in os.environ


def test_tracing_flag_is_read_from_the_environment_only(reload_config):
    """And when it is set, config neither strengthens nor weakens it."""
    reload_config(LANGSMITH_TRACING="true")
    assert os.environ["LANGSMITH_TRACING"] == "true"
