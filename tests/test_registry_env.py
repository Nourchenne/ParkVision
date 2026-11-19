"""Registry CLI reads project credentials without exposing them."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.models import registry


def test_registry_cli_loads_project_env_from_another_directory(tmp_path, monkeypatch, capsys):
    root = tmp_path / "project"
    root.mkdir()
    (root / ".env").write_text(
        "MLFLOW_TRACKING_USERNAME=fixture-user\n"
        "MLFLOW_TRACKING_PASSWORD=fixture-secret\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(registry, "ROOT", root)
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
    observed = {}

    def client(*, tracking_uri, registry_uri):
        observed.update(
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
            username=os.getenv("MLFLOW_TRACKING_USERNAME"),
            password=os.getenv("MLFLOW_TRACKING_PASSWORD"),
        )
        return SimpleNamespace()

    with patch("mlflow.MlflowClient", side_effect=client), \
            patch.object(registry, "expected_alias", return_value=None), \
            patch.object(registry.Registry, "versions", return_value=[]):
        registry.main(["inspect"])

    assert observed == {
        "tracking_uri": registry.ENDPOINT,
        "registry_uri": registry.ENDPOINT,
        "username": "fixture-user",
        "password": "fixture-secret",
    }
    assert "fixture-secret" not in capsys.readouterr().out


@pytest.mark.parametrize("status", [401, 403, 503])
def test_registry_remote_error_is_classified_without_credentials(status):
    error = SimpleNamespace(get_http_status_code=lambda: status)
    with pytest.raises(SystemExit) as stopped:
        registry.exit_on_mlflow_error(error)
    message = str(stopped.value)
    assert str(status) in message
    assert ("authentication or permission" in message) == (status in (401, 403))
