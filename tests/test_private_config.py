from __future__ import annotations

from pathlib import Path

import pytest

from repro_agent.cli.private_config import PrivateConfigError, resolve_llm_settings


def _private_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config"
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)
    return path


def test_private_config_supports_legacy_key_alias_with_explicit_provider(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("REPRO_AGENT_API_KEY", raising=False)
    monkeypatch.delenv("REPRO_AGENT_API_BASE", raising=False)
    monkeypatch.delenv("REPRO_AGENT_MODEL", raising=False)
    config = _private_config(
        tmp_path,
        "\n".join(
            [
                "appid=local-test-appid",
                "REPRO_AGENT_API_BASE=https://gateway.example/v1/",
                "REPRO_AGENT_MODEL=example-model",
            ]
        ),
    )

    settings = resolve_llm_settings(config_file=config)

    assert settings.api_key == "local-test-appid"
    assert settings.api_base == "https://gateway.example/v1"
    assert settings.model == "example-model"


def test_environment_and_cli_take_precedence(monkeypatch, tmp_path: Path) -> None:
    config = _private_config(
        tmp_path,
        "\n".join(
            [
                "REPRO_AGENT_API_KEY=file-key",
                "REPRO_AGENT_API_BASE=https://file.example/v1",
                "REPRO_AGENT_MODEL=file-model",
            ]
        ),
    )
    monkeypatch.setenv("REPRO_AGENT_API_KEY", "environment-key")
    monkeypatch.setenv("REPRO_AGENT_API_BASE", "https://environment.example/v1")
    monkeypatch.setenv("REPRO_AGENT_MODEL", "environment-model")

    settings = resolve_llm_settings("cli-model", config_file=config)

    assert settings.api_key == "environment-key"
    assert settings.api_base == "https://environment.example/v1"
    assert settings.model == "cli-model"


def test_private_config_with_insecure_permissions_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.write_text("REPRO_AGENT_API_KEY=test-key\n", encoding="utf-8")
    config.chmod(0o644)

    with pytest.raises(PrivateConfigError, match="权限过宽"):
        resolve_llm_settings(config_file=config)
