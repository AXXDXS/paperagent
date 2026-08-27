"""Load local-only LLM settings without putting credentials in source control.

The configuration file is deliberately a small, dependency-free ``KEY=VALUE``
or ``KEY: VALUE`` file.  It is intended for a user-owned file such as
``configs/config`` that is excluded from Git, not for an application-wide
configuration format.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_CONFIG_PATH = Path("configs/config")
CONFIG_FILE_ENV_VAR = "REPRO_AGENT_CONFIG_FILE"

_CONFIG_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class PrivateConfigError(ValueError):
    """The private configuration is missing or unsafe to load."""


@dataclass(frozen=True)
class LLMSettings:
    """Resolved LLM settings.  Callers must never log ``api_key``."""

    api_key: str
    api_base: str
    model: str


def load_private_config(config_file: str | Path | None = None) -> dict[str, str]:
    """Read a local config file after checking that it is owner-only.

    A missing default file is valid: environment variables still provide a
    fully supported deployment path.  Explicitly selecting a missing file is
    an error because it is almost certainly a typo.
    """

    explicit_path = config_file is not None or bool(os.environ.get(CONFIG_FILE_ENV_VAR))
    path = Path(config_file or os.environ.get(CONFIG_FILE_ENV_VAR, DEFAULT_CONFIG_PATH))
    if not path.exists():
        if explicit_path:
            raise PrivateConfigError(f"私密配置文件不存在: {path}")
        return {}
    if not path.is_file():
        raise PrivateConfigError(f"私密配置路径不是文件: {path}")

    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise PrivateConfigError(
            f"私密配置文件权限过宽: {path}；请执行 chmod 600 {path}"
        )

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith(("#", ";", "//")):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        separator = "=" if "=" in line else ":" if ":" in line else ""
        if not separator:
            raise PrivateConfigError(f"私密配置第 {line_number} 行格式无效")
        key, value = (part.strip() for part in line.split(separator, 1))
        if not _CONFIG_KEY.fullmatch(key):
            raise PrivateConfigError(f"私密配置第 {line_number} 行的键名无效")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
            value = value[1:-1]
        if not value:
            raise PrivateConfigError(f"私密配置第 {line_number} 行的值不能为空")
        normalized_key = key.upper()
        if normalized_key in values:
            raise PrivateConfigError(f"私密配置重复定义了 {normalized_key}")
        values[normalized_key] = value
    return values


def resolve_llm_settings(
    cli_model: str | None = None,
    *,
    config_file: str | Path | None = None,
) -> LLMSettings:
    """Resolve CLI, environment and private-file LLM settings.

    Precedence is CLI model > environment > private config > safe defaults.
    The credential supports ``APP_ID``/``APPID`` as a migration path for an
    existing local config, but it never infers a provider endpoint or model
    from that alias.  Those non-secret values must be configured explicitly
    when their defaults are not appropriate.
    """

    private = load_private_config(config_file)
    environment_api_key = os.environ.get("REPRO_AGENT_API_KEY")
    legacy_app_id = _first_value(private.get("APP_ID"), private.get("APPID"))
    api_key = _first_value(
        environment_api_key,
        private.get("REPRO_AGENT_API_KEY"),
        legacy_app_id,
    )
    api_base = _first_value(
        os.environ.get("REPRO_AGENT_API_BASE"),
        private.get("REPRO_AGENT_API_BASE"),
        private.get("API_BASE"),
        DEFAULT_API_BASE,
    ).rstrip("/")
    model = _first_value(
        cli_model,
        os.environ.get("REPRO_AGENT_MODEL"),
        private.get("REPRO_AGENT_MODEL"),
        private.get("MODEL"),
        DEFAULT_MODEL,
    )

    parsed_base = urlparse(api_base)
    if parsed_base.scheme != "https" or not parsed_base.netloc:
        raise PrivateConfigError("REPRO_AGENT_API_BASE 必须是完整的 HTTPS URL")
    return LLMSettings(api_key=api_key, api_base=api_base, model=model)


def _first_value(*values: str | None) -> str:
    return next((value.strip() for value in values if value and value.strip()), "")
