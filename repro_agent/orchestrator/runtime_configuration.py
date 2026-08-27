"""Required user configuration for reproducible experiment execution.

The code-analysis agent may prove that an experiment cannot start without a
runtime value (for example a model name) or a credential exposed through an
environment variable.  This module keeps that declaration deterministic and
keeps secret *values* out of persisted Job/Task/intervention payloads.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_ARGUMENT_NAME = re.compile(r"^--?[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_KINDS = {"model_name", "api_base", "credential_env", "other"}
_DELIVERIES = {"environment", "command_argument"}


def normalize_requirements(values: Any) -> list[dict[str, Any]]:
    """Return a bounded, deduplicated list of safe requirement declarations."""

    if not isinstance(values, list):
        return []
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in values[:64]:
        if not isinstance(raw, Mapping):
            continue
        name = str(raw.get("name", "")).strip()
        kind = str(raw.get("kind", "other")).strip().lower()
        delivery = str(raw.get("delivery", "environment")).strip().lower()
        if kind not in _KINDS or delivery not in _DELIVERIES:
            continue
        if not name or len(name) > 128:
            continue
        environment_variable = str(
            raw.get("environment_variable") or (name if delivery == "environment" else "")
        ).strip()
        argument = str(raw.get("argument", "")).strip()
        if delivery == "environment" and not _ENV_NAME.fullmatch(environment_variable):
            continue
        if delivery == "command_argument" and not _ARGUMENT_NAME.fullmatch(argument):
            continue
        if kind == "credential_env" and delivery != "environment":
            continue
        key = (name, delivery)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "name": name,
                "kind": kind,
                "delivery": delivery,
                "environment_variable": environment_variable,
                "argument": argument,
                "required": bool(raw.get("required", True)),
                "reason": str(raw.get("reason", ""))[:2000],
                "source_ref": str(raw.get("source_ref", ""))[:1000],
            }
        )
    return normalized


def missing_requirements(
    requirements: Iterable[Mapping[str, Any]],
    runtime_values: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return required values absent from persisted config or the live process.

    Credential values are checked only in the live process environment.  They
    are never copied into the returned declaration or any persisted state.
    """

    environment = os.environ if environ is None else environ
    missing: list[dict[str, Any]] = []
    for requirement in normalize_requirements(list(requirements)):
        if not requirement["required"]:
            continue
        if requirement["kind"] == "credential_env":
            variable = requirement["environment_variable"]
            if not str(environment.get(variable, "")).strip():
                missing.append(requirement)
            continue
        value = runtime_values.get(requirement["name"])
        if value is None or not str(value).strip():
            missing.append(requirement)
            continue
        if requirement["kind"] == "api_base" and not _api_base_host(str(value)):
            missing.append(requirement)
    return missing


def runtime_network_configuration(
    requirements: Iterable[Mapping[str, Any]],
    runtime_values: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    """Derive the audited egress requirement from confirmed API base URLs.

    Ordinary model names and credentials do not imply network access.  A
    syntactically valid, explicitly required ``api_base`` is the sole switch
    that enables the experiment container's Docker network.
    """

    hosts: list[str] = []
    for requirement in normalize_requirements(list(requirements)):
        if requirement["kind"] != "api_base":
            continue
        value = runtime_values.get(requirement["name"])
        if value is None or not str(value).strip():
            continue
        host = _api_base_host(str(value))
        if not host:
            raise ValueError(
                f"{requirement['name']} must be an http(s) API base URL without embedded credentials"
            )
        if host not in hosts:
            hosts.append(host)
    return bool(hosts), sorted(hosts)


def _api_base_host(value: str) -> str:
    try:
        parsed = urlparse(value.strip())
    except ValueError:
        return ""
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return ""
    return parsed.hostname.lower()


def materialize_runtime_configuration(
    *,
    command: list[str],
    requirements: Iterable[Mapping[str, Any]],
    runtime_values: Mapping[str, Any],
) -> tuple[list[str], dict[str, str], list[str]]:
    """Bind confirmed values to the exact command/environment invocation."""

    materialized_command = [str(part) for part in command]
    environment: dict[str, str] = {}
    secret_environment: list[str] = []
    for requirement in normalize_requirements(list(requirements)):
        if requirement["kind"] == "credential_env":
            variable = requirement["environment_variable"]
            if variable not in secret_environment:
                secret_environment.append(variable)
            continue
        value = runtime_values.get(requirement["name"])
        if value is None or not str(value).strip():
            continue
        text = str(value).strip()
        if requirement["delivery"] == "environment":
            environment[requirement["environment_variable"]] = text
        else:
            materialized_command = _set_command_argument(
                materialized_command, requirement["argument"], text
            )
    return materialized_command, environment, secret_environment


def _set_command_argument(command: list[str], argument: str, value: str) -> list[str]:
    result = list(command)
    for index, part in enumerate(result):
        if part == argument:
            if index + 1 < len(result):
                result[index + 1] = value
            else:
                result.append(value)
            return result
        if part.startswith(argument + "="):
            result[index] = f"{argument}={value}"
            return result
    result.extend([argument, value])
    return result
