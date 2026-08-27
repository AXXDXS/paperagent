"""Canonical, user-reviewable parameters for experiment execution tasks."""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
from typing import Any, Mapping

from repro_agent.domain.enums import ExperimentTier


class ExecutionParameterValidationError(ValueError):
    """The planned experiment parameters cannot safely be presented or run."""


EXECUTION_TIER_NAMES = tuple(tier.value for tier in ExperimentTier)


def execution_parameter_snapshot(
    inputs: Mapping[str, Any],
    *,
    default_execution_image: str,
) -> dict[str, Any]:
    """Return the exact effective execution parameters in a stable JSON shape.

    The values included here are the parameters that can change how an
    experiment executes.  Sandbox guarantees are deliberately explicit as
    well: the repository is always read-only, while network access can only be
    derived from a previously confirmed, required API base (never toggled by
    an arbitrary command edit).
    """

    command = inputs.get("command", [])
    if not isinstance(command, list) or not command:
        raise ExecutionParameterValidationError("command must be a non-empty array")
    if any(not isinstance(part, str) or not part.strip() for part in command):
        raise ExecutionParameterValidationError(
            "command must contain only non-empty string arguments"
        )

    timeout_seconds = inputs.get("timeout_seconds", 3600)
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int):
        raise ExecutionParameterValidationError("timeout_seconds must be an integer")
    if not 1 <= timeout_seconds <= 86_400:
        raise ExecutionParameterValidationError(
            "timeout_seconds must be between 1 and 86400"
        )

    gpu_count = inputs.get("gpu_count", 0)
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int):
        raise ExecutionParameterValidationError("gpu_count must be an integer")
    if gpu_count < 0:
        raise ExecutionParameterValidationError("gpu_count must be non-negative")

    cpu_cores = _bounded_number(
        inputs.get("cpu_cores") or 1.0,
        field_name="cpu_cores",
        minimum=0.1,
        maximum=256.0,
    )
    memory_mb = _bounded_integer(
        inputs.get("memory_mb") or 1024,
        field_name="memory_mb",
        minimum=128,
        maximum=4_194_304,
    )
    disk_mb = _bounded_integer(
        inputs.get("disk_mb") or 4096,
        field_name="disk_mb",
        minimum=128,
        maximum=16_777_216,
    )
    gpu_memory_gb = _bounded_number(
        inputs.get("gpu_memory_gb", 0.0) or 0.0,
        field_name="gpu_memory_gb",
        minimum=0.0,
        maximum=1024.0,
    )

    execution_image = inputs.get("execution_image") or default_execution_image
    if (
        not isinstance(execution_image, str)
        or not execution_image.strip()
        or any(char.isspace() for char in execution_image)
    ):
        raise ExecutionParameterValidationError("execution_image must be a non-empty string")

    working_dir = inputs.get("working_dir", "workspace://repository")
    if not isinstance(working_dir, str) or not working_dir.strip():
        raise ExecutionParameterValidationError("working_dir must be a non-empty string")
    _validate_workspace_path(working_dir.strip(), field_name="working_dir")

    metrics_output_path = inputs.get("metrics_output_path", "output://metrics.json")
    if not isinstance(metrics_output_path, str) or not metrics_output_path.strip():
        raise ExecutionParameterValidationError(
            "metrics_output_path must be a non-empty string"
        )
    _validate_output_path(metrics_output_path.strip())

    tier = inputs.get("tier", "")
    if not isinstance(tier, str):
        raise ExecutionParameterValidationError("tier must be a string")

    experiment_environment = inputs.get("experiment_environment", {}) or {}
    if not isinstance(experiment_environment, Mapping) or any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or not value.strip()
        for key, value in experiment_environment.items()
    ):
        raise ExecutionParameterValidationError(
            "experiment_environment must contain non-empty string keys and values"
        )
    secret_env_vars = inputs.get("experiment_secret_env_vars", []) or []
    if not isinstance(secret_env_vars, list) or any(
        not isinstance(name, str) or not name.strip() for name in secret_env_vars
    ):
        raise ExecutionParameterValidationError(
            "experiment_secret_env_vars must contain non-empty environment-variable names"
        )
    network_enabled = inputs.get("network_enabled", False)
    if not isinstance(network_enabled, bool):
        raise ExecutionParameterValidationError("network_enabled must be boolean")
    network_hosts = inputs.get("network_hosts", []) or []
    if not isinstance(network_hosts, list) or any(
        not isinstance(host, str) or not host.strip() for host in network_hosts
    ):
        raise ExecutionParameterValidationError(
            "network_hosts must contain non-empty host names"
        )
    if network_enabled and not network_hosts:
        raise ExecutionParameterValidationError(
            "network_enabled requires at least one confirmed API host"
        )
    if not network_enabled and network_hosts:
        raise ExecutionParameterValidationError(
            "network_hosts must be empty while networking is disabled"
        )

    return {
        "tier": tier,
        "command": list(command),
        "execution_image": execution_image.strip(),
        "working_dir": working_dir.strip(),
        "timeout_seconds": timeout_seconds,
        "cpu_cores": cpu_cores,
        "memory_mb": memory_mb,
        "disk_mb": disk_mb,
        "gpu_count": gpu_count,
        "gpu_memory_gb": gpu_memory_gb,
        "metrics_output_path": metrics_output_path.strip(),
        "experiment_environment": dict(sorted(experiment_environment.items())),
        # Only names are reviewable/persisted.  Secret values remain solely in
        # the controller process environment.
        "experiment_secret_env_vars": sorted(set(secret_env_vars)),
        "network_enabled": network_enabled,
        "network_hosts": sorted(set(host.strip().lower() for host in network_hosts)),
        "workspace_read_only": True,
    }


def _bounded_number(
    value: Any, *, field_name: str, minimum: float, maximum: float
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExecutionParameterValidationError(f"{field_name} must be a number")
    number = float(value)
    if not minimum <= number <= maximum:
        raise ExecutionParameterValidationError(
            f"{field_name} must be between {minimum} and {maximum}"
        )
    return number


def _bounded_integer(
    value: Any, *, field_name: str, minimum: int, maximum: int
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExecutionParameterValidationError(f"{field_name} must be an integer")
    if not minimum <= value <= maximum:
        raise ExecutionParameterValidationError(
            f"{field_name} must be between {minimum} and {maximum}"
        )
    return value


def _validate_workspace_path(value: str, *, field_name: str) -> None:
    if value.startswith("/"):
        raise ExecutionParameterValidationError(f"{field_name} must stay inside workspace://")
    if "://" in value:
        if not value.startswith("workspace://"):
            raise ExecutionParameterValidationError(
                f"{field_name} must use workspace:// or a relative path"
            )
        relative = value.removeprefix("workspace://")
    else:
        relative = value
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ExecutionParameterValidationError(f"{field_name} must stay inside workspace://")


def _validate_output_path(value: str) -> None:
    if not value.startswith("output://"):
        raise ExecutionParameterValidationError("metrics_output_path must stay in output://")
    relative = value.removeprefix("output://")
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or ".." in path.parts:
        raise ExecutionParameterValidationError(
            "metrics_output_path must stay inside output://"
        )


def execution_parameter_fingerprint(parameters: Mapping[str, Any]) -> str:
    """Hash the reviewed effective parameters using canonical JSON encoding."""

    try:
        encoded = json.dumps(
            parameters,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExecutionParameterValidationError(
            "execution parameters must be losslessly representable as JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def execution_plan_snapshot(
    inputs: Mapping[str, Any],
    *,
    default_execution_image: str,
) -> dict[str, Any]:
    """Validate the complete run plan shown before environment construction.

    Unlike an individual experiment snapshot, the plan contains an exact
    command for every tier and reviews the *base* image.  The immutable image
    digest produced later by the environment builder is a derived artifact and
    therefore does not invalidate this approval.
    """

    raw_commands = inputs.get("tier_commands", {})
    if not isinstance(raw_commands, Mapping):
        raise ExecutionParameterValidationError("tier_commands must be an object")
    if set(raw_commands) != set(EXECUTION_TIER_NAMES):
        missing = sorted(set(EXECUTION_TIER_NAMES) - set(raw_commands))
        extra = sorted(set(raw_commands) - set(EXECUTION_TIER_NAMES))
        raise ExecutionParameterValidationError(
            f"tier_commands must cover exactly all experiment tiers; missing={missing}, extra={extra}"
        )
    tier_commands: dict[str, list[str]] = {}
    for tier in EXECUTION_TIER_NAMES:
        command = raw_commands[tier]
        if not isinstance(command, list) or not command or any(
            not isinstance(part, str) or not part.strip() for part in command
        ):
            raise ExecutionParameterValidationError(
                f"tier_commands.{tier} must be a non-empty string array"
            )
        tier_commands[tier] = list(command)

    base_image = inputs.get("base_image") or default_execution_image
    shared_inputs = dict(inputs)
    shared_inputs.update(
        {
            "tier": "",
            "command": tier_commands[EXECUTION_TIER_NAMES[0]],
            "execution_image": base_image,
        }
    )
    shared = execution_parameter_snapshot(
        shared_inputs,
        default_execution_image=default_execution_image,
    )
    shared.pop("tier", None)
    shared.pop("command", None)
    shared["base_image"] = shared.pop("execution_image")
    return {"tier_commands": tier_commands, **shared}


def execution_plan_fingerprint(plan: Mapping[str, Any]) -> str:
    return execution_parameter_fingerprint(plan)


def has_current_execution_parameter_approval(
    inputs: Mapping[str, Any],
    *,
    next_attempt: int,
    default_execution_image: str,
) -> bool:
    """Whether this exact upcoming attempt has a matching user confirmation."""

    approval = inputs.get("_execution_parameter_approval")
    if not isinstance(approval, Mapping):
        return False
    try:
        fingerprint = execution_parameter_fingerprint(
            execution_parameter_snapshot(
                inputs, default_execution_image=default_execution_image
            )
        )
    except ExecutionParameterValidationError:
        return False
    return (
        approval.get("fingerprint") == fingerprint
        and approval.get("approved_for_attempt") == next_attempt
    )
