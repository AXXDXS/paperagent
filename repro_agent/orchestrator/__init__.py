"""Public orchestration API with lazy exports.

Submodules such as ``runtime_configuration`` are also used by individual
agents.  Importing all orchestrator components here would make that light
dependency load the dispatcher, which imports the agent registry while an
agent module may still be initializing.  Keep the package initializer free of
eager imports and resolve public convenience exports only when requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentDispatcher": ("dispatcher", "AgentDispatcher"),
    "CreateSubagentsResult": ("agent_tools", "CreateSubagentsResult"),
    "CreateSubagentsTool": ("agent_tools", "CreateSubagentsTool"),
    "DispatchOutcome": ("dispatcher", "DispatchOutcome"),
    "GetJobResultTool": ("result_tools", "GetJobResultTool"),
    "InitialPlanner": ("planner", "InitialPlanner"),
    "MainAgent": ("main_agent", "MainAgent"),
    "MainAgentConfig": ("main_agent", "MainAgentConfig"),
    "OutputValidator": ("validator", "OutputValidator"),
    "ReflectionController": ("reflection_controller", "ReflectionController"),
    "Replanner": ("replanner", "Replanner"),
    "SubAgentCreationRecord": ("agent_tools", "SubAgentCreationRecord"),
    "ValidationResult": ("validator", "ValidationResult"),
    "build_task_definition": ("task_factory", "build_task_definition"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Lazily preserve ``from repro_agent.orchestrator import X`` imports."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORTS])
