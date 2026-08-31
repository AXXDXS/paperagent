"""Compatibility imports for the pre-report-lease module name.

New orchestration code uses :mod:`repro_agent.scheduler.agent_reporting`.
Keeping aliases here avoids breaking imports while ensuring there is only one
implementation and one source of reporting/termination semantics.
"""

from repro_agent.scheduler.agent_reporting import (
    AgentReportingPolicy,
    ReportingDecision,
    ReportingOutcome,
    TerminationMode,
    TerminationRecord,
)

LivenessPolicy = AgentReportingPolicy
LivenessDecision = ReportingDecision
LivenessOutcome = ReportingOutcome

__all__ = [
    "AgentReportingPolicy",
    "ReportingDecision",
    "ReportingOutcome",
    "TerminationMode",
    "TerminationRecord",
    "LivenessPolicy",
    "LivenessDecision",
    "LivenessOutcome",
]
