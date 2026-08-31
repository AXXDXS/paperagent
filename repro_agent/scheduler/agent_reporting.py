"""Dynamic sub-agent reporting contracts.

This module deliberately keeps business reporting separate from low-level
activity signals.  A thread, process or container being alive is evidence used
when a report is overdue; it is not itself permission to run forever.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from repro_agent.domain.common import utc_now
from repro_agent.domain.task import AgentReport, Task


class ReportingOutcome(str, Enum):
    NOT_STARTED = "not_started"
    WAITING = "waiting"
    DUE = "due"


@dataclass(frozen=True)
class ReportingDecision:
    outcome: ReportingOutcome
    due_at: Optional[datetime] = None
    seconds_overdue: float = 0.0
    detail: str = ""


class AgentReportingPolicy:
    """Calculates report deadlines and bounded extensions.

    The task's expected duration is the initial reporting contract.  At that
    deadline the main agent pulls status.  A live unfinished task receives a
    new deadline derived from its updated ETA; the hard timeout always remains
    an absolute upper bound.
    """

    def evaluate(
        self, task: Task, *, now: Optional[datetime] = None
    ) -> ReportingDecision:
        now = now or utc_now()
        if task.started_at is None:
            return ReportingDecision(
                ReportingOutcome.NOT_STARTED, detail="task not started"
            )
        due_at = task.next_report_due_at or self.initial_deadline(task)
        if now < due_at:
            return ReportingDecision(
                ReportingOutcome.WAITING,
                due_at=due_at,
                detail=f"next report due at {due_at.isoformat()}",
            )
        overdue = max(0.0, (now - due_at).total_seconds())
        return ReportingDecision(
            ReportingOutcome.DUE,
            due_at=due_at,
            seconds_overdue=overdue,
            detail=f"report deadline overdue by {overdue:.1f}s",
        )

    def initial_deadline(self, task: Task) -> datetime:
        baseline = task.started_at or utc_now()
        seconds = max(1.0, float(task.definition.expected_duration_seconds))
        return self._bounded_deadline(task, baseline, seconds)

    def next_deadline(
        self,
        task: Task,
        report: AgentReport,
        *,
        now: Optional[datetime] = None,
    ) -> datetime:
        now = now or report.reported_at or utc_now()
        requested = report.next_report_after_seconds
        if requested is None or requested <= 0:
            requested = report.eta_seconds
        if requested is None or requested <= 0:
            requested = self.estimate_remaining_seconds(task, now=now)
        return self._bounded_deadline(task, now, max(1.0, float(requested)))

    def estimate_remaining_seconds(
        self, task: Task, *, now: Optional[datetime] = None
    ) -> float:
        """Estimate conservatively when the child cannot supply a fresh ETA."""

        now = now or utc_now()
        report = task.latest_agent_report
        progress = report.progress if report is not None else 0.0
        if task.started_at is not None and 0.0 < progress < 1.0:
            elapsed = max(1.0, (now - task.started_at).total_seconds())
            return max(1.0, elapsed * (1.0 - progress) / progress)
        return max(1.0, float(task.definition.expected_duration_seconds))

    @staticmethod
    def _bounded_deadline(task: Task, baseline: datetime, seconds: float) -> datetime:
        requested = baseline + timedelta(seconds=seconds)
        if task.started_at is None:
            return requested
        hard_deadline = task.started_at + timedelta(
            seconds=max(1.0, float(task.definition.hard_timeout_seconds))
        )
        return min(requested, hard_deadline)


class TerminationMode(str, Enum):
    GRACEFUL = "graceful"
    FORCED = "forced"


@dataclass
class TerminationRecord:
    task_id: str
    mode: TerminationMode
    reason: str
    requested_at: datetime = field(default_factory=utc_now)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "mode": self.mode.value,
            "reason": self.reason,
            "requested_at": self.requested_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
        }
