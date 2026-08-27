from __future__ import annotations

from repro_agent.domain.enums import JobStatus
from repro_agent.orchestrator.main_agent import RunLoopOutcome


def test_iteration_limit_returns_explicit_incomplete_outcome(main_agent) -> None:
    outcome = main_agent.run_until_finished(max_iterations=0)

    assert isinstance(outcome, RunLoopOutcome)
    assert outcome.completed is False
    assert outcome.reason == "iteration_limit"
    assert outcome.iterations == 0


def test_verified_reproduction_gap_is_terminal(main_agent) -> None:
    main_agent.job.status = JobStatus.VERIFIED_REPRODUCTION_GAP

    outcome = main_agent.run_until_finished(max_iterations=1)

    assert outcome.completed is True
    assert outcome.terminal_status == JobStatus.VERIFIED_REPRODUCTION_GAP
    assert outcome.iterations == 0


def test_unchanged_state_does_not_write_duplicate_snapshots(main_agent) -> None:
    main_agent._save_snapshot()
    main_agent._save_snapshot()

    snapshot_dir = main_agent.snapshot_store.root / main_agent.job.job_id
    assert len(list(snapshot_dir.glob("v*.json"))) == 1
