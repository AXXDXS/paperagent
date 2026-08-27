from __future__ import annotations

from repro_agent.domain.enums import JobStatus, ReproductionStatus
from repro_agent.orchestrator.main_agent import MainAgent, MainAgentConfig


def test_mock_pipeline_traverses_all_tiers_without_claiming_real_reproduction(
    job, work_dir, mock_provider
) -> None:
    agent = MainAgent(
        job,
        MainAgentConfig(
            memory_root=str(work_dir / "memory"),
            sandbox_root=str(work_dir / "sandboxes"),
            snapshot_root=str(work_dir / "snapshots"),
            db_path=str(work_dir / "agent.db"),
            model="mock-model",
            mock_execution=True,
            require_execution_parameter_confirmation=False,
            main_loop_wait_seconds=0.001,
        ),
        mock_provider,
    )
    agent.bootstrap()

    outcome = agent.run_until_finished(max_iterations=500)

    assert outcome.completed is True
    assert agent.job.status == JobStatus.USER_REPORT_READY
    assert agent.job.final_reproduction_status == ReproductionStatus.PIPELINE_ONLY
    runs = agent.experiment_run_repo.list_by_job(agent.job.job_id)
    assert len(runs) == 5
    assert all(run.run_type == "mock" for run in runs)
    records = agent.verification_repo.list_by_job(agent.job.job_id)
    assert records and records[-1].mock is True
    assert records[-1].verification_valid is False
