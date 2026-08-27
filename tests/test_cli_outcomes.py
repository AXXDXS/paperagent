from __future__ import annotations

from repro_agent.cli.main import main
from repro_agent.storage.database import Database
from repro_agent.storage.repository import JobRepository


def test_cli_does_not_claim_completion_at_iteration_limit(
    tmp_path, sample_paper, sample_repo, capsys
) -> None:
    exit_code = main(
        [
            "run",
            "--mock",
            "--paper-path",
            str(sample_paper),
            "--repository-path",
            str(sample_repo),
            "--environment-name",
            "emem",
            "--work-dir",
            str(tmp_path / "run"),
            "--max-iterations",
            "0",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 2
    assert "完成" not in output
    assert "达到迭代上限" in output
    assert (tmp_path / "run" / "final_report.json").is_file()

    database = Database(tmp_path / "run" / "repro_agent.db")
    try:
        jobs = JobRepository(database).list_all()
    finally:
        database.close()
    assert len(jobs) == 1
    assert jobs[0].inputs.environment_name == "emem"
    assert (
        tmp_path / "run" / "reports" / jobs[0].job_id / "final_report.json"
    ).is_file()


def test_cli_result_rebuilds_and_prints_the_requested_job_report(
    tmp_path, sample_paper, sample_repo, capsys
) -> None:
    work_dir = tmp_path / "run"
    assert main(
        [
            "run",
            "--mock",
            "--paper-path",
            str(sample_paper),
            "--repository-path",
            str(sample_repo),
            "--work-dir",
            str(work_dir),
            "--max-iterations",
            "0",
        ]
    ) == 2
    capsys.readouterr()

    database = Database(work_dir / "repro_agent.db")
    try:
        jobs = JobRepository(database).list_all()
    finally:
        database.close()

    assert main(
        [
            "result",
            "--job-id",
            jobs[0].job_id,
            "--work-dir",
            str(work_dir),
        ]
    ) == 0
    output = capsys.readouterr().out
    assert f"Job {jobs[0].job_id}" in output
    assert str(work_dir / "reports" / jobs[0].job_id / "final_report.md") in output


def test_cli_resume_continues_the_existing_job(tmp_path, sample_paper, sample_repo, capsys) -> None:
    work_dir = tmp_path / "run"
    assert main(
        [
            "run",
            "--mock",
            "--paper-path",
            str(sample_paper),
            "--repository-path",
            str(sample_repo),
            "--work-dir",
            str(work_dir),
            "--max-iterations",
            "0",
        ]
    ) == 2
    capsys.readouterr()

    database = Database(work_dir / "repro_agent.db")
    try:
        jobs = JobRepository(database).list_all()
    finally:
        database.close()
    assert len(jobs) == 1

    exit_code = main(
        [
            "resume",
            "--mock",
            "--job-id",
            jobs[0].job_id,
            "--work-dir",
            str(work_dir),
            "--max-iterations",
            "0",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 2
    assert "恢复完成" in output
    assert "达到迭代上限" in output
