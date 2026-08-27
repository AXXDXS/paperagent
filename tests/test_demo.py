from __future__ import annotations

import json

from repro_agent.cli.main import main


def test_one_command_demo_runs_complete_offline_pipeline(tmp_path, capsys) -> None:
    work_dir = tmp_path / "demo-output"

    exit_code = main(
        [
            "demo",
            "--work-dir",
            str(work_dir),
            "--max-iterations",
            "500",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Demo 完成" in output
    assert "PIPELINE_ONLY" in output
    assert (work_dir / "repro_agent.db").is_file()
    assert (work_dir / "final_report.md").is_file()
    assert (work_dir / "final_report.json").is_file()

    report = json.loads((work_dir / "final_report.json").read_text(encoding="utf-8"))
    assert report["mock"] is True
    assert report["job"]["status"] == "USER_REPORT_READY"
    assert report["job"]["final_reproduction_status"] == "PIPELINE_ONLY"
    assert [run["tier"] for run in report["experiment_runs"]] == [
        "static_check",
        "unit_test",
        "smoke_test",
        "reduced_experiment",
        "full_experiment",
    ]
    assert report["paper_analysis"]["expected_results"]["accuracy"]["value"] == 0.9
    assert report["verification_records"]
