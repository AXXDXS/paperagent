from __future__ import annotations

import json
from pathlib import Path

from repro_agent.domain.enums import TaskStatus
from repro_agent.domain.task import Task, TaskDefinition
from repro_agent.orchestrator.artifacts import ArtifactResolver


def _completed_task(tmp_path: Path, task_type: str, payload: dict) -> Task:
    task = Task(
        job_id="job_1",
        definition=TaskDefinition(objective=task_type, task_type=task_type),
        status=TaskStatus.SUCCEEDED,
        active_attempt_id=f"attempt_{task_type}",
    )
    path = tmp_path / f"{task_type}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": task.task_id,
                "attempt_id": task.active_attempt_id,
                "task_type": task_type,
                "outcome": "succeeded",
                "payload": payload,
                "artifacts": [],
                "evidence_refs": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    task.outputs = {"result.json": str(path)}
    return task


def test_specification_receives_validated_dependency_payloads(tmp_path: Path) -> None:
    paper = _completed_task(
        tmp_path,
        "paper_analysis",
        {
            "extracted_parameters": [],
            "effective_parameters": {"lr": 0.01},
            "expected_results": {"accuracy": {"value": 0.91}},
        },
    )
    code = _completed_task(
        tmp_path,
        "code_analysis",
        {"entry_points": ["train.py"], "effective_parameters": {"lr": 0.02}},
    )
    resource = _completed_task(
        tmp_path,
        "resource_check",
        {"docker_available": True, "blocking_issues": []},
    )
    spec = Task(
        job_id="job_1",
        definition=TaskDefinition(
            objective="build specification",
            task_type="specification",
            dependencies=[paper.task_id, code.task_id, resource.task_id],
            inputs={"experiment_id": "main", "target_claim": "accuracy"},
        ),
    )

    resolved = ArtifactResolver({t.task_id: t for t in [paper, code, resource]}).resolve(spec)

    assert resolved["paper_findings"]["effective_parameters"] == {"lr": 0.01}
    assert resolved["code_findings"]["entry_points"] == ["train.py"]
    assert resolved["resource_findings"]["docker_available"] is True


def test_dispatch_keeps_persisted_task_inputs_immutable(main_agent, monkeypatch) -> None:
    task = Task(
        job_id=main_agent.job.job_id,
        definition=TaskDefinition(
            objective="immutable inputs",
            task_type="paper_analysis",
            inputs={"original": True},
        ),
    )
    main_agent.scheduler.add_tasks([task])
    main_agent.scheduler.dispatch([task])
    monkeypatch.setattr(
        "repro_agent.orchestrator.main_agent.ArtifactResolver.resolve",
        lambda _self, _task: {
            "original": True,
            "resolved_dependency": {"ok": True},
        },
    )
    captured = {}
    monkeypatch.setattr(
        main_agent.dispatcher,
        "start_async",
        lambda dispatched, **kwargs: captured.update(kwargs),
    )

    main_agent._run_dispatched_task(task)

    assert task.definition.inputs == {"original": True}
    assert captured["resolved_inputs"]["resolved_dependency"] == {"ok": True}
