from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from repro_agent.dynamic_tools.lifecycle import DynamicToolLifecycleManager
from repro_agent.dynamic_tools.models import DynamicToolStatus
from repro_agent.agents.base import AgentRunResult, BaseSubAgent
from repro_agent.agents.registry import SUB_AGENT_REGISTRY
from repro_agent.domain.task import Task
from repro_agent.orchestrator.task_factory import build_task_definition
from repro_agent.orchestrator.interventions import InterventionService
from repro_agent.domain.enums import InterventionStatus
from repro_agent.storage.database import Database
from repro_agent.storage.repository import DynamicToolRepository
from repro_agent.tools.authorization import ToolAuthorizer
from repro_agent.tools.base import ToolInputValidationError
from repro_agent.tools.registry import create_builtin_registry


class _Sandbox:
    def __init__(self, root: Path):
        self.root = root
        self.policy = SimpleNamespace(
            resource_limits=SimpleNamespace(max_tool_calls=20),
        )


class _MeanExecutor:
    def run(self, record, sandbox_ctx, arguments, *, timeout_seconds=30):
        values = arguments["values"]
        return sum(values) / len(values)


def _candidate(code: str | None = None) -> dict:
    return {
        "purpose": "Compute the arithmetic mean of a JSON number array.",
        "functional_key": "statistics.arithmetic_mean",
        "code": code
        or "def arithmetic_mean(values):\n    return sum(values) / len(values)\n",
        "entry_function": "arithmetic_mean",
        "input_schema": {
            "type": "object",
            "properties": {
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 1,
                }
            },
            "required": ["values"],
            "additionalProperties": False,
        },
        "output_schema": {"type": "number"},
        "tests": [{"arguments": {"values": [1, 2, 3]}, "expected": 2.0}],
        "generalization_reason": "Metric and data-analysis tasks repeatedly need a mean.",
        "suggested_task_types": ["verification"],
        "dependencies": [],
        "risk_level": "read_only",
        "requires_network": False,
    }


def _write_sidecar(path: Path, candidate: dict) -> Path:
    path.write_text(json.dumps([candidate]), encoding="utf-8")
    return path


@pytest.fixture()
def lifecycle(tmp_path: Path):
    database = Database(tmp_path / "dynamic.db")
    registry = create_builtin_registry()
    manager = DynamicToolLifecycleManager(
        DynamicToolRepository(database),
        registry,
        executor=_MeanExecutor(),
        verifier=lambda candidate, sandbox: None,
    )
    yield database, registry, manager, _Sandbox(tmp_path)
    database.close()


def test_builtin_tools_are_permanent(lifecycle) -> None:
    _, registry, _, _ = lifecycle
    assert registry.is_permanent("read_file")
    with pytest.raises(ValueError, match="permanent"):
        registry.unregister_dynamic("read_file")


def test_three_independent_reports_activate_and_duplicate_task_does_not_count(
    lifecycle, tmp_path: Path
) -> None:
    _, registry, manager, sandbox = lifecycle
    source = _write_sidecar(tmp_path / "candidate.json", _candidate())

    manager.ingest_sidecar(
        source,
        job_id="job-a",
        task_id="task-1",
        attempt_id="attempt-1",
        task_type="verification",
        sandbox_ctx=sandbox,
    )
    # A retry of the same logical task is not independent support.
    manager.ingest_sidecar(
        source,
        job_id="job-a",
        task_id="task-1",
        attempt_id="attempt-2",
        task_type="verification",
        sandbox_ctx=sandbox,
    )
    assert manager.list_records()[0]["support_count"] == 1

    variant = _candidate(
        "def arithmetic_mean(data):\n    return sum(data) / len(data)\n"
    )
    _write_sidecar(source, variant)
    for index in (2, 3):
        manager.ingest_sidecar(
            source,
            job_id=f"job-{index}",
            task_id=f"task-{index}",
            attempt_id=f"attempt-{index}",
            task_type="verification",
            sandbox_ctx=sandbox,
        )

    record = manager.list_records(include_code=True)[0]
    assert record["support_count"] == 3
    assert record["status"] == DynamicToolStatus.ACTIVE.value
    assert record["life"] == 30
    assert registry.get(record["tool_name"]) is not None
    assert not registry.is_permanent(record["tool_name"])


def test_only_relevant_events_decay_and_pending_expiry_keeps_tombstone(
    lifecycle, tmp_path: Path
) -> None:
    _, registry, manager, sandbox = lifecycle
    source = _write_sidecar(tmp_path / "candidate.json", _candidate())
    manager.ingest_sidecar(
        source,
        job_id="job-a",
        task_id="task-1",
        attempt_id="attempt-1",
        task_type="verification",
        sandbox_ctx=sandbox,
    )
    record = manager.list_records(include_code=True)[0]
    manager.advance_relevant_task_event("paper_analysis")
    assert manager.list_records()[0]["life"] == 10

    for _ in range(10):
        manager.advance_relevant_task_event("verification")
    expired = manager.list_records(include_code=True)[0]
    assert expired["status"] == DynamicToolStatus.EXPIRED.value
    assert expired["code"] == ""
    assert expired["tests"] == []
    assert registry.get(record["tool_name"]) is None


def test_success_resets_active_life_and_failed_call_only_increments_failure(
    lifecycle, tmp_path: Path
) -> None:
    _, registry, manager, sandbox = lifecycle
    source = _write_sidecar(tmp_path / "candidate.json", _candidate())
    for index in range(3):
        manager.ingest_sidecar(
            source,
            job_id=f"job-{index}",
            task_id=f"task-{index}",
            attempt_id=f"attempt-{index}",
            task_type="verification",
            sandbox_ctx=sandbox,
        )
    record = manager.list_records(include_code=True)[0]
    record["life"] = 4
    manager.repository.save(record)

    authorization = ToolAuthorizer(
        registry, invocation_observer=manager.observe_invocation
    ).authorize(
        task_id="consumer",
        task_type="verification",
        allowed_tools=[record["tool_name"]],
        sandbox_ctx=sandbox,
    )
    assert authorization.call(record["tool_name"], values=[2, 4]) == 3.0
    assert manager.repository.get(record["tool_id"])["life"] == 30

    current = manager.repository.get(record["tool_id"])
    current["life"] = 7
    manager.repository.save(current)
    with pytest.raises(ToolInputValidationError):
        authorization.call(record["tool_name"], values=[])
    failed = manager.repository.get(record["tool_id"])
    assert failed["life"] == 7
    assert failed["failure_count"] == 1


class _CandidateReportingAgent(BaseSubAgent):
    task_type = "verification"

    def run(self) -> AgentRunResult:
        self.report_reusable_code_candidate(_candidate())
        payload = {"comparisons": [], "run_actually_executed": True}
        self.write_json_output("result.json", payload)
        return AgentRunResult(succeeded=True, outputs=payload)


def test_dispatcher_persists_sidecar_and_main_ingests_only_after_validation(
    main_agent, monkeypatch
) -> None:
    monkeypatch.setitem(SUB_AGENT_REGISTRY, "verification", _CandidateReportingAgent)
    # Keep this orchestration test independent from Docker; container behavior
    # admission is covered at the lifecycle boundary and in real execution.
    main_agent.dynamic_tool_lifecycle._candidate_verifier = lambda candidate, sandbox: None
    definition = build_task_definition(
        objective="report reusable code",
        task_type="verification",
        extra_allowed_tools=["write_task_output"],
        expected_outputs=["output/result.json"],
    )
    task = Task(job_id=main_agent.job.job_id, definition=definition)
    main_agent.scheduler.add_tasks([task])
    main_agent.scheduler.dispatch([task])
    main_agent._run_dispatched_task(task)
    handle = main_agent.dispatcher.get_handle(task.task_id)
    deadline = time.monotonic() + 5
    while not handle.is_finished() and time.monotonic() < deadline:
        time.sleep(0.01)
    main_agent._collect_finished_subagents()

    # Before independent result validation, no generated-tool record exists.
    assert main_agent.list_dynamic_tools() == []
    main_agent.validate_outputs(main_agent._new_completed_tasks())
    records = main_agent.list_dynamic_tools()
    assert len(records) == 1
    assert records[0]["support_count"] == 1
    assert records[0]["status"] == DynamicToolStatus.PENDING.value
    sandbox = main_agent.sandbox_manager.get(task.task_id)
    assert (sandbox.output_dir / "reusable_code_candidates.json").is_file()
    context = main_agent.context_builder.build(
        job=main_agent.job,
        dag=main_agent.scheduler.dag,
        current_decision="continue",
        recent_events=main_agent.task_repo.list_events(main_agent.job.job_id),
        unresolved_issues=[],
    )
    assert _candidate()["code"] not in context.text


def test_non_read_only_tool_waits_for_explicit_approval_without_failing_job(
    lifecycle, tmp_path: Path, job
) -> None:
    database, registry, manager, sandbox = lifecycle
    value = _candidate()
    value["risk_level"] = "restricted_write"
    source = _write_sidecar(tmp_path / "candidate.json", value)
    for index in range(3):
        manager.ingest_sidecar(
            source,
            job_id=job.job_id,
            task_id=f"task-{index}",
            attempt_id=f"attempt-{index}",
            task_type="verification",
            sandbox_ctx=sandbox,
        )
    record = manager.list_records(include_code=True)[0]
    assert record["status"] == DynamicToolStatus.AWAITING_APPROVAL.value
    assert registry.get(record["tool_name"]) is None

    service = InterventionService(database)
    request = service.create_for_dynamic_tool(
        job, record, source_task_id="task-2"
    )
    resolution = service.resolve(
        request.request_id,
        {"approved": True, "reason": "reviewed"},
        responded_by="owner",
    )
    assert resolution.request.status == InterventionStatus.APPROVED
    assert resolution.job.status == request.previous_job_status
    assert manager.approve(record["tool_id"])
    assert manager.repository.get(record["tool_id"])["status"] == DynamicToolStatus.ACTIVE.value
