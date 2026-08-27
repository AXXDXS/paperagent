"""断点续跑与逐次工具调用审计的回归测试。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from repro_agent.agents.base import AgentRunResult, BaseSubAgent
from repro_agent.agents.registry import SUB_AGENT_REGISTRY
from repro_agent.domain.enums import FailureType, TaskStatus
from repro_agent.domain.task import FailureReport, Task
from repro_agent.orchestrator.main_agent import MainAgent, MainAgentConfig
from repro_agent.orchestrator.task_factory import build_task_definition
from repro_agent.schemas.results import TaskResultEnvelope


def _wait_for_handle(handle, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not handle.is_finished() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert handle.is_finished()


def _config(work_dir: Path) -> MainAgentConfig:
    return MainAgentConfig(
        memory_root=str(work_dir / "memory"),
        sandbox_root=str(work_dir / "sandboxes"),
        snapshot_root=str(work_dir / "snapshots"),
        db_path=str(work_dir / "agent.db"),
        model="mock-model",
        mock_execution=True,
        require_execution_parameter_confirmation=False,
        main_loop_wait_seconds=0.001,
    )


def _register(monkeypatch, task_type: str, agent_cls) -> None:
    monkeypatch.setitem(SUB_AGENT_REGISTRY, task_type, agent_cls)


def _make_task(agent: MainAgent, task_type: str, *, inputs: dict | None = None) -> Task:
    task = Task(
        job_id=agent.job.job_id,
        definition=build_task_definition(
            objective=f"recovery test {task_type}",
            task_type=task_type,
            inputs=inputs or {},
            extra_allowed_tools=["find_files"],
            expected_outputs=["output/result.json"],
        ),
    )
    agent.scheduler.add_tasks([task])
    return task


class _ToolAuditAgent(BaseSubAgent):
    task_type = "tool_audit"

    def run(self) -> AgentRunResult:
        self.call_tool(
            "find_files",
            pattern="*.py",
            root=self.task.definition.inputs["repository_path"],
            max_results=10,
        )
        # 调用完成后故意保持运行，以验证审计并非等任务结束后才落库。
        time.sleep(0.2)
        self.write_json_output("result.json", {"ok": True})
        return AgentRunResult(succeeded=True)


class _CheckpointedAgent(BaseSubAgent):
    task_type = "checkpointed"

    def run(self) -> AgentRunResult:
        response = self.call_llm("return the durable response", tool_names=[])
        if self.task.attempt == 1:
            return AgentRunResult(
                succeeded=False,
                failure_report=FailureReport(
                    failure_type=FailureType.UNKNOWN_ERROR,
                    failed_step="after_llm",
                    error_message="simulated interruption after persisted LLM response",
                ),
            )
        self.write_json_output("result.json", {"response": response.content})
        return AgentRunResult(succeeded=True)


def test_tool_results_are_persisted_before_subagent_finishes(main_agent, sample_repo, monkeypatch):
    _register(monkeypatch, "tool_audit", _ToolAuditAgent)
    task = _make_task(main_agent, "tool_audit", inputs={"repository_path": str(sample_repo)})

    main_agent.scheduler.dispatch([task])
    main_agent._run_dispatched_task(task)

    deadline = time.monotonic() + 3
    records = []
    while time.monotonic() < deadline:
        records = main_agent.tool_invocation_repo.list_by_task(task.task_id)
        if records:
            break
        time.sleep(0.01)

    assert records
    assert records[0]["tool_name"] == "find_files"
    assert records[0]["attempt_id"] == task.active_attempt_id
    assert records[0]["succeeded"] is True
    assert records[0]["result"]

    handle = main_agent.dispatcher.get_handle(task.task_id)
    assert handle is not None
    _wait_for_handle(handle)


def test_retry_guidance_invalidates_the_previous_llm_checkpoint(
    main_agent, mock_provider, monkeypatch
):
    _register(monkeypatch, "checkpointed", _CheckpointedAgent)
    task = _make_task(main_agent, "checkpointed")

    main_agent.scheduler.dispatch([task])
    main_agent._run_dispatched_task(task)
    first_handle = main_agent.dispatcher.get_handle(task.task_id)
    assert first_handle is not None
    _wait_for_handle(first_handle)
    main_agent._collect_finished_subagents()
    assert task.status == TaskStatus.FAILED_RETRYABLE
    # 工具分配权上收后首次派发会先由主智能体定制工具白名单（一次
    # LLM 调用，输出不合法时安全回退模板），随后子智能体自身再调
    # 一次 LLM——因此失败后的计数是 2。
    assert mock_provider.call_count == 2

    main_agent.scheduler.retry(
        task,
        guidance="重试注意事项：不要重复上次中断路径，先检查恢复条件。",
    )
    main_agent.scheduler.refresh_task_states()
    main_agent.scheduler.dispatch([task])
    main_agent._run_dispatched_task(task)
    second_handle = main_agent.dispatcher.get_handle(task.task_id)
    assert second_handle is not None
    _wait_for_handle(second_handle)
    main_agent._collect_finished_subagents()
    main_agent.validate_outputs(main_agent._new_completed_tasks())

    assert task.status == TaskStatus.SUCCEEDED
    # retry_guidance changes task inputs and the model-facing user message, so
    # the failed attempt's response must not be replayed from its checkpoint.
    # 重试派发不再重新定制工具（保留运行期累积授权），因此只新增子
    # 智能体自己的那一次调用。
    assert mock_provider.call_count == 3
    assert "retry_guidance" in task.definition.inputs
    assert "MAIN AGENT RETRY GUIDANCE" in mock_provider.call_log[-1][-1].content


def test_resume_accepts_a_valid_output_left_by_an_interrupted_subagent(
    main_agent, work_dir, mock_provider
):
    task = _make_task(main_agent, "paper_analysis")
    task.status = TaskStatus.RUNNING
    task.active_attempt_id = "attempt_interrupted"
    main_agent.task_repo.save(task)
    main_agent.scheduler.dag.replace_task(task)
    sandbox = main_agent.sandbox_manager.create_sandbox(task)
    result_path = sandbox.output_dir / "result.json"
    result_path.write_text(
        __import__("json").dumps(
            TaskResultEnvelope.succeeded(
                task_id=task.task_id,
                attempt_id=task.active_attempt_id,
                task_type=task.definition.task_type,
                payload={"extracted_parameters": []},
            ).to_dict()
        ),
        encoding="utf-8",
    )

    resumed = MainAgent.resume_from_storage(
        task.job_id,
        main_agent.config,
        mock_provider,
    )
    outcome = resumed.recover_interrupted_tasks()
    restored = resumed.scheduler.dag.get(task.task_id)

    assert outcome.recovered_succeeded_task_ids == [task.task_id]
    assert restored is not None
    assert restored.status == TaskStatus.SUCCEEDED
    assert restored.outputs["result.json"] == str(result_path)


def test_resume_requeues_an_interrupted_subagent_without_a_valid_result(
    main_agent, work_dir, mock_provider
):
    task = _make_task(main_agent, "paper_analysis")
    task.status = TaskStatus.RUNNING
    task.active_attempt_id = "attempt_lost"
    main_agent.task_repo.save(task)
    main_agent.scheduler.dag.replace_task(task)

    resumed = MainAgent.resume_from_storage(
        task.job_id,
        main_agent.config,
        mock_provider,
    )
    outcome = resumed.recover_interrupted_tasks()
    restored = resumed.scheduler.dag.get(task.task_id)

    assert outcome.requeued_task_ids == [task.task_id]
    assert restored is not None
    assert restored.status == TaskStatus.PENDING
    assert restored.active_attempt_id == "attempt_lost"


def test_recovery_never_requeues_while_container_termination_is_unconfirmed(
    main_agent,
) -> None:
    task = _make_task(main_agent, "experiment_execution")
    task.status = TaskStatus.RUNNING
    task.active_attempt_id = "attempt_orphaned"
    main_agent.task_repo.save(task)
    main_agent.scheduler.dag.replace_task(task)
    sandbox = main_agent.sandbox_manager.create_sandbox(task)
    state_path = sandbox.logs_dir / f"{task.active_attempt_id}.execution.json"
    state_path.write_text(
        json.dumps(
            {
                "status": "TERMINATION_FAILED",
                "container_name": "repro-orphaned",
            }
        ),
        encoding="utf-8",
    )

    class UnavailableTerminationBackend:
        @staticmethod
        def cancel(container_name: str) -> bool:
            return False

    main_agent.sandbox_manager.execution_backend = UnavailableTerminationBackend()

    with pytest.raises(RuntimeError, match="termination is unconfirmed"):
        main_agent.recover_interrupted_tasks()
    assert main_agent.scheduler.dag.get(task.task_id).status == TaskStatus.RUNNING


def test_resume_keeps_validated_upstream_tasks_and_runs_only_the_rest(
    job, work_dir, mock_provider
):
    config = _config(work_dir)
    first_agent = MainAgent(job, config, mock_provider)
    first_agent.bootstrap()

    paper_and_code = [
        task
        for task in first_agent.scheduler.dag.all_tasks()
        if task.definition.task_type in {"paper_analysis", "code_analysis"}
    ]
    first_agent.scheduler.refresh_task_states()
    first_agent.scheduler.dispatch(paper_and_code)
    for task in paper_and_code:
        first_agent._run_dispatched_task(task)
    for task in paper_and_code:
        handle = first_agent.dispatcher.get_handle(task.task_id)
        assert handle is not None
        _wait_for_handle(handle)
    first_agent._collect_finished_subagents()
    first_agent.validate_outputs(first_agent._new_completed_tasks())
    attempts_before = {task.task_id: task.attempt for task in paper_and_code}

    resumed = MainAgent.resume_from_storage(job.job_id, config, mock_provider)
    resumed.recover_interrupted_tasks()
    outcome = resumed.run_until_finished(max_iterations=500)

    assert outcome.completed is True
    for task_id, attempt in attempts_before.items():
        restored = resumed.scheduler.dag.get(task_id)
        assert restored is not None
        assert restored.status == TaskStatus.SUCCEEDED
        assert restored.attempt == attempt


def test_recovery_replays_reflection_callbacks_without_duplicate_tasks(
    main_agent, mock_provider
):
    """Cover crashes after validation but before reflection/audit callbacks."""

    reflection = Task(
        job_id=main_agent.job.job_id,
        definition=build_task_definition(
            objective="recover reflection transition",
            task_type="reflection",
            inputs={
                "reflection_round": 1,
                "trigger_metrics": [],
                "creation_key": "reflection:1",
            },
            restrict_tools=[],
        ),
    )
    main_agent.scheduler.add_tasks([reflection])
    reflection.active_attempt_id = "attempt_reflection_committed"
    reflection_sandbox = main_agent.sandbox_manager.create_sandbox(reflection)
    reflection_result = reflection_sandbox.output_dir / "result.json"
    reflection_result.write_text(
        json.dumps(
            TaskResultEnvelope.succeeded(
                task_id=reflection.task_id,
                attempt_id=reflection.active_attempt_id,
                task_type="reflection",
                payload={
                    "likely_source": "config",
                    "hypotheses": [
                        {
                            "category": "C",
                            "description": "check config",
                            "confidence": 0.8,
                            "required_checks": ["batch_size"],
                            "priority": 1,
                        }
                    ],
                },
            ).to_dict()
        ),
        encoding="utf-8",
    )
    reflection.status = TaskStatus.SUCCEEDED
    reflection.outputs = {"result.json": str(reflection_result)}
    main_agent.task_repo.save(reflection)
    main_agent.scheduler.dag.replace_task(reflection)

    resumed = MainAgent.resume_from_storage(
        main_agent.job.job_id, main_agent.config, mock_provider
    )
    resumed.recover_interrupted_tasks()
    reports = resumed.reflection_repo.list_by_job(main_agent.job.job_id)
    audit_tasks = [
        task
        for task in resumed.scheduler.dag.all_tasks()
        if task.definition.inputs.get("audit_hypothesis_id")
    ]
    assert len(reports) == 1
    assert len(audit_tasks) == 1
    assert reports[0].audit_task_ids == [audit_tasks[0].task_id]

    # Simulate the second crash window: validation persisted SUCCEEDED, but the
    # audit-finding callback did not run.
    audit = audit_tasks[0]
    audit.active_attempt_id = "attempt_audit_committed"
    audit_sandbox = resumed.sandbox_manager.create_sandbox(audit)
    audit_result = audit_sandbox.output_dir / "result.json"
    audit_result.write_text(
        json.dumps(
            TaskResultEnvelope.succeeded(
                task_id=audit.task_id,
                attempt_id=audit.active_attempt_id,
                task_type=audit.definition.task_type,
                payload={"unresolved_conflicts": [], "fields": {}},
            ).to_dict()
        ),
        encoding="utf-8",
    )
    audit.status = TaskStatus.SUCCEEDED
    audit.outputs = {"result.json": str(audit_result)}
    resumed.task_repo.save(audit)
    resumed.scheduler.dag.replace_task(audit)

    resumed_again = MainAgent.resume_from_storage(
        main_agent.job.job_id, main_agent.config, mock_provider
    )
    resumed_again.recover_interrupted_tasks()
    restored_reports = resumed_again.reflection_repo.list_by_job(main_agent.job.job_id)
    restored_audits = [
        task
        for task in resumed_again.scheduler.dag.all_tasks()
        if task.definition.inputs.get("audit_hypothesis_id")
    ]
    assert len(restored_reports) == 1
    assert len(restored_audits) == 1
    assert [item.audit_task_id for item in restored_reports[0].audit_findings] == [
        audit.task_id
    ]
    assert resumed_again._pending_audit_task_ids[reports[0].reflection_id] == set()
