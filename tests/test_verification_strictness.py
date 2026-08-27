from __future__ import annotations

from repro_agent.agents.verification.agent import ResultVerificationAgent
from repro_agent.domain.task import Task
from repro_agent.orchestrator.task_factory import build_task_definition
from repro_agent.providers.mock import MockLLMProvider
from repro_agent.sandbox.manager import SandboxManager
from repro_agent.tools.authorization import ToolAuthorizer


def _run_verification(tmp_path, *, mock: bool):
    definition = build_task_definition(
        objective="verify",
        task_type="verification",
        inputs={
            "experiment_spec": {"experiment_id": "exp", "expected_results": {}},
            "experiment_run": {
                "experiment_id": "exp",
                "tier": "full_experiment",
                "run_id": "run-1",
                "exit_code": 0,
                "metrics": {},
                "mock": mock,
            },
        },
    )
    task = Task(job_id="job", definition=definition, active_attempt_id="attempt-1")
    sandbox = SandboxManager(tmp_path / "sandboxes").create_sandbox(task)
    auth = ToolAuthorizer().authorize(
        task_id=task.task_id,
        task_type="verification",
        allowed_tools=definition.allowed_tools,
        sandbox_ctx=sandbox,
    )
    return ResultVerificationAgent(task, auth, MockLLMProvider()).run()


def test_real_verification_fails_closed_without_metrics_or_traceability(tmp_path) -> None:
    result = _run_verification(tmp_path, mock=False)

    assert result.succeeded is False
    assert result.failure_report is not None


def test_mock_verification_is_diagnostic_but_can_finish(tmp_path) -> None:
    result = _run_verification(tmp_path, mock=True)

    assert result.succeeded is True
    assert result.outputs["mock"] is True
    assert result.outputs["verification_valid"] is False
