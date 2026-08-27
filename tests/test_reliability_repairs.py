from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from repro_agent.agents.experiment.agent import (
    ExperimentExecutionAgent,
    ExperimentExecutionResult,
)
from repro_agent.domain.enums import ExperimentTier, FailureType
from repro_agent.agents.paper.agent import normalize_paper_analysis_payload
from repro_agent.agents.verification.agent import ResultVerificationAgent
from repro_agent.domain.task import Task
from repro_agent.evidence.anti_cheat import scan_suspicious_markers
from repro_agent.llm_output import PAPER_ANALYSIS_SCHEMA, StructuredOutputError, parse_structured_json
from repro_agent.orchestrator.task_factory import build_task_definition
from repro_agent.providers.mock import MockLLMProvider
from repro_agent.sandbox.manager import SandboxManager
from repro_agent.tools.authorization import ToolAuthorizer


def _auth(task: Task, sandbox):
    return ToolAuthorizer().authorize(
        task_id=task.task_id,
        task_type=task.definition.task_type,
        allowed_tools=task.definition.allowed_tools,
        sandbox_ctx=sandbox,
    )


def test_staging_namespaces_same_basename_inputs(tmp_path: Path) -> None:
    first = tmp_path / "a" / "repository"
    second = tmp_path / "b" / "repository"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "marker.txt").write_text("first", encoding="utf-8")
    (second / "marker.txt").write_text("second", encoding="utf-8")
    task = Task(
        job_id="job",
        definition=build_task_definition(
            objective="stage",
            task_type="resource_check",
            inputs={"dataset_paths": [str(first), str(second)]},
        ),
    )

    sandbox = SandboxManager(tmp_path / "sandboxes").create_sandbox(task)
    staged = task.definition.inputs["dataset_paths"]

    assert len(staged) == 2
    assert staged[0] != staged[1]
    assert sandbox.resolve_readable_path(staged[0]).endswith("marker.txt") is False
    assert Path(sandbox.resolve_readable_path(staged[0]), "marker.txt").read_text(encoding="utf-8") == "first"
    assert Path(sandbox.resolve_readable_path(staged[1]), "marker.txt").read_text(encoding="utf-8") == "second"


def test_empty_dataset_has_explicit_manifest_semantics() -> None:
    manifest = ExperimentExecutionAgent._dataset_manifest([])

    assert manifest["version"] == 1
    assert manifest["kind"] == "none"
    assert manifest["items"] == []
    assert manifest["digest"].startswith("dataset-manifest:v1:")


@pytest.mark.parametrize(
    "termination_reason",
    ["timeout_killed", "log_limit_exceeded", "disk_limit_exceeded"],
)
def test_execution_resource_limit_is_routed_to_human_resource_intervention(
    termination_reason: str,
) -> None:
    result = ExperimentExecutionResult(
        tier=ExperimentTier.SMOKE_TEST.value,
        exit_code=137,
        stderr_tail="Traceback: process was interrupted",
        termination_reason=termination_reason,
    )

    report = ExperimentExecutionAgent._failure_report_for_unsuccessful_execution(
        result,
        tier=ExperimentTier.SMOKE_TEST,
        metrics_required=False,
    )

    assert report.failure_type == FailureType.RESOURCE_EXCEEDED
    assert report.metadata["termination_reason"] == termination_reason
    assert result.to_dict()["termination_reason"] == termination_reason


def test_execution_program_failure_keeps_existing_code_error_classification() -> None:
    result = ExperimentExecutionResult(
        tier=ExperimentTier.SMOKE_TEST.value,
        exit_code=1,
        stderr_tail="Traceback: application bug",
        termination_reason="completed",
    )

    report = ExperimentExecutionAgent._failure_report_for_unsuccessful_execution(
        result,
        tier=ExperimentTier.SMOKE_TEST,
        metrics_required=False,
    )

    assert report.failure_type == FailureType.CODE_ERROR


def test_llm_json_repair_and_schema_validation() -> None:
    content = "Here is the result:\n```json\n{\"parameters\": [], \"expected_results\": {}, \"notes\": \"ok\",}\n```"
    parsed = parse_structured_json(content, PAPER_ANALYSIS_SCHEMA)
    assert parsed["notes"] == "ok"

    with pytest.raises(StructuredOutputError):
        parse_structured_json('{"parameters": "not-a-list", "expected_results": {}}', PAPER_ANALYSIS_SCHEMA)


def test_paper_analysis_numeric_string_normalization() -> None:
    # Real-model failure mode: metrics arrive as "28.0%"-style strings, pages
    # as integers, confidence as a string, tolerance_type with odd casing.
    content = json.dumps(
        {
            "parameters": [
                {
                    "name": "learning_rate",
                    "value": "3e-4",
                    "experiment_scope": "all",
                    "provenance": "PAPER_EXPLICIT",
                    "page": 5,
                    "section": "4.1",
                    "original_text": "lr = 3e-4",
                    "confidence": "0.9",
                }
            ],
            "expected_results": {
                "WebShop_ExpeL_success_rate": {
                    "value": "28.0%",
                    "tolerance_type": "Absolute",
                    "tolerance": "1,234.5",
                    "tolerance_basis": "paper",
                }
            },
        }
    )
    parsed = parse_structured_json(
        content, PAPER_ANALYSIS_SCHEMA, normalize=normalize_paper_analysis_payload
    )

    parameter = parsed["parameters"][0]
    assert parameter["page"] == "5"
    assert parameter["confidence"] == 0.9
    metric = parsed["expected_results"]["WebShop_ExpeL_success_rate"]
    assert metric["value"] == 28.0
    assert metric["tolerance"] == 1234.5
    assert metric["tolerance_type"] == "absolute"

    # Ambiguous values are never guessed: uncoercible strings still fail closed.
    bad = (
        '{"parameters": [], "expected_results": {"m": '
        '{"value": "see Table 3", "tolerance_type": "absolute", "tolerance": 0.1}}}'
    )
    with pytest.raises(StructuredOutputError):
        parse_structured_json(
            bad, PAPER_ANALYSIS_SCHEMA, normalize=normalize_paper_analysis_payload
        )

    # Booleans are not numbers and must not be coerced.
    boolean_value = (
        '{"parameters": [], "expected_results": {"m": '
        '{"value": true, "tolerance_type": "absolute", "tolerance": 0.1}}}'
    )
    with pytest.raises(StructuredOutputError):
        parse_structured_json(
            boolean_value, PAPER_ANALYSIS_SCHEMA, normalize=normalize_paper_analysis_payload
        )


def test_verifier_reads_real_metrics_instead_of_run_payload(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    metrics = evidence_dir / "metrics.json"
    stdout = evidence_dir / "stdout.log"
    stderr = evidence_dir / "stderr.log"
    state = evidence_dir / "execution.json"
    metrics.write_text('{"accuracy": 0.9}', encoding="utf-8")
    stdout.write_text("training completed", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    state.write_text('{"status": "COMPLETED", "exit_code": 0}', encoding="utf-8")
    def evidence_item(role: str, path: Path, relative_path: str) -> dict:
        return {
            "role": role,
            "path": str(path),
            "relative_path": relative_path,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    evidence = [
        evidence_item("metrics", metrics, "metrics.json"),
        evidence_item("stdout_log", stdout, "stdout.log"),
        evidence_item("stderr_log", stderr, "stderr.log"),
        evidence_item("execution_state", state, "execution.json"),
    ]
    manifest_body = {
        "run_id": "run-1",
        "source": {"sha256": "source"},
        "datasets": [],
        "dataset_manifest": {"kind": "none"},
        "models": [],
        "metrics": {"path": str(metrics)},
        "config_digest": "config",
        "container_digest": "container",
    }
    provenance = {
        **manifest_body,
        "manifest_digest": hashlib.sha256(
            json.dumps(manifest_body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    definition = build_task_definition(
        objective="verify",
        task_type="verification",
        inputs={
            "experiment_spec": {
                "experiment_id": "exp",
                "expected_results": {"accuracy": {"value": 0.9, "tolerance_type": "absolute", "tolerance": 0}},
            },
            "experiment_run": {
                "experiment_id": "exp", "tier": "full_experiment", "run_id": "run-1",
                "exit_code": 0, "metrics": {"accuracy": 0.1}, "mock": False,
                "git_commit": "git", "container_digest": "container", "config_digest": "config",
                "dataset_digest": "dataset-manifest:v1:none", "model_identifier": "model",
                "seed": 1, "hardware_identifier": "cpu", "tier_command_verified": True,
                "artifact_provenance": provenance,
            },
            "verification_evidence": evidence,
            "implementation_summary": "The implementation computes accuracy from model predictions.",
        },
    )
    task = Task(job_id="job", definition=definition, active_attempt_id="attempt-1")
    sandbox = SandboxManager(tmp_path / "sandboxes").create_sandbox(task)
    agent = ResultVerificationAgent(task, _auth(task, sandbox), MockLLMProvider())
    result = agent.run()

    assert result.succeeded is True
    assert result.outputs["independently_read_metrics"] == {"accuracy": 0.9}
    assert result.outputs["verification_valid"] is True

    missing_digest = [dict(item) for item in task.definition.inputs["verification_evidence"]]
    missing_digest[0].pop("sha256")
    evidence_result = agent._verify_evidence(
        missing_digest,
        agent._run_from_dict(task.definition.inputs["experiment_run"]),
    )
    assert evidence_result["verified"] is False
    assert any("execution-time sha256" in error for error in evidence_result["errors"])


def test_anti_cheat_reads_real_code_evidence(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    source = repository / "train.py"
    source.write_text(
        "def train():\n    return 'hard-coded paper result'\n",
        encoding="utf-8",
    )
    definition = build_task_definition(
        objective="verify code",
        task_type="verification",
        inputs={
            "repository_path": str(repository),
            "code_findings": {
                "analysis_evidence": [
                    {"path": "train.py", "start_line": 1, "end_line": 2}
                ]
            },
        },
    )
    task = Task(job_id="job", definition=definition, active_attempt_id="attempt-1")
    sandbox = SandboxManager(tmp_path / "sandboxes").create_sandbox(task)
    agent = ResultVerificationAgent(task, _auth(task, sandbox), MockLLMProvider())

    material, errors = agent._anti_cheat_material(task.definition.inputs)

    assert errors == []
    assert any("hard-coded paper" in item for item in material)
    assert scan_suspicious_markers(*material).passed is False
