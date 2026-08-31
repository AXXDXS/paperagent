"""Deterministic workflow phase advancement for reproduction jobs."""

from __future__ import annotations

import json
import shlex
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from repro_agent.domain.enums import ExperimentTier, JobStatus, ReproductionStatus, TaskStatus
from repro_agent.domain.experiment import ExperimentRun
from repro_agent.domain.job import ReproductionJob
from repro_agent.domain.task import Task
from repro_agent.evaluation.tier_gate import TierGate
from repro_agent.orchestrator.task_factory import build_task_definition
from repro_agent.schemas.results import ResultValidationError, TaskResultEnvelope


@dataclass
class PhaseDecision:
    job_status: JobStatus | None = None
    terminal_status: JobStatus | None = None
    tasks_to_create: list[Task] = field(default_factory=list)
    reproduction_status: ReproductionStatus | None = None
    reason: str = ""


_TIER_JOB_STATUS = {
    ExperimentTier.STATIC_CHECK: JobStatus.UNIT_TEST_RUNNING,
    ExperimentTier.UNIT_TEST: JobStatus.UNIT_TEST_RUNNING,
    ExperimentTier.SMOKE_TEST: JobStatus.SMOKE_TEST_RUNNING,
    ExperimentTier.REDUCED_EXPERIMENT: JobStatus.REDUCED_EXPERIMENT_RUNNING,
    ExperimentTier.FULL_EXPERIMENT: JobStatus.FULL_EXPERIMENT_RUNNING,
}


class PhaseCoordinator:
    def __init__(self):
        self.tier_gate = TierGate()

    def advance(
        self,
        job: ReproductionJob,
        tasks: list[Task],
        runs: list[ExperimentRun],
    ) -> PhaseDecision:
        if job.status in {
            JobStatus.FULLY_REPRODUCED,
            JobStatus.VERIFIED_REPRODUCTION_GAP,
            JobStatus.FAILED,
            JobStatus.BLOCKED_BY_MISSING_RESOURCE,
            JobStatus.CANCELLED,
        }:
            return PhaseDecision(job_status=job.status, terminal_status=job.status)

        terminal_failures = [
            task for task in tasks if task.status == TaskStatus.TERMINAL_FAILURE
        ]
        if terminal_failures:
            return PhaseDecision(
                job_status=JobStatus.FAILED,
                terminal_status=JobStatus.FAILED,
                reason=f"{len(terminal_failures)} task(s) reached terminal failure",
            )

        if job.status in {
            JobStatus.REFLECTION_REQUIRED,
            JobStatus.REFLECTION_PLANNING,
            JobStatus.AUDIT_RUNNING,
            JobStatus.ISSUE_FOUND,
            JobStatus.REPAIR_RUNNING,
            JobStatus.NO_ISSUE_FOUND,
        }:
            return PhaseDecision(job_status=job.status)

        if job.status == JobStatus.RERUN_REQUIRED:
            pending_reruns = [
                task
                for task in tasks
                if task.definition.inputs.get("reflection_id")
                and task.definition.task_type == "experiment_execution"
                and task.status != TaskStatus.SUCCEEDED
            ]
            if pending_reruns:
                return PhaseDecision(job_status=job.status)

        resource = self._latest_succeeded(tasks, "resource_check")
        if resource is not None:
            payload = self._payload(resource)
            if payload and payload.get("blocking_issues"):
                return PhaseDecision(
                    job_status=JobStatus.BLOCKED_BY_MISSING_RESOURCE,
                    terminal_status=JobStatus.BLOCKED_BY_MISSING_RESOURCE,
                    reason="resource check reported blocking issues",
                )

        environment = self._latest_succeeded(tasks, "environment_build")
        if environment is None:
            return PhaseDecision()

        experiment_id = (job.inputs.target_experiments or ["main_experiment"])[0]
        experiment_tasks = [t for t in tasks if t.definition.task_type == "experiment_execution"]
        successful_tiers = {
            run.tier
            for run in runs
            if run.experiment_id == experiment_id and run.exit_code == 0
        }

        if ExperimentTier.FULL_EXPERIMENT not in successful_tiers:
            gate = self.tier_gate.evaluate(experiment_id, runs)
            if not gate.allowed or gate.next_tier is None:
                return PhaseDecision(reason=gate.reason)
            tier = gate.next_tier
            existing = next(
                (
                    t
                    for t in experiment_tasks
                    if t.definition.inputs.get("tier") == tier.value
                    and t.status not in {TaskStatus.CANCELLED, TaskStatus.TERMINAL_FAILURE}
                ),
                None,
            )
            if existing is not None:
                return PhaseDecision(job_status=_TIER_JOB_STATUS[tier])
            command = self._command_for(job, tasks, tier)
            if not command:
                return PhaseDecision(
                    job_status=JobStatus.BLOCKED_BY_MISSING_RESOURCE,
                    terminal_status=JobStatus.BLOCKED_BY_MISSING_RESOURCE,
                    reason="no executable experiment command was discovered or supplied",
                )
            previous = self._task_for_previous_tier(experiment_tasks, tier)
            effective_repository_path = (
                str(previous.definition.inputs.get("repository_path", ""))
                if previous is not None
                else ""
            ) or job.inputs.repository_path
            spec_task = self._latest_succeeded(tasks, "specification")
            resource_task = self._latest_succeeded(tasks, "resource_check")
            spec_payload = self._payload(spec_task) if spec_task else None
            if spec_payload and spec_payload.get("unresolved_conflicts"):
                return PhaseDecision(
                    job_status=JobStatus.BLOCKED_BY_MISSING_RESOURCE,
                    terminal_status=JobStatus.BLOCKED_BY_MISSING_RESOURCE,
                    reason="experiment specification has unresolved conflicts",
                )
            environment_payload = self._payload(environment) or {}
            resource_payload = self._payload(resource_task) if resource_task else {}
            confirmed_plan = job.inputs.confirmed_execution_plan or {}
            execution_timeout_seconds = int(
                confirmed_plan.get(
                    "timeout_seconds", job.inputs.max_runtime_seconds or 600
                )
            )
            tier_report_caps = {
                ExperimentTier.STATIC_CHECK: 300,
                ExperimentTier.UNIT_TEST: 600,
                ExperimentTier.SMOKE_TEST: 900,
                ExperimentTier.REDUCED_EXPERIMENT: 3600,
            }
            expected_duration_seconds = (
                execution_timeout_seconds
                if tier == ExperimentTier.FULL_EXPERIMENT
                else min(
                    execution_timeout_seconds,
                    tier_report_caps.get(tier, execution_timeout_seconds),
                )
            )
            dependencies = [
                item.task_id
                for item in (previous, environment, spec_task, resource_task)
                if item is not None
            ]
            dependencies = list(dict.fromkeys(dependencies))
            manifest = self._execution_manifest(
                spec_payload or {}, resource_payload or {}
            )
            task = Task(
                job_id=job.job_id,
                definition=build_task_definition(
                    objective=f"运行 {tier.value} 层级并记录完整运行证据",
                    task_type="experiment_execution",
                    dependencies=dependencies,
                    inputs={
                        "experiment_id": experiment_id,
                        "tier": tier.value,
                        "command": command,
                        # A validated repair is attempt-scoped.  Carry the exact
                        # repository snapshot used by the preceding successful
                        # tier forward, instead of silently returning to the
                        # user's original broken repository on the next tier.
                        "repository_path": effective_repository_path,
                        "dataset_paths": job.inputs.dataset_paths,
                        "model_paths": job.inputs.model_paths,
                        "checkpoint_paths": job.inputs.checkpoint_paths,
                        "timeout_seconds": execution_timeout_seconds,
                        "metrics_output_path": confirmed_plan.get(
                            "metrics_output_path", "output://metrics.json"
                        ),
                        "execution_manifest": manifest,
                        "execution_image": (
                            environment_payload.get("environment_ref")
                            or environment_payload.get("image_ref", "")
                        ),
                        "environment_backend": environment_payload.get(
                            "environment_backend", "docker"
                        ),
                        "environment_name": environment_payload.get(
                            "environment_name", ""
                        ),
                        "python_version": environment_payload.get(
                            "python_version", "3.11"
                        ),
                        "environment_base_image": environment.definition.inputs.get(
                            "base_image", ""
                        ),
                        # This is the directory users review before dispatch.
                        # Sandbox staging later materializes the repository at
                        # the same workspace location, so confirmation and
                        # actual execution cannot silently diverge.
                        "working_dir": confirmed_plan.get(
                            "working_dir", "workspace://repository"
                        ),
                        "cpu_cores": confirmed_plan.get(
                            "cpu_cores", job.inputs.cpu_cores or 1.0
                        ),
                        "memory_mb": confirmed_plan.get(
                            "memory_mb", job.inputs.memory_mb or 1024
                        ),
                        "disk_mb": confirmed_plan.get(
                            "disk_mb", job.inputs.disk_mb or 4096
                        ),
                        "gpu_count": confirmed_plan.get(
                            "gpu_count", job.inputs.gpu_count or 0
                        ),
                        "gpu_memory_gb": confirmed_plan.get(
                            "gpu_memory_gb", job.inputs.gpu_memory_gb or 0.0
                        ),
                        "experiment_environment": dict(
                            confirmed_plan.get("experiment_environment", {})
                        ),
                        "experiment_secret_env_vars": list(
                            confirmed_plan.get("experiment_secret_env_vars", [])
                        ),
                        "network_enabled": bool(
                            confirmed_plan.get("network_enabled", False)
                        ),
                        "network_hosts": list(
                            confirmed_plan.get("network_hosts", [])
                        ),
                        "tier_command_verified": self._tier_command_verified(
                            job, tasks, tier
                        ),
                        "creation_key": f"experiment:{experiment_id}:{tier.value}",
                    },
                    restrict_tools=["execute_command", "read_file", "hash_path"],
                    expected_outputs=["output/result.json", "output/candidate_memory.md"],
                    expected_duration_seconds=max(1, expected_duration_seconds),
                    # 报备预计时间按门禁层级设置；绝对硬上限仍与用户确认
                    # 的执行超时同尺度，并保留收尾/取消安全余量。
                    soft_timeout_seconds=execution_timeout_seconds + 600,
                    hard_timeout_seconds=execution_timeout_seconds + 1800,
                ),
            )
            return PhaseDecision(job_status=_TIER_JOB_STATUS[tier], tasks_to_create=[task])

        full_task = next(
            (
                task
                for task in reversed(experiment_tasks)
                if task.definition.inputs.get("tier") == ExperimentTier.FULL_EXPERIMENT.value
                and task.status == TaskStatus.SUCCEEDED
            ),
            None,
        )
        verification_candidates = [
            task
            for task in tasks
            if task.definition.task_type == "verification"
            and task.status == TaskStatus.SUCCEEDED
            and (full_task is None or full_task.task_id in task.dependencies)
        ]
        verification = verification_candidates[-1] if verification_candidates else None
        if verification is not None:
            payload = self._payload(verification)
            if payload is None:
                return PhaseDecision(
                    job_status=JobStatus.FAILED,
                    terminal_status=JobStatus.FAILED,
                    reason="validated verification result cannot be read",
                )
            if payload.get("mock"):
                return PhaseDecision(
                    job_status=JobStatus.USER_REPORT_READY,
                    terminal_status=JobStatus.USER_REPORT_READY,
                    reproduction_status=ReproductionStatus.PIPELINE_ONLY,
                    reason="mock execution is diagnostic-only",
                )
            if not payload.get("verification_valid"):
                return PhaseDecision(
                    job_status=JobStatus.FAILED,
                    terminal_status=JobStatus.FAILED,
                    reason="strict verification evidence is incomplete",
                )
            comparisons = payload.get("comparisons") or []
            if comparisons and all(item.get("within_tolerance") for item in comparisons):
                return PhaseDecision(
                    job_status=JobStatus.FULLY_REPRODUCED,
                    terminal_status=JobStatus.FULLY_REPRODUCED,
                    reproduction_status=ReproductionStatus.FULLY_REPRODUCED,
                    reason="all strictly verified metrics are within tolerance",
                )
            return PhaseDecision(
                job_status=JobStatus.REFLECTION_REQUIRED,
                reason="strict verification found an out-of-tolerance result",
            )
        active_verification = next(
            (
                task
                for task in tasks
                if task.definition.task_type == "verification"
                and (full_task is None or full_task.task_id in task.dependencies)
                and task.status not in {TaskStatus.CANCELLED, TaskStatus.TERMINAL_FAILURE}
            ),
            None,
        )
        if active_verification is not None:
            return PhaseDecision(job_status=JobStatus.RESULT_VERIFICATION_RUNNING)

        spec_task = self._latest_succeeded(tasks, "specification")
        code_task = self._latest_succeeded(tasks, "code_analysis")
        dependencies = [
            task.task_id for task in (spec_task, code_task, full_task) if task is not None
        ]
        full_payload = self._payload(full_task) if full_task else {}
        verification_evidence = (full_payload or {}).get("verification_evidence", [])
        verification_task = Task(
            job_id=job.job_id,
            definition=build_task_definition(
                objective="独立验证正式实验指标、证据链与可追溯性",
                task_type="verification",
                dependencies=dependencies,
                inputs={
                    "experiment_id": experiment_id,
                    "repository_path": (
                        str(full_task.definition.inputs.get("repository_path", ""))
                        if full_task is not None
                        else ""
                    )
                    or job.inputs.repository_path,
                    "verification_evidence": verification_evidence,
                    # If prediction/label artifacts are present, the verifier
                    # will independently recompute common metrics from them.
                    "recompute_metrics": any(
                        isinstance(item, dict)
                        and any(token in str(item.get("relative_path", "")).lower() for token in ("predict", "label", "target"))
                        for item in verification_evidence
                    ),
                    "creation_key": f"verification:{experiment_id}:{runs[-1].run_id}",
                },
                restrict_tools=["read_file", "get_file_stat", "hash_path"],
                expected_outputs=["output/result.json", "output/candidate_memory.md"],
            ),
        )
        return PhaseDecision(
            job_status=JobStatus.RESULT_VERIFICATION_RUNNING,
            tasks_to_create=[verification_task],
        )

    @staticmethod
    def _latest_succeeded(tasks: list[Task], task_type: str) -> Task | None:
        found = [
            task
            for task in tasks
            if task.definition.task_type == task_type and task.status == TaskStatus.SUCCEEDED
        ]
        return found[-1] if found else None

    @staticmethod
    def _payload(task: Task) -> dict | None:
        path = task.outputs.get("result.json")
        if not path:
            return None
        try:
            return TaskResultEnvelope.from_file(
                path,
                expected_task_id=task.task_id,
                expected_attempt_id=task.active_attempt_id,
                expected_task_type=task.definition.task_type,
            ).payload
        except ResultValidationError:
            return None

    def _command_for(
        self, job: ReproductionJob, tasks: list[Task], tier: ExperimentTier
    ) -> list[str]:
        confirmed_commands = (
            job.inputs.confirmed_execution_plan.get("tier_commands", {})
            if job.inputs.confirmed_execution_plan
            else {}
        )
        confirmed = confirmed_commands.get(tier.value)
        if isinstance(confirmed, list) and confirmed and all(
            isinstance(part, str) and part for part in confirmed
        ):
            return list(confirmed)
        code_task = self._latest_succeeded(tasks, "code_analysis")
        payload = self._payload(code_task) if code_task else None
        tier_commands = payload.get("tier_commands", {}) if payload else {}
        if tier.value in tier_commands and isinstance(tier_commands[tier.value], list):
            return [str(part) for part in tier_commands[tier.value]]
        if job.inputs.user_run_commands:
            order = list(ExperimentTier)
            index = order.index(tier)
            if len(job.inputs.user_run_commands) >= len(order):
                return shlex.split(job.inputs.user_run_commands[index])
            if tier == ExperimentTier.STATIC_CHECK:
                return ["python", "-m", "compileall", "-q", "."]
            if tier == ExperimentTier.UNIT_TEST:
                return ["python", "-m", "pytest", "-q"]
            return shlex.split(job.inputs.user_run_commands[0])
        entry_points = payload.get("entry_points", []) if payload else []
        if tier == ExperimentTier.STATIC_CHECK:
            return ["python", "-m", "compileall", "-q", "."]
        if tier == ExperimentTier.UNIT_TEST:
            return ["python", "-m", "pytest", "-q"]
        if entry_points:
            return ["python", str(entry_points[0])]
        return []

    def _tier_command_verified(
        self, job: ReproductionJob, tasks: list[Task], tier: ExperimentTier
    ) -> bool:
        confirmed = (
            job.inputs.confirmed_execution_plan.get("tier_commands", {}).get(tier.value)
            if job.inputs.confirmed_execution_plan
            else None
        )
        if isinstance(confirmed, list) and confirmed:
            return True
        if tier in {ExperimentTier.STATIC_CHECK, ExperimentTier.UNIT_TEST}:
            return True
        # Commands inferred by the semantic code-analysis model are useful for
        # diagnostics, but model output alone is not provenance.  Only a full,
        # explicitly supplied five-tier contract can authorize a strict claim.
        return len(job.inputs.user_run_commands) >= len(list(ExperimentTier))

    @staticmethod
    def _execution_manifest(spec: dict, resources: dict) -> dict:
        fields = spec.get("fields", {}) or {}

        def field_value(*names: str):
            for name in names:
                value = fields.get(name)
                if isinstance(value, dict) and "value" in value:
                    return value["value"]
            return None

        config_digest = spec.get("spec_digest", "")
        if not config_digest and spec:
            config_digest = hashlib.sha256(
                json.dumps(spec, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
        gpu_info = resources.get("gpu_info", {}) or {}
        cuda_info = resources.get("cuda_info", {}) or {}
        hardware_identifier = json.dumps(
            {
                "gpu": gpu_info.get("devices") or gpu_info.get("name") or "cpu",
                "cuda": cuda_info.get("version") or cuda_info.get("available", False),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        seed = field_value("seed", "random_seed", "random seed")
        try:
            seed = int(seed) if seed is not None else None
        except (TypeError, ValueError):
            seed = None
        return {
            "config_digest": config_digest,
            "model_identifier": str(
                field_value("model_identifier", "model_name", "model") or ""
            ),
            "seed": seed,
            "hardware_identifier": hardware_identifier,
        }

    @staticmethod
    def _task_for_previous_tier(tasks: list[Task], tier: ExperimentTier) -> Task | None:
        order = list(ExperimentTier)
        index = order.index(tier)
        if index == 0:
            return None
        previous = order[index - 1].value
        matching = [
            task
            for task in tasks
            if task.definition.inputs.get("tier") == previous and task.status == TaskStatus.SUCCEEDED
        ]
        return matching[-1] if matching else None
