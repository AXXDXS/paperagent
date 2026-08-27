"""Durable usage, evidence and experiment-run accounting services."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from repro_agent.domain.common import new_id, utc_now
from repro_agent.domain.enums import ExperimentTier
from repro_agent.domain.experiment import ExperimentRun
from repro_agent.domain.job import ReproductionJob
from repro_agent.domain.task import Task
from repro_agent.evidence.hashing import sha256_of_file


class RuntimeAccountingService:
    def __init__(
        self,
        job: ReproductionJob,
        *,
        job_repo,
        task_repo,
        evidence_repo,
        experiment_run_repo,
        mock_execution: bool,
        input_cost_per_million_usd: float = 0.0,
        output_cost_per_million_usd: float = 0.0,
    ):
        self.job = job
        self.job_repo = job_repo
        self.task_repo = task_repo
        self.evidence_repo = evidence_repo
        self.experiment_run_repo = experiment_run_repo
        self.mock_execution = mock_execution
        self.input_cost_per_million_usd = input_cost_per_million_usd
        self.output_cost_per_million_usd = output_cost_per_million_usd
        self._lock = threading.Lock()

    def record_model_usage(self, params, response) -> None:
        usage = response.usage or {}
        input_tokens = int(
            usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
        )
        output_tokens = int(
            usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
        )
        input_details = (
            usage.get("input_tokens_details")
            or usage.get("prompt_tokens_details")
            or {}
        )
        cached_tokens = int(
            input_details.get("cached_tokens", usage.get("cached_tokens", 0)) or 0
        )
        cache_write_tokens = int(
            input_details.get(
                "cache_write_tokens", usage.get("cache_write_tokens", 0)
            )
            or 0
        )
        cost = (
            input_tokens * self.input_cost_per_million_usd
            + output_tokens * self.output_cost_per_million_usd
        ) / 1_000_000
        with self._lock:
            self.job.model_input_tokens_used += input_tokens
            self.job.model_output_tokens_used += output_tokens
            self.job.model_calls_made += 1
            self.job.model_call_cost_usd += cost
            self.job_repo.save(self.job)
            self.task_repo.record_event(
                self.job.job_id,
                None,
                "model_usage_recorded",
                {
                    "model": params.model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cached_tokens": cached_tokens,
                    "cache_write_tokens": cache_write_tokens,
                    "prompt_cache_key": params.prompt_cache_key,
                    "estimated_cost_usd": cost,
                },
            )

    def budget_limit_reason(self) -> str:
        budget = self.job.budget
        if (
            budget.max_total_runtime_seconds is not None
            and self.job.elapsed_seconds() >= budget.max_total_runtime_seconds
        ):
            return "total_runtime_limit_reached"
        if (
            budget.max_gpu_hours is not None
            and self.job.gpu_hours_used >= budget.max_gpu_hours
        ):
            return "gpu_budget_limit_reached"
        if (
            budget.max_model_call_budget_usd is not None
            and self.job.model_call_cost_usd >= budget.max_model_call_budget_usd
        ):
            return "model_call_budget_limit_reached"
        return ""

    def persist_task_evidence(self, task: Task) -> None:
        for relative_path, absolute_path in sorted(task.outputs.items()):
            path = Path(absolute_path)
            if not path.is_file():
                continue
            evidence_key = hashlib.sha256(
                f"{task.task_id}:{task.active_attempt_id}:{relative_path}".encode("utf-8")
            ).hexdigest()[:24]
            self.evidence_repo.record(
                evidence_id=f"evidence_{evidence_key}",
                job_id=self.job.job_id,
                task_id=task.task_id,
                kind="task_artifact",
                payload={
                    "attempt_id": task.active_attempt_id,
                    "task_type": task.definition.task_type,
                    "path": str(path),
                    "relative_path": relative_path,
                    "sha256": sha256_of_file(path),
                    "size_bytes": path.stat().st_size,
                },
            )

    def persist_experiment_run(self, task: Task, payload: dict[str, Any]) -> None:
        try:
            tier = ExperimentTier(
                payload.get("tier", task.definition.inputs.get("tier", ""))
            )
        except ValueError:
            return
        command = payload.get("command", task.definition.inputs.get("command", []))
        started_at = self._parse_timestamp(payload.get("started_at")) or task.started_at or utc_now()
        completed_at = self._parse_timestamp(payload.get("completed_at")) or utc_now()
        run = ExperimentRun(
            experiment_id=task.definition.inputs.get("experiment_id", "main_experiment"),
            job_id=self.job.job_id,
            tier=tier,
            run_id=payload.get("run_id") or new_id("run"),
            run_type="mock" if payload.get("mock") or self.mock_execution else tier.value,
            git_commit=payload.get("git_commit", ""),
            container_digest=payload.get(
                "container_digest", "mock" if self.mock_execution else ""
            ),
            config_digest=payload.get("config_digest", ""),
            dataset_digest=payload.get("dataset_digest", ""),
            dataset_manifest=payload.get("dataset_manifest", {}),
            model_identifier=payload.get("model_identifier", ""),
            seed=payload.get("seed"),
            hardware_identifier=payload.get("hardware_identifier", ""),
            command=json.dumps(command, ensure_ascii=False)
            if isinstance(command, list)
            else str(command),
            exit_code=payload.get("exit_code"),
            metrics=payload.get("metrics", {}),
            log_path=payload.get("log_path", ""),
            started_at=started_at,
            completed_at=completed_at,
            tier_command_verified=bool(payload.get("tier_command_verified", False)),
        )
        already_persisted = self.experiment_run_repo.exists(run.run_id)
        self.experiment_run_repo.save(run)
        if already_persisted:
            return
        gpu_count = int(task.definition.inputs.get("gpu_count") or 0)
        duration_seconds = float(payload.get("duration_seconds", 0.0) or 0.0)
        if not payload.get("mock") and gpu_count > 0 and duration_seconds > 0:
            self.job.gpu_hours_used += gpu_count * duration_seconds / 3600
            self.job_repo.save(self.job)
        provenance = payload.get("artifact_provenance")
        if isinstance(provenance, dict) and provenance:
            self.evidence_repo.record(
                evidence_id=f"evidence_manifest_{run.run_id}",
                job_id=self.job.job_id,
                task_id=task.task_id,
                kind="execution_manifest",
                payload=provenance,
            )

    @staticmethod
    def _parse_timestamp(value: Any):
        if not value or not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
