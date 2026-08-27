"""ReproductionJob 实体（设计文档 §18.1）与预算/限流参数。

预算相关字段直接对应设计文档 §11.9《防止无限反思和重跑》：
    max_reflection_rounds / max_full_experiment_reruns /
    max_audit_tasks_per_round / max_gpu_budget / max_total_runtime。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from repro_agent.domain.common import iso, new_id, utc_now
from repro_agent.domain.enums import JobStatus, ReproductionStatus


@dataclass
class JobBudget:
    """预算与限流配置（§1.2 可选输入 + §11.9 防止无限反思和重跑）。

    默认值取自设计文档 §11.9 的建议默认值。
    """

    max_parallel_agents: int = 8
    max_reflection_rounds: int = 3
    max_full_experiment_reruns: int = 2
    max_audit_tasks_per_round: int = 12
    max_gpu_hours: Optional[float] = None
    max_total_runtime_seconds: Optional[int] = None
    max_model_call_budget_usd: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_parallel_agents": self.max_parallel_agents,
            "max_reflection_rounds": self.max_reflection_rounds,
            "max_full_experiment_reruns": self.max_full_experiment_reruns,
            "max_audit_tasks_per_round": self.max_audit_tasks_per_round,
            "max_gpu_hours": self.max_gpu_hours,
            "max_total_runtime_seconds": self.max_total_runtime_seconds,
            "max_model_call_budget_usd": self.max_model_call_budget_usd,
        }


@dataclass
class JobInputs:
    """系统输入（§1.2）。"""

    paper_path: str
    repository_path: str
    target_experiments: list[str] = field(default_factory=list)
    appendix_paths: list[str] = field(default_factory=list)
    supplementary_paths: list[str] = field(default_factory=list)
    dataset_paths: list[str] = field(default_factory=list)
    model_paths: list[str] = field(default_factory=list)
    checkpoint_paths: list[str] = field(default_factory=list)
    dataset_download_urls: list[str] = field(default_factory=list)
    user_run_commands: list[str] = field(default_factory=list)
    user_environment_notes: str = ""
    # Human-readable Conda environment name.  Empty means derive it from the
    # repository directory while the content fingerprint remains authoritative
    # for cache validation.
    environment_name: str = ""
    # Non-sensitive values that the target experiment requires at runtime
    # (for example MODEL_NAME or MODEL_API_BASE).  Credential values are
    # deliberately excluded; only their environment-variable names are kept.
    experiment_runtime_config: dict[str, str] = field(default_factory=dict)
    required_experiment_configurations: list[dict[str, Any]] = field(default_factory=list)
    experiment_secret_env_vars: list[str] = field(default_factory=list)
    # Exact user-approved run plan captured after the initial resource check
    # and before environment construction.  Secret values are never stored here; only
    # environment-variable names are persisted.
    confirmed_execution_plan: dict[str, Any] = field(default_factory=dict)
    cpu_cores: Optional[float] = None
    memory_mb: Optional[int] = None
    disk_mb: Optional[int] = None
    gpu_count: Optional[int] = None
    gpu_memory_gb: Optional[float] = None
    max_runtime_seconds: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_path": self.paper_path,
            "repository_path": self.repository_path,
            "target_experiments": self.target_experiments,
            "appendix_paths": self.appendix_paths,
            "supplementary_paths": self.supplementary_paths,
            "dataset_paths": self.dataset_paths,
            "model_paths": self.model_paths,
            "checkpoint_paths": self.checkpoint_paths,
            "dataset_download_urls": self.dataset_download_urls,
            "user_run_commands": self.user_run_commands,
            "user_environment_notes": self.user_environment_notes,
            "environment_name": self.environment_name,
            "experiment_runtime_config": self.experiment_runtime_config,
            "required_experiment_configurations": self.required_experiment_configurations,
            "experiment_secret_env_vars": self.experiment_secret_env_vars,
            "confirmed_execution_plan": self.confirmed_execution_plan,
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "disk_mb": self.disk_mb,
            "gpu_count": self.gpu_count,
            "gpu_memory_gb": self.gpu_memory_gb,
            "max_runtime_seconds": self.max_runtime_seconds,
        }


@dataclass
class ReproductionJob:
    """复现任务实体（设计文档 §18.1 ``ReproductionJob``）。"""

    inputs: JobInputs
    job_id: str = field(default_factory=lambda: new_id("job"))
    budget: JobBudget = field(default_factory=JobBudget)
    status: JobStatus = JobStatus.JOB_CREATED
    reflection_round: int = 0
    full_experiment_rerun_count: int = 0
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    final_reproduction_status: Optional[ReproductionStatus] = None
    # 记录预算耗用情况，供主循环判断是否触及 §11.9 上限
    gpu_hours_used: float = 0.0
    model_call_cost_usd: float = 0.0
    model_input_tokens_used: int = 0
    model_output_tokens_used: int = 0
    model_calls_made: int = 0

    def touch(self) -> None:
        self.updated_at = utc_now()

    def elapsed_seconds(self) -> float:
        return (utc_now() - self.created_at).total_seconds()

    def budget_exhausted(self) -> tuple[bool, str]:
        """检查是否触及任一预算上限，返回 (是否耗尽, 原因)。"""

        if self.reflection_round >= self.budget.max_reflection_rounds:
            return True, "reflection_round_limit_reached"
        if (
            self.full_experiment_rerun_count
            >= self.budget.max_full_experiment_reruns
        ):
            return True, "full_experiment_rerun_limit_reached"
        if (
            self.budget.max_total_runtime_seconds is not None
            and self.elapsed_seconds() >= self.budget.max_total_runtime_seconds
        ):
            return True, "total_runtime_limit_reached"
        if (
            self.budget.max_gpu_hours is not None
            and self.gpu_hours_used >= self.budget.max_gpu_hours
        ):
            return True, "gpu_budget_limit_reached"
        if (
            self.budget.max_model_call_budget_usd is not None
            and self.model_call_cost_usd >= self.budget.max_model_call_budget_usd
        ):
            return True, "model_call_budget_limit_reached"
        return False, ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "inputs": self.inputs.to_dict(),
            "budget": self.budget.to_dict(),
            "status": self.status.value,
            "reflection_round": self.reflection_round,
            "full_experiment_rerun_count": self.full_experiment_rerun_count,
            "created_at": iso(self.created_at),
            "updated_at": iso(self.updated_at),
            "final_reproduction_status": self.final_reproduction_status.value
            if self.final_reproduction_status
            else None,
            "gpu_hours_used": self.gpu_hours_used,
            "model_call_cost_usd": self.model_call_cost_usd,
            "model_input_tokens_used": self.model_input_tokens_used,
            "model_output_tokens_used": self.model_output_tokens_used,
            "model_calls_made": self.model_calls_made,
        }
