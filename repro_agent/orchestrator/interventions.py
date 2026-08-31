"""Human-in-the-loop 请求创建、校验与恢复服务。"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from datetime import timedelta
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

from repro_agent.domain.common import utc_now
from repro_agent.domain.enums import (
    FailureType,
    InterventionKind,
    InterventionStatus,
    JobStatus,
    TaskStatus,
)
from repro_agent.domain.intervention import InterventionRequest
from repro_agent.domain.job import ReproductionJob
from repro_agent.domain.task import Task
from repro_agent.orchestrator.execution_parameters import (
    EXECUTION_TIER_NAMES,
    ExecutionParameterValidationError,
    execution_parameter_fingerprint,
    execution_parameter_snapshot,
    execution_plan_fingerprint,
    execution_plan_snapshot,
)
from repro_agent.orchestrator.runtime_configuration import (
    materialize_runtime_configuration,
    normalize_requirements,
    runtime_network_configuration,
)
from repro_agent.storage.database import Database
from repro_agent.storage.repository import (
    InterventionRepository,
    JobRepository,
    TaskRepository,
)
from repro_agent.tools.authorization import ToolAuthorizer


class InterventionValidationError(ValueError):
    """人工回答不符合请求约束，且没有修改任何持久化状态。"""


@dataclass(frozen=True)
class InterventionResolution:
    request: InterventionRequest
    job: ReproductionJob
    task: Task | None


_PATH_LIST_FIELDS = {"dataset_paths", "model_paths", "checkpoint_paths"}
_STRING_LIST_FIELDS = _PATH_LIST_FIELDS | {
    "dataset_download_urls",
    "user_run_commands",
}
_NUMERIC_FIELDS = {
    "cpu_cores",
    "memory_mb",
    "disk_mb",
    "gpu_count",
    "gpu_memory_gb",
    "max_runtime_seconds",
}
_ALL_INPUT_FIELDS = _STRING_LIST_FIELDS | _NUMERIC_FIELDS | {"user_environment_notes"}


class InterventionService:
    """把“需要人工”转成可恢复状态，而不是一条不可交互的日志。"""

    def __init__(self, database: Database, tool_authorizer: ToolAuthorizer | None = None):
        self.db = database
        self.interventions = InterventionRepository(database)
        self.jobs = JobRepository(database)
        self.tasks = TaskRepository(database)
        self.tool_authorizer = tool_authorizer or ToolAuthorizer()

    # ---- 请求创建 ----

    def create_for_failure(
        self,
        job: ReproductionJob,
        task: Task,
        *,
        timeout_seconds: int | None = None,
    ) -> InterventionRequest:
        failure = task.failure_report
        failure_type = failure.failure_type if failure is not None else FailureType.UNKNOWN_ERROR
        reason = failure.error_message if failure and failure.error_message else task.objective
        requested_tools = self._extract_requested_tools(reason)
        failure_metadata = dict(failure.metadata) if failure is not None else {}

        if failure_type == FailureType.PERMISSION_ERROR:
            kind = InterventionKind.PERMISSION
            allowed_fields: list[str] = []
            if failure_metadata.get("response_mode") == "destructive_action":
                command = failure_metadata.get("command", [])
                question = (
                    f"任务 {task.task_id} 准备执行删除操作。请核对并明确批准或拒绝"
                    f"这条精确命令：{command!r}"
                )
                input_schema = {
                    "type": "object",
                    "properties": {
                        "approved": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["approved"],
                    "additionalProperties": False,
                }
            else:
                question = (
                    f"任务 {task.task_id} 需要额外工具权限。请批准或拒绝；批准不会突破"
                    "该任务类型的风险预算和 forbidden_actions。"
                )
                input_schema = {
                    "type": "object",
                    "properties": {
                        "approved": {"type": "boolean"},
                        "approved_tools": {"type": "array", "items": {"type": "string"}},
                        "reason": {"type": "string"},
                    },
                    "required": ["approved"],
                    "additionalProperties": False,
                }
        elif failure_type == FailureType.MODEL_ERROR:
            kind = InterventionKind.MODEL
            allowed_fields = ["model_paths", "checkpoint_paths", "user_environment_notes"]
            question = f"任务 {task.task_id} 缺少可用模型或 checkpoint，请提供替代路径。"
            input_schema = self._job_input_schema(allowed_fields)
        elif failure_type == FailureType.DATA_ERROR:
            kind = InterventionKind.USER_DATA
            allowed_fields = ["dataset_paths", "dataset_download_urls", "user_environment_notes"]
            question = f"任务 {task.task_id} 缺少可用数据，请提供数据路径或下载地址。"
            input_schema = self._job_input_schema(allowed_fields)
        elif failure_type == FailureType.RESOURCE_EXCEEDED:
            kind = InterventionKind.RESOURCE
            allowed_fields = [
                "cpu_cores",
                "memory_mb",
                "disk_mb",
                "gpu_count",
                "gpu_memory_gb",
                "max_runtime_seconds",
                "user_environment_notes",
            ]
            question = f"任务 {task.task_id} 超出资源预算，请提供可用资源或运行时限制。"
            input_schema = self._job_input_schema(allowed_fields)
        else:
            kind = InterventionKind.GENERIC
            allowed_fields = sorted(_ALL_INPUT_FIELDS)
            question = (
                f"任务 {task.task_id} 需要人工信息才能继续。请根据失败原因补充运行输入。"
            )
            input_schema = self._job_input_schema(allowed_fields)

        return self._create(
            job,
            task,
            kind=kind,
            question=question,
            reason=reason,
            input_schema=input_schema,
            metadata={
                "failure_type": failure_type.value,
                "allowed_input_fields": allowed_fields,
                "requested_tools": requested_tools,
                "failure_report": failure.to_dict() if failure is not None else None,
                **failure_metadata,
            },
            timeout_seconds=timeout_seconds,
        )

    def create_for_tool_grant(
        self,
        job: ReproductionJob,
        task: Task,
        *,
        tool_name: str,
        reason: str = "",
        rationale: str = "",
        timeout_seconds: int | None = None,
    ) -> InterventionRequest:
        """运行期缺工具升级（ASK_USER 分支）专用的人工介入请求。

        与 ``create_for_failure`` 的 PERMISSION 分支不同：这不是任务
        失败后的"事后审批"，而是子智能体**仍然挂着**等待裁决的
        "事中审批"——人工批准后由主智能体把工具注入挂起中的子智能体
        并唤醒它原地继续（见 ``AgentDispatcher.resume_escalation``），
        而不是把任务重置 PENDING 重启。因此 metadata 里携带
        ``response_mode="tool_grant_escalation"`` 供恢复路径区分。

        若当前 Job 已有其他挂起中的人工介入，返回已有的那个请求
        **且不占用它**（调用方应检查返回的 request 是否属于本次升级，
        不是则视为创建失败，fail closed）。
        """

        existing = self.interventions.get_pending_for_job(job.job_id)
        if existing is not None:
            return existing

        question = (
            f"任务 {task.task_id} 的子智能体在执行中请求补授工具 '{tool_name}'，"
            f"主智能体拿不准是否应授予。请批准或拒绝；批准不会突破该任务"
            "类型的风险预算和 forbidden_actions。"
        )
        return self._create(
            job,
            task,
            kind=InterventionKind.PERMISSION,
            question=question,
            reason=reason or f"主智能体对补授工具 '{tool_name}' 的裁决为 ask_user",
            input_schema={
                "type": "object",
                "properties": {
                    "approved": {"type": "boolean"},
                    "approved_tools": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
                "required": ["approved"],
                "additionalProperties": False,
            },
            metadata={
                "response_mode": "tool_grant_escalation",
                "failure_type": FailureType.PERMISSION_ERROR.value,
                "allowed_input_fields": [],
                "requested_tools": [tool_name],
                "escalation_task_id": task.task_id,
                "escalation_rationale": rationale,
                "failure_report": None,
            },
            timeout_seconds=timeout_seconds,
        )

    def create_for_missing_resources(
        self,
        job: ReproductionJob,
        task: Task,
        blocking_issues: list[str],
        *,
        missing_required_resources: list[dict[str, Any]] | None = None,
        timeout_seconds: int | None = None,
    ) -> InterventionRequest:
        missing_required_resources = [
            dict(item)
            for item in (missing_required_resources or [])
            if isinstance(item, dict)
        ]
        text = "\n".join(str(item) for item in blocking_issues)
        lower = text.lower()
        missing_kinds = {
            str(item.get("kind"))
            for item in missing_required_resources
            if item.get("kind")
        }
        has_model_inputs = bool(
            task.definition.inputs.get("model_paths")
            or task.definition.inputs.get("checkpoint_paths")
        )
        has_data_inputs = bool(task.definition.inputs.get("dataset_paths"))
        needs_data = "dataset" in lower or "data" in lower or has_data_inputs or "dataset" in missing_kinds
        needs_model = (
            "checkpoint" in lower
            or "model" in lower
            or has_model_inputs
            or bool({"model", "checkpoint"}.intersection(missing_kinds))
        )
        dataset_names = [
            str(item.get("name"))
            for item in missing_required_resources
            if item.get("kind") == "dataset" and item.get("name")
        ]
        model_names = [
            str(item.get("name"))
            for item in missing_required_resources
            if item.get("kind") in {"model", "checkpoint"} and item.get("name")
        ]
        if needs_data and needs_model:
            kind = InterventionKind.RESOURCE
            allowed_fields = ["dataset_paths", "model_paths", "checkpoint_paths"]
            named = [
                *(f"数据集 {name}" for name in dataset_names),
                *(f"模型/checkpoint {name}" for name in model_names),
            ]
            question = (
                "实验规格要求的运行资源未就绪"
                + (f"（{', '.join(named)}）" if named else "")
                + "。请提供对应的可访问路径。"
            )
        elif needs_model:
            kind = InterventionKind.MODEL
            allowed_fields = ["model_paths", "checkpoint_paths"]
            question = (
                "实验规格要求以下模型或 checkpoint，但当前工作区和已配置路径中未找到："
                f"{', '.join(model_names)}。请提供可访问的替代路径。"
                if model_names
                else "资源检查发现模型或 checkpoint 不可用，请提供可访问的替代路径。"
            )
        elif needs_data:
            kind = InterventionKind.USER_DATA
            allowed_fields = ["dataset_paths"]
            if dataset_names:
                question = (
                    "实验规格要求以下数据集，但当前工作区和已配置路径中未找到："
                    f"{', '.join(dataset_names)}。请提供可访问的数据集路径。"
                )
            else:
                question = "资源检查发现数据不可用，请提供可访问的数据集路径。"
        else:
            kind = InterventionKind.RESOURCE
            allowed_fields = [
                "dataset_paths",
                "model_paths",
                "checkpoint_paths",
                "cpu_cores",
                "memory_mb",
                "disk_mb",
                "gpu_count",
                "gpu_memory_gb",
                "max_runtime_seconds",
            ]
            question = "资源检查发现阻塞项，请提供替代资源信息。"
        return self._create(
            job,
            task,
            kind=kind,
            question=question,
            reason=text,
            input_schema=self._job_input_schema(allowed_fields),
            metadata={
                "allowed_input_fields": allowed_fields,
                "blocking_issues": list(blocking_issues),
                "missing_required_resources": missing_required_resources,
                "rerun_resource_check": True,
            },
            timeout_seconds=timeout_seconds,
        )

    def create_for_missing_command(
        self,
        job: ReproductionJob,
        *,
        timeout_seconds: int | None = None,
    ) -> InterventionRequest:
        fields = ["user_run_commands", "max_runtime_seconds"]
        return self._create(
            job,
            None,
            kind=InterventionKind.COMMAND,
            question="未发现可执行的实验入口，请提供至少一条实验运行命令。",
            reason="no executable experiment command was discovered or supplied",
            input_schema=self._job_input_schema(fields),
            metadata={"allowed_input_fields": fields},
            timeout_seconds=timeout_seconds,
        )

    def create_for_execution_parameters(
        self,
        job: ReproductionJob,
        task: Task,
        *,
        default_execution_image: str,
        timeout_seconds: int | None = None,
    ) -> InterventionRequest:
        """Pause immediately before an experiment run for explicit user review.

        This is intentionally a ``COMMAND`` intervention rather than a tool
        permission prompt: it protects the complete experiment invocation
        (including image, working directory and resource request), not every
        individual tool call made while preparing the experiment.
        """

        try:
            parameters = execution_parameter_snapshot(
                task.definition.inputs,
                default_execution_image=default_execution_image,
            )
        except ExecutionParameterValidationError as exc:
            raise InterventionValidationError(
                f"cannot request confirmation for invalid execution parameters: {exc}"
            ) from exc

        return self._create(
            job,
            task,
            kind=InterventionKind.COMMAND,
            question=(
                "即将启动实验执行。请核对下方完整运行参数；可提交 command、"
                "execution_image、working_dir、timeout_seconds、CPU/内存/磁盘或 GPU 的"
                "修改（也可修改 metrics_output_path），并将 approved 设为 true 后才会开始执行。"
                "网络状态由已确认的必需 API Base 决定并显示在参数快照中；"
                "代码工作区固定为只读。"
            ),
            reason="experiment execution parameters require user confirmation",
            input_schema={
                "type": "object",
                "properties": {
                    "approved": {"type": "boolean"},
                    "command": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 1,
                        "maxItems": 256,
                    },
                    "execution_image": {"type": "string", "minLength": 1, "maxLength": 512},
                    "working_dir": {"type": "string", "minLength": 1, "maxLength": 512},
                    "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 86400},
                    "cpu_cores": {"type": "number", "minimum": 0.1, "maximum": 256},
                    "memory_mb": {"type": "integer", "minimum": 128, "maximum": 4194304},
                    "disk_mb": {"type": "integer", "minimum": 128, "maximum": 16777216},
                    "gpu_count": {"type": "integer", "minimum": 0, "maximum": 32},
                    "gpu_memory_gb": {"type": "number", "minimum": 0, "maximum": 1024},
                    "metrics_output_path": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 512,
                    },
                    "reason": {"type": "string", "maxLength": 4000},
                },
                "required": ["approved"],
                "additionalProperties": False,
            },
            metadata={
                "response_mode": "execution_parameters",
                "proposed_parameters": parameters,
                "parameter_fingerprint": execution_parameter_fingerprint(parameters),
                "execution_parameter_default_image": default_execution_image,
            },
            timeout_seconds=timeout_seconds,
        )

    def create_for_pre_environment_execution_plan(
        self,
        job: ReproductionJob,
        task: Task,
        plan_inputs: dict[str, Any],
        *,
        requirements: list[dict[str, Any]],
        missing: list[dict[str, Any]],
        default_execution_image: str,
        timeout_seconds: int | None = None,
    ) -> InterventionRequest:
        """Collect missing runtime config and approve the plan in one request."""

        try:
            plan = execution_plan_snapshot(
                plan_inputs,
                default_execution_image=default_execution_image,
            )
        except ExecutionParameterValidationError as exc:
            raise InterventionValidationError(
                f"cannot request confirmation for invalid execution plan: {exc}"
            ) from exc
        task.definition.inputs["_pre_environment_execution_plan_candidate"] = plan
        normalized_requirements = normalize_requirements(requirements)
        normalized_missing = normalize_requirements(missing)
        value_names = [
            item["name"]
            for item in normalized_missing
            if item["kind"] != "credential_env"
        ]
        secret_names = [
            item["environment_variable"]
            for item in normalized_missing
            if item["kind"] == "credential_env"
        ]
        command_schema = {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 1,
            "maxItems": 256,
        }
        return self._create(
            job,
            task,
            kind=InterventionKind.COMMAND,
            question=(
                "资源检查已经完成，环境构建尚未开始。请在同一次回答中提供缺失的"
                "模型名/API 地址等非敏感配置、确认凭证环境变量，并核对五级实验"
                "命令、基础镜像、工作目录、超时及 CPU/内存/磁盘/GPU 参数；"
                "将 approved 设为 true 后才会开始环境构建。"
            ),
            reason=(
                "missing required runtime configuration and execution plan require "
                "one combined confirmation before environment construction"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "approved": {"type": "boolean"},
                    "values": {
                        "type": "object",
                        "properties": {
                            name: {"type": "string", "minLength": 1, "maxLength": 2048}
                            for name in value_names
                        },
                        "required": value_names,
                        "additionalProperties": False,
                    },
                    "confirmed_secret_env_vars": {
                        "type": "array",
                        "items": {"type": "string", "enum": secret_names},
                        "uniqueItems": True,
                    },
                    "tier_commands": {
                        "type": "object",
                        "properties": {
                            tier: command_schema for tier in EXECUTION_TIER_NAMES
                        },
                        "required": list(EXECUTION_TIER_NAMES),
                        "additionalProperties": False,
                    },
                    "base_image": {"type": "string", "minLength": 1, "maxLength": 512},
                    "working_dir": {"type": "string", "minLength": 1, "maxLength": 512},
                    "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 86400},
                    "cpu_cores": {"type": "number", "minimum": 0.1, "maximum": 256},
                    "memory_mb": {"type": "integer", "minimum": 128, "maximum": 4194304},
                    "disk_mb": {"type": "integer", "minimum": 128, "maximum": 16777216},
                    "gpu_count": {"type": "integer", "minimum": 0, "maximum": 32},
                    "gpu_memory_gb": {"type": "number", "minimum": 0, "maximum": 1024},
                    "metrics_output_path": {"type": "string", "minLength": 1, "maxLength": 512},
                    "reason": {"type": "string", "maxLength": 4000},
                },
                "required": [
                    "approved",
                    *(["values"] if value_names else []),
                    *(["confirmed_secret_env_vars"] if secret_names else []),
                ],
                "additionalProperties": False,
            },
            metadata={
                "response_mode": "pre_environment_execution_plan",
                "proposed_plan": plan,
                "plan_fingerprint": execution_plan_fingerprint(plan),
                "execution_parameter_default_image": default_execution_image,
                "requirements": normalized_requirements,
                "missing_requirements": normalized_missing,
                "required_value_names": value_names,
                "required_secret_env_vars": secret_names,
                "credential_values_must_not_be_persisted": True,
            },
            timeout_seconds=timeout_seconds,
        )

    def create_for_required_experiment_configuration(
        self,
        job: ReproductionJob,
        task: Task,
        missing: list[dict[str, Any]],
        *,
        timeout_seconds: int | None = None,
    ) -> InterventionRequest:
        """Collect must-have experiment configuration before execution.

        Credential values are never accepted in the response.  The user sets
        them in the controller process environment and confirms only the
        variable names, so intervention persistence cannot leak an API key.
        """

        requirements = normalize_requirements(missing)
        value_names = [
            item["name"] for item in requirements if item["kind"] != "credential_env"
        ]
        secret_names = [
            item["environment_variable"]
            for item in requirements
            if item["kind"] == "credential_env"
        ]
        descriptions = [
            f"{item['name']} ({item['kind']}): {item['reason']} [{item['source_ref']}]"
            for item in requirements
        ]
        return self._create(
            job,
            task,
            kind=InterventionKind.MODEL,
            question=(
                "代码与实验参数已确认，但仍缺少会使实验必然失败的运行配置。"
                "请在 values 中提供模型名/API 地址等非敏感值；对于 API Key/Token，"
                "请先在运行 ReproAgent 的环境中设置对应环境变量，再仅确认变量名。"
            ),
            reason="\n".join(descriptions),
            input_schema={
                "type": "object",
                "properties": {
                    "values": {
                        "type": "object",
                        "properties": {
                            name: {"type": "string", "minLength": 1, "maxLength": 2048}
                            for name in value_names
                        },
                        "required": value_names,
                        "additionalProperties": False,
                    },
                    "confirmed_secret_env_vars": {
                        "type": "array",
                        "items": {"type": "string", "enum": secret_names},
                        "uniqueItems": True,
                    },
                },
                "required": [
                    *(["values"] if value_names else []),
                    *(["confirmed_secret_env_vars"] if secret_names else []),
                ],
                "additionalProperties": False,
            },
            metadata={
                "response_mode": "required_experiment_configuration",
                "requirements": requirements,
                "required_value_names": value_names,
                "required_secret_env_vars": secret_names,
                "credential_values_must_not_be_persisted": True,
            },
            timeout_seconds=timeout_seconds,
        )

    def create_for_spec_conflicts(
        self,
        job: ReproductionJob,
        task: Task,
        *,
        conflict_fields: list[str],
        primary_values: dict[str, Any],
        timeout_seconds: int | None = None,
    ) -> InterventionRequest:
        return self._create(
            job,
            task,
            kind=InterventionKind.GENERIC,
            question=(
                "实验规格存在来源冲突。请明确每个冲突字段的最终值，或显式"
                "批准当前按来源优先级选择的主值。"
            ),
            reason="unresolved experiment specification conflicts",
            input_schema={
                "type": "object",
                "properties": {
                    "approve_primary_values": {"type": "boolean"},
                    "resolved_values": {"type": "object"},
                },
                "minProperties": 1,
                "additionalProperties": False,
            },
            metadata={
                "response_mode": "spec_conflict",
                "conflict_fields": conflict_fields,
                "primary_values": primary_values,
            },
            timeout_seconds=timeout_seconds,
        )

    def create_for_dynamic_tool(
        self,
        job: ReproductionJob,
        record: dict[str, Any],
        *,
        source_task_id: str,
        timeout_seconds: int | None = None,
    ) -> InterventionRequest:
        """Request explicit activation for a generated non-read-only tool."""

        return self._create(
            job,
            None,
            kind=InterventionKind.GENERIC,
            question=(
                "动态工具候选已取得 3 个独立任务证据并通过沙箱行为测试，"
                "但其风险级别不是只读。请核对用途、Schema、风险和代码哈希后"
                "明确批准或拒绝激活。"
            ),
            reason="non-read-only generated tool requires explicit activation",
            input_schema={
                "type": "object",
                "properties": {
                    "approved": {"type": "boolean"},
                    "reason": {"type": "string", "maxLength": 4000},
                },
                "required": ["approved"],
                "additionalProperties": False,
            },
            metadata={
                "response_mode": "dynamic_tool_activation",
                "dynamic_tool_id": record.get("tool_id", ""),
                "tool_name": record.get("tool_name", ""),
                "purpose": record.get("purpose", ""),
                "risk_level": record.get("risk_level", ""),
                "input_schema": record.get("input_schema", {}),
                "output_schema": record.get("output_schema", {}),
                "code_hash": record.get("code_hash", ""),
                "source_task_id": source_task_id,
            },
            timeout_seconds=timeout_seconds,
        )

    def _create(
        self,
        job: ReproductionJob,
        task: Task | None,
        *,
        kind: InterventionKind,
        question: str,
        reason: str,
        input_schema: dict[str, Any],
        metadata: dict[str, Any],
        timeout_seconds: int | None,
    ) -> InterventionRequest:
        existing = self.interventions.get_pending_for_job(job.job_id)
        if existing is not None:
            return existing

        previous_status = job.status
        request = InterventionRequest(
            job_id=job.job_id,
            task_id=task.task_id if task is not None else None,
            kind=kind,
            question=question,
            reason=reason,
            input_schema=input_schema,
            previous_job_status=previous_status,
            metadata=metadata,
            expires_at=(
                utc_now() + timedelta(seconds=timeout_seconds)
                if timeout_seconds is not None and timeout_seconds > 0
                else None
            ),
        )
        job.status = self._waiting_job_status(kind)
        if task is not None:
            task.status = self._waiting_task_status(kind)
        self.interventions.create_and_pause(request, job, task)
        self.tasks.record_event(
            job.job_id,
            request.task_id,
            "intervention_requested",
            {
                "request_id": request.request_id,
                "kind": kind.value,
                "question": question,
                "expires_at": request.to_dict()["expires_at"],
            },
        )
        return request

    # ---- 回答、拒绝与超时 ----

    def resolve(
        self,
        request_id: str,
        payload: dict[str, Any],
        *,
        responded_by: str = "user",
        resume_task: bool = True,
    ) -> InterventionResolution:
        """回答请求并同步当前 MainAgent 的内存视图，之后可直接续跑。

        ``resume_task=False`` 专用于工具升级场景：任务对应的子智能体
        线程仍挂起等待裁决（没有失败、不需要重启），因此**不能**把
        任务重置为 PENDING——状态恢复（WAITING_FOR_PERMISSION →
        RUNNING）由调用方在唤醒子智能体时同步完成。
        """
        request, job, task = self._load_pending(request_id)
        if request.expires_at is not None and utc_now() >= request.expires_at:
            if request.metadata.get("response_mode") == "dynamic_tool_activation":
                return self._finish_dynamic_tool_decision(
                    request,
                    job,
                    status=InterventionStatus.EXPIRED,
                    payload={"approved": False, "reason": "intervention deadline elapsed"},
                    responded_by="system",
                )
            return self._finish_without_resume(
                request,
                job,
                task,
                status=InterventionStatus.EXPIRED,
                payload={"reason": "intervention deadline elapsed"},
                responded_by="system",
            )

        if not isinstance(payload, dict):
            raise InterventionValidationError("response must be a JSON object")

        if request.metadata.get("response_mode") == "dynamic_tool_activation":
            unknown = set(payload) - {"approved", "reason"}
            if unknown or not isinstance(payload.get("approved"), bool):
                raise InterventionValidationError(
                    "dynamic-tool response requires boolean approved and optional reason"
                )
            normalized = {
                "approved": payload["approved"],
                "reason": str(payload.get("reason", ""))[:4000],
            }
            status = (
                InterventionStatus.APPROVED
                if normalized["approved"]
                else InterventionStatus.REJECTED
            )
        elif request.kind == InterventionKind.PERMISSION:
            normalized = self._validate_permission_response(request, task, payload)
            if not normalized["approved"]:
                return self._finish_without_resume(
                    request,
                    job,
                    task,
                    status=InterventionStatus.REJECTED,
                    payload=normalized,
                    responded_by=responded_by,
                )
            if task is None:
                raise InterventionValidationError("permission request has no target task")
            if request.metadata.get("response_mode") == "destructive_action":
                approvals = task.definition.inputs.setdefault(
                    "_destructive_action_approvals", []
                )
                approvals.append(
                    {
                        "fingerprint": request.metadata["command_fingerprint"],
                        "command": list(request.metadata.get("command", [])),
                        "approved_for_attempt": task.attempt + 1,
                        "request_id": request.request_id,
                    }
                )
            else:
                for name in normalized["approved_tools"]:
                    if name not in task.definition.allowed_tools:
                        task.definition.allowed_tools.append(name)
            status = InterventionStatus.APPROVED
        elif request.metadata.get("response_mode") == "required_experiment_configuration":
            if task is None:
                raise InterventionValidationError(
                    "required-configuration request has no target task"
                )
            normalized = self._validate_required_experiment_configuration_response(
                request, payload
            )
            self._apply_required_experiment_configuration(job, task, normalized)
            status = InterventionStatus.RESOLVED
        elif request.metadata.get("response_mode") == "pre_environment_execution_plan":
            if task is None:
                raise InterventionValidationError(
                    "pre-environment execution-plan request has no target task"
                )
            normalized = self._validate_pre_environment_execution_plan_response(
                request, job, task, payload
            )
            if not normalized["approved"]:
                return self._finish_without_resume(
                    request,
                    job,
                    task,
                    status=InterventionStatus.REJECTED,
                    payload=normalized,
                    responded_by=responded_by,
                )
            self._apply_pre_environment_execution_plan(
                job, task, normalized, request
            )
            status = InterventionStatus.RESOLVED
        elif request.metadata.get("response_mode") == "execution_parameters":
            if task is None:
                raise InterventionValidationError(
                    "execution-parameter request has no target task"
                )
            normalized = self._validate_execution_parameter_response(
                request, task, payload
            )
            if not normalized["approved"]:
                return self._finish_without_resume(
                    request,
                    job,
                    task,
                    status=InterventionStatus.REJECTED,
                    payload=normalized,
                    responded_by=responded_by,
                )
            self._apply_execution_parameter_confirmation(task, normalized, request)
            status = InterventionStatus.RESOLVED
        elif request.metadata.get("response_mode") == "spec_conflict":
            normalized = self._validate_spec_conflict_response(request, payload)
            if task is None:
                raise InterventionValidationError("spec conflict request has no target task")
            task.definition.inputs["user_overrides"] = dict(
                normalized["resolved_values"]
            )
            status = InterventionStatus.RESOLVED
        else:
            normalized = self._validate_job_input_response(request, payload)
            self._apply_job_inputs(job, task, normalized)
            status = InterventionStatus.RESOLVED

        request.status = status
        request.response = normalized
        request.responded_by = self._validated_actor(responded_by)
        request.responded_at = utc_now()
        job.status = request.previous_job_status
        if task is not None:
            if resume_task:
                self._prepare_task_for_resume(task)
            else:
                # 工具升级场景：子智能体线程还活着，保留任务执行现场，
                # 只把状态从等待权限拉回运行中（由调用方在唤醒线程
                # 前后同步设置；这里至少不能把它误置为 PENDING）。
                if task.status == TaskStatus.WAITING_FOR_PERMISSION:
                    task.status = TaskStatus.RUNNING
                    task.touch()
        self.interventions.resolve_with_state(request, job, task)
        self.tasks.record_event(
            job.job_id,
            request.task_id,
            "intervention_resolved",
            {
                "request_id": request.request_id,
                "status": request.status.value,
                "responded_by": request.responded_by,
                "response_fields": sorted(normalized),
            },
        )
        return InterventionResolution(request=request, job=job, task=task)

    def reject(
        self,
        request_id: str,
        *,
        reason: str = "user rejected the intervention",
        responded_by: str = "user",
    ) -> InterventionResolution:
        request, job, task = self._load_pending(request_id)
        if request.metadata.get("response_mode") == "dynamic_tool_activation":
            return self._finish_dynamic_tool_decision(
                request,
                job,
                status=InterventionStatus.REJECTED,
                payload={
                    "approved": False,
                    "reason": self._bounded_string(reason, "reason", 4_000),
                },
                responded_by=responded_by,
            )
        return self._finish_without_resume(
            request,
            job,
            task,
            status=InterventionStatus.REJECTED,
            payload={"reason": self._bounded_string(reason, "reason", 4_000)},
            responded_by=responded_by,
        )

    def expire_overdue(self, job_id: str) -> list[InterventionResolution]:
        resolutions: list[InterventionResolution] = []
        for request in self.interventions.list_by_job(job_id, InterventionStatus.PENDING):
            if request.expires_at is None or utc_now() < request.expires_at:
                continue
            job = self.jobs.get(request.job_id)
            task = self.tasks.get(request.task_id) if request.task_id else None
            if job is None:
                continue
            if request.metadata.get("response_mode") == "dynamic_tool_activation":
                resolutions.append(
                    self._finish_dynamic_tool_decision(
                        request,
                        job,
                        status=InterventionStatus.EXPIRED,
                        payload={
                            "approved": False,
                            "reason": "intervention deadline elapsed",
                        },
                        responded_by="system",
                    )
                )
                continue
            resolutions.append(
                self._finish_without_resume(
                    request,
                    job,
                    task,
                    status=InterventionStatus.EXPIRED,
                    payload={"reason": "intervention deadline elapsed"},
                    responded_by="system",
                )
            )
        return resolutions

    def _finish_without_resume(
        self,
        request: InterventionRequest,
        job: ReproductionJob,
        task: Task | None,
        *,
        status: InterventionStatus,
        payload: dict[str, Any],
        responded_by: str,
    ) -> InterventionResolution:
        request.status = status
        request.response = payload
        request.responded_by = self._validated_actor(responded_by)
        request.responded_at = utc_now()
        job.status = JobStatus.FAILED
        if task is not None:
            task.status = TaskStatus.TERMINAL_FAILURE
            task.completed_at = utc_now()
            task.lease_owner = None
            task.lease_expires_at = None
        self.interventions.resolve_with_state(request, job, task)
        self.tasks.record_event(
            job.job_id,
            request.task_id,
            "intervention_closed_without_resume",
            {
                "request_id": request.request_id,
                "status": status.value,
                "responded_by": request.responded_by,
                "reason": payload.get("reason", ""),
            },
        )
        return InterventionResolution(request=request, job=job, task=task)

    def _finish_dynamic_tool_decision(
        self,
        request: InterventionRequest,
        job: ReproductionJob,
        *,
        status: InterventionStatus,
        payload: dict[str, Any],
        responded_by: str,
    ) -> InterventionResolution:
        """Close optional tool activation without failing the reproduction Job."""

        request.status = status
        request.response = payload
        request.responded_by = self._validated_actor(responded_by)
        request.responded_at = utc_now()
        job.status = request.previous_job_status
        self.interventions.resolve_with_state(request, job, None)
        self.tasks.record_event(
            job.job_id,
            None,
            "dynamic_tool_activation_closed",
            {
                "request_id": request.request_id,
                "status": status.value,
                "responded_by": request.responded_by,
            },
        )
        return InterventionResolution(request=request, job=job, task=None)

    # ---- 校验与状态变换 ----

    def _load_pending(
        self, request_id: str
    ) -> tuple[InterventionRequest, ReproductionJob, Task | None]:
        request = self.interventions.get(request_id)
        if request is None:
            raise KeyError(f"unknown intervention request: {request_id}")
        if not request.is_pending:
            raise InterventionValidationError(
                f"intervention {request_id} is already {request.status.value}"
            )
        job = self.jobs.get(request.job_id)
        if job is None:
            raise KeyError(f"unknown job: {request.job_id}")
        task = self.tasks.get(request.task_id) if request.task_id else None
        if request.task_id and task is None:
            raise KeyError(f"unknown task: {request.task_id}")
        return request, job, task

    def _validate_permission_response(
        self,
        request: InterventionRequest,
        task: Task | None,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if request.metadata.get("response_mode") == "destructive_action":
            unknown = set(payload) - {"approved", "reason"}
            if unknown:
                raise InterventionValidationError(
                    "unknown destructive-action response field(s): "
                    + ", ".join(sorted(unknown))
                )
            if not isinstance(payload.get("approved"), bool):
                raise InterventionValidationError(
                    "destructive-action response requires boolean 'approved'"
                )
            fingerprint = request.metadata.get("command_fingerprint")
            command = request.metadata.get("command")
            if not isinstance(fingerprint, str) or not fingerprint or not isinstance(command, list):
                raise InterventionValidationError(
                    "destructive-action request is missing its bound command"
                )
            return {
                "approved": payload["approved"],
                "approved_tools": [],
                "reason": self._bounded_string(payload.get("reason", ""), "reason", 4_000),
                "command_fingerprint": fingerprint,
            }

        unknown = set(payload) - {"approved", "approved_tools", "reason"}
        if unknown:
            raise InterventionValidationError(
                f"unknown permission response field(s): {', '.join(sorted(unknown))}"
            )
        if not isinstance(payload.get("approved"), bool):
            raise InterventionValidationError("permission response requires boolean 'approved'")
        reason = self._bounded_string(payload.get("reason", ""), "reason", 4_000)
        if not payload["approved"]:
            return {"approved": False, "approved_tools": [], "reason": reason}
        if task is None:
            raise InterventionValidationError("permission request has no target task")
        tool_names = payload.get("approved_tools", request.metadata.get("requested_tools", []))
        tool_names = self._string_list(tool_names, "approved_tools", max_items=32)
        denials = self.tool_authorizer.validate_human_approval(
            task_type=task.definition.task_type,
            tool_names=tool_names,
            forbidden_actions=task.definition.forbidden_actions,
        )
        if denials:
            detail = "; ".join(f"{item.tool_name}: {item.reason}" for item in denials)
            raise InterventionValidationError(
                "approval would cross a non-overridable security boundary: " + detail
            )
        return {"approved": True, "approved_tools": tool_names, "reason": reason}

    def _validate_pre_environment_execution_plan_response(
        self,
        request: InterventionRequest,
        job: ReproductionJob,
        task: Task,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        allowed = {
            "approved",
            "values",
            "confirmed_secret_env_vars",
            "tier_commands",
            "base_image",
            "working_dir",
            "timeout_seconds",
            "cpu_cores",
            "memory_mb",
            "disk_mb",
            "gpu_count",
            "gpu_memory_gb",
            "metrics_output_path",
            "reason",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise InterventionValidationError(
                "unknown pre-environment execution-plan response field(s): "
                + ", ".join(sorted(unknown))
            )
        if not isinstance(payload.get("approved"), bool):
            raise InterventionValidationError(
                "pre-environment execution-plan response requires boolean 'approved'"
            )
        default_image = request.metadata.get("execution_parameter_default_image")
        if not isinstance(default_image, str) or not default_image.strip():
            raise InterventionValidationError(
                "execution-plan request is missing its default image"
            )
        candidate = task.definition.inputs.get(
            "_pre_environment_execution_plan_candidate"
        )
        if not isinstance(candidate, dict):
            raise InterventionValidationError(
                "execution-plan candidate is missing from the target task"
            )
        try:
            current = execution_plan_snapshot(
                candidate, default_execution_image=default_image
            )
        except ExecutionParameterValidationError as exc:
            raise InterventionValidationError(
                f"current execution plan is invalid: {exc}"
            ) from exc
        if execution_plan_fingerprint(current) != request.metadata.get(
            "plan_fingerprint"
        ):
            raise InterventionValidationError(
                "execution plan changed while awaiting confirmation; a new confirmation is required"
            )
        reason = self._bounded_string(payload.get("reason", ""), "reason", 4_000)
        if not payload["approved"]:
            return {"approved": False, "reason": reason, "updates": {}}

        configuration = self._validate_required_experiment_configuration_response(
            request,
            {
                "values": payload.get("values", {}),
                "confirmed_secret_env_vars": payload.get(
                    "confirmed_secret_env_vars", []
                ),
            },
        )

        updates: dict[str, Any] = {}
        if "tier_commands" in payload:
            raw_commands = payload["tier_commands"]
            if not isinstance(raw_commands, dict):
                raise InterventionValidationError("tier_commands must be an object")
            if set(raw_commands) != set(EXECUTION_TIER_NAMES):
                raise InterventionValidationError(
                    "tier_commands must cover exactly: "
                    + ", ".join(EXECUTION_TIER_NAMES)
                )
            updates["tier_commands"] = {
                tier: self._string_list(
                    raw_commands[tier], f"tier_commands.{tier}", max_items=256
                )
                for tier in EXECUTION_TIER_NAMES
            }
            if any(not command for command in updates["tier_commands"].values()):
                raise InterventionValidationError(
                    "every tier command must contain at least one argument"
                )
        if "base_image" in payload:
            image = self._bounded_string(payload["base_image"], "base_image", 512).strip()
            if not image or any(char.isspace() for char in image):
                raise InterventionValidationError(
                    "base_image must be a non-empty image reference without whitespace"
                )
            updates["base_image"] = image
        if "working_dir" in payload:
            updates["working_dir"] = self._execution_working_dir(payload["working_dir"])
        if "timeout_seconds" in payload:
            timeout = payload["timeout_seconds"]
            if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 1_800:
                raise InterventionValidationError(
                    "timeout_seconds must be an integer between 1 and 1800"
                )
            updates["timeout_seconds"] = timeout
        if "cpu_cores" in payload:
            value = payload["cpu_cores"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.1 <= float(value) <= 256:
                raise InterventionValidationError(
                    "cpu_cores must be a number between 0.1 and 256"
                )
            updates["cpu_cores"] = float(value)
        for field_name, upper in (("memory_mb", 4_194_304), ("disk_mb", 16_777_216)):
            if field_name not in payload:
                continue
            value = payload[field_name]
            if isinstance(value, bool) or not isinstance(value, int) or not 128 <= value <= upper:
                raise InterventionValidationError(
                    f"{field_name} must be an integer between 128 and {upper}"
                )
            updates[field_name] = value
        if "gpu_count" in payload:
            value = payload["gpu_count"]
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 32:
                raise InterventionValidationError(
                    "gpu_count must be an integer between 0 and 32"
                )
            updates["gpu_count"] = value
        if "gpu_memory_gb" in payload:
            value = payload["gpu_memory_gb"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1024:
                raise InterventionValidationError(
                    "gpu_memory_gb must be a number between 0 and 1024"
                )
            updates["gpu_memory_gb"] = float(value)
        if "metrics_output_path" in payload:
            updates["metrics_output_path"] = self._metrics_output_path(
                payload["metrics_output_path"]
            )

        effective_inputs = dict(current)
        effective_inputs.update(updates)
        requirements = normalize_requirements(
            request.metadata.get("requirements", [])
        )
        runtime_values = dict(job.inputs.experiment_runtime_config)
        runtime_values.update(configuration["values"])
        materialized_commands: dict[str, list[str]] = {}
        experiment_environment: dict[str, str] = {}
        secret_environment: list[str] = []
        for tier, command in effective_inputs["tier_commands"].items():
            materialized, environment, secrets = materialize_runtime_configuration(
                command=list(command),
                requirements=requirements,
                runtime_values=runtime_values,
            )
            materialized_commands[tier] = materialized
            experiment_environment.update(environment)
            secret_environment.extend(secrets)
        try:
            network_enabled, network_hosts = runtime_network_configuration(
                requirements, runtime_values
            )
        except ValueError as exc:
            raise InterventionValidationError(str(exc)) from exc
        effective_inputs.update(
            {
                "tier_commands": materialized_commands,
                "experiment_environment": experiment_environment,
                "experiment_secret_env_vars": sorted(set(secret_environment)),
                "network_enabled": network_enabled,
                "network_hosts": network_hosts,
            }
        )
        try:
            effective = execution_plan_snapshot(
                effective_inputs, default_execution_image=default_image
            )
        except ExecutionParameterValidationError as exc:
            raise InterventionValidationError(
                f"updated execution plan is invalid: {exc}"
            ) from exc
        return {
            "approved": True,
            "reason": reason,
            "updates": updates,
            "values": configuration["values"],
            "confirmed_secret_env_vars": configuration[
                "confirmed_secret_env_vars"
            ],
            "effective_plan": effective,
            "plan_fingerprint": execution_plan_fingerprint(effective),
        }

    def _validate_execution_parameter_response(
        self,
        request: InterventionRequest,
        task: Task,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        allowed = {
            "approved",
            "command",
            "execution_image",
            "working_dir",
            "timeout_seconds",
            "cpu_cores",
            "memory_mb",
            "disk_mb",
            "gpu_count",
            "gpu_memory_gb",
            "metrics_output_path",
            "reason",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise InterventionValidationError(
                "unknown execution-parameter response field(s): "
                + ", ".join(sorted(unknown))
            )
        if not isinstance(payload.get("approved"), bool):
            raise InterventionValidationError(
                "execution-parameter response requires boolean 'approved'"
            )

        default_image = request.metadata.get("execution_parameter_default_image")
        if not isinstance(default_image, str) or not default_image.strip():
            raise InterventionValidationError(
                "execution-parameter request is missing its default image"
            )
        try:
            current = execution_parameter_snapshot(
                task.definition.inputs, default_execution_image=default_image
            )
        except ExecutionParameterValidationError as exc:
            raise InterventionValidationError(
                f"current execution parameters are invalid: {exc}"
            ) from exc
        if execution_parameter_fingerprint(current) != request.metadata.get(
            "parameter_fingerprint"
        ):
            raise InterventionValidationError(
                "execution parameters changed while awaiting confirmation; "
                "a new confirmation is required"
            )

        reason = self._bounded_string(payload.get("reason", ""), "reason", 4_000)
        if not payload["approved"]:
            return {"approved": False, "reason": reason, "updates": {}}

        updates: dict[str, Any] = {}
        if "command" in payload:
            updates["command"] = self._string_list(
                payload["command"], "command", max_items=256
            )
        if "execution_image" in payload:
            image = self._bounded_string(
                payload["execution_image"], "execution_image", 512
            )
            if not image.strip() or any(char.isspace() for char in image):
                raise InterventionValidationError(
                    "execution_image must be a non-empty image reference without whitespace"
                )
            updates["execution_image"] = image.strip()
        if "working_dir" in payload:
            updates["working_dir"] = self._execution_working_dir(payload["working_dir"])
        if "timeout_seconds" in payload:
            timeout = payload["timeout_seconds"]
            if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 1_800:
                raise InterventionValidationError(
                    "timeout_seconds must be an integer between 1 and 1800"
                )
            updates["timeout_seconds"] = timeout
        if "gpu_count" in payload:
            gpu_count = payload["gpu_count"]
            if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or not 0 <= gpu_count <= 32:
                raise InterventionValidationError(
                    "gpu_count must be an integer between 0 and 32"
                )
            updates["gpu_count"] = gpu_count
        if "cpu_cores" in payload:
            cpu_cores = payload["cpu_cores"]
            if (
                isinstance(cpu_cores, bool)
                or not isinstance(cpu_cores, (int, float))
                or not 0.1 <= float(cpu_cores) <= 256
            ):
                raise InterventionValidationError(
                    "cpu_cores must be a number between 0.1 and 256"
                )
            updates["cpu_cores"] = float(cpu_cores)
        for field_name, upper in (
            ("memory_mb", 4_194_304),
            ("disk_mb", 16_777_216),
        ):
            if field_name not in payload:
                continue
            value = payload[field_name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 128 <= value <= upper
            ):
                raise InterventionValidationError(
                    f"{field_name} must be an integer between 128 and {upper}"
                )
            updates[field_name] = value
        if "gpu_memory_gb" in payload:
            gpu_memory_gb = payload["gpu_memory_gb"]
            if (
                isinstance(gpu_memory_gb, bool)
                or not isinstance(gpu_memory_gb, (int, float))
                or not 0 <= float(gpu_memory_gb) <= 1024
            ):
                raise InterventionValidationError(
                    "gpu_memory_gb must be a number between 0 and 1024"
                )
            updates["gpu_memory_gb"] = float(gpu_memory_gb)
        if "metrics_output_path" in payload:
            updates["metrics_output_path"] = self._metrics_output_path(
                payload["metrics_output_path"]
            )

        effective_inputs = dict(task.definition.inputs)
        effective_inputs.update(updates)
        try:
            effective = execution_parameter_snapshot(
                effective_inputs, default_execution_image=default_image
            )
        except ExecutionParameterValidationError as exc:
            raise InterventionValidationError(
                f"updated execution parameters are invalid: {exc}"
            ) from exc
        return {
            "approved": True,
            "reason": reason,
            "updates": updates,
            "effective_parameters": effective,
            "parameter_fingerprint": execution_parameter_fingerprint(effective),
        }

    def _validate_required_experiment_configuration_response(
        self,
        request: InterventionRequest,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        unknown = set(payload) - {"values", "confirmed_secret_env_vars"}
        if unknown:
            raise InterventionValidationError(
                "unknown required-configuration response field(s): "
                + ", ".join(sorted(unknown))
            )
        required_values = set(request.metadata.get("required_value_names", []))
        raw_values = payload.get("values", {})
        if not isinstance(raw_values, dict):
            raise InterventionValidationError("values must be an object")
        if set(raw_values) != required_values:
            raise InterventionValidationError(
                "values must cover exactly the requested configuration names; "
                f"missing={sorted(required_values - set(raw_values))}, "
                f"extra={sorted(set(raw_values) - required_values)}"
            )
        values = {
            str(name): self._bounded_string(value, str(name), 2048).strip()
            for name, value in raw_values.items()
        }
        if any(not value for value in values.values()):
            raise InterventionValidationError("required configuration values must not be empty")

        required_secrets = set(request.metadata.get("required_secret_env_vars", []))
        confirmed = set(
            self._string_list(
                payload.get("confirmed_secret_env_vars", []),
                "confirmed_secret_env_vars",
                max_items=64,
            )
        )
        if confirmed != required_secrets:
            raise InterventionValidationError(
                "confirmed_secret_env_vars must cover exactly the requested variables; "
                f"missing={sorted(required_secrets - confirmed)}, "
                f"extra={sorted(confirmed - required_secrets)}"
            )
        absent = sorted(name for name in required_secrets if not os.environ.get(name, "").strip())
        if absent:
            raise InterventionValidationError(
                "the following credential environment variables are not set in the "
                "current ReproAgent process: " + ", ".join(absent)
            )
        return {
            "values": values,
            "confirmed_secret_env_vars": sorted(confirmed),
        }

    def _validate_job_input_response(
        self, request: InterventionRequest, payload: dict[str, Any]
    ) -> dict[str, Any]:
        allowed = set(request.metadata.get("allowed_input_fields", []))
        unknown = set(payload) - allowed
        if unknown:
            raise InterventionValidationError(
                f"unknown or disallowed response field(s): {', '.join(sorted(unknown))}"
            )
        if not payload:
            raise InterventionValidationError("response must provide at least one requested field")
        normalized: dict[str, Any] = {}
        for name, value in payload.items():
            if name in _STRING_LIST_FIELDS:
                normalized[name] = self._string_list(value, name)
                if not normalized[name]:
                    raise InterventionValidationError(f"{name} must not be empty")
                if name == "dataset_download_urls":
                    for url in normalized[name]:
                        parsed = urlparse(url)
                        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                            raise InterventionValidationError(
                                "dataset_download_urls must contain only http(s) URLs"
                            )
            elif name == "user_environment_notes":
                normalized[name] = self._bounded_string(value, name, 10_000)
                if not normalized[name].strip():
                    raise InterventionValidationError("user_environment_notes must not be empty")
            elif name in _NUMERIC_FIELDS:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise InterventionValidationError(f"{name} must be a positive number")
                if value <= 0:
                    raise InterventionValidationError(f"{name} must be greater than zero")
                if name in {
                    "gpu_count",
                    "memory_mb",
                    "disk_mb",
                    "max_runtime_seconds",
                } and not isinstance(value, int):
                    raise InterventionValidationError(f"{name} must be an integer")
                normalized[name] = value
            else:  # 领域模型新增字段但未同步校验器时 fail closed。
                raise InterventionValidationError(f"unsupported response field: {name}")
        return normalized

    @staticmethod
    def _validate_spec_conflict_response(
        request: InterventionRequest, payload: dict[str, Any]
    ) -> dict[str, Any]:
        unknown = set(payload) - {"approve_primary_values", "resolved_values"}
        if unknown:
            raise InterventionValidationError(
                f"unknown spec conflict response field(s): {', '.join(sorted(unknown))}"
            )
        approve = payload.get("approve_primary_values", False)
        if not isinstance(approve, bool):
            raise InterventionValidationError("approve_primary_values must be boolean")
        resolved = payload.get("resolved_values", {})
        if not isinstance(resolved, dict):
            raise InterventionValidationError("resolved_values must be an object")
        conflict_fields = set(request.metadata.get("conflict_fields", []))
        if approve:
            resolved = dict(request.metadata.get("primary_values", {}))
        elif set(resolved) != conflict_fields:
            missing = sorted(conflict_fields - set(resolved))
            extra = sorted(set(resolved) - conflict_fields)
            raise InterventionValidationError(
                f"resolved_values must cover exactly the conflict fields; "
                f"missing={missing}, extra={extra}"
            )
        if not resolved:
            raise InterventionValidationError(
                "approve_primary_values or resolved_values is required"
            )
        return {
            "approve_primary_values": approve,
            "resolved_values": resolved,
        }

    @staticmethod
    def _apply_job_inputs(
        job: ReproductionJob, task: Task | None, values: dict[str, Any]
    ) -> None:
        if set(values) & {
            "user_run_commands",
            "cpu_cores",
            "memory_mb",
            "disk_mb",
            "gpu_count",
            "gpu_memory_gb",
            "max_runtime_seconds",
        }:
            job.inputs.confirmed_execution_plan = {}
        for name, value in values.items():
            setattr(job.inputs, name, value)
        if task is None:
            return
        for name, value in values.items():
            task.definition.inputs[name] = value
        if task.definition.task_type == "resource_check":
            task.definition.inputs.update(
                {
                    "dataset_paths": list(job.inputs.dataset_paths),
                    "model_paths": list(job.inputs.model_paths),
                    "checkpoint_paths": list(job.inputs.checkpoint_paths),
                    "requested_cpu_cores": job.inputs.cpu_cores,
                    "requested_memory_mb": job.inputs.memory_mb,
                    "requested_disk_mb": job.inputs.disk_mb,
                    "requested_gpu_count": job.inputs.gpu_count,
                    "requested_gpu_memory_gb": job.inputs.gpu_memory_gb,
                }
            )
            if any(
                task.definition.inputs.get(field)
                for field in ("dataset_paths", "model_paths", "checkpoint_paths")
            ) and "check_path_resource" not in task.definition.allowed_tools:
                task.definition.allowed_tools.append("check_path_resource")
        if task.definition.task_type == "environment_build" and "user_environment_notes" in values:
            task.definition.inputs["dependencies_hint"] = values["user_environment_notes"]
        if task.definition.task_type == "experiment_execution":
            if values.get("user_run_commands"):
                task.definition.inputs["command"] = shlex.split(values["user_run_commands"][0])
            if "max_runtime_seconds" in values:
                task.definition.inputs["timeout_seconds"] = values["max_runtime_seconds"]

    @staticmethod
    def _apply_required_experiment_configuration(
        job: ReproductionJob,
        task: Task,
        normalized: dict[str, Any],
    ) -> None:
        # Any newly supplied model/API/credential binding changes the effective
        # run plan and therefore invalidates an older approval.
        job.inputs.confirmed_execution_plan = {}
        job.inputs.experiment_runtime_config.update(normalized["values"])
        job.inputs.experiment_secret_env_vars = sorted(
            set(job.inputs.experiment_secret_env_vars)
            | set(normalized["confirmed_secret_env_vars"])
        )
        requirements = normalize_requirements(
            job.inputs.required_experiment_configurations
            or task.definition.inputs.get("required_experiment_configurations", [])
        )
        command, environment, secret_environment = materialize_runtime_configuration(
            command=list(task.definition.inputs.get("command", [])),
            requirements=requirements,
            runtime_values=job.inputs.experiment_runtime_config,
        )
        try:
            network_enabled, network_hosts = runtime_network_configuration(
                requirements, job.inputs.experiment_runtime_config
            )
        except ValueError as exc:
            raise InterventionValidationError(str(exc)) from exc
        execution_manifest = dict(
            task.definition.inputs.get("execution_manifest", {}) or {}
        )
        for requirement in requirements:
            if requirement["kind"] != "model_name":
                continue
            model_name = job.inputs.experiment_runtime_config.get(requirement["name"])
            if model_name:
                execution_manifest["model_identifier"] = str(model_name)
                break
        task.definition.inputs.update(
            {
                "command": command,
                "experiment_runtime_config": dict(job.inputs.experiment_runtime_config),
                "experiment_environment": environment,
                "experiment_secret_env_vars": secret_environment,
                "network_enabled": network_enabled,
                "network_hosts": network_hosts,
                "required_experiment_configurations": requirements,
                "execution_manifest": execution_manifest,
            }
        )

    @staticmethod
    def _apply_pre_environment_execution_plan(
        job: ReproductionJob,
        task: Task,
        normalized: dict[str, Any],
        request: InterventionRequest,
    ) -> None:
        plan = normalized["effective_plan"]
        job.inputs.experiment_runtime_config.update(normalized["values"])
        job.inputs.experiment_secret_env_vars = sorted(
            set(job.inputs.experiment_secret_env_vars)
            | set(normalized["confirmed_secret_env_vars"])
        )
        job.inputs.confirmed_execution_plan = dict(plan)
        job.inputs.max_runtime_seconds = int(plan["timeout_seconds"])
        job.inputs.cpu_cores = float(plan["cpu_cores"])
        job.inputs.memory_mb = int(plan["memory_mb"])
        job.inputs.disk_mb = int(plan["disk_mb"])
        job.inputs.gpu_count = int(plan["gpu_count"])
        job.inputs.gpu_memory_gb = float(plan["gpu_memory_gb"])
        task.definition.inputs.update(
            {
                "base_image": plan["base_image"],
                "cpu_cores": job.inputs.cpu_cores,
                "memory_mb": job.inputs.memory_mb,
                "disk_mb": job.inputs.disk_mb,
                "gpu_count": job.inputs.gpu_count,
                "gpu_memory_gb": job.inputs.gpu_memory_gb,
                "_pre_environment_execution_plan_candidate": dict(plan),
                "_pre_environment_execution_plan_approval": {
                    "fingerprint": normalized["plan_fingerprint"],
                    "request_id": request.request_id,
                },
            }
        )

    @staticmethod
    def _apply_execution_parameter_confirmation(
        task: Task,
        normalized: dict[str, Any],
        request: InterventionRequest,
    ) -> None:
        parameters = normalized["effective_parameters"]
        task.definition.inputs.update(
            {
                "command": list(parameters["command"]),
                "execution_image": parameters["execution_image"],
                "working_dir": parameters["working_dir"],
                "timeout_seconds": parameters["timeout_seconds"],
                "cpu_cores": parameters["cpu_cores"],
                "memory_mb": parameters["memory_mb"],
                "disk_mb": parameters["disk_mb"],
                "gpu_count": parameters["gpu_count"],
                "gpu_memory_gb": parameters["gpu_memory_gb"],
                "metrics_output_path": parameters["metrics_output_path"],
                # This approval is provenance for the exact command bound to
                # this tier, including a command inferred by code analysis.
                "tier_command_verified": True,
                "_execution_parameter_approval": {
                    "fingerprint": normalized["parameter_fingerprint"],
                    "approved_for_attempt": task.attempt + 1,
                    "request_id": request.request_id,
                },
            }
        )

    @staticmethod
    def _execution_working_dir(value: Any) -> str:
        working_dir = InterventionService._bounded_string(value, "working_dir", 512).strip()
        if not working_dir:
            raise InterventionValidationError("working_dir must not be empty")
        if working_dir.startswith("/"):
            raise InterventionValidationError("working_dir must stay inside the workspace")
        if "://" in working_dir:
            if not working_dir.startswith("workspace://"):
                raise InterventionValidationError("working_dir must use workspace:// or a relative path")
            relative = working_dir.removeprefix("workspace://")
        else:
            relative = working_dir
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            raise InterventionValidationError("working_dir must stay inside the workspace")
        return working_dir

    @staticmethod
    def _metrics_output_path(value: Any) -> str:
        path = InterventionService._bounded_string(
            value, "metrics_output_path", 512
        ).strip()
        if not path.startswith("output://"):
            raise InterventionValidationError(
                "metrics_output_path must stay in output://"
            )
        relative = path.removeprefix("output://")
        candidate = PurePosixPath(relative)
        if not relative or candidate.is_absolute() or ".." in candidate.parts:
            raise InterventionValidationError(
                "metrics_output_path must stay inside the output directory"
            )
        return path

    @staticmethod
    def _prepare_task_for_resume(task: Task) -> None:
        task.status = TaskStatus.PENDING
        task.assigned_agent = None
        task.dispatched_at = None
        task.started_at = None
        task.completed_at = None
        task.heartbeat = None
        task.last_push_heartbeat = None
        task.last_pull_heartbeat = None
        task.latest_agent_report = None
        task.next_report_due_at = None
        task.report_sequence = 0
        task.overrun_report_count = 0
        task.reporting_exhausted = False
        task.failure_report = None
        task.last_activity_signature = ""
        task.lease_owner = None
        task.lease_expires_at = None

    @staticmethod
    def _waiting_job_status(kind: InterventionKind) -> JobStatus:
        if kind == InterventionKind.MODEL:
            return JobStatus.WAITING_FOR_MODEL
        if kind == InterventionKind.PERMISSION:
            return JobStatus.WAITING_FOR_PERMISSION
        return JobStatus.WAITING_FOR_USER_DATA

    @staticmethod
    def _waiting_task_status(kind: InterventionKind) -> TaskStatus:
        if kind == InterventionKind.PERMISSION:
            return TaskStatus.WAITING_FOR_PERMISSION
        if kind in {InterventionKind.USER_DATA, InterventionKind.MODEL, InterventionKind.RESOURCE}:
            return TaskStatus.WAITING_FOR_USER_DATA
        return TaskStatus.WAITING_FOR_INPUT

    @staticmethod
    def _job_input_schema(fields: list[str]) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        for name in fields:
            if name in _STRING_LIST_FIELDS:
                properties[name] = {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                }
            elif name == "user_environment_notes":
                properties[name] = {"type": "string", "minLength": 1}
            elif name in {"gpu_count", "max_runtime_seconds"}:
                properties[name] = {"type": "integer", "minimum": 1}
            elif name in {"memory_mb", "disk_mb"}:
                properties[name] = {"type": "integer", "minimum": 128}
            elif name == "cpu_cores":
                properties[name] = {"type": "number", "minimum": 0.1}
            elif name == "gpu_memory_gb":
                properties[name] = {"type": "number", "exclusiveMinimum": 0}
        return {
            "type": "object",
            "properties": properties,
            "minProperties": 1,
            "additionalProperties": False,
        }

    @staticmethod
    def _extract_requested_tools(message: str) -> list[str]:
        candidates = re.findall(r"tool ['\"]([A-Za-z0-9_.-]+)['\"]", message)
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _string_list(value: Any, name: str, *, max_items: int = 100) -> list[str]:
        if not isinstance(value, list) or len(value) > max_items:
            raise InterventionValidationError(
                f"{name} must be an array with at most {max_items} strings"
            )
        result = []
        for item in value:
            if not isinstance(item, str) or not item.strip() or len(item) > 8_192:
                raise InterventionValidationError(
                    f"{name} must contain non-empty strings up to 8192 characters"
                )
            result.append(item)
        return result

    @staticmethod
    def _bounded_string(value: Any, name: str, limit: int) -> str:
        if not isinstance(value, str) or len(value) > limit:
            raise InterventionValidationError(
                f"{name} must be a string up to {limit} characters"
            )
        return value

    @staticmethod
    def _validated_actor(value: str) -> str:
        if not isinstance(value, str) or not value.strip() or len(value) > 200:
            raise InterventionValidationError(
                "responded_by must be a non-empty string up to 200 characters"
            )
        return value.strip()
