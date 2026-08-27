"""资源检查子智能体（设计文档 §9.3）。

职责：检查数据、模型、checkpoint、GPU、显存、CPU/内存、磁盘、CUDA、
驱动、许可证和访问限制。缺少关键资源时不自行决定阻塞策略，而是把
``ResourceStatus`` 明确报告给主智能体，由主智能体决定是否询问用户/
阻塞下游任务（§9.3 最后一句）。

工具方面全部使用只读探测工具（find_named_resource /
check_path_resource / check_disk_space / check_gpu / check_cuda），风险预算
READ_ONLY。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from repro_agent.agents.base import AgentRunResult, BaseSubAgent
from repro_agent.domain.enums import ResourceStatus
from repro_agent.domain.resource_requirements import (
    normalize_required_resources,
    resource_name_key,
)


@dataclass
class ResourceCheckResult:
    dataset_status: dict[str, str] = field(default_factory=dict)
    model_status: dict[str, str] = field(default_factory=dict)
    checkpoint_status: dict[str, str] = field(default_factory=dict)
    gpu_info: dict[str, Any] = field(default_factory=dict)
    cuda_info: dict[str, Any] = field(default_factory=dict)
    disk_info: dict[str, Any] = field(default_factory=dict)
    required_resources: list[dict[str, Any]] = field(default_factory=list)
    required_resource_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    missing_required_resources: list[dict[str, Any]] = field(default_factory=list)
    blocking_issues: list[str] = field(default_factory=list)
    audit_checks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_status": self.dataset_status,
            "model_status": self.model_status,
            "checkpoint_status": self.checkpoint_status,
            "gpu_info": self.gpu_info,
            "cuda_info": self.cuda_info,
            "disk_info": self.disk_info,
            "required_resources": self.required_resources,
            "required_resource_status": self.required_resource_status,
            "missing_required_resources": self.missing_required_resources,
            "blocking_issues": self.blocking_issues,
            "audit_checks": self.audit_checks,
        }


_BLOCKING_STATUSES = {
    ResourceStatus.MISSING.value,
    ResourceStatus.REQUIRES_AUTHORIZATION.value,
    ResourceStatus.REQUIRES_CREDENTIALS.value,
    ResourceStatus.RESTRICTED.value,
}


class ResourceCheckAgent(BaseSubAgent):
    task_type = "resource_check"
    system_prompt = (
        "你是 ReproAgent 系统的资源检查子智能体。你的任务是检查数据集、"
        "模型、checkpoint 是否存在及可用，检查 GPU/显存/CUDA/驱动/磁盘"
        "空间，并汇报清晰的资源状态。当发现关键资源缺失时，不要自行决定"
        "如何处理，只需要如实报告状态，由主智能体决定后续动作。"
    )

    def run(self) -> AgentRunResult:
        inputs = self.task.definition.inputs
        dataset_paths: list[str] = inputs.get("dataset_paths", [])
        model_paths: list[str] = inputs.get("model_paths", [])
        checkpoint_paths: list[str] = inputs.get("checkpoint_paths", [])
        repository_path: str = inputs.get("repository_path", "")
        experiment_spec = inputs.get("experiment_spec", {}) or {}
        declared_resources = experiment_spec.get("required_resources")
        if not isinstance(declared_resources, list):
            declared_resources = (experiment_spec.get("resources") or {}).get(
                "required", []
            )
        required_resources = normalize_required_resources(declared_resources)

        result = ResourceCheckResult(
            required_resources=required_resources,
            audit_checks=[
                str(item)
                for item in inputs.get("required_checks", [])
                if str(item).strip()
            ],
        )

        for path in dataset_paths:
            status = self.call_tool_checkpointed(
                f"dataset_resource:{path}", "check_path_resource", path=path, kind="data"
            )
            result.dataset_status[path] = status.get("status", ResourceStatus.UNKNOWN.value)

        for path in model_paths:
            status = self.call_tool_checkpointed(
                f"model_resource:{path}", "check_path_resource", path=path, kind="model"
            )
            result.model_status[path] = status.get("status", ResourceStatus.UNKNOWN.value)

        for path in checkpoint_paths:
            status = self.call_tool_checkpointed(
                f"checkpoint_resource:{path}",
                "check_path_resource",
                path=path,
                kind="checkpoint",
            )
            result.checkpoint_status[path] = status.get("status", ResourceStatus.UNKNOWN.value)

        result.gpu_info = self.call_tool_checkpointed("gpu_info", "check_gpu")
        result.cuda_info = self.call_tool_checkpointed("cuda_info", "check_cuda")
        result.disk_info = self.call_tool_checkpointed("disk_info", "check_disk_space", path=".")

        explicit_paths = {
            "dataset": result.dataset_status,
            "model": result.model_status,
            "checkpoint": result.checkpoint_status,
        }
        required_counts = {
            kind: sum(
                1
                for item in required_resources
                if item["required"] and item["kind"] == kind
            )
            for kind in explicit_paths
        }
        for index, requirement in enumerate(required_resources):
            if not requirement["required"]:
                continue
            kind = requirement["kind"]
            available_explicit = [
                path
                for path, status in explicit_paths.get(kind, {}).items()
                if status not in _BLOCKING_STATUSES
            ]
            matched_explicit = [
                path
                for path in available_explicit
                if resource_name_key(requirement["name"])
                in resource_name_key(path)
            ]
            # When the spec has exactly one resource of this kind, a single
            # user-provided path is an explicit binding even if its basename is
            # generic (for example /mnt/data/current).
            if (
                not matched_explicit
                and required_counts.get(kind) == 1
                and len(available_explicit) == 1
            ):
                matched_explicit = list(available_explicit)

            discovery: dict[str, Any] = {}
            if matched_explicit:
                status = ResourceStatus.AVAILABLE_BUT_UNVERIFIED.value
                discovery = {
                    "source": "user_provided_path",
                    "matched_paths": matched_explicit,
                }
            else:
                status, discovery = self._discover_in_repository(
                    repository_path,
                    requirement,
                    checkpoint_key=f"required_resource:{index}",
                )

            record = {
                **requirement,
                "status": status,
                "discovery": discovery,
            }
            result.required_resource_status[requirement["resource_id"]] = record
            if status in _BLOCKING_STATUSES:
                result.missing_required_resources.append(record)

        all_statuses = {
            **result.dataset_status,
            **result.model_status,
            **result.checkpoint_status,
        }
        result.blocking_issues = [
            f"{path}: {status}"
            for path, status in all_statuses.items()
            if status in _BLOCKING_STATUSES
        ]
        result.blocking_issues.extend(
            f"required {item['kind']} '{item['name']}': {item['status']}"
            for item in result.missing_required_resources
        )
        requested_gpu_count = self._non_negative_int(
            inputs.get("requested_gpu_count"), default=0
        )
        requested_gpu_memory_gb = self._non_negative_float(
            inputs.get("requested_gpu_memory_gb"), default=0.0
        )
        if requested_gpu_count:
            available_count = int(result.gpu_info.get("gpu_count", 0) or 0)
            if not result.gpu_info.get("available") or available_count < requested_gpu_count:
                result.blocking_issues.append(
                    f"requested {requested_gpu_count} GPU(s), but only {available_count} are available"
                )
            elif requested_gpu_memory_gb:
                required_mb = int(requested_gpu_memory_gb * 1024)
                eligible = [
                    gpu
                    for gpu in result.gpu_info.get("gpus", [])
                    if int(gpu.get("memory_total_mb", 0) or 0) >= required_mb
                ]
                if len(eligible) < requested_gpu_count:
                    result.blocking_issues.append(
                        f"requested {requested_gpu_memory_gb:g} GiB VRAM on each of "
                        f"{requested_gpu_count} GPU(s), but only {len(eligible)} device(s) qualify"
                    )
        requested_disk_mb = self._non_negative_int(
            inputs.get("requested_disk_mb"), default=0
        )
        if requested_disk_mb and int(result.disk_info.get("free_bytes", 0) or 0) < requested_disk_mb * 1024 * 1024:
            result.blocking_issues.append(
                f"requested {requested_disk_mb} MiB disk, but the sandbox host has insufficient free space"
            )

        result_payload = result.to_dict()
        self.write_json_output("result.json", result_payload)
        self.write_candidate_memory(self._render_candidate_memory(result))

        return AgentRunResult(
            succeeded=True,
            outputs=result_payload,
            candidate_memory_written=True,
        )

    @staticmethod
    def _non_negative_int(value: Any, *, default: int) -> int:
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return default
        return value

    @staticmethod
    def _non_negative_float(value: Any, *, default: float) -> float:
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            return default
        return float(value)

    def _discover_in_repository(
        self,
        repository_path: str,
        requirement: dict[str, Any],
        *,
        checkpoint_key: str,
    ) -> tuple[str, dict[str, Any]]:
        if not repository_path or "find_named_resource" not in self.granted_tools:
            return ResourceStatus.MISSING.value, {
                "source": "repository_search_unavailable",
                "candidates": [],
            }
        names = [requirement["name"], *requirement.get("aliases", [])]
        last_result: dict[str, Any] = {}
        for alias_index, name in enumerate(names):
            last_result = self.call_tool_checkpointed(
                f"{checkpoint_key}:{alias_index}",
                "find_named_resource",
                root=repository_path,
                name=name,
                kind=requirement["kind"],
            )
            if last_result.get("status") not in _BLOCKING_STATUSES:
                return str(last_result["status"]), {
                    "source": "repository_search",
                    "matched_name": name,
                    "candidates": list(last_result.get("candidates", [])),
                    "truncated": bool(last_result.get("truncated", False)),
                }
        return ResourceStatus.MISSING.value, {
            "source": "repository_search",
            "candidates": list(last_result.get("candidates", [])),
            "truncated": bool(last_result.get("truncated", False)),
        }

    def _render_candidate_memory(self, result: ResourceCheckResult) -> str:
        lines = [
            f"# resource_check.{self.task.task_id}",
            "",
            "## 摘要 (L1)",
            f"GPU 可用: {result.gpu_info.get('available')}; "
            f"阻塞性问题数: {len(result.blocking_issues)}",
            "",
            "## 细节 (L2)",
            f"- dataset_status: {result.dataset_status}",
            f"- model_status: {result.model_status}",
            f"- checkpoint_status: {result.checkpoint_status}",
            f"- required_resource_status: {result.required_resource_status}",
            f"- disk_free_gb: {result.disk_info.get('free_gb')}",
            "",
            "## 证据 (L3)",
        ]
        for issue in result.blocking_issues:
            lines.append(f"- 阻塞问题: {issue}")
        return "\n".join(lines) + "\n"
