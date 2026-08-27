"""沙箱管理器：为每个任务创建/回收 ``TaskSandbox``（设计文档 §12）。

与调度器、工具授权的关系：
    orchestrator 在派发任务前调用 ``SandboxManager.create_sandbox``
    得到一个 ``TaskSandbox``，再用它构造 ``ToolAuthorizer.authorize``
    所需的 ``sandbox_ctx`` 参数——沙箱和工具授权是两个独立关注点
    （"能访问哪些路径" vs "能调用哪些工具"），组合起来才构成完整的
    子智能体能力边界。
"""

from __future__ import annotations

import logging
import hashlib
import re
import shutil
from pathlib import Path
import math

from repro_agent.domain.task import Task
from repro_agent.sandbox.policy import SandboxPolicy, SandboxResourceLimits
from repro_agent.sandbox.paths import to_virtual_path
from repro_agent.sandbox.workspace import TaskSandbox

logger = logging.getLogger(__name__)

# 任务输入字段中，凡是"宿主机路径"语义的键，都必须在创建沙箱时
# 拷贝进 input/ 并把 inputs 中的值原地替换为沙箱内路径——子智能体
# 代码（agents/*/agent.py）永远只从 task.definition.inputs 里读到
# 沙箱内路径，天然无法越权触达宿主机文件系统，不需要每个子智能体
# 自己判断"这个路径是不是在沙箱外"。
#
# 单值字段（一个键对应一个路径字符串）与列表字段（一个键对应多个
# 路径）分开处理。
_SINGLE_PATH_INPUT_KEYS = ("paper_path", "repository_path")
_LIST_PATH_INPUT_KEYS = ("appendix_paths", "dataset_paths", "model_paths", "checkpoint_paths", "files")


class SandboxManager:
    """管理某个 Job 下所有任务沙箱的生命周期。"""

    def __init__(self, job_root: str | Path, *, execution_backend=None, execution_image: str = "python:3.11-slim"):
        self.job_root = Path(job_root)
        self.sandbox_root = self.job_root / "sandbox"
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
        self._sandboxes: dict[str, TaskSandbox] = {}
        self.execution_backend = execution_backend
        self.execution_image = execution_image

    def create_sandbox(
        self,
        task: Task,
        *,
        allow_network: bool | None = None,
        extra_readable_roots: list[str] | None = None,
        resource_limits: SandboxResourceLimits | None = None,
    ) -> TaskSandbox:
        attempt_id = task.active_attempt_id or f"attempt_{task.attempt}"
        root = self.sandbox_root / f"task_{task.task_id}" / attempt_id
        inputs = task.definition.inputs
        if allow_network is None:
            requested_network = inputs.get("network_enabled", False)
            if not isinstance(requested_network, bool):
                raise ValueError("task network_enabled must be boolean")
            allow_network = requested_network
        if resource_limits is None:
            requested_gpu_count = inputs.get("gpu_count", 0) or 0
            if isinstance(requested_gpu_count, bool) or not isinstance(
                requested_gpu_count, int
            ) or requested_gpu_count < 0:
                raise ValueError("task gpu_count must be a non-negative integer")
            cpu_cores = self._positive_number(
                inputs.get("cpu_cores"), default=1.0, name="cpu_cores"
            )
            memory_mb = self._positive_integer(
                inputs.get("memory_mb"), default=1024, name="memory_mb"
            )
            disk_mb = self._positive_integer(
                inputs.get("disk_mb"), default=4096, name="disk_mb"
            )
            gpu_memory_gb = inputs.get("gpu_memory_gb")
            if gpu_memory_gb in (None, 0, 0.0):
                gpu_memory_mb = None
            else:
                gpu_memory_mb = math.ceil(
                    self._positive_number(
                        gpu_memory_gb, default=0.0, name="gpu_memory_gb"
                    )
                    * 1024
                )
            resource_limits = SandboxResourceLimits(
                cpu_cores=cpu_cores,
                memory_mb=memory_mb,
                disk_mb=disk_mb,
                gpu_count=requested_gpu_count,
                gpu_memory_mb=gpu_memory_mb,
            )
        policy = SandboxPolicy(
            task_id=task.task_id,
            allow_network=allow_network,
            readable_extra_roots=extra_readable_roots or [],
            resource_limits=resource_limits,
            soft_timeout_seconds=task.definition.soft_timeout_seconds,
            hard_timeout_seconds=task.definition.hard_timeout_seconds,
            attempt_number=task.attempt,
            approved_destructive_command_fingerprints=[
                str(item.get("fingerprint"))
                for item in task.definition.inputs.get(
                    "_destructive_action_approvals", []
                )
                if isinstance(item, dict)
                and item.get("approved_for_attempt") == task.attempt
                and item.get("fingerprint")
            ],
        )
        sandbox = TaskSandbox(
            task_id=task.task_id,
            attempt_id=attempt_id,
            root=root,
            policy=policy,
            extra_readable_roots=[Path(p) for p in (extra_readable_roots or [])],
            execution_backend=self.execution_backend,
            execution_image=(
                task.definition.inputs.get("execution_image") or self.execution_image
            ),
        )
        self._stage_declared_inputs(task, sandbox)
        self._sandboxes[task.task_id] = sandbox
        return sandbox

    @staticmethod
    def _positive_number(value, *, default: float, name: str) -> float:
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"task {name} must be a positive number")
        return float(value)

    @staticmethod
    def _positive_integer(value, *, default: int, name: str) -> int:
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"task {name} must be a positive integer")
        return value

    def _stage_declared_inputs(self, task: Task, sandbox: TaskSandbox) -> None:
        """把任务定义中所有声明为宿主机路径的输入拷贝进沙箱 input/，
        并将 ``task.definition.inputs`` 中对应字段原地重写为
        ``input://xxx`` 虚拟路径（借鉴 DeerFlow 的虚拟路径映射，见
        ``sandbox/paths.py`` 顶部说明）。

        子智能体（``agents/*/agent.py``）读到的 ``inputs`` 字段永远是
        虚拟路径字符串，既看不到、也不需要知道宿主机真实路径在哪里——
        即使子智能体的 Prompt/输出被意外泄露给用户或写入日志，暴露的
        也只是 ``input://paper.txt`` 这种不包含任何宿主机文件系统信息
        的句柄，而不是真实的绝对路径。
        """

        inputs = task.definition.inputs
        if not inputs:
            return

        for key in _SINGLE_PATH_INPUT_KEYS:
            value = inputs.get(key)
            if not value or not isinstance(value, str):
                continue
            virtual = self._safe_stage(
                sandbox, value, dest_relative_path=self._namespaced_destination(key, value)
            )
            if virtual is not None:
                inputs[key] = virtual
                if key == "repository_path" and task.definition.task_type in {
                    "environment_build", "coding", "experiment_execution"
                }:
                    staged_repository = Path(sandbox.resolve_readable_path(virtual))
                    writable_repository = sandbox.workspace_dir / "repository"
                    if staged_repository.is_dir():
                        shutil.copytree(
                            staged_repository,
                            writable_repository,
                            dirs_exist_ok=True,
                            symlinks=True,
                            ignore_dangling_symlinks=True,
                        )
                        inputs["repository_workdir"] = "workspace://repository"

        for key in _LIST_PATH_INPUT_KEYS:
            values = inputs.get(key)
            if not isinstance(values, list):
                continue
            staged_values = []
            for index, v in enumerate(values):
                if not isinstance(v, str):
                    continue
                virtual = self._safe_stage(
                    sandbox,
                    v,
                    dest_relative_path=self._namespaced_destination(key, v, index=index),
                )
                if virtual is not None:
                    staged_values.append(virtual)
            inputs[key] = staged_values

        # Verification evidence is a labelled list because the verifier must
        # distinguish metrics, logs, execution state and arbitrary artifacts.
        # Stage each source under a role + content-addressed namespace and
        # rewrite only the path field visible to the child agent.
        evidence = inputs.get("verification_evidence")
        if isinstance(evidence, list):
            staged_evidence = []
            for index, item in enumerate(evidence):
                if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                    continue
                source_path = item["path"]
                role = str(item.get("role", "artifact"))
                destination = self._namespaced_destination(
                    f"verification/{self._slug(role)}", source_path, index=index
                )
                virtual = self._safe_stage(
                    sandbox, source_path, dest_relative_path=destination
                )
                if virtual is None:
                    continue
                staged_item = dict(item)
                staged_item["path"] = virtual
                staged_evidence.append(staged_item)
            inputs["verification_evidence"] = staged_evidence

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "-", value).strip("-") or "artifact"

    @classmethod
    def _namespaced_destination(
        cls, key: str, source_path: str, *, index: int = 0
    ) -> str:
        source = Path(source_path)
        digest = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:16]
        basename = cls._slug(source.name or "resource")
        return f"{cls._slug(key)}/{digest}-{index}-{basename}"

    def _safe_stage(
        self,
        sandbox: TaskSandbox,
        source_path: str,
        *,
        dest_relative_path: str | None = None,
    ) -> str | None:
        try:
            return sandbox.stage_input_file_as_virtual_path(
                source_path, dest_relative_path=dest_relative_path
            )
        except FileNotFoundError:
            # 不能把缺失输入从列表中静默删掉，否则 resource_check 会
            # 误判为“用户没有声明任何资源”。返回一个位于沙箱 input/
            # 下、确定不存在的虚拟句柄，让只读资源探测工具安全地产生
            # MISSING，同时不向子智能体泄露宿主机绝对路径。
            logger.warning("input source not found while staging sandbox: %s", source_path)
            digest = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:12]
            basename = Path(source_path).name or "resource"
            return to_virtual_path("input", f"_missing/{digest}-{basename}")

    def get(self, task_id: str) -> TaskSandbox | None:
        return self._sandboxes.get(task_id)

    def cleanup(
        self,
        task_id: str,
        *,
        keep_output: bool = True,
        human_confirmed: bool = False,
    ) -> bool:
        """清理任务沙箱；删除目录前必须由控制面传入人工确认。"""

        sandbox = self._sandboxes.get(task_id)
        if sandbox is None:
            return False
        if not human_confirmed:
            logger.warning(
                "sandbox cleanup for task %s skipped: human confirmation required",
                task_id,
            )
            return False
        import shutil

        for d in (sandbox.tmp_dir, sandbox.workspace_dir):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True, exist_ok=True)
        if not keep_output and sandbox.output_dir.exists():
            shutil.rmtree(sandbox.output_dir, ignore_errors=True)
            sandbox.output_dir.mkdir(parents=True, exist_ok=True)
        return True
