"""实验执行子智能体（设计文档 §9.7、§10 实验分级执行）。

职责：按序执行静态检查 → 单元测试 → 数据加载测试 → 模型前向测试 →
单 Batch 测试 → 冒烟测试 → 缩小规模实验 → 正式实验 → 评测程序。

**硬约束（§9.7）**："实验执行子智能体不能在运行过程中修改代码"。
本实现完全不向该子智能体授予 ``write_file``/``git_worktree_apply``
等写代码类工具，只授予 ``execute_command``（跑命令）+
``write_task_output``（写自己的结果文件）+ 只读文件工具——即使
任务定义意外把写文件工具加进了 ``allowed_tools``，orchestrator 在
构造该任务的 ``allowed_tools`` 时也不会包含它们（分工在
``orchestrator/task_factory.py`` 中通过任务类型固定模板体现）。

**硬约束（§10 开头）**："系统不得直接运行正式实验"——每一级必须
基于前一级通过后才能晋级，本类的 ``run_tier`` 方法要求调用方显式
声明 ``tier``，而 orchestrator 的分级门禁（gating）逻辑
（见 evaluation/tier_gate.py）负责保证正式实验任务只有在前四级全部
通过后才会被创建到 DAG 里，子智能体本身不做"是否该升级"的决策。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import hashlib
import json
import re
from pathlib import Path

from repro_agent.agents.base import AgentRunResult, BaseSubAgent
from repro_agent.domain.enums import ExperimentTier, FailureType
from repro_agent.domain.task import FailureReport
from repro_agent.evidence.provenance import RunRecord
from repro_agent.domain.common import new_id


_TIER_SUCCESS_CRITERIA = {
    ExperimentTier.STATIC_CHECK: "exit_code == 0",
    ExperimentTier.UNIT_TEST: "exit_code == 0 且测试全部通过",
    ExperimentTier.SMOKE_TEST: "数据加载、前向、反向传播、checkpoint 保存、评测程序均成功",
    ExperimentTier.REDUCED_EXPERIMENT: "loss 下降、梯度正常、指标合理",
    ExperimentTier.FULL_EXPERIMENT: "完整训练/评测流程成功且产出可追溯的指标文件",
}


# 退出码为 0 但 stderr 命中以下任一特征时，视为包装脚本掩盖了子进程
# 失败（典型形态：``uv run ... ; echo done``、``set +e``、无 ``set -e``
# 的多步 shell 脚本）。假成功会污染后续所有阶段，宁可误报交给重规划。
_MASKED_FAILURE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(?:error|fatal|critical)\s*:", re.IGNORECASE),
    re.compile(r"traceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r"\bcommand not found\b", re.IGNORECASE),
    re.compile(r"\bmodulenotfounderror\b", re.IGNORECASE),
    re.compile(r"\bimporterror\b", re.IGNORECASE),
    re.compile(r"\bmemoryerror\b", re.IGNORECASE),
    re.compile(r"\bsegmentation fault\b", re.IGNORECASE),
    re.compile(r"\bno module named\b", re.IGNORECASE),
)


# These values are emitted by the isolated execution backend when it, rather
# than the experiment process, stops a run to enforce a configured limit.
# Keep the classification tied to this explicit backend evidence: an ordinary
# program exit code (including 137) is not enough to infer resource pressure.
_RESOURCE_LIMIT_TERMINATION_REASONS = frozenset(
    {
        "timeout_killed",
        "log_limit_exceeded",
        "disk_limit_exceeded",
        "oom_killed",
    }
)

_RESOURCE_EXHAUSTION_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
    "out of memory",
    "cannot allocate memory",
    "memoryerror",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "std::bad_alloc",
    "oom-kill",
    "oom killed",
)

_CODE_ERROR_MARKERS = (
    "traceback",
    "syntaxerror",
    "indentationerror",
    "assertionerror",
    "attributeerror",
    "nameerror",
    "typeerror",
    "valueerror",
    "keyerror",
    "indexerror",
    "runtimeerror",
    "notimplementederror",
    "pytest.fail",
    " failed ",
)


@dataclass
class ExperimentExecutionResult:
    tier: str
    command: list[str] = field(default_factory=list)
    exit_code: int = -1
    stdout_tail: str = ""
    stderr_tail: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    run_id: str = ""
    container_digest: str = ""
    mock: bool = False
    git_commit: str = ""
    config_digest: str = ""
    dataset_digest: str = ""
    dataset_manifest: dict[str, Any] = field(default_factory=dict)
    model_identifier: str = ""
    seed: int | None = None
    hardware_identifier: str = ""
    log_path: str = ""
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0
    termination_reason: str = ""
    artifact_provenance: dict[str, Any] = field(default_factory=dict)
    verification_evidence: list[dict[str, Any]] = field(default_factory=list)
    tier_command_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "metrics": self.metrics,
            "run_id": self.run_id,
            "container_digest": self.container_digest,
            "mock": self.mock,
            "git_commit": self.git_commit,
            "config_digest": self.config_digest,
            "dataset_digest": self.dataset_digest,
            "dataset_manifest": self.dataset_manifest,
            "model_identifier": self.model_identifier,
            "seed": self.seed,
            "hardware_identifier": self.hardware_identifier,
            "log_path": self.log_path,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "termination_reason": self.termination_reason,
            "artifact_provenance": self.artifact_provenance,
            "verification_evidence": self.verification_evidence,
            "tier_command_verified": self.tier_command_verified,
        }


class ExperimentExecutionAgent(BaseSubAgent):
    task_type = "experiment_execution"
    system_prompt = (
        "你是 ReproAgent 系统的实验执行子智能体。你的任务是按照给定的分级"
        "（静态检查/单元测试/冒烟测试/缩小规模实验/正式实验）执行命令并"
        "收集结果。你绝对不能修改任何代码文件，只能执行命令、读取日志和"
        "输出结果文件。"
    )

    def run(self) -> AgentRunResult:
        inputs = self.task.definition.inputs
        tier_name = inputs.get("tier", ExperimentTier.STATIC_CHECK.value)
        command = inputs.get("command", [])
        timeout_seconds = inputs.get("timeout_seconds", 3600)
        metrics_output_path = inputs.get("metrics_output_path", "")
        repository_path = inputs.get("repository_path", "")
        # ``working_dir`` is the user-confirmed runtime parameter.  The
        # staging fallback remains for older persisted tasks that predate the
        # confirmation gate.
        working_dir = inputs.get("working_dir", inputs.get("repository_workdir", "."))
        execution_manifest = inputs.get("execution_manifest", {}) or {}
        experiment_environment = dict(inputs.get("experiment_environment", {}) or {})
        experiment_secret_env_vars = list(
            inputs.get("experiment_secret_env_vars", []) or []
        )

        source_evidence = self._hash_optional(repository_path)
        dataset_evidence = [
            item
            for path in inputs.get("dataset_paths", [])
            if (item := self._hash_optional(path))
        ]
        model_evidence = [
            item
            for path in inputs.get("model_paths", [])
            if (item := self._hash_optional(path))
        ]

        if not command:
            failure = FailureReport(
                failure_type=FailureType.INPUT_MISSING,
                failed_step="prepare_command",
                error_message="experiment execution task missing 'command'",
                likely_causes=["主智能体生成任务定义时未提供可执行命令"],
                recommended_action="由主智能体补充 command 字段后重试",
            )
            return AgentRunResult(succeeded=False, failure_report=failure)

        try:
            tier = ExperimentTier(tier_name)
        except ValueError:
            tier = ExperimentTier.STATIC_CHECK

        exec_result = self.call_tool(
            "execute_command",
            command=command,
            timeout_seconds=timeout_seconds,
            working_dir=working_dir,
            environment={
                "REPRO_AGENT_TIER": tier.value,
                # REPRO_AGENT_OUTPUT_DIR / REPRO_AGENT_METRICS_PATH 由执行后端
                # 自行设置（Docker 指向容器挂载 /output，Conda 指向宿主机任务
                # 输出目录）；在此传入会被 Conda 后端视为保留变量冲突。
                "PYTHONDONTWRITEBYTECODE": "1",
                **experiment_environment,
            },
            passthrough_environment=experiment_secret_env_vars,
            gpu_count=int(inputs.get("gpu_count") or 0),
            allow_network=bool(inputs.get("network_enabled", False)),
            workspace_read_only=True,
        )

        dataset_manifest = self._dataset_manifest(dataset_evidence)
        dataset_digest = dataset_manifest["digest"]
        source_digest = source_evidence.get("sha256", "") if source_evidence else ""
        git_commit = source_evidence.get("git_commit", "") if source_evidence else ""
        if not git_commit and source_digest:
            git_commit = f"source-sha256:{source_digest}"
        model_identifier = execution_manifest.get("model_identifier", "")
        if not model_identifier and model_evidence:
            model_identifier = f"sha256:{self._digest_evidence(model_evidence)}"
        if not model_identifier and source_digest:
            model_identifier = f"repository-bundled:{source_digest}"

        result = ExperimentExecutionResult(
            tier=tier.value,
            command=command,
            exit_code=exec_result.get("exit_code", -1),
            stdout_tail=exec_result.get("stdout", "")[-4000:],
            stderr_tail=exec_result.get("stderr", "")[-4000:],
            run_id=new_id("run"),
            container_digest=exec_result.get("container_digest", ""),
            mock=bool(exec_result.get("mock", False)),
            git_commit=git_commit,
            config_digest=str(execution_manifest.get("config_digest", "")),
            dataset_digest=dataset_digest,
            dataset_manifest=dataset_manifest,
            model_identifier=model_identifier,
            seed=execution_manifest.get("seed"),
            hardware_identifier=str(execution_manifest.get("hardware_identifier", "")),
            log_path=exec_result.get("stdout_log_path", ""),
            started_at=exec_result.get("started_at", ""),
            completed_at=exec_result.get("completed_at", ""),
            duration_seconds=float(exec_result.get("duration_seconds", 0.0) or 0.0),
            termination_reason=str(exec_result.get("termination_reason", "")),
            tier_command_verified=bool(inputs.get("tier_command_verified", False)),
        )

        if metrics_output_path and result.exit_code == 0:
            result.metrics = self._read_metrics_file(metrics_output_path)

        verification_evidence = self._verification_evidence(exec_result, metrics_output_path)
        result.verification_evidence = verification_evidence
        metrics_evidence = next(
            (
                {
                    "path": item["path"],
                    "sha256": item.get("sha256", ""),
                    "size_bytes": item.get("size_bytes", 0),
                }
                for item in verification_evidence
                if item.get("role") == "metrics"
            ),
            {},
        )
        provenance_body = {
            "run_id": result.run_id,
            "source": source_evidence,
            "datasets": dataset_evidence,
            "dataset_manifest": dataset_manifest,
            "models": model_evidence,
            "metrics": metrics_evidence,
            "config_digest": result.config_digest,
            "container_digest": result.container_digest,
        }
        result.artifact_provenance = {
            **provenance_body,
            "manifest_digest": hashlib.sha256(
                json.dumps(
                    provenance_body, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }

        # smoke_test/reduced 级别是验证可跑性，pytest 等命令不会产出
        # metrics 文件；仅 full_experiment 要求 metrics 输出。
        metrics_required = (
            not result.mock and tier == ExperimentTier.FULL_EXPERIMENT
        )
        # 包装脚本（例如 ``uv run ... ; echo done``）可能吞掉子进程失败并以
        # 退出码 0 结束。此类“假成功”会污染后续所有阶段，因此在命令自称
        # 成功时扫描 stderr 中的硬错误特征。
        masked_failure = (
            self._masked_failure_reason(result)
            if not result.mock and result.exit_code == 0
            else ""
        )
        succeeded = (
            result.exit_code == 0
            and (bool(result.metrics) or not metrics_required)
            and not masked_failure
        )
        result_payload = result.to_dict()
        result_payload["success_criteria"] = _TIER_SUCCESS_CRITERIA.get(tier, "")
        self.write_json_output("result.json", result_payload)
        self.write_candidate_memory(self._render_candidate_memory(result, succeeded))

        failure_report = None
        if not succeeded:
            failure_report = self._failure_report_for_unsuccessful_execution(
                result,
                tier=tier,
                metrics_required=metrics_required,
                masked_failure=masked_failure,
            )

        return AgentRunResult(
            succeeded=succeeded,
            outputs=result_payload,
            candidate_memory_written=True,
            failure_report=failure_report,
        )

    @staticmethod
    def _masked_failure_reason(result: ExperimentExecutionResult) -> str:
        """检测退出码为 0 时被掩盖的硬失败。

        返回 stderr 中首条命中硬错误特征的行（截断），未命中时返回空串。
        仅在 ``exit_code == 0`` 时由调用方使用。
        """
        stderr = result.stderr_tail or ""
        for pattern in _MASKED_FAILURE_PATTERNS:
            for line in stderr.splitlines():
                if pattern.search(line):
                    return line.strip()[:500]
        return ""

    @staticmethod
    def _failure_report_for_unsuccessful_execution(
        result: ExperimentExecutionResult,
        *,
        tier: ExperimentTier,
        metrics_required: bool,
        masked_failure: str = "",
    ) -> FailureReport:
        """Classify an unsuccessful run from explicit execution evidence.

        Resource-limit terminations are controller facts, so they must take
        precedence over any partial stderr (which may contain a traceback from
        a process interrupted while unwinding).  This lets the replanner route
        the failure to the existing human resource-confirmation path.
        """

        if result.termination_reason in _RESOURCE_LIMIT_TERMINATION_REASONS:
            return FailureReport(
                failure_type=FailureType.RESOURCE_EXCEEDED,
                failed_step=f"execute_tier_{tier.value}",
                error_message=(
                    "experiment execution was stopped by the sandbox resource "
                    f"guard: {result.termination_reason}"
                ),
                partial_outputs=["output/result.json"],
                likely_causes=[
                    "实验超过了已确认的运行时、日志或磁盘资源限制"
                ],
                recommended_action=(
                    "请求用户确认可用资源或调整运行时限制后，再创建新的执行尝试"
                ),
                metadata={"termination_reason": result.termination_reason},
            )

        diagnostic = "\n".join(
            part for part in (result.stderr_tail, result.stdout_tail) if part
        )
        if masked_failure and result.exit_code == 0:
            error_message = (
                "命令以退出码 0 结束，但 stderr 记录了硬错误——包装脚本很可能"
                f"掩盖了子进程失败：{masked_failure}"
            )
        elif result.exit_code == 0 and metrics_required and not result.metrics:
            error_message = "experiment tier did not produce the planned metrics output"
        else:
            error_message = diagnostic[-4000:]

        diagnostic_lower = f" {diagnostic.lower()} "
        resource_marker = next(
            (
                marker
                for marker in _RESOURCE_EXHAUSTION_MARKERS
                if marker in diagnostic_lower
            ),
            "",
        )
        likely_sigkill_oom = result.exit_code == 137 and (
            not diagnostic.strip() or " killed " in diagnostic_lower
        )
        if resource_marker or likely_sigkill_oom:
            return FailureReport(
                failure_type=FailureType.RESOURCE_EXCEEDED,
                failed_step=f"execute_tier_{tier.value}",
                error_message=error_message or "process was killed with exit code 137",
                partial_outputs=["output/result.json"],
                likely_causes=["实验耗尽了容器内存或 GPU 显存资源"],
                recommended_action=(
                    "请求用户确认更高的内存/显存限制或降低 batch size 后重试"
                ),
                metadata={
                    "termination_reason": result.termination_reason,
                    "exit_code": result.exit_code,
                    "resource_marker": resource_marker or "exit_137_sigkill",
                    "tier": tier.value,
                    "command": list(result.command),
                    "stdout_tail": result.stdout_tail[-4000:],
                    "stderr_tail": result.stderr_tail[-4000:],
                },
            )
        # 被掩盖的普通代码失败意味着包装脚本/命令本身有问题；明确的
        # ModuleNotFoundError 是例外，不论退出码是否被包装器吞掉，都应
        # 交给现有环境的依赖修复路径。
        missing_dependency = bool(
            re.search(
                r"(?:ModuleNotFoundError|ImportError):\s*No module named\s*['\"]",
                diagnostic,
                re.IGNORECASE,
            )
        )
        is_code_error = not missing_dependency and (
            bool(masked_failure)
            or any(marker in diagnostic_lower for marker in _CODE_ERROR_MARKERS)
        )

        return FailureReport(
            failure_type=(
                FailureType.CODE_ERROR
                if is_code_error
                else FailureType.ENVIRONMENT_ERROR
            ),
            failed_step=f"execute_tier_{tier.value}",
            error_message=error_message,
            partial_outputs=["output/result.json"],
            likely_causes=["命令执行失败，详见 stderr"],
            recommended_action="交由错误诊断子智能体分析",
            metadata={
                "termination_reason": result.termination_reason,
                "exit_code": result.exit_code,
                "tier": tier.value,
                "command": list(result.command),
                "stdout_tail": result.stdout_tail[-4000:],
                "stderr_tail": result.stderr_tail[-4000:],
            },
        )

    def _hash_optional(self, path: str) -> dict[str, Any]:
        if not path:
            return {}
        try:
            return self.call_tool("hash_path", path=path)
        except Exception:
            return {}

    @staticmethod
    def _digest_evidence(records: list[dict[str, Any]]) -> str:
        if not records:
            return ""
        payload = [
            {"path": item.get("path", ""), "sha256": item.get("sha256", "")}
            for item in records
        ]
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    @classmethod
    def _dataset_manifest(cls, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Make the no-external-data case explicit instead of using empty hash."""

        kind = "external" if records else "none"
        items = [
            {"path": item.get("path", ""), "sha256": item.get("sha256", "")}
            for item in records
        ]
        semantic = {"version": 1, "kind": kind, "items": items}
        digest = hashlib.sha256(
            json.dumps(semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {**semantic, "digest": f"dataset-manifest:v1:{digest}"}

    @staticmethod
    def _verification_evidence(
        exec_result: dict[str, Any], metrics_output_path: str
    ) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        for role, key, digest_key, size_key in (
            ("stdout_log", "stdout_log_path", "stdout_log_sha256", "stdout_log_size_bytes"),
            ("stderr_log", "stderr_log_path", "stderr_log_sha256", "stderr_log_size_bytes"),
            (
                "execution_state",
                "execution_state_path",
                "execution_state_sha256",
                "execution_state_size_bytes",
            ),
        ):
            path = exec_result.get(key)
            if path and (role != "execution_state" or Path(path).is_file()):
                evidence.append(
                    {
                        "role": role,
                        "path": str(path),
                        "sha256": str(exec_result.get(digest_key, "")),
                        "size_bytes": int(exec_result.get(size_key, 0) or 0),
                    }
                )
        configured_name = metrics_output_path.rsplit("/", 1)[-1]
        for item in exec_result.get("output_artifacts", []) or []:
            if not isinstance(item, dict) or not item.get("path"):
                continue
            relative_path = str(item.get("relative_path", ""))
            lower_path = relative_path.lower()
            if relative_path == configured_name:
                role = "metrics"
            elif "predict" in lower_path:
                role = "predictions"
            elif "label" in lower_path or "target" in lower_path:
                role = "labels"
            else:
                role = "artifact"
            evidence.append({role_key: value for role_key, value in {
                "role": role,
                "path": str(item["path"]),
                "relative_path": str(item.get("relative_path", "")),
                "size_bytes": item.get("size_bytes", 0),
                "sha256": str(item.get("sha256", "")),
            }.items() if value is not None})
        return evidence

    def _read_metrics_file(self, path: str) -> dict[str, float]:
        try:
            content = self.call_tool("read_file", path=path)
        except Exception:
            return {}
        try:
            data = json.loads(content.get("content", "{}"))
            return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}

    def _render_candidate_memory(self, result: ExperimentExecutionResult, succeeded: bool) -> str:
        return (
            f"# experiment_execution.{self.task.task_id}\n\n"
            "## 摘要 (L1)\n"
            f"层级: {result.tier}, 成功: {succeeded}, exit_code: {result.exit_code}\n\n"
            "## 细节 (L2)\n"
            f"- command: {result.command}\n"
            f"- metrics: {result.metrics}\n\n"
            "## 证据 (L3)\n"
            f"- run_id: {result.run_id}\n"
            f"- stderr_tail: {result.stderr_tail[-500:]}\n"
        )
