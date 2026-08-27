"""输出验证器（设计文档 §19 ``main_agent.validate_outputs``）。

主智能体收到子智能体的 ``AgentRunResult`` 后，不能无条件相信"子智能体
自称成功"，必须依据任务定义的 ``expected_outputs``/``completion_criteria``
做独立校验——这与 §9.9 "子智能体不能自行宣布整个实验复现成功"的原则
（也是设计文档 §3 原则 21）一脉相承，只是把校验范围从"实验复现"
扩大到"每一个任务的输出"。

校验通过 → 任务标记 SUCCEEDED；
校验失败 → 任务标记 VALIDATION_FAILED，交给 ``classify_failure``
（replanner.py）决定下一步动作，而不是直接判定为终止失败。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from repro_agent.domain.task import Task
from repro_agent.sandbox.manager import SandboxManager
from repro_agent.schemas.results import ResultValidationError, TaskResultEnvelope


@dataclass
class ValidationResult:
    task_id: str
    passed: bool
    reasons: list[str] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)


class OutputValidator:
    """依据任务定义的 expected_outputs 校验子智能体产出。"""

    def __init__(self, sandbox_manager: SandboxManager):
        self.sandbox_manager = sandbox_manager

    def validate(self, task: Task, agent_succeeded: bool) -> ValidationResult:
        if not agent_succeeded:
            return ValidationResult(task_id=task.task_id, passed=False, reasons=["子智能体报告执行失败"])

        sandbox = self.sandbox_manager.get(task.task_id)
        if sandbox is None:
            return ValidationResult(
                task_id=task.task_id, passed=False, reasons=["找不到任务对应的沙箱，无法校验产出"]
            )

        return self._validate_produced(task, sandbox.collect_outputs())

    def validate_recovered_output(self, task: Task, output_dir: str | Path) -> ValidationResult:
        """校验进程中断前已落盘的 output/ 目录。

        该路径只用于恢复器；它不会重新运行子 Agent，也不会相信数据库中
        的旧状态，而是重新读取 result envelope 和所有 expected outputs。
        """

        root = Path(output_dir)
        produced = {
            str(path.relative_to(root)): str(path)
            for path in root.rglob("*")
            if path.is_file() and not path.is_symlink()
        } if root.is_dir() else {}
        return self._validate_produced(task, produced)

    @staticmethod
    def _validate_produced(task: Task, produced: dict[str, str]) -> ValidationResult:
        reasons: list[str] = []
        for expected in task.definition.expected_outputs:
            # expected_outputs 里的路径约定为相对 output/ 目录的路径，
            # 与 write_task_output 工具的写入目标保持一致。
            rel = expected.split("output/", 1)[-1] if "output/" in expected else expected
            if rel not in produced:
                reasons.append(f"缺少预期产物文件: {expected}")

        result_path = produced.get("result.json")
        if result_path:
            try:
                TaskResultEnvelope.from_file(
                    result_path,
                    expected_task_id=task.task_id,
                    expected_attempt_id=task.active_attempt_id,
                    expected_task_type=task.definition.task_type,
                )
            except ResultValidationError as exc:
                reasons.append(f"结果契约校验失败: {exc}")

        for criterion in task.definition.completion_criteria:
            # 完成标准的具体校验逻辑因任务而异，这里只做存在性记录，
            # 真正的语义校验（如"指标在容差范围内"）由结果验证子智能体
            # 或更专门的校验器（evaluation/ 模块）完成，避免在这个
            # 通用校验器里堆砌大量 if-else 特判。
            reasons.append(f"[待人工/专项校验] {criterion}")

        hard_failures = [r for r in reasons if not r.startswith("[待人工/专项校验]")]
        passed = len(hard_failures) == 0

        return ValidationResult(task_id=task.task_id, passed=passed, reasons=reasons, outputs=produced)
