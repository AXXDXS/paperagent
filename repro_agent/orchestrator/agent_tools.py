"""Main-agent-only tools for creating and launching sub-agents.

This module deliberately lives in ``orchestrator`` rather than the global
``tools`` package.  Its tool schema can be exposed to a future main-agent LLM,
but it is never registered in ``default_registry()`` and is never injected into
``BaseSubAgent``.  That keeps the creation boundary explicit: only the main
agent owns this tool, and child agents cannot recursively create more agents.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from repro_agent.domain.enums import FailureType, TaskStatus
from repro_agent.domain.task import FailureReport
from repro_agent.orchestrator.artifacts import ArtifactResolutionError, ArtifactResolver
from repro_agent.orchestrator.dispatcher import AgentDispatcher
from repro_agent.scheduler.scheduler import TaskScheduler


@dataclass(frozen=True)
class SubAgentCreationRecord:
    task_id: str
    status: str
    attempt_id: str = ""
    assigned_agent: str = ""
    error: str = ""


@dataclass(frozen=True)
class CreateSubagentsResult:
    requested_task_ids: list[str]
    records: list[SubAgentCreationRecord]

    @property
    def started_task_ids(self) -> list[str]:
        return [record.task_id for record in self.records if record.status == "started"]

    @property
    def failed_task_ids(self) -> list[str]:
        return [record.task_id for record in self.records if record.status == "failed"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_task_ids": list(self.requested_task_ids),
            "started_task_ids": self.started_task_ids,
            "failed_task_ids": self.failed_task_ids,
            "records": [asdict(record) for record in self.records],
        }


class CreateSubagentsTool:
    """Batch-create all dispatched sub-agents selected in one scheduler step."""

    name = "create_subagents"
    description = (
        "Create and asynchronously start one or more already-dispatched sub-agents. "
        "This tool is restricted to the main orchestrator."
    )

    def __init__(
        self,
        scheduler: TaskScheduler,
        dispatcher: AgentDispatcher,
        *,
        tool_allocation_planner=None,
    ):
        self._scheduler = scheduler
        self._dispatcher = dispatcher
        # 主智能体的规划期工具分配器（可选）：派发前按任务内容定制
        # allowed_tools，任务类型模板降级为参考。为 None 时保持模板
        # 原样下发（旧行为，单测/未配置 LLM 时）。
        self._tool_allocation_planner = tool_allocation_planner

    @classmethod
    def to_openai_tool(cls) -> dict[str, Any]:
        """Return a standard function-tool schema for future main-LLM use."""

        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_ids": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                            "minItems": 1,
                            "description": (
                                "Task ids that the scheduler has already moved to "
                                "DISPATCHED in this scheduling step."
                            ),
                        }
                    },
                    "required": ["task_ids"],
                    "additionalProperties": False,
                },
            },
        }

    def call(self, *, task_ids: list[str]) -> CreateSubagentsResult:
        """Resolve inputs and start all requested agents without blocking.

        A single call accepts the complete scheduler batch.  Failures are
        isolated per task so one invalid task cannot prevent other independent
        tasks in the same batch from starting.
        """

        requested = list(dict.fromkeys(str(task_id) for task_id in task_ids if task_id))
        if not requested:
            raise ValueError("create_subagents requires at least one non-empty task id")

        tasks_by_id = {
            task.task_id: task for task in self._scheduler.dag.all_tasks()
        }
        resolver = ArtifactResolver(tasks_by_id)
        records: list[SubAgentCreationRecord] = []

        for task_id in requested:
            task = tasks_by_id.get(task_id)
            if task is None:
                records.append(
                    SubAgentCreationRecord(
                        task_id=task_id,
                        status="failed",
                        error="task is not present in the main-agent DAG",
                    )
                )
                continue
            if task.status != TaskStatus.DISPATCHED:
                records.append(
                    SubAgentCreationRecord(
                        task_id=task_id,
                        status="failed",
                        attempt_id=task.active_attempt_id,
                        assigned_agent=task.assigned_agent or "",
                        error=(
                            "task must be DISPATCHED before create_subagents; "
                            f"current status is {task.status.value}"
                        ),
                    )
                )
                continue

            try:
                resolved_inputs = resolver.resolve(task)
            except ArtifactResolutionError as exc:
                self._scheduler.mark_failed(
                    task,
                    TaskStatus.TERMINAL_FAILURE,
                    FailureReport(
                        failure_type=FailureType.DEPENDENCY_ERROR,
                        failed_step="resolve_dependency_artifacts",
                        error_message=str(exc),
                    ),
                )
                records.append(
                    SubAgentCreationRecord(
                        task_id=task_id,
                        status="failed",
                        attempt_id=task.active_attempt_id,
                        assigned_agent=task.assigned_agent or "",
                        error=str(exc),
                    )
                )
                continue

            try:
                self._customize_tool_allocation(task, resolved_inputs)
                self._scheduler.mark_running(task)
                self._dispatcher.start_async(task, resolved_inputs=resolved_inputs)
            except Exception as exc:  # noqa: BLE001 - isolate one failed launch
                self._scheduler.mark_failed(
                    task,
                    TaskStatus.FAILED_RETRYABLE,
                    FailureReport(
                        failure_type=FailureType.TOOL_ERROR,
                        failed_step="create_subagent",
                        error_message=str(exc),
                        likely_causes=["子 Agent 创建或后台线程启动失败"],
                        recommended_action="检查 Agent 注册、沙箱和工具授权后重试",
                    ),
                )
                records.append(
                    SubAgentCreationRecord(
                        task_id=task_id,
                        status="failed",
                        attempt_id=task.active_attempt_id,
                        assigned_agent=task.assigned_agent or "",
                        error=str(exc),
                    )
                )
                continue

            records.append(
                SubAgentCreationRecord(
                    task_id=task_id,
                    status="started",
                    attempt_id=task.active_attempt_id,
                    assigned_agent=task.assigned_agent or "",
                )
            )

        result = CreateSubagentsResult(requested_task_ids=requested, records=records)
        self._scheduler.task_repo.record_event(
            self._scheduler.job_id,
            None,
            "create_subagents_tool_called",
            result.to_dict(),
        )
        return result

    def _customize_tool_allocation(self, task, resolved_inputs) -> None:
        """派发前由主智能体定制任务的工具白名单（模板仅作参考）。

        工具分配权上收的规划期落地：主智能体结合任务目标、输入、
        类型模板（参考）和工具白名单/风险值，为任务实例裁剪出
        实际需要的 ``allowed_tools``。只在**首次派发**（attempt == 1）
        时定制——重试派发保留运行期累积的授权（人工批准追加、
        主智能体补授的工具都在 allowed_tools 里，不能被重新裁掉）。
        定制失败一律回退模板，绝不阻塞派发。
        """

        planner = self._tool_allocation_planner
        if planner is None:
            return
        if task.attempt != 1:
            # 重试派发：保留既有白名单（含运行期追加的授权）。
            return
        try:
            customized, source = planner.plan_allowed_tools(
                task_type=task.definition.task_type,
                objective=task.definition.objective,
                inputs=resolved_inputs or task.definition.inputs,
                template_tools=list(task.definition.allowed_tools),
                forbidden_actions=task.definition.forbidden_actions,
            )
        except Exception:  # noqa: BLE001 - 定制失败不阻塞派发
            return
        if customized == list(task.definition.allowed_tools):
            return
        previous = list(task.definition.allowed_tools)
        task.definition.allowed_tools = list(customized)
        self._scheduler.task_repo.save(task)
        self._scheduler.task_repo.record_event(
            self._scheduler.job_id,
            task.task_id,
            "tool_allocation_customized",
            {
                "attempt_id": task.active_attempt_id,
                "previous_tools": previous,
                "allowed_tools": list(customized),
                "source": source,
            },
        )
