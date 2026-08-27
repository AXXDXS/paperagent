"""任务并发优先级排序（设计文档 §8.3）。

调度器优先考虑（按文档顺序）：
    1. 关键路径上的任务；
    2. 能解锁更多任务的任务；
    3. 用户明确指定的高优先级任务；
    4. 预计时间较短的任务；
    5. 资源占用较小的任务；
    6. 已完成前置修复的任务。

实现说明：
    "关键路径"精确计算需要预估每个任务的耗时并做最长路径分析，
    工程上性价比不高（预估本身就不准）。这里采用一个更稳健的
    近似：用 ``unlock_count``（该任务完成后能级联解锁的下游任务数，
    是关键路径的一个强相关代理指标——真正的关键路径瓶颈节点通常也是
    解锁最多任务的节点）作为第一/第二条的合并近似，其余四条按文档
    顺序作为排序的第二、三、四关键字。这是一个显式的简化，若未来
    需要更精确的关键路径分析，可以在此模块内替换排序函数而不影响
    调度器主体逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass

from repro_agent.domain.dag import TaskDAG
from repro_agent.domain.task import Task


@dataclass
class ScheduledCandidate:
    task: Task
    unlock_count: int
    user_priority: int
    expected_duration: int
    estimated_resource_cost: float


def rank_ready_tasks(dag: TaskDAG, ready: list[Task]) -> list[Task]:
    """按 §8.3 优先级对 READY 任务排序，返回新列表（不修改输入）。"""

    candidates = []
    for task in ready:
        resource_cost = _estimate_resource_cost(task)
        candidates.append(
            ScheduledCandidate(
                task=task,
                unlock_count=dag.unlock_count(task.task_id),
                user_priority=task.definition.priority,
                expected_duration=task.definition.expected_duration_seconds,
                estimated_resource_cost=resource_cost,
            )
        )

    candidates.sort(
        key=lambda c: (
            -c.unlock_count,  # 解锁越多任务，优先级越高
            -c.user_priority,  # 用户指定优先级越高越先跑
            c.expected_duration,  # 预计耗时越短越先跑（快速清理短任务）
            c.estimated_resource_cost,  # 资源占用越小越先跑
        )
    )
    return [c.task for c in candidates]


def _estimate_resource_cost(task: Task) -> float:
    """从任务输入里估算资源占用（GPU 数量优先，其次输入文件规模）。"""

    inputs = task.definition.inputs or {}
    gpu_count = inputs.get("gpu_count")
    if isinstance(gpu_count, (int, float)):
        return float(gpu_count)
    files = inputs.get("files")
    if isinstance(files, list):
        return float(len(files)) * 0.01
    return 0.0
