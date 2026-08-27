"""任务 DAG（设计文档 §3 原则 4-6、§6 任务拆解、§8 并发调度）。

设计要求（对应设计文档条款）：
    - 原则 4：所有任务形成任务 DAG；
    - 原则 5：有前置依赖的任务不能提前执行；
    - 原则 6：无依赖任务应尽量并行。

本模块只负责 DAG 结构本身的维护（增删节点、环检测、就绪判定、
关键路径估算），具体的调度决策（并发数、优先级排序）留给
``repro_agent.scheduler``，遵循单一职责原则，便于独立测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from repro_agent.domain.enums import TaskStatus
from repro_agent.domain.task import Task


class CycleDetectedError(RuntimeError):
    """任务 DAG 中检测到环，违反设计文档 §3 原则 4。"""


@dataclass
class TaskDAG:
    """内存态的任务依赖图，任务本身的权威状态仍以数据库为准（§3 原则 15-16）。

    这里维护的是"结构"（谁依赖谁），任务的 ``status`` 字段仍然来自
    ``Task`` 对象本身，调用方应保证传入的 ``Task`` 是从存储层读出的
    最新快照，而不是长期持有的陈旧引用。
    """

    job_id: str
    tasks: dict[str, Task] = field(default_factory=dict)
    # children[task_id] = 依赖 task_id 的下游任务集合，用于 O(1) 查找
    # "完成这个任务后，还能解锁哪些任务"（调度优先级会用到，见 §8.3）。
    _children: dict[str, set[str]] = field(default_factory=dict)

    def add_task(self, task: Task) -> None:
        if task.job_id != self.job_id:
            raise ValueError(
                f"task {task.task_id} belongs to job {task.job_id}, "
                f"not {self.job_id}"
            )
        self.tasks[task.task_id] = task
        self._children.setdefault(task.task_id, set())
        for dep in task.dependencies:
            self._children.setdefault(dep, set()).add(task.task_id)
        self._check_no_cycle()

    def remove_task(self, task_id: str) -> None:
        self.tasks.pop(task_id, None)
        self._children.pop(task_id, None)
        for children in self._children.values():
            children.discard(task_id)

    def replace_task(self, task: Task) -> None:
        """用新快照替换已存在的任务节点（结构不变，只更新状态等字段）。"""

        self.tasks[task.task_id] = task

    def add_dependency(self, task_id: str, dependency_id: str) -> None:
        """Add one runtime prerequisite while keeping the child index valid."""

        task = self.tasks.get(task_id)
        if task is None:
            raise KeyError(f"unknown task: {task_id}")
        if dependency_id not in self.tasks:
            raise KeyError(f"unknown dependency task: {dependency_id}")
        if dependency_id in task.dependencies:
            return
        task.dependencies.append(dependency_id)
        self._children.setdefault(dependency_id, set()).add(task_id)
        try:
            self._check_no_cycle()
        except CycleDetectedError:
            task.dependencies.remove(dependency_id)
            self._children.get(dependency_id, set()).discard(task_id)
            raise

    def replace_dependency(
        self, task_id: str, dependency_id: str, replacements: list[str]
    ) -> None:
        """Atomically replace one dependency edge with validated subtask edges."""

        task = self.tasks.get(task_id)
        if task is None:
            raise KeyError(f"unknown task: {task_id}")
        if dependency_id not in task.dependencies:
            return
        missing = [item for item in replacements if item not in self.tasks]
        if missing:
            raise KeyError(f"unknown replacement dependencies: {missing}")
        original = list(task.dependencies)
        rewritten: list[str] = []
        for item in original:
            if item == dependency_id:
                rewritten.extend(replacements)
            else:
                rewritten.append(item)
        task.dependencies[:] = list(dict.fromkeys(rewritten))
        self._children.get(dependency_id, set()).discard(task_id)
        for replacement in replacements:
            self._children.setdefault(replacement, set()).add(task_id)
        try:
            self._check_no_cycle()
        except CycleDetectedError:
            task.dependencies[:] = original
            for replacement in replacements:
                self._children.get(replacement, set()).discard(task_id)
            self._children.setdefault(dependency_id, set()).add(task_id)
            raise

    def children_of(self, task_id: str) -> set[str]:
        return set(self._children.get(task_id, set()))

    def all_tasks(self) -> list[Task]:
        return list(self.tasks.values())

    def get(self, task_id: str) -> Task | None:
        return self.tasks.get(task_id)

    # ---- 依赖满足性判定 ----

    def dependencies_satisfied(self, task: Task) -> bool:
        """依赖是否全部成功（原则 5：有前置依赖的任务不能提前执行）。"""

        for dep_id in task.dependencies:
            dep = self.tasks.get(dep_id)
            if dep is None or dep.status != TaskStatus.SUCCEEDED:
                return False
        return True

    def dependency_failed(self, task: Task) -> bool:
        """依赖中是否存在终止性失败（用于把下游任务标记为阻塞/终止失败）。"""

        for dep_id in task.dependencies:
            dep = self.tasks.get(dep_id)
            if dep is not None and dep.status in {
                TaskStatus.TERMINAL_FAILURE,
                TaskStatus.CANCELLED,
            }:
                return True
        return False

    def ready_tasks(self) -> list[Task]:
        """返回所有可以进入调度队列的任务（原则 6）。

        正确的调用序列是先 ``unblock_or_block()`` 把满足依赖条件的
        PENDING/BLOCKED 任务转为 READY 状态，再调用本方法取出所有
        READY 任务交给调度器排序/派发。因此这里既要覆盖"已经是
        READY 状态"的任务，也要覆盖"尚未被刷新、但依赖已满足"的
        PENDING/BLOCKED 任务（后者是为了在调用方忘记先调用
        ``unblock_or_block()`` 时依然能得到正确结果，不强依赖调用顺序）。
        """

        ready = []
        for task in self.tasks.values():
            if task.status == TaskStatus.READY:
                ready.append(task)
                continue
            if task.status not in {TaskStatus.PENDING, TaskStatus.BLOCKED}:
                continue
            if self.dependencies_satisfied(task):
                ready.append(task)
        return ready

    def unblock_or_block(self) -> list[Task]:
        """刷新 PENDING/BLOCKED 任务的状态，返回状态发生变化的任务列表。"""

        changed = []
        for task in self.tasks.values():
            if task.status not in {TaskStatus.PENDING, TaskStatus.BLOCKED}:
                continue
            if self.dependency_failed(task):
                if task.status != TaskStatus.BLOCKED:
                    task.status = TaskStatus.BLOCKED
                    changed.append(task)
                continue
            if self.dependencies_satisfied(task):
                if task.status != TaskStatus.READY:
                    task.status = TaskStatus.READY
                    changed.append(task)
        return changed

    # ---- 关键路径 / 解锁能力评估（供调度优先级使用，§8.3）----

    def unlock_count(self, task_id: str) -> int:
        """该任务完成后，可能被解锁的下游任务数量（含传递闭包）。"""

        seen: set[str] = set()
        stack = list(self._children.get(task_id, set()))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self._children.get(current, set()))
        return len(seen)

    def topological_order(self) -> list[str]:
        """返回拓扑排序（用于展示/校验），检测到环会抛出异常。"""

        in_degree = {tid: 0 for tid in self.tasks}
        for task in self.tasks.values():
            for dep in task.dependencies:
                if dep in in_degree:
                    in_degree[task.task_id] += 1
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        order: list[str] = []
        while queue:
            current = queue.pop(0)
            order.append(current)
            for child in self._children.get(current, set()):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        if len(order) != len(self.tasks):
            raise CycleDetectedError(
                f"job {self.job_id} 的任务 DAG 中检测到环，无法完成拓扑排序"
            )
        return order

    def _check_no_cycle(self) -> None:
        self.topological_order()

    def summary(self) -> dict[str, int]:
        """按状态统计任务数量，供主智能体上下文摘要使用（§16）。"""

        counts: dict[str, int] = {}
        for task in self.tasks.values():
            counts[task.status.value] = counts.get(task.status.value, 0) + 1
        return counts
