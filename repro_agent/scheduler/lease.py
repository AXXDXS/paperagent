"""任务执行租约（Lease）机制。

复用来源说明：
    该机制直接借鉴 DeerFlow 的 RunStore「Lease + 心跳」多 Worker
    运行归属模型（见 ``doc/DeerFlow_架构分析.md`` 第 6.2 节
    ``RunStore：Lease/心跳模式实现多 Worker 运行归属``）：

    - 一个任务被某个执行单元（子智能体 Worker）领取后，Worker 必须
      周期性地"续租"（renew lease）证明自己仍然存活并持有该任务的
      执行权；
    - 续租是一个原子操作：续期所有权的同时检查是否存在挂起的取消
      请求，取消动作因此总能被正在续租的持有者感知到；
    - 租约过期后，调度器的对账（reconcile）逻辑可以安全地把任务
      标记为疑似卡死 / 允许被别的 Worker 重新认领，而不会出现两个
      Worker 都认为自己持有任务所有权的竞态。

为什么要在单机 MVP 里也引入这个机制（而不是等到分布式部署才加）：
    设计文档 §13.3/§13.4 明确要求"软超时"要能区分"运行缓慢"和
    "卡死"，而心跳/租约模型天然把这两者的判定标准分离开——没有
    续租 ≠ 没有心跳 ≠ 真的卡死，三者独立判断比"单一超时计时器"
    更不容易误杀正常运行中的长任务，也为未来扩展成多进程/多机
    子智能体池预留了接口，不需要日后重写调度器核心。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from repro_agent.domain.common import utc_now
from repro_agent.domain.task import Task


@dataclass
class LeaseRenewalResult:
    granted: bool
    reason: str = ""
    cancel_requested: bool = False


class LeaseManager:
    """维护任务级执行租约，纯内存 + 数据库落盘双重记录。

    数据库落盘（``task.lease_owner``/``task.lease_expires_at``）保证
    进程重启后仍能恢复租约状态，内存中的 ``_cancel_requests`` 则用于
    "取消请求"这种轻量、高频的信号传递（对应 DeerFlow 文档中
    "非 Owner Worker 收到取消请求后，把中断请求持久化下来，真正的
    Owner 在下一次续租时观察到该请求"的设计）。
    """

    def __init__(self, lease_duration_seconds: float = 60.0):
        self.lease_duration_seconds = lease_duration_seconds
        self._cancel_requests: set[str] = set()

    def acquire(self, task: Task, owner: str, *, now: Optional[datetime] = None) -> bool:
        """尝试为任务分配执行租约，成功返回 True。"""

        now = now or utc_now()
        if task.lease_owner and task.lease_expires_at and task.lease_expires_at > now:
            # 已经被其它 Worker 持有且未过期
            return task.lease_owner == owner
        task.lease_owner = owner
        task.lease_expires_at = now + timedelta(seconds=self.lease_duration_seconds)
        return True

    def renew(self, task: Task, owner: str, *, now: Optional[datetime] = None) -> LeaseRenewalResult:
        """续租：同时检查取消请求，是 DeerFlow 设计中的关键原子操作。"""

        now = now or utc_now()
        if task.lease_owner != owner:
            return LeaseRenewalResult(granted=False, reason="not_lease_owner")
        task.lease_expires_at = now + timedelta(seconds=self.lease_duration_seconds)
        cancel_requested = task.task_id in self._cancel_requests
        if cancel_requested:
            self._cancel_requests.discard(task.task_id)
        return LeaseRenewalResult(granted=True, cancel_requested=cancel_requested)

    def release(self, task: Task) -> None:
        task.lease_owner = None
        task.lease_expires_at = None

    def is_expired(self, task: Task, *, now: Optional[datetime] = None) -> bool:
        now = now or utc_now()
        if task.lease_expires_at is None:
            return False
        return task.lease_expires_at <= now

    def request_cancel(self, task_id: str) -> None:
        self._cancel_requests.add(task_id)
