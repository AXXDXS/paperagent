"""子智能体"存活性"管理：push 汇报 + pull 探测 + 死亡判定 + 强制终止。

用户需求原文拆解为四个明确的机制点，本模块负责后三个（push 汇报的
产生方在 ``agents/base.py::BaseSubAgent.report_progress``，属于子
智能体一侧；本模块站在主智能体一侧消费/裁决这些汇报）：

1. push：子智能体运行线程主动调用 ``report_progress()``，
   通过 ``AgentDispatcher`` 注入的回调写入
   ``TaskScheduler.report_heartbeat()``（心跳落库），必须保证
   "子智能体先主动汇报"——本模块从不伪造/补写一条"push"来源的心跳，
   只有子智能体自己调用回调才会产生 ``reported_by="push"`` 的记录。
2. pull：主智能体调用 ``get_subagent_status(task_id)`` 主动查询子
   智能体当前状态；本模块提供 ``LivenessPolicy.should_force_pull()``
   判断"距离上一次 push 心跳是否已经超过宽限期（默认 120 秒，对应
   用户需求原文"超时 2 分钟没有汇报"）"，超过则主循环应调用
   ``get_subagent_status`` 做一次强制查询，查询本身不算作 push 心跳，
   只会在 ``Heartbeat.reported_by`` 记为 ``"pull"``，不会重置卡死计时。
3. 死亡判定：``LivenessPolicy.judge()`` 综合"pull 探测是否成功
   （线程是否还活着）"和"心跳/日志是否有进展"两个信号，只有两者都
   说明任务已经不可能再推进时才判定为"确认死亡"，避免把"运行缓慢但
   仍然存活"的任务误杀。
4. 强制终止：``ForcedTerminationPolicy`` 封装"先尝试优雅信号
   （``LeaseManager.request_cancel`` + ``threading.Event.set()``），
   给予 grace period 等待子智能体自己在检查点（``BaseSubAgent.
   check_cancellation()``）响应退出；如果 grace period 内仍未退出，
   才执行 forced kill（放弃 join、标记线程为悬挂、按失败处理并留痕，
   代价是可能遗留悬挂资源/未完成写入——这与用户描述的权衡完全一致）"
   这一整套流程，供 ``MainAgent`` 在主循环中调用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from repro_agent.domain.common import utc_now
from repro_agent.domain.task import Heartbeat, Task

logger = logging.getLogger(__name__)


class LivenessOutcome(str, Enum):
    """一次存活性裁决的结果。"""

    ALIVE_REPORTING = "alive_reporting"  # push 心跳新鲜，一切正常
    OVERDUE_NEEDS_PULL = "overdue_needs_pull"  # 超过宽限期未 push，需要主动 pull 探测
    ALIVE_AFTER_PULL = "alive_after_pull"  # pull 探测证实仍在运行，只是汇报不及时
    CONFIRMED_DEAD = "confirmed_dead"  # pull 探测也无法证明存活，判定死亡


@dataclass
class LivenessDecision:
    outcome: LivenessOutcome
    detail: str = ""
    seconds_since_last_push: Optional[float] = None


class LivenessPolicy:
    """依据"上一次 push 心跳的新鲜度"判断是否需要强制 pull / 判定死亡。

    与 ``scheduler.timeout_policy.TimeoutPolicy`` 的关系：
        ``TimeoutPolicy`` 判断的是"软/硬超时"（任务总耗时是否超过
        预算），是"这个任务花的时间是否合理"；本策略判断的是
        "子智能体最近一次主动汇报是否还新鲜"，是"这个子智能体是否
        还活着、还在按要求跟主智能体沟通"，两者是正交的两个维度：
        一个任务完全可能"没有超过硬超时，但已经连续 2 分钟没有汇报"，
        也完全可能"心跳很新鲜，但已经超过硬超时"（心跳新鲜只说明
        没死，不代表任务可以无限跑下去）。主循环里两者都要检查，
        互不替代。
    """

    def __init__(self, *, grace_seconds_default: int = 120):
        self.grace_seconds_default = grace_seconds_default

    def evaluate_push_freshness(self, task: Task, *, now: Optional[datetime] = None) -> LivenessDecision:
        """检查是否已经超过"未汇报宽限期"，尚未触发 pull 探测前调用。"""

        now = now or utc_now()
        # 注意：不能用 `x or default`——liveness_grace_seconds=0 是一个
        # 合法的"零宽限期"配置（测试/调试场景常用），`0 or default` 会
        # 因为 0 是 falsy 而被错误地替换成默认值 120，导致"配置为 0"
        # 和"未配置（使用默认值）"这两种语义被意外混淆。只有真正的
        # "未设置"（None）才应该回退到默认值。
        grace = (
            task.definition.liveness_grace_seconds
            if task.definition.liveness_grace_seconds is not None
            else self.grace_seconds_default
        )

        if task.started_at is None:
            return LivenessDecision(LivenessOutcome.ALIVE_REPORTING, "task not started yet")

        last_push_at = self._last_push_timestamp(task)
        baseline = last_push_at or task.started_at
        elapsed = (now - baseline).total_seconds()

        if elapsed < grace:
            return LivenessDecision(
                LivenessOutcome.ALIVE_REPORTING,
                f"last push {elapsed:.0f}s ago, within grace {grace}s",
                seconds_since_last_push=elapsed,
            )

        return LivenessDecision(
            LivenessOutcome.OVERDUE_NEEDS_PULL,
            f"no push heartbeat for {elapsed:.0f}s (grace={grace}s), forcing status pull",
            seconds_since_last_push=elapsed,
        )

    def judge_after_pull(
        self,
        task: Task,
        *,
        pull_succeeded: bool,
        pull_detail: str = "",
    ) -> LivenessDecision:
        """已经执行过一次强制 ``get_subagent_status`` 查询之后，据其结果裁决。

        ``pull_succeeded`` 语义上等价于"查询到子智能体运行线程仍然
        存活（线程未退出/未抛出未捕获异常）"。只有查询本身也失败
        （线程已经不存在、或返回的状态表明已经僵死）才判定为确认死亡；
        查询成功但汇报的进度依然是旧值，只说明"活着但迟迟没有新进展"，
        这属于 ``TimeoutPolicy`` 的 STALLED 范畴，交由硬超时兜底，
        本策略不会仅凭"进度没变化"就判死刑——避免和已有的软/硬超时
        机制产生重复或冲突的判定路径。
        """

        if pull_succeeded:
            return LivenessDecision(
                LivenessOutcome.ALIVE_AFTER_PULL,
                pull_detail or "status pull confirmed the sub-agent thread is still alive",
            )
        return LivenessDecision(
            LivenessOutcome.CONFIRMED_DEAD,
            pull_detail or "status pull could not confirm sub-agent is alive",
        )

    @staticmethod
    def _last_push_timestamp(task: Task) -> Optional[datetime]:
        hb = task.last_push_heartbeat or task.heartbeat
        if hb is None or hb.reported_by != "push":
            return None
        return hb.updated_at


class TerminationMode(str, Enum):
    GRACEFUL = "graceful"
    FORCED = "forced"


@dataclass
class TerminationRecord:
    task_id: str
    mode: TerminationMode
    reason: str
    requested_at: datetime = field(default_factory=utc_now)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "mode": self.mode.value,
            "reason": self.reason,
            "requested_at": self.requested_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
