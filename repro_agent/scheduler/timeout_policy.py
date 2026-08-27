"""软/硬超时判定策略（设计文档 §13.3、§13.4）。

复用来源说明：
    - "软超时不立即终止，而是先检查活跃信号"的思路直接对应设计文档
      §13.3 的四项检查（日志是否增长、心跳是否正常、是否有新工具调用、
      CPU/GPU 是否有活动），本实现用「心跳时间戳 + 心跳进度 + 日志游标
      是否推进」三个廉价信号做近似判断（真实系统里 CPU/GPU 探测由
      沙箱层负责上报，这里预留了 ``resource_active`` 回调钩子）。
    - 具体的"处理"分支（正常运行延长/运行缓慢更新预计时间/卡死终止/
      已出错生成报告）参考了 DeepCode 规划阶段的重试退避思想
      （``workflows/agent_orchestration_engine.py`` 中
      ``_adjust_params_for_retry`` 每次重试反而降低 token 预算而不是
      提高——即"给出问题诊断后调整而不是无脑重试"），在这里体现为
      ``TimeoutDecision`` 携带的诊断信息会被主智能体用于分类失败原因
      （§19 主循环 ``classify_failure``）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

from repro_agent.domain.common import utc_now
from repro_agent.domain.task import Task


class TimeoutOutcome(str, Enum):
    HEALTHY = "healthy"  # 未超时，或超时但活跃信号正常，继续观察
    SLOW_BUT_ALIVE = "slow_but_alive"  # 软超时但仍在推进，延长预计时间
    STALLED = "stalled"  # 软超时且无活跃信号，疑似卡死
    HARD_TIMEOUT = "hard_timeout"  # 达到硬超时，必须终止


@dataclass
class TimeoutDecision:
    outcome: TimeoutOutcome
    elapsed_seconds: float
    detail: str = ""


ResourceActivityProbe = Callable[[Task], bool]
"""外部资源活跃度探测钩子（例如沙箱层上报 CPU/GPU 是否有新增使用量）。

在没有真实沙箱资源监控时，默认认为"无法探测视为不活跃"，
偏保守（宁可多观察一轮，也不要因为漏判活跃信号而误杀任务）。
"""


def _default_resource_probe(_: Task) -> bool:
    return False


class TimeoutPolicy:
    """封装软/硬超时的判定逻辑，供调度器在每个轮询周期调用。"""

    def __init__(
        self,
        *,
        heartbeat_stale_multiplier: float = 3.0,
        resource_probe: Optional[ResourceActivityProbe] = None,
    ):
        # 超过 heartbeat_interval 的多少倍没有心跳，视为心跳中断。
        self.heartbeat_stale_multiplier = heartbeat_stale_multiplier
        self.resource_probe = resource_probe or _default_resource_probe

    def evaluate(self, task: Task, *, now: Optional[datetime] = None) -> TimeoutDecision:
        now = now or utc_now()
        if task.started_at is None:
            return TimeoutDecision(TimeoutOutcome.HEALTHY, 0.0, "not started")

        elapsed = (now - task.started_at).total_seconds()
        definition = task.definition

        if elapsed >= definition.hard_timeout_seconds:
            return TimeoutDecision(
                TimeoutOutcome.HARD_TIMEOUT,
                elapsed,
                f"elapsed {elapsed:.0f}s >= hard_timeout {definition.hard_timeout_seconds}s",
            )

        if elapsed < definition.soft_timeout_seconds:
            return TimeoutDecision(TimeoutOutcome.HEALTHY, elapsed, "within soft timeout")

        # 已超过软超时，检查活跃信号（§13.3）
        heartbeat_ok = self._heartbeat_is_fresh(task, now, definition.heartbeat_interval_seconds)
        log_growing = self._log_is_growing(task)
        resource_active = self.resource_probe(task)

        if heartbeat_ok or log_growing or resource_active:
            return TimeoutDecision(
                TimeoutOutcome.SLOW_BUT_ALIVE,
                elapsed,
                "soft timeout exceeded but activity signals present "
                f"(heartbeat_ok={heartbeat_ok}, log_growing={log_growing}, "
                f"resource_active={resource_active})",
            )

        return TimeoutDecision(
            TimeoutOutcome.STALLED,
            elapsed,
            "soft timeout exceeded with no activity signal, suspected stall",
        )

    def _heartbeat_is_fresh(
        self, task: Task, now: datetime, heartbeat_interval: int
    ) -> bool:
        if task.heartbeat is None:
            return False
        stale_after = heartbeat_interval * self.heartbeat_stale_multiplier
        age = (now - task.heartbeat.updated_at).total_seconds()
        return age <= stale_after

    def _log_is_growing(self, task: Task) -> bool:
        if task.heartbeat is None:
            return False
        current_position = str(task.heartbeat.last_log_position)
        grown = current_position != task.last_activity_signature
        if grown:
            task.last_activity_signature = current_position
        return grown
