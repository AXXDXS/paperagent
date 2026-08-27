"""回归测试：子智能体存活性检测（push 宽限期 -> 强制 pull -> 死亡判定 -> 终止）。

覆盖需求点：
    - 心跳新鲜时不触发 pull；
    - 超过 ``liveness_grace_seconds`` 未收到 push 心跳时，主循环应
      通过 ``get_subagent_status`` 强制 pull 一次；
    - pull 探测到线程仍存活 -> ALIVE_AFTER_PULL，不终止；
    - pull 探测到线程已不存在 -> CONFIRMED_DEAD -> 触发终止流程；
    - 优雅取消信号在宽限期内被子智能体感知并退出 -> GRACEFUL 终止；
    - 优雅信号无响应 -> 兜底 FORCED 终止，且任务被标记为 HARD_TIMEOUT。
"""

from __future__ import annotations

import threading
import time
from datetime import timedelta

from repro_agent.agents.base import AgentRunResult, BaseSubAgent, CancellationRequested
from repro_agent.agents.registry import SUB_AGENT_REGISTRY
from repro_agent.domain.enums import TaskStatus
from repro_agent.domain.task import Heartbeat, Task
from repro_agent.orchestrator.task_factory import build_task_definition
from repro_agent.domain.common import utc_now
from repro_agent.scheduler.subagent_liveness import LivenessOutcome, TerminationMode


class _CooperativeLongRunningAgent(BaseSubAgent):
    """一个会定期检查取消信号、能够响应优雅终止的长任务子智能体。"""

    task_type = "cooperative_long_running"

    def run(self) -> AgentRunResult:
        self.report_progress(0.05, "started", eta_seconds=100.0)
        for _ in range(200):
            self.check_cancellation()
            time.sleep(0.02)
        self.write_json_output("result.json", {"ok": True})
        return AgentRunResult(succeeded=True, outputs={"ok": True})


class _UnresponsiveAgent(BaseSubAgent):
    """一个完全不检查取消信号、模拟"卡死不响应优雅信号"的子智能体。"""

    task_type = "unresponsive"

    def run(self) -> AgentRunResult:
        self.report_progress(0.05, "started", eta_seconds=100.0)
        # 故意长时间阻塞且不调用 check_cancellation()，模拟对优雅信号
        # 完全无响应的场景，迫使 MainAgent 走到 force_kill 分支。
        time.sleep(5.0)
        return AgentRunResult(succeeded=True, outputs={"ok": True})


def _make_task(main_agent, task_type: str, *, liveness_grace_seconds: int = 1) -> Task:
    definition = build_task_definition(
        objective=f"test task for {task_type}",
        task_type=task_type,
        extra_allowed_tools=["write_task_output"],
        expected_outputs=["output/result.json"],
        heartbeat_interval_seconds=1,
    )
    definition.liveness_grace_seconds = liveness_grace_seconds
    task = Task(job_id=main_agent.job.job_id, definition=definition)
    main_agent.scheduler.add_tasks([task])
    return task


def test_fresh_push_heartbeat_does_not_trigger_pull(main_agent, monkeypatch):
    monkeypatch.setitem(SUB_AGENT_REGISTRY, "cooperative_long_running", _CooperativeLongRunningAgent)
    task = _make_task(main_agent, "cooperative_long_running", liveness_grace_seconds=100)

    main_agent.scheduler.dispatch([task])
    main_agent._run_dispatched_task(task)

    # 心跳新鲜（grace=100s 远大于刚启动的等待时间），不应触发 pull。
    time.sleep(0.05)
    decision = main_agent.liveness_policy.evaluate_push_freshness(task)
    assert decision.outcome == LivenessOutcome.ALIVE_REPORTING

    # 主动请求终止，避免子线程残留影响后续测试（非本用例断言点）。
    handle = main_agent.dispatcher.get_handle(task.task_id)
    handle.request_graceful_cancel()
    deadline = time.monotonic() + 3
    while handle.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)


def test_overdue_push_triggers_pull_and_confirms_alive(main_agent, monkeypatch):
    monkeypatch.setitem(SUB_AGENT_REGISTRY, "cooperative_long_running", _CooperativeLongRunningAgent)
    task = _make_task(main_agent, "cooperative_long_running", liveness_grace_seconds=0)

    main_agent.scheduler.dispatch([task])
    main_agent._run_dispatched_task(task)

    # grace=0：几乎立刻就会被判定为"超过宽限期未汇报"，需要强制 pull。
    # 此时线程仍然存活（还在 200 次循环里），pull 应确认存活而不是死亡。
    time.sleep(0.05)
    main_agent._check_subagent_liveness()

    # 存活确认后任务不应该被标记为失败/终止。
    assert task.status == TaskStatus.RUNNING

    handle = main_agent.dispatcher.get_handle(task.task_id)
    assert handle is not None
    handle.request_graceful_cancel()
    deadline = time.monotonic() + 3
    while handle.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)


def test_pull_probe_does_not_replace_last_push_heartbeat(main_agent, monkeypatch):
    task = _make_task(main_agent, "cooperative_long_running", liveness_grace_seconds=100)
    push = Heartbeat(progress=0.4, current_step="training", reported_by="push")
    pull = Heartbeat(progress=0.4, current_step="training (alive)", reported_by="pull")
    task.heartbeat = push
    task.last_push_heartbeat = push

    class _Handle:
        @staticmethod
        def pull_status():
            return pull

    monkeypatch.setattr(main_agent.dispatcher, "get_handle", lambda task_id: _Handle())

    pulled = main_agent.get_subagent_status(task.task_id)

    assert pulled is not None and pulled.reported_by == "pull"
    assert task.heartbeat is push
    assert task.last_push_heartbeat is push
    assert task.last_pull_heartbeat is pulled


def test_confirmed_dead_after_pull_fails_marks_task_failed(main_agent, monkeypatch):
    """线程已经真正退出（is_alive()==False）时，pull 探测应判定死亡并终止任务。"""

    monkeypatch.setitem(SUB_AGENT_REGISTRY, "cooperative_long_running", _CooperativeLongRunningAgent)
    task = _make_task(main_agent, "cooperative_long_running", liveness_grace_seconds=0)

    main_agent.scheduler.dispatch([task])
    main_agent._run_dispatched_task(task)

    handle = main_agent.dispatcher.get_handle(task.task_id)
    # 强制让线程立即结束，模拟"子智能体线程已经不存在"的死亡场景。
    handle.request_graceful_cancel()
    deadline = time.monotonic() + 3
    while handle.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not handle.is_alive()

    # 此时任务状态机仍然认为它是 RUNNING（线程结果还没被主循环收集），
    # 手动调用一次存活性检查，验证会被判定为 CONFIRMED_DEAD 并终止。
    task.status = TaskStatus.RUNNING
    main_agent.scheduler.dag.replace_task(task)
    main_agent._check_subagent_liveness()

    assert task.status == TaskStatus.HARD_TIMEOUT
    assert main_agent.dispatcher.get_handle(task.task_id) is None
    assert task.task_id not in main_agent._pending_validation


def test_graceful_cancellation_terminates_cooperative_agent(main_agent, monkeypatch):
    monkeypatch.setitem(SUB_AGENT_REGISTRY, "cooperative_long_running", _CooperativeLongRunningAgent)
    task = _make_task(main_agent, "cooperative_long_running")

    main_agent.scheduler.dispatch([task])
    main_agent._run_dispatched_task(task)
    handle = main_agent.dispatcher.get_handle(task.task_id)

    main_agent._terminate_subagent(task, handle, reason="test-induced graceful termination")

    assert task.status == TaskStatus.HARD_TIMEOUT
    assert main_agent._termination_log[-1].mode == TerminationMode.GRACEFUL
    assert main_agent.dispatcher.get_handle(task.task_id) is None


def test_forced_termination_when_agent_does_not_respond_to_graceful_signal(main_agent, monkeypatch):
    monkeypatch.setitem(SUB_AGENT_REGISTRY, "unresponsive", _UnresponsiveAgent)
    # heartbeat_interval_seconds 决定了 _terminate_subagent 的宽限等待时长，
    # 设小一点让测试快速完成（不影响断言的正确性）。
    definition = build_task_definition(
        objective="unresponsive test task",
        task_type="unresponsive",
        extra_allowed_tools=["write_task_output"],
        expected_outputs=["output/result.json"],
        heartbeat_interval_seconds=1,
    )
    task = Task(job_id=main_agent.job.job_id, definition=definition)
    main_agent.scheduler.add_tasks([task])

    main_agent.scheduler.dispatch([task])
    main_agent._run_dispatched_task(task)
    handle = main_agent.dispatcher.get_handle(task.task_id)

    time.sleep(0.05)  # 确保线程已经进入 5s 的阻塞 sleep
    main_agent._terminate_subagent(task, handle, reason="test-induced forced termination")

    assert task.status == TaskStatus.HARD_TIMEOUT
    assert main_agent._termination_log[-1].mode == TerminationMode.FORCED
    assert handle.forced_killed is True
    assert main_agent.dispatcher.get_handle(task.task_id) is None


def test_hard_timeout_retains_lease_until_execution_termination(main_agent):
    definition = build_task_definition(
        objective="timeout lease",
        task_type="paper_analysis",
        hard_timeout_seconds=1,
    )
    task = Task(job_id=main_agent.job.job_id, definition=definition)
    main_agent.scheduler.add_tasks([task])
    main_agent.scheduler.dispatch([task])
    main_agent.scheduler.mark_running(task)
    task.started_at = utc_now() - timedelta(seconds=10)

    result = main_agent.scheduler.check_timeouts()

    assert result.hard_timeout_tasks == [task]
    assert task.status == TaskStatus.HARD_TIMEOUT
    assert task.lease_owner is not None
