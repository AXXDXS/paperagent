"""回归测试：子智能体异步派发 + push 心跳 + 校验通过后才回收句柄。

覆盖需求点：
    - AgentDispatcher.start_async 立即返回、不阻塞调用方；
    - 子智能体运行期间至少产生一次 push 心跳（由 dispatcher 自动首尾
      汇报兜底保证）；
    - MainAgent._collect_finished_subagents 能正确探测线程已结束；
    - 只有 validate_outputs 判定通过后才会 discard_handle（在此之前
      dispatcher.get_handle 必须仍然能取到句柄）。
"""

from __future__ import annotations

import time

from repro_agent.agents.base import AgentRunResult, BaseSubAgent
from repro_agent.agents.registry import SUB_AGENT_REGISTRY
from repro_agent.domain.enums import TaskStatus
from repro_agent.domain.task import Task
from repro_agent.orchestrator.task_factory import build_task_definition
from repro_agent.tools.registry import default_registry


class _InstantSuccessAgent(BaseSubAgent):
    """一个可控的测试用子智能体：立即返回成功，并写出预期产物。"""

    task_type = "instant_success"

    def run(self) -> AgentRunResult:
        self.report_progress(0.5, "halfway")
        self.write_json_output("result.json", {"ok": True})
        return AgentRunResult(succeeded=True, outputs={"ok": True})


class _SlowThenSuccessAgent(BaseSubAgent):
    """模拟一个需要一小段时间才能完成的子智能体，用于测试轮询逻辑。"""

    task_type = "slow_success"

    def run(self) -> AgentRunResult:
        self.report_progress(0.1, "starting", eta_seconds=1.0)
        time.sleep(0.3)
        self.report_progress(0.9, "almost_done", eta_seconds=0.1)
        self.write_json_output("result.json", {"ok": True})
        return AgentRunResult(succeeded=True, outputs={"ok": True})


class _AlwaysFailAgent(BaseSubAgent):
    task_type = "instant_fail"

    def run(self) -> AgentRunResult:
        from repro_agent.domain.enums import FailureType
        from repro_agent.domain.task import FailureReport

        return AgentRunResult(
            succeeded=False,
            failure_report=FailureReport(
                failure_type=FailureType.UNKNOWN_ERROR,
                failed_step="deliberate_failure",
                error_message="test-induced failure",
            ),
        )


def _register_test_agent(monkeypatch, task_type: str, agent_cls) -> None:
    """临时把测试用子智能体注册进全局注册表，测试结束后自动恢复。"""

    monkeypatch.setitem(SUB_AGENT_REGISTRY, task_type, agent_cls)


def _make_task(main_agent, task_type: str, expected_outputs=None) -> Task:
    definition = build_task_definition(
        objective=f"test task for {task_type}",
        task_type=task_type,
        extra_allowed_tools=["write_task_output"],
        expected_outputs=expected_outputs or ["output/result.json"],
    )
    task = Task(job_id=main_agent.job.job_id, definition=definition)
    main_agent.scheduler.add_tasks([task])
    return task


def test_start_async_returns_immediately_and_produces_push_heartbeat(main_agent, monkeypatch):
    _register_test_agent(monkeypatch, "instant_success", _InstantSuccessAgent)
    task = _make_task(main_agent, "instant_success")

    main_agent.scheduler.dispatch([task])
    started_at = time.monotonic()
    main_agent._run_dispatched_task(task)
    elapsed = time.monotonic() - started_at

    # start_async 不应该阻塞：即便子智能体内部有 sleep，也应该几乎立刻返回。
    assert elapsed < 0.2

    handle = main_agent.dispatcher.get_handle(task.task_id)
    assert handle is not None

    # 等待后台线程完成
    deadline = time.monotonic() + 5
    while not handle.is_finished() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert handle.is_finished()

    # push 心跳必须由子智能体自己触发过（至少一次 reported_by="push"）。
    assert task.heartbeat is not None
    assert task.heartbeat.reported_by == "push"


def test_collect_finished_subagents_then_validate_then_discard(main_agent, monkeypatch):
    _register_test_agent(monkeypatch, "instant_success", _InstantSuccessAgent)
    task = _make_task(main_agent, "instant_success")

    main_agent.scheduler.dispatch([task])
    main_agent._run_dispatched_task(task)

    handle = main_agent.dispatcher.get_handle(task.task_id)
    deadline = time.monotonic() + 5
    while not handle.is_finished() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert handle.is_finished()

    # 线程结束但尚未校验：句柄必须仍然存在（呼应"验证通过后才能关闭"）。
    main_agent._collect_finished_subagents()
    assert task.task_id in main_agent._pending_validation
    assert main_agent.dispatcher.get_handle(task.task_id) is not None
    assert task.status == TaskStatus.RUNNING

    # 校验通过后，任务状态变为 SUCCEEDED，句柄被回收。
    completed = main_agent._new_completed_tasks()
    assert [t.task_id for t in completed] == [task.task_id]
    main_agent.validate_outputs(completed)

    assert task.status == TaskStatus.SUCCEEDED
    assert task.task_id not in main_agent._pending_validation
    assert main_agent.dispatcher.get_handle(task.task_id) is None


def test_failed_subagent_is_recycled_without_waiting_for_validation(main_agent, monkeypatch):
    _register_test_agent(monkeypatch, "instant_fail", _AlwaysFailAgent)
    task = _make_task(main_agent, "instant_fail")

    main_agent.scheduler.dispatch([task])
    main_agent._run_dispatched_task(task)

    handle = main_agent.dispatcher.get_handle(task.task_id)
    deadline = time.monotonic() + 5
    while not handle.is_finished() and time.monotonic() < deadline:
        time.sleep(0.02)

    main_agent._collect_finished_subagents()

    # 失败任务不进入待校验队列，直接被标记失败并回收句柄。
    assert task.task_id not in main_agent._pending_validation
    assert task.status == TaskStatus.FAILED_RETRYABLE
    assert main_agent.dispatcher.get_handle(task.task_id) is None


def test_multiple_subagents_run_concurrently_in_background_threads(main_agent, monkeypatch):
    _register_test_agent(monkeypatch, "slow_success", _SlowThenSuccessAgent)
    task_a = _make_task(main_agent, "slow_success")
    task_b = _make_task(main_agent, "slow_success")

    main_agent.scheduler.dispatch([task_a, task_b])
    started_at = time.monotonic()
    main_agent._run_dispatched_task(task_a)
    main_agent._run_dispatched_task(task_b)
    dispatch_elapsed = time.monotonic() - started_at

    # 两个任务各自 sleep 0.3s，但派发本身应该几乎不耗时（证明是并行的
    # 后台线程而不是顺序阻塞执行）。
    assert dispatch_elapsed < 0.2

    handle_a = main_agent.dispatcher.get_handle(task_a.task_id)
    handle_b = main_agent.dispatcher.get_handle(task_b.task_id)
    deadline = time.monotonic() + 5
    while (not handle_a.is_finished() or not handle_b.is_finished()) and time.monotonic() < deadline:
        time.sleep(0.02)

    assert handle_a.is_finished()
    assert handle_b.is_finished()


def test_create_subagents_is_a_main_only_batch_tool(main_agent, monkeypatch):
    _register_test_agent(monkeypatch, "slow_success", _SlowThenSuccessAgent)
    task_a = _make_task(main_agent, "slow_success")
    task_b = _make_task(main_agent, "slow_success")
    main_agent.scheduler.dispatch([task_a, task_b])

    schema = main_agent.create_subagents_tool.to_openai_tool()
    assert schema["function"]["name"] == "create_subagents"
    assert schema["function"]["parameters"]["required"] == ["task_ids"]
    # The global registry is child-facing; the creation tool must never appear there.
    assert default_registry().get("create_subagents") is None

    result = main_agent.create_subagents_tool.call(
        task_ids=[task_a.task_id, task_b.task_id]
    )

    assert result.started_task_ids == [task_a.task_id, task_b.task_id]
    assert result.failed_task_ids == []
    assert task_a.status == TaskStatus.RUNNING
    assert task_b.status == TaskStatus.RUNNING
    assert main_agent.dispatcher.get_handle(task_a.task_id) is not None
    assert main_agent.dispatcher.get_handle(task_b.task_id) is not None
    events = main_agent.task_repo.list_events(main_agent.job.job_id)
    event = next(
        item for item in events if item["event_type"] == "create_subagents_tool_called"
    )
    assert event["payload"]["started_task_ids"] == [task_a.task_id, task_b.task_id]


def test_create_subagents_tool_rejects_non_dispatched_task(main_agent) -> None:
    task = _make_task(main_agent, "instant_success")

    result = main_agent.create_subagents_tool.call(task_ids=[task.task_id])

    assert result.started_task_ids == []
    assert result.failed_task_ids == [task.task_id]
    assert "must be DISPATCHED" in result.records[0].error
    assert main_agent.dispatcher.get_handle(task.task_id) is None
