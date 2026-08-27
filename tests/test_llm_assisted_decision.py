"""回归测试：§16 上下文编排接入 §19 失败分类决策点的 LLM 兜底分支。

覆盖需求点：
    - 已被确定性规则表覆盖的 16 种已知失败类型完全不受影响，不触发
      任何 LLM 调用（``Replanner.classify_failure`` 原有行为保持不变）；
    - 只有 ``FailureType.UNKNOWN_ERROR`` 才会委托给 ``llm_fallback``；
    - ``MainAgentLLMDecisionMaker`` 能正确编排 §16 九段上下文、调用
      LLM、把合法 JSON 响应解析为 ``FailureDecision``；
    - LLM 返回非法 JSON / 非法决策值 / 调用报错时，均安全降级为
      ``TERMINAL_FAILURE``，不会让异常向上传播打断主循环；
    - ``MainAgent._classify_failure_via_llm`` 端到端：真正通过
      ``_handle_failed_task`` 触发、且会记录 ``failure_classified_by_llm``
      审计事件。
"""

from __future__ import annotations

import json

import pytest

from repro_agent.domain.enums import FailureDecision, FailureType
from repro_agent.domain.job import ReproductionJob
from repro_agent.domain.task import FailureReport, Task
from repro_agent.orchestrator.llm_decision import MainAgentLLMDecisionMaker
from repro_agent.orchestrator.main_agent import MainAgent
from repro_agent.orchestrator.replanner import Replanner
from repro_agent.orchestrator.task_factory import build_task_definition
from repro_agent.providers.base import LLMResponse
from repro_agent.providers.mock import MockLLMProvider
from repro_agent.providers.prompt_cache import (
    STABLE_PROMPT_PREFIX,
    STABLE_PROMPT_PREFIX_VERSION,
)


def _make_task(
    *,
    failure_type: FailureType | None,
    attempt: int = 0,
    max_attempts: int = 3,
    job_id: str = "job-1",
) -> Task:
    definition = build_task_definition(
        objective="测试任务",
        task_type="coding",
        max_attempts=max_attempts,
    )
    task = Task(job_id=job_id, definition=definition)
    task.attempt = attempt
    if failure_type is not None:
        task.failure_report = FailureReport(
            failure_type=failure_type,
            failed_step="tool_call",
            error_message="boom",
            likely_causes=["未知原因"],
            recommended_action="人工排查",
        )
    return task


# ---- Replanner 分支路由：规则覆盖 vs LLM 兜底 ----


def test_known_failure_types_never_trigger_llm_fallback():
    """16 种已知失败类型必须走确定性规则，绝不调用 llm_fallback。"""

    replanner = Replanner()
    calls = []

    def _should_not_be_called(task: Task) -> FailureDecision:
        calls.append(task.task_id)
        return FailureDecision.ASK_USER

    known_types = [t for t in FailureType if t != FailureType.UNKNOWN_ERROR]
    assert len(known_types) == len(list(FailureType)) - 1

    for failure_type in known_types:
        task = _make_task(failure_type=failure_type)
        decision = replanner.classify_failure(task, llm_fallback=_should_not_be_called)
        assert isinstance(decision, FailureDecision)

    assert calls == [], "已知失败类型不应触发 LLM 兜底回调"


def test_unknown_error_delegates_to_llm_fallback_when_provided():
    replanner = Replanner()
    task = _make_task(failure_type=FailureType.UNKNOWN_ERROR)

    called_with = []

    def _fallback(t: Task) -> FailureDecision:
        called_with.append(t.task_id)
        return FailureDecision.RETRY

    decision = replanner.classify_failure(task, llm_fallback=_fallback)
    assert decision == FailureDecision.RETRY
    assert called_with == [task.task_id]


def test_unknown_error_without_llm_fallback_still_safely_degrades_to_terminal_failure():
    """不传 llm_fallback 时行为与改造前一致：安全降级为终止，不抛异常。"""

    replanner = Replanner()
    task = _make_task(failure_type=FailureType.UNKNOWN_ERROR)
    decision = replanner.classify_failure(task)
    assert decision == FailureDecision.TERMINAL_FAILURE


def test_max_attempts_exhausted_short_circuits_before_llm_fallback():
    """达到最大重试次数的升级判断优先于"是否需要 LLM 兜底"，不应调用 LLM。"""

    replanner = Replanner()
    task = _make_task(failure_type=FailureType.UNKNOWN_ERROR, attempt=3, max_attempts=3)

    def _should_not_be_called(t: Task) -> FailureDecision:
        raise AssertionError("不应该调用 LLM 兜底")

    decision = replanner.classify_failure(task, llm_fallback=_should_not_be_called)
    assert decision == FailureDecision.TERMINAL_FAILURE


# ---- MainAgentLLMDecisionMaker：上下文编排 + 解析 + 安全降级 ----


def _decision_maker(mock_provider: MockLLMProvider, context_builder) -> MainAgentLLMDecisionMaker:
    return MainAgentLLMDecisionMaker(context_builder, mock_provider, model="mock-model")


def test_llm_decision_maker_parses_valid_json_response(main_agent: MainAgent):
    provider = MockLLMProvider(
        fallback_response=LLMResponse(
            content=json.dumps({"decision": "split", "reason": "任务范围过大"})
        )
    )
    maker = _decision_maker(provider, main_agent.context_builder)
    task = _make_task(failure_type=FailureType.UNKNOWN_ERROR)

    result = maker.classify_failure_with_llm(
        job=main_agent.job,
        dag=main_agent.scheduler.dag,
        task=task,
        recent_events=[],
    )

    assert result.decision == FailureDecision.SPLIT
    assert result.reason == "任务范围过大"
    assert result.fallback_used is False
    # 编排的上下文应该被真正送进了 LLM（校验 call_log 里出现了任务信息）。
    assert len(provider.call_log) == 1
    system_message = provider.call_log[0][0]
    assert system_message.content.startswith(STABLE_PROMPT_PREFIX)
    assert provider.params_log[0].prompt_cache_key.startswith(
        f"{STABLE_PROMPT_PREFIX_VERSION}:"
    )
    user_message = provider.call_log[0][-1]
    assert task.task_id in user_message.content
    context = json.loads(user_message.content)
    assert context["schema_version"] == "1.0"
    assert context["context_type"] == "main_agent_decision"
    assert [segment["name"] for segment in context["segments"]] == [
        "job_status",
        "dag_summary",
        "current_decision",
        "memory_l0_index",
        "recent_events",
        "unresolved_issues",
        "budget",
    ]
    assert context["segments"][1]["kind"] == "task_graph"
    assert context["segments"][1]["source"] == "scheduler"
    assert context["segments"][2]["content"]["description"].startswith("任务 ")
    assert context["compression"]["dropped_segment_names"] == []


def test_llm_decision_maker_falls_back_on_invalid_json(main_agent: MainAgent):
    provider = MockLLMProvider(fallback_response=LLMResponse(content="not a json at all"))
    maker = _decision_maker(provider, main_agent.context_builder)
    task = _make_task(failure_type=FailureType.UNKNOWN_ERROR)

    result = maker.classify_failure_with_llm(
        job=main_agent.job, dag=main_agent.scheduler.dag, task=task, recent_events=[]
    )

    assert result.decision == FailureDecision.TERMINAL_FAILURE
    assert result.fallback_used is True


def test_llm_decision_maker_falls_back_on_illegal_decision_value(main_agent: MainAgent):
    provider = MockLLMProvider(
        fallback_response=LLMResponse(
            content=json.dumps({"decision": "reboot_the_universe", "reason": "不合法的决策"})
        )
    )
    maker = _decision_maker(provider, main_agent.context_builder)
    task = _make_task(failure_type=FailureType.UNKNOWN_ERROR)

    result = maker.classify_failure_with_llm(
        job=main_agent.job, dag=main_agent.scheduler.dag, task=task, recent_events=[]
    )

    assert result.decision == FailureDecision.TERMINAL_FAILURE
    assert result.fallback_used is True


def test_llm_decision_maker_falls_back_when_llm_call_raises(main_agent: MainAgent):
    from repro_agent.providers.base import LLMProviderError

    class _AlwaysFailProvider:
        def complete(self, messages, params):
            raise LLMProviderError("network exploded", is_retryable=False)

    maker = _decision_maker(_AlwaysFailProvider(), main_agent.context_builder)
    task = _make_task(failure_type=FailureType.UNKNOWN_ERROR)

    result = maker.classify_failure_with_llm(
        job=main_agent.job, dag=main_agent.scheduler.dag, task=task, recent_events=[]
    )

    assert result.decision == FailureDecision.TERMINAL_FAILURE
    assert result.fallback_used is True


def test_context_envelope_remains_valid_json_when_segments_are_truncated(
    main_agent: MainAgent,
):
    context = main_agent.context_builder.build(
        job=main_agent.job,
        dag=main_agent.scheduler.dag,
        current_decision="x" * 20_000,
        recent_events=[{"event_type": "large", "payload": "y" * 20_000}],
        unresolved_issues=[],
        max_tokens=128,
    )

    payload = json.loads(context.text)
    assert payload["schema_version"] == "1.0"
    assert payload["compression"]["max_tokens"] == 128
    assert payload["compression"]["dropped_segment_names"]
    assert all(isinstance(segment["content"], (dict, list, str)) for segment in payload["segments"])
    assert any(
        segment["metadata"].get("content_truncated")
        for segment in payload["segments"]
    )


# ---- MainAgent 端到端接入 ----


def test_main_agent_handle_failed_task_uses_llm_for_unknown_error_and_records_event(
    main_agent: MainAgent,
):
    """端到端：UNKNOWN_ERROR 任务经 _handle_failed_task 触发 LLM 分类，
    且记录了可审计的 failure_classified_by_llm 事件。
    """

    main_agent.llm_decision_maker.llm_provider = MockLLMProvider(
        fallback_response=LLMResponse(
            content=json.dumps({"decision": "ask_user", "reason": "需要人工介入"})
        )
    )

    task = _make_task(failure_type=FailureType.UNKNOWN_ERROR, job_id=main_agent.job.job_id)
    main_agent.scheduler.dag.add_task(task)

    main_agent._handle_failed_task(task)

    events = main_agent.task_repo.list_events(main_agent.job.job_id)
    llm_events = [e for e in events if e["event_type"] == "failure_classified_by_llm"]
    assert len(llm_events) == 1
    assert llm_events[0]["payload"]["decision"] == "ask_user"
    assert llm_events[0]["payload"]["fallback_used"] is False

    classified_events = [e for e in events if e["event_type"] == "failure_classified"]
    assert classified_events[-1]["payload"]["decision"] == "ask_user"


def test_main_agent_handle_failed_task_known_type_does_not_call_llm(main_agent: MainAgent):
    """已知失败类型（如 TRANSIENT_ERROR）走纯规则路径，不应产生 LLM 分类事件。"""

    provider = main_agent.llm_decision_maker.llm_provider
    original_call_count = getattr(provider, "call_count", 0)

    task = _make_task(failure_type=FailureType.TRANSIENT_ERROR, job_id=main_agent.job.job_id)
    main_agent.scheduler.dag.add_task(task)

    main_agent._handle_failed_task(task)

    events = main_agent.task_repo.list_events(main_agent.job.job_id)
    llm_events = [e for e in events if e["event_type"] == "failure_classified_by_llm"]
    assert llm_events == []
    assert getattr(provider, "call_count", 0) == original_call_count
    guidance = task.definition.inputs["retry_guidance"]
    assert isinstance(guidance, str)
    assert guidance.startswith("重试注意事项：")
    assert "人工排查" in guidance
    assert task.failure_report is None

    requeue_events = [e for e in events if e["event_type"] == "task_requeued"]
    assert requeue_events[-1]["payload"]["retry_guidance"] == guidance
