"""端到端回归测试：反思闭环（§11.2、§19）。

覆盖两条互斥的分支，呼应用户明确提出的约束——"如果审查确定整个流程
没有问题，不用为了对齐论文中的目标而重跑"：

    1. gap 检测 -> 触发 reflection -> reflection 产出假设 -> 派发审计
       任务 -> 审计任务均未发现问题 -> 汇总为 ``NO_OBVIOUS_ERROR_FOUND``
       -> **不重跑**，直接把 Job 推进到终态 ``VERIFIED_REPRODUCTION_GAP``。
    2. 同样的触发路径，但审计任务确认了具体错误（配置冲突）->
       ``CONFIG_ERROR_CONFIRMED`` -> 派发 repair 任务 -> repair 完成 ->
       派发"最小范围重跑"任务，Job 状态推进到 ``RERUN_REQUIRED``。

两个测试都复用 ``main_agent`` fixture（真实 SQLite + 真实沙箱 +
真实 AgentDispatcher 后台线程，只有 LLM 是 mock），并通过临时注册测试
专用子智能体类的方式驱动任务真正跑完一次完整的
dispatch -> start_async -> collect -> validate 流程，
这样才能验证 ``MainAgent`` 里反思闭环的胶水逻辑（
``_on_task_validated_for_reflection`` / ``_advance_reflection_pipeline``
等）而不是绕过它们直接操纵内部状态。
"""

from __future__ import annotations

import time

from repro_agent.agents.base import AgentRunResult, BaseSubAgent
from repro_agent.agents.registry import SUB_AGENT_REGISTRY
from repro_agent.domain.enums import ExperimentTier, JobStatus, ReproductionStatus, TaskStatus
from repro_agent.domain.experiment import ExperimentRun
from repro_agent.domain.task import Task
from repro_agent.orchestrator.task_factory import build_task_definition


# ---- 测试专用子智能体：verification / reflection / 审计 / 修复 ----


class _GapVerificationAgent(BaseSubAgent):
    """产出一个明显超出容差的指标对比，用于触发反思。"""

    task_type = "verification"

    def run(self) -> AgentRunResult:
        self.write_json_output(
            "result.json",
            {
                "run_actually_executed": True,
                "comparisons": [
                    {
                        "metric": "accuracy",
                        "paper_value": 0.95,
                        "reproduced_value": 0.60,
                        "tolerance_type": "absolute",
                        "tolerance": 0.02,
                        "within_tolerance": False,
                    }
                ]
            },
        )
        return AgentRunResult(succeeded=True, outputs={"ok": True})


class _OneHypothesisReflectionAgent(BaseSubAgent):
    """只产出一条 C 类假设（对应 specification 审计任务）。"""

    task_type = "reflection"

    def run(self) -> AgentRunResult:
        self.write_json_output(
            "result.json",
            {
                "likely_source": "possible_param_mismatch",
                "hypotheses": [
                    {
                        "category": "C",
                        "description": "怀疑实验规格里存在未对齐的参数",
                        "confidence": 0.7,
                        "required_checks": ["compare batch_size"],
                        "priority": 5,
                    }
                ],
            },
        )
        return AgentRunResult(succeeded=True, outputs={"ok": True})


class _CleanSpecificationAuditAgent(BaseSubAgent):
    """审计任务：未发现任何未解决的字段冲突（无问题分支）。"""

    task_type = "specification"

    def run(self) -> AgentRunResult:
        self.write_json_output("result.json", {"unresolved_conflicts": [], "fields": {}})
        return AgentRunResult(succeeded=True, outputs={"ok": True})


class _ConflictSpecificationAuditAgent(BaseSubAgent):
    """审计任务：确认存在未解决的字段冲突（确认问题分支）。"""

    task_type = "specification"

    def run(self) -> AgentRunResult:
        self.write_json_output(
            "result.json",
            {
                "unresolved_conflicts": [
                    {"field": "batch_size", "paper_value": 32, "code_value": 64}
                ],
                "fields": {},
            },
        )
        return AgentRunResult(succeeded=True, outputs={"ok": True})


class _InstantSuccessAgent(BaseSubAgent):
    """通用的"立即成功"子智能体，用于修复任务。"""

    task_type = "coding"

    def run(self) -> AgentRunResult:
        payload = {"ok": True}
        if self.task.definition.task_type == "specification":
            payload = {
                "experiment_id": "main_experiment",
                "target_claim": "reproduce_main_result",
                "fields": {},
                "unresolved_conflicts": [],
            }
        self.write_json_output("result.json", payload)
        return AgentRunResult(succeeded=True, outputs={"ok": True})


def _register(monkeypatch, task_type: str, agent_cls) -> None:
    monkeypatch.setitem(SUB_AGENT_REGISTRY, task_type, agent_cls)


def _run_task_to_completion(main_agent, task: Task, *, timeout: float = 5.0) -> None:
    """把一个已加入调度器的任务完整跑一遍 dispatch -> ... -> validate。

    这是驱动 ``MainAgent`` 反思闭环胶水逻辑的关键：只有真正走完
    ``validate_outputs``（从而触发 ``_on_task_validated_for_reflection``）
    才能验证 main_agent.py 里新增的回调是否被正确调用。
    """

    main_agent.scheduler.dispatch([task])
    main_agent._run_dispatched_task(task)

    handle = main_agent.dispatcher.get_handle(task.task_id)
    assert handle is not None, f"task {task.task_id} has no handle after dispatch"
    deadline = time.monotonic() + timeout
    while not handle.is_finished() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert handle.is_finished(), f"task {task.task_id} did not finish in time"

    main_agent._collect_finished_subagents()
    completed = main_agent._new_completed_tasks()
    if task.task_id in [t.task_id for t in completed]:
        main_agent.validate_outputs(completed)


def _make_and_add_task(main_agent, task_type: str, *, inputs=None) -> Task:
    definition = build_task_definition(
        objective=f"test task for {task_type}",
        task_type=task_type,
        inputs=inputs or {},
        extra_allowed_tools=["write_task_output"],
    )
    task = Task(job_id=main_agent.job.job_id, definition=definition)
    main_agent.scheduler.add_tasks([task])
    return task


def _trigger_reflection_from_gap(main_agent, monkeypatch, reflection_agent_cls) -> None:
    """跑完 verification -> gap 检测 -> reflection 两步，直到审计任务派发。"""

    _register(monkeypatch, "verification", _GapVerificationAgent)
    verification_task = _make_and_add_task(main_agent, "verification")
    _run_task_to_completion(main_agent, verification_task)
    assert verification_task.status == TaskStatus.SUCCEEDED

    # 手动驱动一次 gap 检测 + reflection 触发（等价于 step() 里对应片段）。
    gap_decision = main_agent._check_result_gap()
    assert gap_decision is not None
    assert gap_decision.should_reflect is True

    _register(monkeypatch, "reflection", reflection_agent_cls)
    main_agent._trigger_reflection(gap_decision.reason)
    assert main_agent.job.status == JobStatus.REFLECTION_REQUIRED

    reflection_tasks = [
        t for t in main_agent.scheduler.dag.all_tasks() if t.definition.task_type == "reflection"
    ]
    assert len(reflection_tasks) == 1
    _run_task_to_completion(main_agent, reflection_tasks[0])
    assert reflection_tasks[0].status == TaskStatus.SUCCEEDED

    # reflection 成功后应该已经构造出一份 ReflectionReport 并派发了审计任务。
    assert len(main_agent._reflection_reports) == 1


def test_no_issue_found_skips_rerun_and_reaches_verified_gap(main_agent, monkeypatch):
    """核心约束：审计确认无明显问题时，不重跑，直接进入终态。"""

    _trigger_reflection_from_gap(main_agent, monkeypatch, _OneHypothesisReflectionAgent)

    report = main_agent._reflection_reports[0]
    audit_tasks = [
        t
        for t in main_agent.scheduler.dag.all_tasks()
        if t.definition.inputs.get("reflection_id") == report.reflection_id
        and t.definition.inputs.get("audit_hypothesis_id")
    ]
    assert len(audit_tasks) == 1
    assert audit_tasks[0].definition.task_type == "specification"

    _register(monkeypatch, "specification", _CleanSpecificationAuditAgent)
    _run_task_to_completion(main_agent, audit_tasks[0])
    assert audit_tasks[0].status == TaskStatus.SUCCEEDED

    # 该轮审计任务已全部完成，推进反思流水线做汇总。
    main_agent._advance_reflection_pipeline()

    assert report.audit_result is not None
    assert report.audit_result.value == "NO_OBVIOUS_ERROR_FOUND"
    assert report.rerun_triggered is False
    assert report.repair_task_ids == []

    # 不允许产生任何 repair/experiment_execution 重跑任务。
    all_task_types = [t.definition.task_type for t in main_agent.scheduler.dag.all_tasks()]
    assert "experiment_execution" not in all_task_types

    assert main_agent.job.status == JobStatus.VERIFIED_REPRODUCTION_GAP
    assert main_agent.job.final_reproduction_status == ReproductionStatus.VERIFIED_REPRODUCTION_GAP
    # 预算未被无谓消耗：没有发生重跑，重跑计数应保持为 0。
    assert main_agent.job.full_experiment_rerun_count == 0


def test_confirmed_issue_triggers_repair_then_minimum_rerun(main_agent, monkeypatch):
    """对照分支：审计确认具体错误时，应该走 repair -> 最小范围重跑。"""

    main_agent.job.inputs.user_run_commands = ["python train.py"]
    experiment_id = main_agent.job.inputs.target_experiments[0]
    for tier in ExperimentTier:
        main_agent.experiment_run_repo.save(
            ExperimentRun(
                job_id=main_agent.job.job_id,
                experiment_id=experiment_id,
                tier=tier,
                exit_code=0,
                container_digest="sha256:test",
                command='["python", "train.py"]',
            )
        )

    _trigger_reflection_from_gap(main_agent, monkeypatch, _OneHypothesisReflectionAgent)

    report = main_agent._reflection_reports[0]
    audit_tasks = [
        t
        for t in main_agent.scheduler.dag.all_tasks()
        if t.definition.inputs.get("reflection_id") == report.reflection_id
        and t.definition.inputs.get("audit_hypothesis_id")
    ]
    assert len(audit_tasks) == 1

    _register(monkeypatch, "specification", _ConflictSpecificationAuditAgent)
    _run_task_to_completion(main_agent, audit_tasks[0])
    assert audit_tasks[0].status == TaskStatus.SUCCEEDED

    main_agent._advance_reflection_pipeline()

    assert report.audit_result is not None
    assert report.audit_result.value == "CONFIG_ERROR_CONFIRMED"
    assert main_agent.job.status == JobStatus.REPAIR_RUNNING
    assert len(report.repair_task_ids) == 1

    repair_task = main_agent.scheduler.dag.get(report.repair_task_ids[0])
    assert repair_task is not None
    _register(monkeypatch, repair_task.definition.task_type, _InstantSuccessAgent)
    _run_task_to_completion(main_agent, repair_task)
    assert repair_task.status == TaskStatus.SUCCEEDED

    # 修复完成后推进流水线，应该派发最小范围重跑任务。
    main_agent._advance_reflection_pipeline()

    assert report.rerun_triggered is True
    rerun_tasks = [
        t for t in main_agent.scheduler.dag.all_tasks() if t.definition.task_type == "experiment_execution"
    ]
    assert len(rerun_tasks) == 1
    assert rerun_tasks[0].definition.inputs.get("reflection_id") == report.reflection_id
    assert rerun_tasks[0].definition.inputs.get("tier") == ExperimentTier.FULL_EXPERIMENT.value
    assert rerun_tasks[0].definition.inputs.get("command") == ["python", "train.py"]
    assert rerun_tasks[0].definition.inputs.get("parent_run_id")
    assert main_agent.job.status == JobStatus.RERUN_REQUIRED
    assert main_agent.job.full_experiment_rerun_count == 1
