"""工具分配权上收（主智能体统一分配/补授工具）的核心机制回归测试。

覆盖需求点：
    1. ``ToolAuthorization.grant_additional``：运行期动态授权并入后，
       子智能体的下一次 ``call`` 能直接使用该工具（原地继续的物质基础）；
    2. ``escalation_handler``：调用"已注册但未分配"的工具时先升级给主
       智能体裁决，而不是直接失败；裁决拒绝时抛 ``ToolGrantDeniedError``；
    3. ``ToolGrantDecisionMaker``：确定性规则（未注册/超风险预算）直接
       DENY；LLM 裁决 GRANT；LLM 失败安全降级 ASK_USER；同 (task, tool)
       裁决只做一次（缓存）；
    4. ``ToolAllocationPlanner``：LLM 输出的幻觉工具名被过滤、
       ``write_task_output`` 强制保留、失败回退模板；
    5. dispatcher 挂起-批准-唤醒链路：子智能体线程阻塞在升级等待上，
       人工批准后 ``resume_escalation`` 注入工具并唤醒，线程**原地继续**
       完成原工具调用（不重启、任务上下文不丢）。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from repro_agent.domain.enums import TaskStatus, ToolGrantDecision
from repro_agent.domain.task import Task
from repro_agent.orchestrator.dispatcher import AgentDispatcher, SubAgentHandle
from repro_agent.orchestrator.tool_grant import (
    ToolAllocationPlanner,
    ToolGrantDecisionMaker,
    extract_requested_tool_names,
)
from repro_agent.orchestrator.task_factory import build_task_definition
from repro_agent.providers.base import LLMResponse
from repro_agent.providers.mock import MockLLMProvider
from repro_agent.sandbox.manager import SandboxManager
from repro_agent.storage.database import Database
from repro_agent.storage.repository import TaskRepository
from repro_agent.tools.authorization import ToolAuthorizer, ToolAuthorization
from repro_agent.tools.base import ToolGrantDeniedError, ToolPermissionError


# ---------------------------------------------------------------------------
# 基础构造辅助
# ---------------------------------------------------------------------------


def _grant_llm(payload: dict) -> MockLLMProvider:
    return MockLLMProvider(
        fallback_response=LLMResponse(content=json.dumps(payload, ensure_ascii=False))
    )


def _make_authorization(tmp_path: Path, allowed_tools: list[str]) -> tuple[ToolAuthorization, SandboxManager, Task]:
    sandbox_manager = SandboxManager(str(tmp_path / "sandbox"))
    authorizer = ToolAuthorizer()
    definition = build_task_definition(objective="只读分析", task_type="paper_analysis")
    definition.allowed_tools = list(allowed_tools)
    task = Task(job_id="job-1", definition=definition)
    sandbox = sandbox_manager.create_sandbox(task)
    authorization = authorizer.authorize(
        task_id=task.task_id,
        task_type=definition.task_type,
        allowed_tools=definition.allowed_tools,
        sandbox_ctx=sandbox,
    )
    return authorization, sandbox_manager, task


# ---------------------------------------------------------------------------
# 1. 运行期动态授权
# ---------------------------------------------------------------------------


def test_grant_additional_extends_authorization(tmp_path):
    """grant_additional 之后 call 未授权工具应能直接执行成功。"""
    authorization, _, _ = _make_authorization(tmp_path, ["list_directory"])
    assert "read_file" not in authorization.granted_tool_names

    read_spec = ToolAuthorizer().registry.get("read_file")
    assert read_spec is not None
    authorization.grant_additional(read_spec)

    assert "read_file" in authorization.granted_tool_names
    # describe_granted 也能暴露新工具（供下一次 LLM 调用使用）
    tools = authorization.describe_granted(["read_file"])
    assert tools[0]["function"]["name"] == "read_file"


def test_grant_additional_is_thread_safe_under_concurrent_reads(tmp_path):
    """主智能体线程 grant_additional 的同时子智能体线程读工具名不崩溃。"""
    authorization, _, _ = _make_authorization(tmp_path, ["list_directory"])
    authorizer = ToolAuthorizer()
    errors: list[BaseException] = []

    def reader():
        try:
            for _ in range(200):
                authorization.granted_tool_names
                authorization.describe_granted(None)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=reader)
    thread.start()
    for name in ["read_file", "find_files", "grep_files"]:
        spec = authorizer.registry.get(name)
        if spec is not None:
            authorization.grant_additional(spec)
    thread.join(timeout=5)
    assert errors == []


# ---------------------------------------------------------------------------
# 2. 升级钩子（ToolAuthorization.call 的两种未授权行为）
# ---------------------------------------------------------------------------


def test_escalation_handler_grant_lets_call_continue_in_place(tmp_path):
    """升级裁决为补授时，call 原地继续执行原工具——不重启、不丢上下文。"""
    authorization, _, _ = _make_authorization(tmp_path, ["list_directory"])
    authorizer = ToolAuthorizer()

    # 在沙箱可读目录里准备一个真实文件，验证工具真的被执行
    sandbox_dir = authorization._sandbox_ctx.input_dir  # noqa: SLF001
    (Path(sandbox_dir) / "notes.txt").write_text("hello", encoding="utf-8")

    def handler(tool_name, arguments):
        # 模拟主智能体裁决 GRANT：把 spec 并入授权对象
        spec = authorizer.registry.get(tool_name)
        assert spec is not None
        authorization.grant_additional(spec)

    authorization.set_escalation_handler(handler)
    # read_file 在 paper_analysis 风险预算内，应能通过升级获得执行
    result = authorization.call("read_file", path="notes.txt")
    assert result["content"] == "hello"


def test_escalation_handler_deny_raises_tool_grant_denied(tmp_path):
    authorization, _, _ = _make_authorization(tmp_path, ["list_directory"])

    def handler(tool_name, arguments):
        raise ToolGrantDeniedError(tool_name, "任务不需要该工具")

    authorization.set_escalation_handler(handler)
    with pytest.raises(ToolGrantDeniedError):
        authorization.call("read_file", path="x.txt")


def test_without_handler_behaves_like_before(tmp_path):
    """未注入 handler 时保持旧行为：直接抛 ToolPermissionError。"""
    authorization, _, _ = _make_authorization(tmp_path, ["list_directory"])
    with pytest.raises(ToolPermissionError):
        authorization.call("read_file", path="x.txt")


# ---------------------------------------------------------------------------
# 3. 裁决器
# ---------------------------------------------------------------------------


def test_decision_maker_denies_unregistered_tool_without_llm():
    maker = ToolGrantDecisionMaker(ToolAuthorizer())
    outcome = maker.adjudicate(
        task_id="t1",
        task_type="paper_analysis",
        objective="分析论文",
        inputs={},
        allowed_tools=["read_file"],
        forbidden_actions=[],
        tool_name="no_such_tool",
    )
    assert outcome.decision == ToolGrantDecision.DENY
    assert "not registered" in outcome.reason


def test_decision_maker_denies_tool_over_risk_budget_without_llm():
    """超风险预算是硬边界：主智能体裁量不能突破，直接 DENY。"""
    maker = ToolGrantDecisionMaker(ToolAuthorizer())
    outcome = maker.adjudicate(
        task_id="t1",
        task_type="paper_analysis",  # READ_ONLY 预算
        objective="分析论文",
        inputs={},
        allowed_tools=["read_file"],
        forbidden_actions=[],
        tool_name="execute_command",  # HIGH_RISK
    )
    assert outcome.decision == ToolGrantDecision.DENY


def test_decision_maker_llm_grant_and_cached_adjudication():
    provider = _grant_llm({"decision": "grant", "reason": "任务确实需要读 PDF"})
    maker = ToolGrantDecisionMaker(ToolAuthorizer(), provider)

    kwargs = dict(
        task_id="t1",
        task_type="paper_analysis",
        objective="解析 PDF 中的表格",
        inputs={},
        allowed_tools=["read_file"],
        forbidden_actions=[],
        tool_name="read_pdf_text",
    )
    outcome = maker.adjudicate(**kwargs)
    assert outcome.decision == ToolGrantDecision.GRANT
    assert outcome.source == "llm"

    # 第二次同 (task, tool) 命中缓存，不再调 LLM
    calls_before = provider.call_count
    outcome2 = maker.adjudicate(**kwargs)
    assert outcome2.decision == ToolGrantDecision.GRANT
    assert outcome2.source == "cache"
    assert provider.call_count == calls_before


def test_decision_maker_llm_deny():
    provider = _grant_llm({"decision": "deny", "reason": "现有工具足够"})
    maker = ToolGrantDecisionMaker(ToolAuthorizer(), provider)
    outcome = maker.adjudicate(
        task_id="t1",
        task_type="paper_analysis",
        objective="分析论文",
        inputs={},
        allowed_tools=["read_file"],
        forbidden_actions=[],
        tool_name="grep_files",
    )
    assert outcome.decision == ToolGrantDecision.DENY
    assert outcome.source == "llm"


def test_decision_maker_llm_failure_falls_back_to_ask_user():
    """LLM 失败既不能放权也不能杀任务：安全降级为 ASK_USER。"""

    class BoomLLM:
        def complete(self, messages, params):
            raise RuntimeError("network down")

    maker = ToolGrantDecisionMaker(ToolAuthorizer(), BoomLLM(), max_llm_retries=1)
    outcome = maker.adjudicate(
        task_id="t1",
        task_type="paper_analysis",
        objective="分析论文",
        inputs={},
        allowed_tools=["read_file"],
        forbidden_actions=[],
        tool_name="grep_files",
    )
    assert outcome.decision == ToolGrantDecision.ASK_USER
    assert outcome.source == "fallback"


def test_decision_maker_without_llm_asks_user():
    maker = ToolGrantDecisionMaker(ToolAuthorizer(), None)
    outcome = maker.adjudicate(
        task_id="t1",
        task_type="paper_analysis",
        objective="分析论文",
        inputs={},
        allowed_tools=["read_file"],
        forbidden_actions=[],
        tool_name="grep_files",
    )
    assert outcome.decision == ToolGrantDecision.ASK_USER


# ---------------------------------------------------------------------------
# 4. 规划期分配器
# ---------------------------------------------------------------------------


def test_allocation_planner_falls_back_to_template_on_llm_failure():
    class BoomLLM:
        def complete(self, messages, params):
            raise RuntimeError("network down")

    planner = ToolAllocationPlanner(ToolAuthorizer(), BoomLLM(), max_llm_retries=1)
    template = ["read_file", "find_files", "grep_files", "write_task_output"]
    tools, source = planner.plan_allowed_tools(
        task_type="paper_analysis",
        objective="分析论文",
        inputs={},
        template_tools=template,
    )
    assert tools == template
    assert source.startswith("template-fallback")


def test_allocation_planner_filters_hallucinated_names():
    """LLM 幻觉出的工具名被丢弃，write_task_output 强制保留。"""
    provider = _grant_llm(
        {
            "allowed_tools": ["read_file", "hack_the_planet", "write_task_output"],
            "reason": "最小集",
        }
    )
    planner = ToolAllocationPlanner(ToolAuthorizer(), provider)
    tools, source = planner.plan_allowed_tools(
        task_type="paper_analysis",
        objective="只读论文",
        inputs={},
        template_tools=["read_file", "find_files", "grep_files", "write_task_output"],
    )
    assert source == "llm-customized"
    assert "read_file" in tools
    assert "write_task_output" in tools
    assert "hack_the_planet" not in tools
    # 风险预算外的高危工具即便被 LLM 输出也不可能出现
    assert "execute_command" not in tools


def test_allocation_planner_never_returns_high_risk_for_read_only_tasks():
    provider = _grant_llm(
        {"allowed_tools": ["execute_command", "write_file"], "reason": "越权请求"}
    )
    planner = ToolAllocationPlanner(ToolAuthorizer(), provider)
    tools, _ = planner.plan_allowed_tools(
        task_type="paper_analysis",
        objective="只读论文",
        inputs={},
        template_tools=["read_file", "write_task_output"],
    )
    # 全部是幻觉/越权名称 -> 过滤后只剩强制项，回退模板
    assert "execute_command" not in tools
    assert "write_file" not in tools
    assert "read_file" in tools


# ---------------------------------------------------------------------------
# 5. 错误消息工具名提取
# ---------------------------------------------------------------------------


def test_extract_requested_tool_names_from_call_error():
    message = "task t1 is not authorized to call tool 'inspect_pdf_page'"
    assert extract_requested_tool_names(message) == ["inspect_pdf_page"]


def test_extract_requested_tool_names_from_describe_error():
    message = (
        "task t1 requested tool description(s) not granted: read_pdf_text, "
        "inspect_pdf_page (granted=['list_directory'])"
    )
    assert extract_requested_tool_names(message) == [
        "read_pdf_text",
        "inspect_pdf_page",
    ]


# ---------------------------------------------------------------------------
# 6. dispatcher 挂起-批准-唤醒链路（集成级）
# ---------------------------------------------------------------------------


def _wait_for(predicate, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    return None


@pytest.fixture()
def dispatcher_env(tmp_path):
    """构造带真实 SQLite/sandbox 的 dispatcher 环境（不经过 MainAgent）。"""
    from repro_agent.domain.job import JobInputs, ReproductionJob
    from repro_agent.storage.repository import JobRepository

    db = Database(tmp_path / "state.db")
    task_repo = TaskRepository(db)
    job = ReproductionJob(
        inputs=JobInputs(
            paper_path=str(tmp_path / "paper.txt"),
            repository_path=str(tmp_path / "repo"),
        )
    )
    JobRepository(db).save(job)
    sandbox_manager = SandboxManager(str(tmp_path / "sandbox"))
    authorizer = ToolAuthorizer()

    definition = build_task_definition(objective="读取论文文件", task_type="paper_analysis")
    # 故意只给 list_directory，模拟主智能体裁剪后的白名单
    definition.allowed_tools = ["list_directory", "write_task_output"]
    task = Task(job_id=job.job_id, definition=definition)
    task.status = TaskStatus.RUNNING
    task.attempt = 1
    task.active_attempt_id = "attempt_1"
    task_repo.save(task)
    sandbox_manager.create_sandbox(task)

    # 裁决器：LLM 恒定返回 ask_user -> 走人工挂起分支
    provider = _grant_llm({"decision": "ask_user", "reason": "拿不准"})
    maker = ToolGrantDecisionMaker(authorizer, provider)
    escalation_requests: list[tuple[str, str]] = []

    def fake_request_human(task_obj, tool_name, reason):
        escalation_requests.append((task_obj.task_id, tool_name))
        return f"req-{len(escalation_requests)}"

    dispatcher = AgentDispatcher(
        sandbox_manager,
        authorizer,
        provider,
        task_repo,
        tool_grant_decision_maker=maker,
        request_human_tool_grant=fake_request_human,
        escalation_wait_poll_seconds=0.02,
    )
    return dispatcher, task, task_repo, escalation_requests


def test_dispatcher_escalation_approval_resumes_in_place(dispatcher_env):
    """核心场景：升级挂起 -> 人工批准 -> 子智能体线程原地继续完成任务。"""
    dispatcher, task, task_repo, escalation_requests = dispatcher_env

    # 在沙箱里准备一个可读文件
    sandbox = dispatcher.sandbox_manager.get(task.task_id)
    assert sandbox is not None
    (Path(sandbox.input_dir) / "notes.txt").write_text("hello paper", encoding="utf-8")

    authorization = dispatcher.tool_authorizer.authorize(
        task_id=task.task_id,
        task_type=task.definition.task_type,
        allowed_tools=task.definition.allowed_tools,
        sandbox_ctx=sandbox,
        attempt_id=task.active_attempt_id,
    )
    handle = SubAgentHandle(
        task, lambda: None, attempt_id=task.active_attempt_id,
        on_progress_push=dispatcher._on_progress_push,
    )
    # 复刻 start_async 中的 handler 注入
    authorization.set_escalation_handler(
        lambda tool_name, arguments: dispatcher._handle_tool_escalation(
            task, authorization, handle, sandbox, tool_name, arguments
        )
    )

    outcome: dict = {}

    def subagent_thread():
        # 子智能体调用未分配的工具：触发升级 -> 挂起 -> 批准后原地继续
        try:
            result = authorization.call("read_file", path="notes.txt")
            outcome["result"] = result
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc

    thread = threading.Thread(target=subagent_thread, daemon=True)
    thread.start()

    # 等待升级请求登记
    escalation = _wait_for(lambda: dispatcher.get_pending_escalation(task.task_id))
    assert escalation is not None
    assert escalation.tool_name == "read_file"
    assert escalation_requests and escalation_requests[0][1] == "read_file"
    # 线程还在挂起（没有结果也没有错误）
    assert "result" not in outcome and "error" not in outcome

    # 人工批准：注入工具并唤醒
    resumed = dispatcher.resume_escalation(
        task.task_id, approved_tools=["read_file"], reason="人工批准"
    )
    assert resumed is True

    thread.join(timeout=10)
    assert not thread.is_alive()

    # 子智能体原地继续并成功读到文件内容——没有重启、没有异常
    assert "error" not in outcome, f"unexpected error: {outcome.get('error')}"
    result = outcome.get("result")
    assert result is not None and result["content"] == "hello paper"

    # 工具已并入授权对象与任务定义（重试后仍然生效）
    assert "read_file" in authorization.granted_tool_names
    assert "read_file" in task.definition.allowed_tools
    # 升级登记项已被清理
    assert dispatcher.get_pending_escalation(task.task_id) is None


def test_dispatcher_escalation_rejection_fails_subagent(dispatcher_env):
    """人工拒绝：挂起线程被唤醒并收到明确的拒绝裁决异常。"""
    dispatcher, task, task_repo, escalation_requests = dispatcher_env

    sandbox = dispatcher.sandbox_manager.get(task.task_id)
    assert sandbox is not None
    authorization = dispatcher.tool_authorizer.authorize(
        task_id=task.task_id,
        task_type=task.definition.task_type,
        allowed_tools=task.definition.allowed_tools,
        sandbox_ctx=sandbox,
        attempt_id=task.active_attempt_id,
    )
    handle = SubAgentHandle(
        task, lambda: None, attempt_id=task.active_attempt_id,
        on_progress_push=dispatcher._on_progress_push,
    )
    authorization.set_escalation_handler(
        lambda tool_name, arguments: dispatcher._handle_tool_escalation(
            task, authorization, handle, sandbox, tool_name, arguments
        )
    )

    outcome: dict = {}

    def subagent_thread():
        try:
            authorization.call("read_file", path="notes.txt")
            outcome["result"] = True
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc

    thread = threading.Thread(target=subagent_thread, daemon=True)
    thread.start()

    escalation = _wait_for(lambda: dispatcher.get_pending_escalation(task.task_id))
    assert escalation is not None

    failed = dispatcher.fail_escalation(task.task_id, reason="人工拒绝")
    assert failed is True

    thread.join(timeout=10)
    assert not thread.is_alive()
    error = outcome.get("error")
    assert isinstance(error, ToolGrantDeniedError)
    assert "人工拒绝" in str(error)
    # 工具没有被补授
    assert "read_file" not in authorization.granted_tool_names


def test_dispatcher_escalation_cancel_signal_interrupts_wait(dispatcher_env):
    """优雅取消信号能中断挂起等待，不会让线程永远阻塞。"""
    dispatcher, task, _, _ = dispatcher_env

    sandbox = dispatcher.sandbox_manager.get(task.task_id)
    assert sandbox is not None
    authorization = dispatcher.tool_authorizer.authorize(
        task_id=task.task_id,
        task_type=task.definition.task_type,
        allowed_tools=task.definition.allowed_tools,
        sandbox_ctx=sandbox,
        attempt_id=task.active_attempt_id,
    )
    handle = SubAgentHandle(
        task, lambda: None, attempt_id=task.active_attempt_id,
        on_progress_push=dispatcher._on_progress_push,
    )
    authorization.set_escalation_handler(
        lambda tool_name, arguments: dispatcher._handle_tool_escalation(
            task, authorization, handle, sandbox, tool_name, arguments
        )
    )

    outcome: dict = {}

    def subagent_thread():
        try:
            authorization.call("read_file", path="notes.txt")
            outcome["result"] = True
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc

    thread = threading.Thread(target=subagent_thread, daemon=True)
    thread.start()

    escalation = _wait_for(lambda: dispatcher.get_pending_escalation(task.task_id))
    assert escalation is not None

    # 主智能体发出优雅取消信号
    handle.request_graceful_cancel()
    thread.join(timeout=10)
    assert not thread.is_alive()
    # 线程以异常退出（CancellationRequested），不是无限挂起
    assert "error" in outcome
