"""回归测试：工具风险分级授权 + 子智能体只能使用主智能体下发的工具。

覆盖需求点：
    - 只读类任务（如 paper_analysis）即使 allowed_tools 里出现了高危
      工具名，也会被风险预算拒绝，不会被授予；
    - 高危任务类型（如 coding）可以拿到写文件/执行命令类工具；
    - write_task_output 对所有任务类型都是"始终允许"的最小能力；
    - 子智能体调用未被授权的工具会抛出 ToolPermissionError，而不是
      静默失败或绕过限制直接访问文件系统。
"""

from __future__ import annotations

import pytest

from repro_agent.sandbox.manager import SandboxManager
from repro_agent.tools.authorization import ToolAuthorizer, risk_allowed
from repro_agent.tools.base import ToolPermissionError, ToolRiskLevel
from repro_agent.domain.task import Task
from repro_agent.orchestrator.task_factory import build_task_definition


def test_risk_allowed_denies_high_risk_tool_for_read_only_task_type():
    assert risk_allowed("paper_analysis", ToolRiskLevel.READ_ONLY, tool_name="read_file") is True
    assert risk_allowed("paper_analysis", ToolRiskLevel.HIGH_RISK, tool_name="execute_command") is False


def test_risk_allowed_grants_high_risk_tool_for_high_budget_task_type():
    assert risk_allowed("coding", ToolRiskLevel.HIGH_RISK, tool_name="execute_command") is True


def test_write_task_output_always_allowed_regardless_of_risk_budget():
    # write_task_output 属于 _ALWAYS_ALLOWED_TOOLS 豁免名单，即便任务
    # 类型的风险预算是最低的 READ_ONLY，也不应被拒绝。
    assert risk_allowed("paper_analysis", ToolRiskLevel.RESTRICTED_WRITE, tool_name="write_task_output") is True


def test_authorizer_denies_tools_exceeding_task_type_risk_budget(tmp_path):
    sandbox_manager = SandboxManager(str(tmp_path / "job_root"))
    authorizer = ToolAuthorizer()

    definition = build_task_definition(
        objective="尝试越权",
        task_type="paper_analysis",
        # 显式在 allowed_tools 里"意外"混入高危工具，模拟任务定义配置
        # 疏漏的场景——授权层必须依然拒绝，而不是信任 allowed_tools。
        extra_allowed_tools=["execute_command", "write_file"],
    )
    task = Task(job_id="job-1", definition=definition)
    sandbox = sandbox_manager.create_sandbox(task)

    authorization = authorizer.authorize(
        task_id=task.task_id,
        task_type=task.definition.task_type,
        allowed_tools=task.definition.allowed_tools,
        sandbox_ctx=sandbox,
    )

    granted_names = set(authorization.granted_tool_names)
    assert "execute_command" not in granted_names
    assert "write_file" not in granted_names
    assert "read_file" in granted_names  # 只读工具应当正常授予
    denied_names = {d.tool_name for d in authorization.denials}
    assert {"execute_command", "write_file"} <= denied_names


def test_tool_authorization_call_raises_permission_error_for_ungranted_tool(tmp_path):
    sandbox_manager = SandboxManager(str(tmp_path / "job_root"))
    authorizer = ToolAuthorizer()

    definition = build_task_definition(objective="只读分析", task_type="paper_analysis")
    task = Task(job_id="job-1", definition=definition)
    sandbox = sandbox_manager.create_sandbox(task)

    authorization = authorizer.authorize(
        task_id=task.task_id,
        task_type=task.definition.task_type,
        allowed_tools=task.definition.allowed_tools,
        sandbox_ctx=sandbox,
    )

    with pytest.raises(ToolPermissionError):
        authorization.call("execute_command", command="echo hi")


def test_base_sub_agent_call_tool_propagates_permission_error(tmp_path):
    """子智能体基类调用未授权工具时应该抛出 ToolPermissionError 而不是被吞掉。"""

    from repro_agent.agents.base import BaseSubAgent
    from repro_agent.providers.mock import MockLLMProvider

    sandbox_manager = SandboxManager(str(tmp_path / "job_root"))
    authorizer = ToolAuthorizer()
    definition = build_task_definition(objective="只读分析", task_type="paper_analysis")
    task = Task(job_id="job-1", definition=definition)
    sandbox = sandbox_manager.create_sandbox(task)
    authorization = authorizer.authorize(
        task_id=task.task_id,
        task_type=task.definition.task_type,
        allowed_tools=task.definition.allowed_tools,
        sandbox_ctx=sandbox,
    )

    agent = BaseSubAgent(task, authorization, MockLLMProvider())
    with pytest.raises(ToolPermissionError):
        agent.call_tool("execute_command", command="echo hi")


def test_restrict_tools_narrows_template_to_intersection():
    """``restrict_tools`` 应该在类型模板范围内做交集收窄，而不是追加。"""

    definition = build_task_definition(
        objective="扫描代码仓库",
        task_type="code_analysis",
        restrict_tools=["find_files", "grep_files"],
    )

    # code_analysis 模板还包含 list_directory/read_file/get_file_stat，
    # 但这个具体任务实例声明了自己只需要 find_files/grep_files，
    # write_task_output 作为始终保留的最小能力也应该在。
    assert set(definition.allowed_tools) == {"find_files", "grep_files", "write_task_output"}


def test_restrict_tools_cannot_grant_tools_outside_template():
    """``restrict_tools`` 只能收窄，不能用来"越权"拿到模板之外的工具。"""

    definition = build_task_definition(
        objective="尝试用 restrict_tools 越权",
        task_type="paper_analysis",
        # execute_command 不在 paper_analysis 的标准模板里，应该被忽略。
        restrict_tools=["read_file", "execute_command"],
    )

    assert "execute_command" not in definition.allowed_tools
    assert "read_file" in definition.allowed_tools


def test_restrict_tools_none_keeps_full_template_for_backward_compatibility():
    """不传 ``restrict_tools`` 时保留旧行为：使用完整类型模板。"""

    default_definition = build_task_definition(objective="默认", task_type="code_analysis")
    from repro_agent.orchestrator.task_factory import STANDARD_TOOL_TEMPLATES

    assert set(default_definition.allowed_tools) == set(
        STANDARD_TOOL_TEMPLATES["code_analysis"]
    )


def test_describe_granted_returns_only_requested_subset(tmp_path):
    """``ToolAuthorization.describe_granted`` 按需返回子集，而不是 granted 全集。"""

    sandbox_manager = SandboxManager(str(tmp_path / "job_root"))
    definition = build_task_definition(
        objective="只读分析",
        task_type="paper_analysis",
        restrict_tools=["read_file", "read_pdf_text"],
    )
    task = Task(job_id="job-1", definition=definition)
    authorization = ToolAuthorizer().authorize(
        task_id=task.task_id,
        task_type=task.definition.task_type,
        allowed_tools=task.definition.allowed_tools,
        sandbox_ctx=sandbox_manager.create_sandbox(task),
    )

    subset = authorization.describe_granted(["read_file"])
    assert [tool["function"]["name"] for tool in subset] == ["read_file"]

    full = authorization.describe_granted(None)
    assert {tool["function"]["name"] for tool in full} == set(authorization.granted_tool_names)


def test_describe_granted_rejects_tool_not_granted(tmp_path):
    """请求一个未被授权的工具描述时应该直接报错，而不是静默忽略。"""

    sandbox_manager = SandboxManager(str(tmp_path / "job_root"))
    definition = build_task_definition(objective="只读分析", task_type="paper_analysis")
    task = Task(job_id="job-1", definition=definition)
    authorization = ToolAuthorizer().authorize(
        task_id=task.task_id,
        task_type=task.definition.task_type,
        allowed_tools=task.definition.allowed_tools,
        sandbox_ctx=sandbox_manager.create_sandbox(task),
    )

    with pytest.raises(ToolPermissionError):
        authorization.describe_granted(["execute_command"])


def test_forbidden_actions_override_allowed_tool_list(tmp_path):
    sandbox_manager = SandboxManager(str(tmp_path / "job_root"))
    task = Task(
        job_id="job-1",
        definition=build_task_definition(
            objective="deny read",
            task_type="paper_analysis",
            forbidden_actions=["read_file"],
        ),
    )
    authorization = ToolAuthorizer().authorize(
        task_id=task.task_id,
        task_type=task.definition.task_type,
        allowed_tools=["read_file"],
        forbidden_actions=task.definition.forbidden_actions,
        sandbox_ctx=sandbox_manager.create_sandbox(task),
    )

    assert "read_file" not in authorization.granted_tool_names
    assert any(denial.reason == "explicitly forbidden" for denial in authorization.denials)
