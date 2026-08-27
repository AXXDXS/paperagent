"""工具（Tool）系统：文件查找/阅读/资源检查等能力的统一封装与分级授权。

核心设计（响应用户需求"文件查找、文件阅读、资源阅读等封装为工具，
子 agent 只能使用主 agent 传给他的、低风险的工具"）：

    1. 所有"动手能力"都注册为 ``ToolSpec``，携带显式的
       ``ToolRiskLevel``（只读 / 受限写 / 高危），见 ``base.py``。
    2. 全局 ``ToolRegistry``（``registry.py``）只应被主智能体侧的
       ``ToolAuthorizer`` 访问；子智能体运行时代码不导入它。
    3. 主智能体通过 ``ToolAuthorizer.authorize()`` 依据任务的
       ``allowed_tools`` 白名单和任务类型的风险预算，生成一个
       裁剪过的 ``ToolAuthorization``，这才是真正下发给子智能体的
       对象——子智能体只能调用被授予的工具，调用未授权工具会抛出
       ``ToolPermissionError``。

详见 ``authorization.py`` 顶部的完整设计说明，以及仓库根目录
``paper_agent/CHANGES_AND_DESIGN_NOTES.md`` 中的"工具权限体系"章节。
"""

from repro_agent.tools.authorization import (
    TASK_TYPE_RISK_BUDGET,
    ToolAuthorization,
    ToolAuthorizer,
    ToolDenial,
    risk_allowed,
)
from repro_agent.tools.base import (
    INVALID_TOOL_OUTPUT,
    InvalidToolOutputError,
    SandboxContext,
    ToolExecutionError,
    ToolInputValidationError,
    ToolInvocationLog,
    ToolPermissionError,
    ToolRiskLevel,
    ToolOutputSpec,
    ToolSpec,
)
from repro_agent.tools.registry import ToolRegistry, default_registry

__all__ = [
    "TASK_TYPE_RISK_BUDGET",
    "SandboxContext",
    "ToolAuthorization",
    "ToolAuthorizer",
    "ToolDenial",
    "ToolExecutionError",
    "InvalidToolOutputError",
    "INVALID_TOOL_OUTPUT",
    "ToolInputValidationError",
    "ToolInvocationLog",
    "ToolPermissionError",
    "ToolRegistry",
    "ToolRiskLevel",
    "ToolOutputSpec",
    "ToolSpec",
    "default_registry",
    "risk_allowed",
]
