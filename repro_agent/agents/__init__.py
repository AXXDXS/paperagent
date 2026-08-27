"""十个子智能体（设计文档 §9）。

所有子智能体都继承 ``BaseSubAgent``（见 base.py），只能通过构造函数
注入的 ``ToolAuthorization`` 调用被主智能体授权的工具，详见
``tools/authorization.py`` 顶部的完整设计说明。
"""

from repro_agent.agents.base import AgentRunResult, BaseSubAgent
from repro_agent.agents.registry import SUB_AGENT_REGISTRY, get_agent_class

__all__ = [
    "SUB_AGENT_REGISTRY",
    "AgentRunResult",
    "BaseSubAgent",
    "get_agent_class",
]
