"""子智能体工厂：按 ``task_type`` 映射到对应的子智能体类（设计文档 §9）。

orchestrator 通过这个注册表把任务分发给正确的子智能体实现，不需要
在主循环里写一长串 if/elif 判断 task_type。
"""

from __future__ import annotations

from repro_agent.agents.base import BaseSubAgent
from repro_agent.agents.code.agent import CodeAnalysisAgent
from repro_agent.agents.coding.agent import CodingAgent
from repro_agent.agents.environment.agent import EnvironmentBuildAgent
from repro_agent.agents.experiment.agent import ExperimentExecutionAgent
from repro_agent.agents.paper.agent import PaperAnalysisAgent
from repro_agent.agents.reflection.agent import ReflectionAgent
from repro_agent.agents.resource.agent import ResourceCheckAgent
from repro_agent.agents.specification.agent import ExperimentSpecificationAgent
from repro_agent.agents.verification.agent import ResultVerificationAgent

SUB_AGENT_REGISTRY: dict[str, type[BaseSubAgent]] = {
    "paper_analysis": PaperAnalysisAgent,
    "code_analysis": CodeAnalysisAgent,
    "resource_check": ResourceCheckAgent,
    "specification": ExperimentSpecificationAgent,
    "environment_build": EnvironmentBuildAgent,
    "coding": CodingAgent,
    "experiment_execution": ExperimentExecutionAgent,
    "verification": ResultVerificationAgent,
    "reflection": ReflectionAgent,
}


def get_agent_class(task_type: str) -> type[BaseSubAgent]:
    agent_cls = SUB_AGENT_REGISTRY.get(task_type)
    if agent_cls is None:
        raise ValueError(f"no sub-agent registered for task_type='{task_type}'")
    return agent_cls
