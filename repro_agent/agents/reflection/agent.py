"""反思智能体（设计文档 §9.10、§11）。

职责（§9.10 严格限定）：
    只负责 1) 分析结果差距 2) 检查当前证据 3) 识别可能的错误来源
    4) 生成审计检查项 5) 对检查项排序 6) 向主智能体建议需要创建的
    任务 7) 判断差距更可能来自执行错误/配置差异/随机性/论文未披露
    细节。

**硬约束（§9.10）**：反思智能体不直接修改代码、不直接修改配置、
不直接运行正式实验、不宣布论文有问题、不宣布复现失败——这些都是
"决策"，只有主智能体才能做。本实现只授予该智能体只读工具（阅读
记忆/日志/证据），且返回值只包含"假设 + 建议审计任务清单"，不包含
任何执行动作，从返回类型上就杜绝了"反思智能体越权做决策"的可能。

检查维度覆盖 §11.3 的 A-E 五大类（论文理解/代码路径/参数/数据/
模型是否正确），用于生成结构化的审计检查项建议。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from repro_agent.agents.base import AgentRunResult, BaseSubAgent
from repro_agent.domain.experiment import MetricComparison
from repro_agent.llm_output import REFLECTION_SCHEMA, parse_structured_json

_CHECK_DIMENSIONS = {
    "A": "论文理解是否正确",
    "B": "代码路径是否正确",
    "C": "参数是否正确",
    "D": "数据是否正确",
    "E": "模型是否正确",
}


@dataclass
class SuggestedAuditTask:
    dimension: str
    description: str
    priority: int

    def to_dict(self) -> dict[str, Any]:
        return {"dimension": self.dimension, "description": self.description, "priority": self.priority}


@dataclass
class ReflectionAnalysis:
    likely_source: str = "unknown"  # "execution_error" | "config_difference" | "randomness" | "undisclosed_detail"
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    suggested_audit_tasks: list[SuggestedAuditTask] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "likely_source": self.likely_source,
            "hypotheses": self.hypotheses,
            "suggested_audit_tasks": [t.to_dict() for t in self.suggested_audit_tasks],
        }


class ReflectionAgent(BaseSubAgent):
    task_type = "reflection"
    system_prompt = (
        "你是 ReproAgent 系统的反思智能体。当复现结果与论文结果差距较大时，"
        "你需要分析可能原因，检查证据，识别可能的错误来源，生成结构化的"
        "审计检查项并排序，判断差距更可能来自执行错误、配置差异、随机性"
        "还是论文未披露细节。你绝对不能：修改代码、修改配置、运行正式"
        "实验、宣布论文有问题、宣布复现失败——这些决策只能由主智能体做出。"
        "你只能提出假设和建议，不能执行任何动作。"
    )

    def run(self) -> AgentRunResult:
        inputs = self.task.definition.inputs
        trigger_metrics = inputs.get("trigger_metrics", [])
        available_evidence = inputs.get("available_evidence", {})

        prompt = self._build_prompt(trigger_metrics, available_evidence)
        # 可用证据已经由主智能体在 inputs.available_evidence 中整理好，
        # 反思本身是纯假设生成，不需要模型再自己发起工具调用去翻记忆/
        # 日志（也避免反思智能体越权直接读取任意文件）。
        response = self.call_llm(
            prompt,
            temperature=0.4,
            tool_names=[],
            output_schema=REFLECTION_SCHEMA,
            output_schema_name="reflection_analysis",
        )
        analysis = self._parse_llm_output(response.content)

        result_payload = analysis.to_dict()
        self.write_json_output("result.json", result_payload)
        self.write_candidate_memory(self._render_candidate_memory(analysis))

        return AgentRunResult(succeeded=True, outputs=result_payload, candidate_memory_written=True)

    def _build_prompt(
        self, trigger_metrics: list[dict[str, Any]], available_evidence: dict[str, Any]
    ) -> str:
        return (
            f"触发反思的指标差距: {json.dumps(trigger_metrics, ensure_ascii=False)}\n"
            f"当前可用证据: {json.dumps(available_evidence, ensure_ascii=False)[:6000]}\n\n"
            "请分析可能的差距来源，覆盖以下检查维度：\n"
            + "\n".join(f"- {k}: {v}" for k, v in _CHECK_DIMENSIONS.items())
            + "\n\n请输出 JSON:\n"
            '{"likely_source": "execution_error|config_difference|randomness|undisclosed_detail", '
            '"hypotheses": [{"category": "A", "description": "...", "priority": 1, '
            '"confidence": 0.7, "required_checks": ["..."]}], '
            '"suggested_audit_tasks": [{"dimension": "B", "description": "...", "priority": 1}]}'
        )

    def _parse_llm_output(self, content: str) -> ReflectionAnalysis:
        data = parse_structured_json(
            content, REFLECTION_SCHEMA, label="reflection output"
        )
        audit_tasks = [
            SuggestedAuditTask(
                dimension=t.get("dimension", ""),
                description=t.get("description", ""),
                priority=int(t.get("priority", 0)),
            )
            for t in data["suggested_audit_tasks"]
        ]
        audit_tasks.sort(key=lambda t: t.priority, reverse=True)
        return ReflectionAnalysis(
            likely_source=data["likely_source"],
            hypotheses=data["hypotheses"],
            suggested_audit_tasks=audit_tasks,
        )

    def _render_candidate_memory(self, analysis: ReflectionAnalysis) -> str:
        lines = [
            f"# reflection.{self.task.task_id}",
            "",
            "## 摘要 (L1)",
            f"可能来源: {analysis.likely_source}; 建议审计任务数: {len(analysis.suggested_audit_tasks)}",
            "",
            "## 细节 (L2)",
        ]
        for h in analysis.hypotheses:
            lines.append(f"- [{h.get('category')}] {h.get('description')} (置信度: {h.get('confidence')})")
        lines.extend(["", "## 证据 (L3)"])
        for t in analysis.suggested_audit_tasks:
            lines.append(f"- 建议审计: [{t.dimension}] {t.description} (优先级 {t.priority})")
        return "\n".join(lines) + "\n"
