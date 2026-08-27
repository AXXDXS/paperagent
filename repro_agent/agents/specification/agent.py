"""实验规格子智能体（设计文档 §9.4）。

职责：把论文分析、代码分析和用户输入合并成统一的 ``ExperimentSpec``。
每个字段必须记录来源（``FieldProvenance``），**存在冲突时不能静默
覆盖**——本实现把冲突检测做成显式的、独立于 LLM 输出的确定性逻辑：
同一字段如果论文分析和代码分析给出的值不同，直接记录到
``ProvenancedField.conflicting_values``，不依赖 LLM 自己判断"要不要
报告冲突"（LLM 存在遗漏报告冲突的风险，这是一条不能只靠 Prompt
约束的规则，必须用代码兜底）。

风险预算：``specification`` -> RESTRICTED_WRITE（比纯分析任务多了
写自己输出文件的权限，但仍然不能执行命令/修改代码仓库）。
"""

from __future__ import annotations

import json
import hashlib
from typing import Any

from repro_agent.agents.base import AgentRunResult, BaseSubAgent
from repro_agent.domain.enums import FieldProvenance, ToleranceType
from repro_agent.domain.experiment import ExperimentSpec, ExpectedResult, ProvenancedField
from repro_agent.domain.resource_requirements import infer_required_resources
from repro_agent.orchestrator.runtime_configuration import normalize_requirements


# 来源优先级：数字越大优先级越高，用于"非冲突"情况下选择哪个值作为
# 主值（冲突情况下两个值都会被记录，不会被这个优先级掩盖）。
_PROVENANCE_PRIORITY = {
    FieldProvenance.USER_PROVIDED: 6,
    FieldProvenance.PAPER_EXPLICIT: 5,
    FieldProvenance.APPENDIX_EXPLICIT: 4,
    FieldProvenance.CODE_EFFECTIVE: 3,
    FieldProvenance.CODE_DEFAULT: 2,
    FieldProvenance.FRAMEWORK_DEFAULT: 1,
    FieldProvenance.AGENT_INFERRED: 0,
}


class ExperimentSpecificationAgent(BaseSubAgent):
    task_type = "specification"
    system_prompt = (
        "你是 ReproAgent 系统的实验规格子智能体。你的任务是把论文分析、"
        "代码分析和用户提供的信息合并成统一的实验复现规格，包括预期结果"
        "及其容差、模型/数据集/训练/评测配置。每个字段都必须标注来源类型。"
        "如果同一字段在不同来源之间出现冲突的值，你必须如实报告冲突，"
        "不能自行选择一个值静默覆盖另一个。"
    )

    def run(self) -> AgentRunResult:
        inputs = self.task.definition.inputs
        paper_findings: dict[str, Any] = inputs.get("paper_findings", {})
        code_findings: dict[str, Any] = inputs.get("code_findings", {})
        user_overrides: dict[str, Any] = inputs.get("user_overrides", {})
        experiment_id: str = inputs.get("experiment_id", "experiment")
        target_claim: str = inputs.get("target_claim", "reproduce_main_result")

        spec = ExperimentSpec(
            experiment_id=experiment_id,
            target_claim=target_claim,
            job_id=self.task.job_id,
        )

        self._merge_fields(spec, paper_findings, code_findings, user_overrides)
        self._merge_expected_results(spec, paper_findings, user_overrides)

        required_resources = infer_required_resources(
            paper_findings, code_findings, user_overrides
        )
        required_user_configuration = normalize_requirements(
            code_findings.get("required_user_configuration", [])
        )
        spec.resources = {
            "required": required_resources,
            "required_user_configuration": required_user_configuration,
        }

        unresolved = spec.unresolved_conflicts()

        result_payload = spec.to_dict()
        # Keep these two execution-gate inputs at the top level as well as in
        # ``resources``.  The top-level contract makes downstream orchestration
        # and persisted-job migration deterministic, while ``resources`` keeps
        # the canonical ExperimentSpec self-contained.
        result_payload["required_resources"] = required_resources
        result_payload["required_user_configuration"] = required_user_configuration
        result_payload["unresolved_conflicts"] = unresolved
        result_payload["audit_checks"] = [
            str(item)
            for item in inputs.get("required_checks", [])
            if str(item).strip()
        ]
        semantic_spec = {
            key: value
            for key, value in result_payload.items()
            if key not in {"spec_id", "created_at", "spec_digest"}
        }
        result_payload["spec_digest"] = hashlib.sha256(
            json.dumps(
                semantic_spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        result_payload["frozen"] = not unresolved
        self.write_json_output("result.json", result_payload)
        self.write_candidate_memory(self._render_candidate_memory(spec, unresolved))

        return AgentRunResult(succeeded=True, outputs=result_payload, candidate_memory_written=True)

    def _merge_fields(
        self,
        spec: ExperimentSpec,
        paper_findings: dict[str, Any],
        code_findings: dict[str, Any],
        user_overrides: dict[str, Any],
    ) -> None:
        """按 §9.4 的来源优先级合并字段，检测并记录冲突。"""

        candidates: dict[str, list[tuple[Any, FieldProvenance, str]]] = {}

        paper_parameters = paper_findings.get("effective_parameters") or {}
        # Version-1 paper results only exposed a rich list. Normalize it here so
        # persisted jobs remain resumable after the contract upgrade.
        if not paper_parameters:
            paper_parameters = {
                item.get("name"): item.get("value")
                for item in (paper_findings.get("extracted_parameters") or [])
                if isinstance(item, dict) and item.get("name")
            }
        for name, value in paper_parameters.items():
            candidates.setdefault(name, []).append((value, FieldProvenance.PAPER_EXPLICIT, "paper_analysis"))
        for name, value in (code_findings.get("effective_parameters") or {}).items():
            candidates.setdefault(name, []).append((value, FieldProvenance.CODE_EFFECTIVE, "code_analysis"))
        for name, value in user_overrides.items():
            candidates.setdefault(name, []).append((value, FieldProvenance.USER_PROVIDED, "user_input"))

        for name, entries in candidates.items():
            # 按优先级排序，取最高优先级作为主值
            entries_sorted = sorted(
                entries, key=lambda e: _PROVENANCE_PRIORITY.get(e[1], -1), reverse=True
            )
            primary_value, primary_provenance, primary_ref = entries_sorted[0]

            distinct_values = {str(v) for v, _, _ in entries}
            conflicts = []
            user_resolved = primary_provenance == FieldProvenance.USER_PROVIDED
            if len(distinct_values) > 1 and not user_resolved:
                for value, provenance, ref in entries_sorted[1:]:
                    if value != primary_value:
                        conflicts.append(
                            {"value": value, "provenance": provenance.value, "source_ref": ref}
                        )

            spec.fields[name] = ProvenancedField(
                value=primary_value,
                provenance=primary_provenance,
                source_ref=primary_ref,
                confidence=1.0 if not conflicts else 0.5,
                conflicting_values=conflicts,
            )

    def _merge_expected_results(
        self,
        spec: ExperimentSpec,
        paper_findings: dict[str, Any],
        user_overrides: dict[str, Any],
    ) -> None:
        expected = (user_overrides.get("expected_results")
                    or paper_findings.get("expected_results")
                    or {})
        for metric, meta in expected.items():
            try:
                tolerance_type = ToleranceType(meta.get("tolerance_type", "absolute"))
            except ValueError:
                tolerance_type = ToleranceType.ABSOLUTE
            spec.expected_results[metric] = ExpectedResult(
                metric=metric,
                value=float(meta.get("value", 0.0)),
                tolerance_type=tolerance_type,
                tolerance=float(meta.get("tolerance", 0.0)),
                tolerance_basis=meta.get("tolerance_basis", ""),
            )

    def _render_candidate_memory(self, spec: ExperimentSpec, unresolved: list[str]) -> str:
        lines = [
            f"# specification.{self.task.task_id}",
            "",
            "## 摘要 (L1)",
            f"实验 {spec.experiment_id}，共 {len(spec.fields)} 个字段，"
            f"{len(unresolved)} 个未解决冲突，"
            f"{len(spec.resources.get('required', []))} 个运行必需资源。",
            "",
            "## 细节 (L2)",
        ]
        for name, field_ in spec.fields.items():
            lines.append(f"- {name} = {field_.value} (来源: {field_.provenance.value})")
        for resource in spec.resources.get("required", []):
            lines.append(
                f"- required_resource: {resource['kind']} {resource['name']} "
                f"(来源: {resource.get('source_ref', '')})"
            )
        lines.extend(["", "## 证据 (L3)"])
        for name in unresolved:
            field_ = spec.fields[name]
            lines.append(f"- 冲突字段 [{name}]: 主值={field_.value}, 其他={field_.conflicting_values}")
        return "\n".join(lines) + "\n"
