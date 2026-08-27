"""论文分析子智能体（设计文档 §9.1，正文/附录拆分版）。

职责：解析论文结构、提取方法概要与复现参数。按用户方案，论文不再由
单个子智能体一次性阅读，而是拆成两类并行任务：

- ``scope="body"``（正文任务）：阅读正文页范围，输出
  1. ``method_summary`` —— 方法/流程 Pipeline 的总结性描述；
  2. ``parameters`` —— 复现所需四类参数（训练/模型/数据/评测）的
     名称 + 值（附轻量证据字段：页码、置信度）；
  3. ``expected_results`` —— 主实验指标（验证闭环的比对基准）。

- ``scope="appendix"``（附录任务）：阅读附录页范围/附录文件，**只**
  输出与复现直接相关的训练参数；补充性说明、探索性实验、健壮性
  论证等一律忽略。不输出方法概要与主实验指标。

两类任务共享同一个输出瘦身契约：不再要求模型抄写原文引用
（``original_text``），证据字段收敛为 page + confidence，provenance
由代码按 scope 确定性填充（正文 → PAPER_EXPLICIT，附录 →
APPENDIX_EXPLICIT），下游 ``merge_paper_findings`` 依赖该优先级
（正文显式值 > 附录显式值 > 代码有效值）合并。

风险预算：``paper_analysis`` -> READ_ONLY（见
``tools/authorization.py::TASK_TYPE_RISK_BUDGET``），因此本子智能体
默认只能拿到只读工具；论文分析本质上是纯读取+推理，唯一写权限是
``write_task_output``（所有任务类型默认授予的最小写权限）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repro_agent.agents.base import AgentRunResult, BaseSubAgent
from repro_agent.domain.enums import FieldProvenance, ToleranceType
from repro_agent.llm_output import (
    PAPER_ANALYSIS_SCHEMA,
    StructuredOutputError,
    parse_structured_json,
)


_TOLERANCE_TYPES = frozenset(item.value for item in ToleranceType)
_THOUSANDS_SEPARATED = re.compile(r"-?\d{1,3}(,\d{3})+(?:\.\d+)?")


def _coerce_number(value: Any) -> Any:
    """Coerce unambiguous numeric strings to numbers, else return unchanged.

    Models routinely emit metrics as ``"28.0"``, ``"28.0%"`` or ``"1,234.5"``.
    Only presentations that decode deterministically are rewritten; anything
    else is returned untouched so the strict schema check can reject it.
    """

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip().rstrip("%％").strip()
    if not text:
        return value
    if _THOUSANDS_SEPARATED.fullmatch(text):
        text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return value


def _stringify_page(value: Any) -> Any:
    """Coerce integer/float page numbers to the string form the schema requires."""

    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    return value


def normalize_paper_analysis_payload(data: Any) -> Any:
    """Deterministic presentation fixes applied before strict schema validation.

    Real-model responses frequently express numbers as strings (``"28.0"``,
    ``"28.0%"``, ``"1,234.5"``), pages as integers, and enum values with
    unexpected casing.  These are presentation variants of a valid answer,
    not guesses: coercion only rewrites values that decode unambiguously,
    and ``parse_structured_json`` still applies the strict local schema
    afterwards, so anything unfixable fails closed exactly as before.
    """

    if not isinstance(data, dict):
        return data
    parameters = data.get("parameters")
    if isinstance(parameters, list):
        for item in parameters:
            if not isinstance(item, dict):
                continue
            if "confidence" in item:
                item["confidence"] = _coerce_number(item["confidence"])
            if "page" in item:
                item["page"] = _stringify_page(item["page"])
    expected = data.get("expected_results")
    if isinstance(expected, dict):
        for meta in expected.values():
            if not isinstance(meta, dict):
                continue
            for key in ("value", "tolerance"):
                if key in meta:
                    meta[key] = _coerce_number(meta[key])
            tolerance_type = meta.get("tolerance_type")
            if isinstance(tolerance_type, str):
                folded = (
                    tolerance_type.strip().lower().replace(" ", "_").replace("-", "_")
                )
                if folded in _TOLERANCE_TYPES:
                    meta["tolerance_type"] = folded
    return data


@dataclass
class ExtractedParameter:
    """§9.1 参数提取格式（瘦身后）。original_text/section 仅为兼容旧
    持久化产物保留字段，新契约不再要求模型填写。"""

    name: str
    value: Any
    experiment_scope: str
    provenance: FieldProvenance
    page: str = ""
    section: str = ""
    original_text: str = ""
    confidence: float = 1.0
    is_inferred: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "experiment_scope": self.experiment_scope,
            "provenance": self.provenance.value,
            "page": self.page,
            "section": self.section,
            "original_text": self.original_text,
            "confidence": self.confidence,
            "is_inferred": self.is_inferred,
        }


_TYPE_RULES_NOTE = (
    "输出类型硬性要求: 数字类 value/confidence/tolerance 必须是 JSON 数字"
    "（不加引号、百分号或千分位逗号，如 28.0）；文字类 value 用简短字符串；"
    "page 必须是字符串（如 \"5\"）；tolerance_type 只能取 "
    "absolute/relative/std_multiple。\n"
)


class PaperAnalysisAgent(BaseSubAgent):
    task_type = "paper_analysis"
    system_prompt = (
        "你是 ReproAgent 系统的论文分析子智能体。你阅读论文的一个指定部分"
        "（正文或附录），提取方法概要与复现所需参数。你只能使用被授权的"
        "只读工具获取论文内容，不能执行任何写入或命令类操作。输出必须是"
        "严格的 JSON，不要抄写原文整句，参数值保持简短精确。"
    )

    # ---- 输入契约 ----
    # scope: "body" | "appendix"（旧任务缺省视为 body 全文，向后兼容）
    # page_range: [start, end]（原文页码，闭区间），仅作用于主论文 PDF

    def run(self) -> AgentRunResult:
        inputs = self.task.definition.inputs
        paper_path = inputs.get("paper_path", "")
        appendix_paths = inputs.get("appendix_paths", [])
        supplementary_paths = inputs.get("supplementary_paths", [])
        declared_files = inputs.get("files", [])
        target_experiments = inputs.get("target_experiments", [])
        scope = inputs.get("scope") or "body"
        page_range = inputs.get("page_range") or None

        if scope not in ("body", "appendix"):
            raise StructuredOutputError(
                f"paper_analysis task has unknown scope {scope!r} (expected 'body' or 'appendix')"
            )

        sources = [paper_path, *appendix_paths, *supplementary_paths, *declared_files]
        # Preserve order while removing duplicates.  All paths have already been
        # staged into the task's read-only input directory by SandboxManager.
        sources = list(dict.fromkeys(path for path in sources if path))
        paper_text = self._read_sources(
            sources, main_paper_path=paper_path, page_range=page_range
        )

        # 论文内容已经通过 read_pdf_text 读入 prompt；模型在阅读提取文本
        # 时如果对某个具体位置（典型如表格列对应关系）确实无法理解，
        # 允许它调用 inspect_pdf_page 查看那一页的原始版面做仲裁。
        # 按实际授权过滤是为了兼容旧持久化 Job（授权清单里还没有该工具
        # 时自动退回旧行为）。
        inspection_tools = [
            name for name in ("inspect_pdf_page",) if name in self.granted_tools
        ]
        prompt = self._build_prompt(
            paper_text,
            target_experiments,
            scope=scope,
            page_range=page_range,
            inspection_available=bool(inspection_tools),
            audit_hypothesis=str(inputs.get("audit_hypothesis", "")),
            required_checks=list(inputs.get("required_checks", []) or []),
        )
        response = self.call_llm(
            prompt,
            temperature=0.2,
            tool_names=inspection_tools,
            output_schema=PAPER_ANALYSIS_SCHEMA,
            output_schema_name="paper_analysis",
        )

        (
            parameters,
            method_summary,
            extraction_notes,
            expected_results,
        ) = self._parse_llm_output(response.content, scope=scope)

        effective_parameters = {
            parameter.name: parameter.value for parameter in parameters if parameter.name
        }

        result_payload: dict[str, Any] = {
            "scope": scope,
            "paper_path": paper_path,
            "page_range": list(page_range) if page_range else None,
            "source_files": sources,
            "target_experiments": target_experiments,
            "method_summary": method_summary,
            "extracted_parameters": [p.to_dict() for p in parameters],
            # Stable downstream contract.  Keep extracted_parameters for rich
            # evidence, while specification consumes this normalized mapping.
            "effective_parameters": effective_parameters,
            "expected_results": expected_results,
            "notes": extraction_notes,
        }
        self.write_json_output("result.json", result_payload)
        self.write_candidate_memory(
            self._render_candidate_memory(scope, method_summary, parameters)
        )

        return AgentRunResult(
            succeeded=True,
            outputs=result_payload,
            candidate_memory_written=True,
            raw_llm_responses=[response.content],
        )

    def _read_sources(
        self,
        source_paths: list[str],
        *,
        main_paper_path: str,
        page_range: list[int] | tuple[int, int] | None,
    ) -> str:
        """Read every declared source exactly once.

        The page range applies to the main paper PDF only; standalone
        appendix/supplement files are read in full.  A bounded per-source
        excerpt prevents one large source from starving the others out of
        the prompt.
        """

        chunks: list[str] = []
        per_source_limit = max(8_000, 80_000 // max(1, len(source_paths)))
        for index, source_path in enumerate(source_paths):
            tool_name = (
                "read_pdf_text"
                if Path(source_path).suffix.lower() == ".pdf"
                else "read_file"
            )
            kwargs: dict[str, Any] = {"path": source_path}
            if tool_name == "read_pdf_text" and source_path == main_paper_path and page_range:
                kwargs["start_page"] = int(page_range[0])
                kwargs["end_page"] = int(page_range[1])
            result = self.call_tool_checkpointed(
                f"paper_source_{index}", tool_name, **kwargs
            )
            content = result.get("content", "")[:per_source_limit]
            chunks.append(f"\n===== SOURCE: {source_path} =====\n{content}")
        return "\n".join(chunks)

    def _build_prompt(
        self,
        paper_text: str,
        target_experiments: list[str],
        *,
        scope: str = "body",
        page_range: list[int] | tuple[int, int] | None = None,
        inspection_available: bool = False,
        audit_hypothesis: str = "",
        required_checks: list[str] | None = None,
    ) -> str:
        tool_note = ""
        if inspection_available:
            tool_note = (
                "关于工具：你有一个 inspect_pdf_page 工具，可以查看某个 PDF 页的"
                "原始版面文本（保留列与表格对齐）。它是最后手段：只有当提取文本中"
                "某个具体位置（如表格数值与表头的对应关系、公式符号断裂）"
                "确实无法理解、且无法从上下文推断时才调用它，调用时必须在 "
                "reason 参数里写明具体哪里无法理解。\n\n"
            )
        audit_note = ""
        if audit_hypothesis.strip() or required_checks:
            audit_note = (
                f"本次是证据审计。审计假设: {audit_hypothesis.strip()}\n"
                f"必须逐项核查: {required_checks or []}\n"
                "只基于文中可定位证据提取；无法确认的检查项要在 notes 中明确说明。\n\n"
            )

        if scope == "appendix":
            task_instruction = (
                "请阅读以下论文附录材料，只提取与复现实验直接相关的训练参数。\n\n"
                "明确忽略（不要输出）：补充性背景说明、相关工作的扩展讨论、"
                "探索性/附加实验、健壮性或泛化性分析、证明与推导、与目标实验"
                "无关的数据展示。\n\n"
                "重点提取：训练超参数（学习率、轮数、batch size、优化器等）、"
                "训练流程实现细节、数据构造与预处理细节、环境与硬件配置，"
                "以及目标实验在附录中的补充数值。\n\n"
                f"目标复现实验: {target_experiments}（作为相关性锚点，"
                "与它无关的附录内容一律忽略）\n\n"
                "请以 JSON 格式输出，格式为:\n"
                '{"parameters": [{"name": "...", "value": "...", '
                '"page": "15", "confidence": 0.9}], "notes": "..."}\n'
                "（附录任务不要输出 method_summary 和 expected_results。）\n\n"
                f"{_TYPE_RULES_NOTE}\n"
                f"附录内容：\n{paper_text[:80000]}"
            )
            return tool_note + audit_note + task_instruction

        range_note = ""
        if page_range:
            range_note = f"（正文页范围：第 {page_range[0]}-{page_range[1]} 页）"
        return (
            tool_note
            + audit_note
            + "请阅读以下论文正文"
            + range_note
            + "，完成三项任务：\n\n"
            "一、方法/流程概要（method_summary）：用 5-10 句话总结论文的方法 "
            "Pipeline——整体流程、核心组件及其衔接关系、训练/推理的执行顺序。"
            "只描述方法结构，不要罗列参数数值。\n\n"
            "二、复现参数提取（parameters）：提取复现实验所需的全部参数，覆盖"
            "四类——训练参数（学习率、轮数、batch size、优化器等）、模型参数"
            "（架构、规模、预训练权重、解码温度等）、数据参数（数据集、划分、"
            "预处理）、评测参数（指标定义、评测协议、few-shot 设置等）。\n\n"
            "三、主实验指标（expected_results）：从正文主结果表提取目标实验的"
            f"指标。目标复现实验: {target_experiments}\n\n"
            "附录中通常还有更多训练细节，由并行的附录任务负责；本任务只处理"
            "正文，不要臆造附录内容。\n\n"
            "请以 JSON 格式输出，格式为:\n"
            '{"method_summary": "...", '
            '"parameters": [{"name": "...", "value": "...", '
            '"page": "5", "confidence": 0.9}], '
            '"expected_results": {"metric_name": {"value": 0.0, '
            '"tolerance_type": "absolute", "tolerance": 0.0, '
            '"tolerance_basis": "paper reported variance or explicit user rule"}}, '
            '"notes": "..."}\n\n'
            f"{_TYPE_RULES_NOTE}\n"
            f"论文正文内容：\n{paper_text[:80000]}"
        )

    def _parse_llm_output(
        self, content: str, *, scope: str = "body"
    ) -> tuple[list[ExtractedParameter], str, str, dict[str, Any]]:
        data = parse_structured_json(
            content,
            PAPER_ANALYSIS_SCHEMA,
            label="paper analysis output",
            normalize=normalize_paper_analysis_payload,
        )

        # provenance 由 scope 确定性决定，不依赖模型自报：正文显式值优先级
        # 高于附录显式值（见 specification._PROVENANCE_PRIORITY）。
        provenance = (
            FieldProvenance.APPENDIX_EXPLICIT
            if scope == "appendix"
            else FieldProvenance.PAPER_EXPLICIT
        )

        parameters = []
        for item in data.get("parameters", []):
            confidence = item.get("confidence", 1.0)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 1.0
            parameters.append(
                ExtractedParameter(
                    name=item.get("name", ""),
                    value=item.get("value"),
                    experiment_scope=scope,
                    provenance=provenance,
                    page=str(item.get("page", "") or ""),
                    section=str(item.get("section", "") or ""),
                    original_text=str(item.get("original_text", "") or ""),
                    confidence=confidence,
                    is_inferred=bool(item.get("is_inferred", False)),
                )
            )

        expected_results = data.get("expected_results") or {}
        method_summary = str(data.get("method_summary", "") or "")

        if scope == "body" and not expected_results:
            # 验证闭环（ResultVerificationAgent）以 expected_results 为比对
            # 基准：正文任务拿不到任何主实验指标意味着复现成功与否无法
            # 判定，fail-closed 让任务失败进入重试，而不是静默产出一份
            # 无法验证的"成功"结果。
            raise StructuredOutputError(
                "paper analysis output is invalid: body scope returned no "
                "expected_results; the verification loop has no metric baseline"
            )

        return parameters, method_summary, data.get("notes", ""), expected_results

    def _render_candidate_memory(
        self,
        scope: str,
        method_summary: str,
        parameters: list[ExtractedParameter],
    ) -> str:
        lines = [
            f"# paper.{self.task.task_id}",
            "",
            "## 摘要 (L1)",
            f"[{scope}] 提取到 {len(parameters)} 个参数"
            + (f"，方法概要 {len(method_summary)} 字。" if method_summary else "。"),
        ]
        if method_summary:
            lines.extend(["", "## 方法概要", method_summary])
        lines.extend(["", "## 细节 (L2)"])
        for p in parameters:
            lines.append(
                f"- {p.name} = {p.value} (来源: {p.provenance.value}, "
                f"页码: {p.page or '?'}, 置信度: {p.confidence})"
            )
        return "\n".join(lines) + "\n"
