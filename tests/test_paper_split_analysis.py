"""论文分析"正文 + 附录"拆分方案的单元测试。

覆盖链条上的每一环：
    probe_paper_structure（附录边界探测）
    -> extract_pdf_text 页范围（原文页码保留）
    -> InitialPlanner 任务拆分（正文 + 附录，超长分片）
    -> PaperAnalysisAgent scope 感知（prompt 差异、页范围工具参数、
       body 的 expected_results fail-closed）
    -> merge_paper_findings（正文优先、附录补缺、审计信息）
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from repro_agent.agents.paper.agent import PaperAnalysisAgent
from repro_agent.domain.job import JobInputs, ReproductionJob
from repro_agent.llm_output import StructuredOutputError
from repro_agent.orchestrator.artifacts import merge_paper_findings
from repro_agent.orchestrator.planner import (
    APPENDIX_CHUNK_CHAR_LIMIT,
    InitialPlanner,
    _split_appendix_pages,
)
from repro_agent.orchestrator.task_factory import build_task_definition
from repro_agent.paper_input import PaperStructure, extract_pdf_text, probe_paper_structure
from repro_agent.providers.base import LLMResponse
from repro_agent.providers.mock import MockLLMProvider
from repro_agent.sandbox.manager import SandboxManager
from repro_agent.tools.authorization import ToolAuthorizer


# --------------------------------------------------------------------------
# 测试 PDF 生成器：pypdf 手写内容流（BT/Tj/T*），足以让 extract_text
# 还原出行结构，供探测/页范围/agent 全链路测试使用。
# --------------------------------------------------------------------------

def _make_pdf(path: Path, pages: list[list[str]]) -> Path:
    writer = PdfWriter()
    font = DictionaryObject()
    font[NameObject("/Type")] = NameObject("/Font")
    font[NameObject("/Subtype")] = NameObject("/Type1")
    font[NameObject("/BaseFont")] = NameObject("/Helvetica")
    font_ref = writer._add_object(font)

    for lines in pages:
        page = writer.add_blank_page(width=612, height=792)
        resources = DictionaryObject()
        fonts = DictionaryObject()
        fonts[NameObject("/F1")] = font_ref
        resources[NameObject("/Font")] = fonts
        page[NameObject("/Resources")] = resources
        parts = ["BT", "/F1 12 Tf", "14 TL", "1 0 0 1 72 720 Tm"]
        for index, line in enumerate(lines):
            if index:
                parts.append("T*")
            escaped = (
                line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            )
            parts.append(f"({escaped}) Tj")
        parts.append("ET")
        stream = DecodedStreamObject()
        stream.set_data("\n".join(parts).encode("latin-1", "replace"))
        page[NameObject("/Contents")] = writer._add_object(stream)

    with open(path, "wb") as handle:
        writer.write(handle)
    return path


@pytest.fixture()
def paper_with_appendix(tmp_path: Path) -> Path:
    """5 页论文：3 页正文 + References + 2 页附录。"""
    return _make_pdf(
        tmp_path / "paper.pdf",
        [
            ["1 Introduction", "We study learning agents."],
            ["2 Method", "Our method uses three rounds of experience collection."],
            ["3 Experiments", "Main results are reported in Table 1."],
            ["References", "Author 2024. A paper about agents."],
            ["Appendix", "A Hyperparameters", "learning rate is 3e-4"],
            ["A.2 More details", "batch size is 32"],
        ],
    )


@pytest.fixture()
def paper_without_appendix(tmp_path: Path) -> Path:
    return _make_pdf(
        tmp_path / "no_appendix.pdf",
        [
            ["1 Introduction", "Short paper."],
            ["References", "Author 2024."],
        ],
    )


# --------------------------------------------------------------------------
# 结构探测与页范围提取
# --------------------------------------------------------------------------

def test_probe_detects_appendix_after_references(paper_with_appendix: Path) -> None:
    structure = probe_paper_structure(paper_with_appendix)

    assert structure.total_pages == 6
    assert structure.appendix_start_page == 5
    assert structure.body_pages == (1, 4)
    assert structure.appendix_pages == (5, 6)
    assert structure.has_appendix
    assert structure.appendix_char_count > 0


def test_probe_without_appendix_falls_back_to_whole_document(paper_without_appendix: Path) -> None:
    structure = probe_paper_structure(paper_without_appendix)

    assert structure.has_appendix is False
    assert structure.appendix_pages is None
    assert structure.body_pages == (1, 2)


def test_extract_pdf_text_page_range_keeps_original_page_numbers(paper_with_appendix: Path) -> None:
    text = extract_pdf_text(paper_with_appendix, start_page=5, end_page=6)

    assert "--- Page 5 ---" in text
    assert "--- Page 6 ---" in text
    assert "--- Page 1 ---" not in text
    assert "Hyperparameters" in text

    with pytest.raises(Exception, match="invalid page range"):
        extract_pdf_text(paper_with_appendix, start_page=5, end_page=99)


# --------------------------------------------------------------------------
# 规划器拆分
# --------------------------------------------------------------------------

def _job(paper_path: str) -> ReproductionJob:
    return ReproductionJob(
        inputs=JobInputs(
            paper_path=paper_path,
            repository_path="/nonexistent/repo",
            target_experiments=["main_experiment"],
        )
    )


def test_planner_splits_paper_into_body_and_appendix_tasks(paper_with_appendix: Path) -> None:
    tasks = InitialPlanner().plan_initial_tasks(_job(str(paper_with_appendix)))

    paper_tasks = [t for t in tasks if t.definition.task_type == "paper_analysis"]
    assert len(paper_tasks) == 2

    body = next(t for t in paper_tasks if t.definition.inputs["scope"] == "body")
    appendix = next(t for t in paper_tasks if t.definition.inputs["scope"] == "appendix")

    assert body.definition.inputs["page_range"] == [1, 4]
    assert appendix.definition.inputs["page_range"] == [5, 6]
    assert body.definition.inputs["creation_key"] == "initial:paper_analysis:body"
    assert appendix.definition.inputs["creation_key"] == "initial:paper_analysis:appendix:0"
    # 两任务零依赖 -> 可并行
    assert body.definition.dependencies == []
    assert appendix.definition.dependencies == []

    # 规格必须汇总正文与附录；资源门禁随后消费规格，而不是提前检查。
    specification = next(
        t for t in tasks if t.definition.task_type == "specification"
    )
    resource = next(t for t in tasks if t.definition.task_type == "resource_check")
    assert {body.task_id, appendix.task_id}.issubset(
        set(specification.definition.dependencies)
    )
    assert resource.definition.dependencies == [specification.task_id]


def test_planner_without_appendix_falls_back_to_single_task(paper_without_appendix: Path) -> None:
    tasks = InitialPlanner().plan_initial_tasks(_job(str(paper_without_appendix)))

    paper_tasks = [t for t in tasks if t.definition.task_type == "paper_analysis"]
    assert len(paper_tasks) == 1
    assert paper_tasks[0].definition.inputs["scope"] == "body"
    assert "page_range" not in paper_tasks[0].definition.inputs


def test_planner_non_pdf_paper_uses_single_task(tmp_path: Path) -> None:
    text_paper = tmp_path / "paper.txt"
    text_paper.write_text("plain text paper", encoding="utf-8")

    tasks = InitialPlanner().plan_initial_tasks(_job(str(text_paper)))
    paper_tasks = [t for t in tasks if t.definition.task_type == "paper_analysis"]
    assert len(paper_tasks) == 1


def test_appendix_chunking_splits_by_pages_and_caps_chunks() -> None:
    # 100k 字符附录 -> ceil(100k/40k)=3 片；页数 30 -> 每片 10 页。
    structure = PaperStructure(
        total_pages=40,
        appendix_start_page=11,
        body_pages=(1, 10),
        appendix_pages=(11, 40),
        appendix_char_count=100_000,
    )
    ranges = _split_appendix_pages(
        structure, char_limit=APPENDIX_CHUNK_CHAR_LIMIT, max_chunks=4
    )
    assert len(ranges) == 3
    assert ranges[0] == (11, 20)
    assert ranges[-1][1] == 40
    # 覆盖完整、无重叠
    covered = [page for start, end in ranges for page in range(start, end + 1)]
    assert covered == list(range(11, 41))

    # 超限封顶：1000k 字符也最多 4 片
    huge = PaperStructure(
        total_pages=400,
        appendix_start_page=11,
        body_pages=(1, 10),
        appendix_pages=(11, 400),
        appendix_char_count=1_000_000,
    )
    assert (
        len(
            _split_appendix_pages(
                huge, char_limit=APPENDIX_CHUNK_CHAR_LIMIT, max_chunks=4
            )
        )
        == 4
    )


# --------------------------------------------------------------------------
# 子智能体 scope 行为
# --------------------------------------------------------------------------

def _build_agent(tmp_path: Path, pdf_path: Path, scope: str, page_range, provider):
    task_inputs = {
        "paper_path": str(pdf_path),
        "target_experiments": ["main_experiment"],
        "scope": scope,
        "files": [str(pdf_path)],
        "creation_key": f"test:{scope}",
    }
    if page_range:
        task_inputs["page_range"] = list(page_range)

    from repro_agent.domain.task import Task

    task = Task(
        job_id="job_split_test",
        definition=build_task_definition(
            objective="test paper analysis",
            task_type="paper_analysis",
            inputs=task_inputs,
            restrict_tools=["read_file", "read_pdf_text", "inspect_pdf_page"],
        ),
    )
    sandbox = SandboxManager(tmp_path / "sandboxes").create_sandbox(task)
    authorization = ToolAuthorizer().authorize(
        task_id=task.task_id,
        task_type="paper_analysis",
        allowed_tools=task.definition.allowed_tools,
        sandbox_ctx=sandbox,
    )
    agent = PaperAnalysisAgent(task, authorization, provider, model="mock-model")
    return agent, authorization


def test_appendix_agent_reads_page_range_and_returns_training_params(
    tmp_path: Path, paper_with_appendix: Path
) -> None:
    provider = MockLLMProvider(
        fallback_response=LLMResponse(
            content=json.dumps(
                {
                    "parameters": [
                        {"name": "learning_rate", "value": 3e-4, "page": "5", "confidence": 0.9}
                    ],
                    "notes": "appendix only",
                }
            )
        )
    )
    agent, authorization = _build_agent(
        tmp_path, paper_with_appendix, "appendix", [5, 6], provider
    )
    result = agent.run()

    assert result.succeeded
    payload = result.outputs
    assert payload["scope"] == "appendix"
    assert payload["page_range"] == [5, 6]
    assert payload["extracted_parameters"][0]["provenance"] == "APPENDIX_EXPLICIT"
    assert "method_summary" not in payload or payload.get("method_summary") == ""

    # 工具调用带了页范围参数，且只读取了附录页
    pdf_calls = [
        log for log in authorization.invocation_log if log.tool_name == "read_pdf_text"
    ]
    assert pdf_calls, "appendix agent must read the PDF via read_pdf_text"
    assert pdf_calls[0].arguments.get("start_page") == 5
    assert pdf_calls[0].arguments.get("end_page") == 6

    # 附录 prompt：训练参数聚焦 + 忽略清单；输出格式示例不要求 method_summary
    prompt = provider.call_log[0][-1].content
    assert "训练参数" in prompt
    assert "明确忽略" in prompt
    assert '{"method_summary"' not in prompt
    assert '{"parameters"' in prompt


def test_body_agent_requires_expected_results_fail_closed(
    tmp_path: Path, paper_with_appendix: Path
) -> None:
    provider = MockLLMProvider(
        fallback_response=LLMResponse(
            content=json.dumps({"parameters": [{"name": "seed", "value": 42}]})
        )
    )
    agent, _ = _build_agent(tmp_path, paper_with_appendix, "body", [1, 4], provider)

    with pytest.raises(StructuredOutputError, match="expected_results"):
        agent.run()


def test_body_agent_returns_method_summary_and_paper_explicit_params(
    tmp_path: Path, paper_with_appendix: Path
) -> None:
    provider = MockLLMProvider(
        fallback_response=LLMResponse(
            content=json.dumps(
                {
                    "method_summary": "Agent collects experiences, extracts insights, "
                    "and retrieves them at evaluation time.",
                    "parameters": [
                        {"name": "rounds", "value": 3, "page": "2", "confidence": 0.95}
                    ],
                    "expected_results": {
                        "success_rate": {
                            "value": 0.59,
                            "tolerance_type": "absolute",
                            "tolerance": 0.02,
                        }
                    },
                    "notes": "body only",
                }
            )
        )
    )
    agent, authorization = _build_agent(
        tmp_path, paper_with_appendix, "body", [1, 4], provider
    )
    result = agent.run()

    assert result.succeeded
    payload = result.outputs
    assert payload["scope"] == "body"
    assert payload["page_range"] == [1, 4]
    assert "collects experiences" in payload["method_summary"]
    assert payload["extracted_parameters"][0]["provenance"] == "PAPER_EXPLICIT"
    assert payload["expected_results"]["success_rate"]["value"] == 0.59
    assert payload["effective_parameters"] == {"rounds": 3}

    pdf_calls = [
        log for log in authorization.invocation_log if log.tool_name == "read_pdf_text"
    ]
    assert pdf_calls[0].arguments.get("start_page") == 1
    assert pdf_calls[0].arguments.get("end_page") == 4

    prompt = provider.call_log[0][-1].content
    assert "方法/流程概要" in prompt
    assert "expected_results" in prompt


# --------------------------------------------------------------------------
# 合并逻辑
# --------------------------------------------------------------------------

def _body_payload() -> dict:
    return {
        "scope": "body",
        "page_range": [1, 12],
        "method_summary": "pipeline summary",
        "extracted_parameters": [
            {"name": "rounds", "value": 3, "provenance": "PAPER_EXPLICIT"},
            {"name": "model", "value": "gpt-3.5", "provenance": "PAPER_EXPLICIT"},
        ],
        "effective_parameters": {"rounds": 3, "model": "gpt-3.5"},
        "expected_results": {"success_rate": {"value": 0.59, "tolerance_type": "absolute", "tolerance": 0.02}},
        "notes": "body notes",
    }


def _appendix_payload() -> dict:
    return {
        "scope": "appendix",
        "page_range": [13, 38],
        "extracted_parameters": [
            {"name": "model", "value": "gpt-4", "provenance": "APPENDIX_EXPLICIT"},
            {"name": "batch_size", "value": 32, "provenance": "APPENDIX_EXPLICIT"},
        ],
        "effective_parameters": {"model": "gpt-4", "batch_size": 32},
        "notes": "appendix notes",
    }


def test_merge_paper_findings_body_wins_and_appendix_fills_gaps() -> None:
    merged = merge_paper_findings([_appendix_payload(), _body_payload()])

    # 正文专属字段保留
    assert merged["method_summary"] == "pipeline summary"
    assert merged["expected_results"]["success_rate"]["value"] == 0.59

    # 参数全量合并；同名正文值优先
    assert merged["effective_parameters"] == {
        "rounds": 3,
        "model": "gpt-3.5",
        "batch_size": 32,
    }
    names = [p["name"] for p in merged["extracted_parameters"]]
    assert names == ["rounds", "model", "model", "batch_size"]

    # 审计信息与备注
    assert len(merged["paper_analysis_parts"]) == 2
    assert merged["paper_analysis_parts"][0]["scope"] == "body"
    assert "[body] body notes" in merged["notes"]
    assert "[appendix] appendix notes" in merged["notes"]


def test_merge_paper_findings_handles_appendix_only_and_empty() -> None:
    # 只有附录（正文任务失败的降级场景）：附录升格为基底
    merged = merge_paper_findings([_appendix_payload()])
    assert merged.get("method_summary", "") == ""
    assert merged["effective_parameters"]["batch_size"] == 32

    assert merge_paper_findings([]) == {}
