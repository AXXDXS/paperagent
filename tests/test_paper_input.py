from __future__ import annotations

import sys
import types

import pytest

from repro_agent.paper_input import (
    PaperInputError,
    extract_pdf_page_original,
    extract_pdf_text,
)


class _Page:
    """最简假页：不支持 visitor_text，模拟旧版 pypdf 或测试替身。"""

    def __init__(self, text: str):
        self.text = text

    def extract_text(self) -> str:
        return self.text


class _VisitorPage:
    """支持 visitor_text 的假页，按 pypdf 的调用约定回放预置片段。

    fragments: list of (text, x, y, font_size, base_font)
    """

    def __init__(self, fragments: list[tuple[str, float, float, float, str]]):
        self._fragments = fragments

    def _plain(self) -> str:
        return "\n".join(text for text, *_ in self._fragments)

    def extract_text(self, *, visitor_text=None):
        if visitor_text is None:
            return self._plain()
        for text, x, y, size, font in self._fragments:
            visitor_text(
                text,
                (1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 1.0, x, y),
                {"/BaseFont": font},
                size,
            )
        return self._plain()


def _fake_pdf_module(pages) -> types.ModuleType:
    return types.SimpleNamespace(
        PdfReader=lambda _: types.SimpleNamespace(is_encrypted=False, pages=pages)
    )


def test_pdf_extraction_preserves_page_boundaries(tmp_path, monkeypatch) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.7\nplaceholder")
    fake_module = _fake_pdf_module(
        [_Page("First page"), _Page("Second page")]
    )
    monkeypatch.setitem(sys.modules, "pypdf", fake_module)

    text = extract_pdf_text(path)

    assert "--- Page 1 ---" in text
    assert "First page" in text
    assert "--- Page 2 ---" in text


def test_non_pdf_content_with_pdf_suffix_is_rejected(tmp_path) -> None:
    path = tmp_path / "paper.pdf"
    path.write_text("not really a PDF", encoding="utf-8")

    with pytest.raises(PaperInputError, match="header"):
        extract_pdf_text(path)


def test_markdown_structure_from_font_signals(tmp_path, monkeypatch) -> None:
    """字号/字重信号 → 标题层级；题注斜体；列表项；行首符号转义。"""
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.7\nplaceholder")
    regu = "/MM+NimbusRomNo9L-Regu"
    medi = "/HL+NimbusRomNo9L-Medi"
    fragments = [
        # 论文主标题：14.3pt 粗体（正文 10pt 的 1.43 倍）→ #
        ("ExpeL: LLM Agents Are Experiential Learners", 100.0, 700.0, 14.3, medi),
        # 作者行：12pt 粗体但多逗号长行 → 保持普通段落
        (
            "Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, Gao Huang",
            60.0, 660.0, 12.0, medi,
        ),
        # Abstract 关键词 + 粗体 → ##
        ("Abstract", 280.0, 620.0, 10.9, medi),
        # 正文若干行（10pt 常规体，构成主导字号）
        ("The recent surge in research interest in applying large language models.", 54.0, 600.0, 10.0, regu),
        ("Our agent autonomously gathers experiences and extracts knowledge.", 54.0, 580.0, 10.0, regu),
        # 编号小节：10.9pt 粗体，编号深度 2 → ###
        ("4.4 Transfer Learning", 54.0, 560.0, 10.9, medi),
        # 表格题注（编号后紧跟冒号）→ 整行斜体
        ("Table 1: Main results on HotpotQA.", 54.0, 540.0, 9.0, regu),
        # 项目符号列表 → Markdown 列表项
        ("• Experience improves agents.", 54.0, 520.0, 10.0, regu),
        # 行首星号是文字（通讯作者脚注）→ 转义
        ("*Corresponding author.", 54.0, 500.0, 9.0, regu),
    ]
    fake_module = _fake_pdf_module([_VisitorPage(fragments)])
    monkeypatch.setitem(sys.modules, "pypdf", fake_module)

    text = extract_pdf_text(path)

    assert "# ExpeL: LLM Agents Are Experiential Learners" in text
    assert "## Abstract" in text
    assert "### 4.4 Transfer Learning" in text
    assert "*Table 1: Main results on HotpotQA.*" in text
    assert "- Experience improves agents." in text
    assert "\\*Corresponding author." in text
    # 作者行不能被误判为标题
    assert "# Andrew Zhao" not in text
    assert "## Andrew Zhao" not in text
    assert "Andrew Zhao, Daniel Huang" in text
    # 页码边界标记保留
    assert "--- Page 1 ---" in text


def test_plain_numbered_headings_annotated_without_font_signals(
    tmp_path, monkeypatch
) -> None:
    """无字体信号（页面对象不支持 visitor）时退回正则标注。"""
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.7\nplaceholder")
    fake_module = _fake_pdf_module(
        [
            _Page(
                "1 Introduction\n"
                "2 We observe that this is a long sentence in body text.\n"
                "Figure 1 shows the overview of our method.\n"
                "Abstract\n"
                "• first item\n"
                "# hashtag-like line\n"
            )
        ]
    )
    monkeypatch.setitem(sys.modules, "pypdf", fake_module)

    text = extract_pdf_text(path)

    # 编号标题（短行、无句尾标点）→ ##；正文里的 "2 We observe..." 不判
    assert "## 1 Introduction" in text
    assert "## 2 We observe" not in text
    # 正文引用 "Figure 1 shows..." 不是题注（无冒号/句点紧跟编号）
    assert "*Figure 1 shows*" not in text
    # 整行关键词 → ##
    assert "## Abstract" in text
    # 项目符号 → 列表项
    assert "- first item" in text
    # 行首 # 是文字 → 转义
    assert "\\# hashtag-like line" in text


def test_output_format_text_keeps_plain_text(tmp_path, monkeypatch) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.7\nplaceholder")
    fake_module = _fake_pdf_module([_Page("1 Introduction\nBody text.")])
    monkeypatch.setitem(sys.modules, "pypdf", fake_module)

    text = extract_pdf_text(path, output_format="text")

    assert "1 Introduction" in text
    assert "# " not in text
    assert "--- Page 1 ---" in text


def test_fragment_mismatch_falls_back_to_plain(tmp_path, monkeypatch) -> None:
    """visitor 片段覆盖不足（丢内容）时必须退回权威纯文本。"""

    class _LossyVisitorPage(_VisitorPage):
        def extract_text(self, *, visitor_text=None):
            plain = self._plain()
            if visitor_text is None:
                return plain
            # 只回放前半部分片段，模拟网关/版本差异导致 visitor 丢内容
            for text, x, y, size, font in self._fragments[:1]:
                visitor_text(
                    text,
                    (1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0, 1.0, x, y),
                    {"/BaseFont": font},
                    size,
                )
            return plain

    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.7\nplaceholder")
    regu = "/MM+NimbusRomNo9L-Regu"
    fragments = [
        ("First half of the sentence.", 54.0, 700.0, 10.0, regu),
        ("Second half must not be lost.", 54.0, 680.0, 10.0, regu),
    ]
    fake_module = _fake_pdf_module([_LossyVisitorPage(fragments)])
    monkeypatch.setitem(sys.modules, "pypdf", fake_module)

    text = extract_pdf_text(path)

    # 退回权威文本：两半内容都在
    assert "First half of the sentence." in text
    assert "Second half must not be lost." in text


def test_invalid_output_format_is_rejected(tmp_path) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.7\nplaceholder")

    with pytest.raises(PaperInputError, match="output_format"):
        extract_pdf_text(path, output_format="html")


def test_extract_pdf_page_original_returns_single_page(
    tmp_path, monkeypatch
) -> None:
    """单页原始提取：返回指定页内容、模式与总页数。"""

    class _LayoutPage:
        def __init__(self, text: str):
            self._text = text

        def extract_text(self, *, extraction_mode=None, visitor_text=None):
            if extraction_mode == "layout":
                return f"layout view: {self._text}"
            return self._text

    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.7\nplaceholder")
    fake_module = _fake_pdf_module(
        [_LayoutPage("first"), _LayoutPage("second")]
    )
    monkeypatch.setitem(sys.modules, "pypdf", fake_module)

    content, mode, total_pages = extract_pdf_page_original(path, 2)

    assert content == "layout view: second"
    assert mode == "layout"
    assert total_pages == 2


def test_extract_pdf_page_original_falls_back_to_linear_text(
    tmp_path, monkeypatch
) -> None:
    """版面模式不可用（抛异常/空白）时退回线性文本并标明 mode='text'。"""

    class _BrokenLayoutPage:
        def __init__(self, text: str):
            self._text = text

        def extract_text(self, *, extraction_mode=None, visitor_text=None):
            if extraction_mode == "layout":
                raise RuntimeError("layout unsupported")
            return self._text

    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.7\nplaceholder")
    fake_module = _fake_pdf_module([_BrokenLayoutPage("linear only")])
    monkeypatch.setitem(sys.modules, "pypdf", fake_module)

    content, mode, total_pages = extract_pdf_page_original(path, 1)

    assert content == "linear only"
    assert mode == "text"
    assert total_pages == 1


def test_extract_pdf_page_original_validates_page_number(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.7\nplaceholder")
    fake_module = _fake_pdf_module([_Page("only page")])
    monkeypatch.setitem(sys.modules, "pypdf", fake_module)

    with pytest.raises(PaperInputError, match="out of range"):
        extract_pdf_page_original(path, 2)
    with pytest.raises(PaperInputError, match="page_number"):
        extract_pdf_page_original(path, "1")  # type: ignore[arg-type]


class _FakeSandboxContext:
    """满足 SandboxContext 协议的最小桩：路径解析为绝对路径。"""

    task_id = "task_test"

    def __init__(self, root: Path):
        self._root = root

    def resolve_readable_path(self, relative_path: str) -> str:
        return str(self._root / relative_path)

    def resolve_writable_path(self, relative_path: str) -> str:
        return str(self._root / relative_path)

    def resolve_output_path(self, relative_path: str) -> str:
        return str(self._root / relative_path)

    def network_allowed(self) -> bool:
        return False


def test_inspect_pdf_page_tool_requires_specific_reason(
    tmp_path, monkeypatch
) -> None:
    """收紧条件：泛泛/过短的 reason 必须被拒绝，不触碰 PDF。"""

    from repro_agent.tools.base import ToolExecutionError
    from repro_agent.tools.filesystem_tools import inspect_pdf_page

    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.7\nplaceholder")
    fake_module = _fake_pdf_module([_Page("content")])
    monkeypatch.setitem(sys.modules, "pypdf", fake_module)
    ctx = _FakeSandboxContext(tmp_path)

    with pytest.raises(ToolExecutionError, match="reason"):
        inspect_pdf_page(ctx, "paper.pdf", 1, reason="看不懂")
    with pytest.raises(ToolExecutionError, match="reason"):
        inspect_pdf_page(ctx, "paper.pdf", 1, reason="   ")


def test_inspect_pdf_page_tool_returns_page_with_reason(
    tmp_path, monkeypatch
) -> None:
    from repro_agent.tools.filesystem_tools import inspect_pdf_page

    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.7\nplaceholder")
    fake_module = _fake_pdf_module([_Page("page one"), _Page("page two")])
    monkeypatch.setitem(sys.modules, "pypdf", fake_module)
    ctx = _FakeSandboxContext(tmp_path)

    result = inspect_pdf_page(
        ctx,
        "paper.pdf",
        2,
        reason="Table 3 的数值列与表头在提取文本中错位，无法确定对应关系",
    )

    assert result["page_number"] == 2
    assert result["total_pages"] == 2
    assert "page two" in result["content"]
    assert result["mode"] in ("layout", "text")
    assert result["reason"].startswith("Table 3")
