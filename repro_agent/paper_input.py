"""Explicit paper input parsing with fail-closed PDF handling.

PDF 文本提取产出 Markdown：通过 pypdf 的 ``visitor_text`` 回调采集每个
文本片段的字号/字体/坐标，把同一视觉行的片段聚合成行，再依据
"字号相对正文的比例 + 字体粗细 + 编号/关键词模式"把行标注为标题层级、
图表题注或列表项。PDF 内嵌图片不提取（pypdf 只读文本层），题注作为
文字保留。

结构增强是尽力而为的附加层，内容完整性以 ``extract_text()`` 的返回为
权威：片段重建文本与权威文本的双向覆盖率不足（或页面对象不支持
visitor）时，自动退回纯文本行 + 正则结构标注。fail-closed 语义不变
（伪 PDF / 加密 / 超限 / 无文本层一律报错）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


class PaperInputError(ValueError):
    pass


# --------------------------------------------------------------------------
# 结构标注：字体信号与文本模式
# --------------------------------------------------------------------------

# BaseFont 名中常见的粗体后缀（Nimbus 的 -Medi、LaTeX 的 CMBX 等）。
_BOLD_FONT_MARKERS = ("Medi", "Bold", "Heav", "Black", "Semibd", "Demi", "CMBX")

# "1 Introduction" / "4.4 Transfer Learning" / "A.1 Appendix Details"。
_NUMBERED_HEADING_RE = re.compile(
    r"^(\d{1,2}(?:\.\d{1,2}){0,3}|[A-Z](?:\.\d{1,2}){0,3})\.?\s+\S"
)
# 题注开头："Table 7: ..." / "Figure 1. ..." / "Fig. 3: ..." / "Algorithm 2: ..."。
# 要求编号后紧跟冒号或句点，避免误伤正文中的 "Figure 1 shows that ..." 引用；
# 缩写形式 "Fig." 只认冒号——句点式 "Fig. 6. As we can see ..." 是正文引用
# 而非题注，无法可靈区分时宁可不标（fail-closed）。
_CAPTION_RE = re.compile(
    r"^(?:(?:Table|Figure|Algorithm)\.?\s+\d+\s*[.:]\s|Fig\.\s+\d+\s*:\s)",
    re.IGNORECASE,
)
# 列表项开头的项目符号（bullet、中点、小方块等）。
_BULLET_RE = re.compile(r"^[\u2022\u00b7\u25e6\u2023\u25aa\u2043]\s+")
# 整行由 Markdown 结构符号组成（水平线 / setext 下划线风险）。
_RULE_LINE_RE = re.compile(r"^[-=_*`~#>\s]{3,}$")
# 纯数字/百分比/标点行（表格单元格常见形态，不是标题）。
_NUMERIC_LINE_RE = re.compile(r"^[\W\d\s]+$")
# 行首会被 Markdown 当作结构语法的字符。
_LEADING_SYNTAX_RE = re.compile(r"^(?:[#>`])")
# 行首强调标记（"*Corresponding author." 中的星号是文字，不是强调）。
_LEADING_EMPHASIS_RE = re.compile(r"^[*_](?=\S)")

# Algorithm 伪代码块的控制流关键字（粗体短行，但不是标题）。
_PSEUDOCODE_KEYWORDS = frozenset(
    {
        "Initialize",
        "Input",
        "Output",
        "Require",
        "Ensure",
        "while",
        "for",
        "if",
        "then",
        "else",
        "end",
        "do",
        "return",
    }
)

_KEYWORD_HEADINGS = frozenset(
    {
        "abstract",
        "introduction",
        "related work",
        "background",
        "preliminaries",
        "method",
        "methodology",
        "experiments",
        "experimental setup",
        "results",
        "evaluation",
        "discussion",
        "conclusion",
        "conclusions",
        "future work",
        "acknowledgments",
        "acknowledgements",
        "acknowledgment",
        "acknowledgement",
        "references",
        "appendix",
        "appendices",
    }
)


@dataclass
class _Fragment:
    """visitor 回调采集到的一个文本片段及其版式信号。"""

    text: str
    x: float
    y: float
    size: float
    bold: bool


@dataclass
class _Line:
    """聚合后的视觉行：文本 + 行级字号/粗体占比。"""

    text: str
    size: float
    bold_ratio: float
    degenerate: bool = False
    """整行片段都落在 (0, 0) 退化坐标上（表格/变换后内容），不可作标题。"""


def _collect_fragments(page) -> list[_Fragment]:
    """采集一页的文本片段；页面对象不支持 visitor 时返回空列表。"""

    fragments: list[_Fragment] = []

    def _visitor(text, cm, tm, font_dict, font_size):
        if not text or not text.strip():
            return
        base_font = ""
        if font_dict:
            try:
                base_font = str(font_dict.get("/BaseFont", ""))
            except Exception:
                base_font = ""
        x, y = 0.0, 0.0
        if tm and len(tm) >= 6:
            try:
                x, y = float(tm[4]), float(tm[5])
            except (TypeError, ValueError):
                x, y = 0.0, 0.0
        fragments.append(
            _Fragment(
                text=text.replace("\n", " "),
                x=x,
                y=y,
                size=float(font_size or 0.0),
                bold=any(marker in base_font for marker in _BOLD_FONT_MARKERS),
            )
        )

    try:
        page.extract_text(visitor_text=_visitor)
    except TypeError:
        # 页面对象的 extract_text 不接受 visitor 参数（测试替身或旧版
        # pypdf）：放弃字体信号，退回纯文本 + 正则标注。
        return []
    except Exception:
        return []
    return fragments


def _dominant_size(fragments: list[_Fragment]) -> float:
    """按字符数加权的主导字号（正文字号），作为标题判定的基准。"""

    weights: dict[float, int] = {}
    for frag in fragments:
        if frag.size <= 0:
            continue
        key = round(frag.size * 2) / 2
        weights[key] = weights.get(key, 0) + len(frag.text)
    if not weights:
        return 0.0
    return max(weights.items(), key=lambda item: item[1])[0]


def _group_into_lines(
    fragments: list[_Fragment], body_size: float
) -> list[_Line]:
    """按调用顺序把片段聚成视觉行（遵循内容流顺序，天然兼容分栏）。

    行判定：与当前行锚定 y 的距离在容差内（约半个正文字号，容忍
    上标/下标的基线偏移）。行内按 x 排序，片段间横向间距明显时补空格。
    """

    tolerance = max(2.5, body_size * 0.45) if body_size > 0 else 3.0
    raw_lines: list[list[_Fragment]] = []
    for frag in fragments:
        if raw_lines and abs(frag.y - raw_lines[-1][0].y) <= tolerance:
            raw_lines[-1].append(frag)
        else:
            raw_lines.append([frag])

    reference_size = body_size or 10.0
    lines: list[_Line] = []
    for parts in raw_lines:
        parts.sort(key=lambda frag: frag.x)
        pieces: list[str] = []
        prev_end: float | None = None
        char_total = bold_chars = 0
        size_weight = 0.0
        for frag in parts:
            frag_size = frag.size if frag.size > 0 else reference_size
            if prev_end is not None:
                gap = frag.x - prev_end
                threshold = max(0.8, 0.18 * frag_size)
                if (
                    gap > threshold
                    and pieces
                    and not pieces[-1].endswith((" ", "-"))
                    and not frag.text.startswith(" ")
                ):
                    pieces.append(" ")
            pieces.append(frag.text)
            # 用 0.5 * 字号 * 字符数粗略估计片段宽度，用于下一个间隙判断。
            prev_end = frag.x + len(frag.text) * 0.5 * frag_size
            char_total += len(frag.text)
            if frag.bold:
                bold_chars += len(frag.text)
            size_weight += len(frag.text) * frag_size
        text = re.sub(r"\s+", " ", "".join(pieces)).strip()
        if not text:
            continue
        degenerate = all(
            frag.x == 0.0 and frag.y == 0.0 for frag in parts
        ) and len(parts) > 0
        lines.append(
            _Line(
                text=text,
                size=(size_weight / char_total) if char_total else 0.0,
                bold_ratio=(bold_chars / char_total) if char_total else 0.0,
                degenerate=degenerate,
            )
        )
    return lines


def _heading_level(
    text: str, *, size: float, bold_ratio: float, body_size: float, degenerate: bool = False
) -> int | None:
    """判定一行是否是标题，返回 Markdown 层级（None 表示普通段落）。

    规则按可靠度排序：
        1. 编号标题（"4.4 Transfer Learning"）：层级由编号深度决定，
           需要粗体或明显大于正文的字号佐证；
        2. 整行是常见章节关键词（Abstract/References/...）：##；
        3. 粗体且字号 >= 1.3 倍正文：论文主标题 #；
        4. 粗体短行且不含多个逗号：无编号小标题 ###（附录模板标题等）。
    """

    if not text or len(text) > 150:
        return None
    # 退化坐标行（表格/变换后内容）、纯数字行、过短行都不是标题。
    if degenerate or _NUMERIC_LINE_RE.match(text) or len(text) < 8:
        return None
    # Algorithm 伪代码的控制流行（"Initialize:" / "while ... do" / "end if"
    # / "return B"）通常是粗体短行，但把它们标成标题只会制造噪音。
    first_word = text.rstrip(":").strip().split(" ", 1)[0].rstrip(":")
    if first_word in _PSEUDOCODE_KEYWORDS:
        return None
    size_ratio = (size / body_size) if body_size > 0 else 1.0
    bold = bold_ratio >= 0.5
    emphasized = bold or size_ratio >= 1.06

    numbered = _NUMBERED_HEADING_RE.match(text)
    if numbered and emphasized:
        depth = numbered.group(1).count(".") + 1
        return min(1 + depth, 4)

    keyword = text.rstrip(":").strip().lower()
    if keyword in _KEYWORD_HEADINGS and emphasized:
        return 2

    if bold and size_ratio >= 1.3 and len(text.split()) <= 16 and text.count(",") < 3:
        return 1

    if bold and len(text) <= 90 and len(text.split()) <= 12 and text.count(",") < 2:
        return 3

    return None


def _escape_leading_markdown(text: str) -> str:
    """转义会被 Markdown 误解析的行首字符，保证内容原样保留。"""

    if _LEADING_SYNTAX_RE.match(text) or _RULE_LINE_RE.match(text):
        return "\\" + text
    if _LEADING_EMPHASIS_RE.match(text):
        return "\\" + text
    return text


def _render_lines_markdown(lines: list[_Line], body_size: float) -> str:
    out: list[str] = []
    for line in lines:
        text = line.text
        if _CAPTION_RE.match(text):
            out.append(f"*{text}*")
            continue
        level = _heading_level(
            text,
            size=line.size,
            bold_ratio=line.bold_ratio,
            body_size=body_size,
            degenerate=line.degenerate,
        )
        if level:
            out.append(f"{'#' * level} {text}")
            continue
        bullet = _BULLET_RE.match(text)
        if bullet:
            out.append(f"- {text[bullet.end():].strip()}")
            continue
        out.append(_escape_leading_markdown(text))
    return "\n".join(out)


def _render_plain_lines(page_text: str) -> str:
    """无字体信号时的正则结构标注（保守：只标高置信模式）。"""

    out: list[str] = []
    for raw in page_text.splitlines():
        text = re.sub(r"\s+", " ", raw).strip()
        if not text:
            continue
        if _CAPTION_RE.match(text):
            out.append(f"*{text}*")
            continue
        numbered = _NUMBERED_HEADING_RE.match(text)
        if (
            numbered
            and len(text) <= 150
            and len(text.split()) <= 12
            and not text.endswith((".", ","))
        ):
            depth = numbered.group(1).count(".") + 1
            out.append(f"{'#' * min(1 + depth, 4)} {text}")
            continue
        keyword = text.rstrip(":").strip().lower()
        if keyword in _KEYWORD_HEADINGS and len(text.split()) <= 4:
            out.append(f"## {text}")
            continue
        bullet = _BULLET_RE.match(text)
        if bullet:
            out.append(f"- {text[bullet.end():].strip()}")
            continue
        out.append(_escape_leading_markdown(text))
    return "\n".join(out)


def _coverage(candidate: str, reference: str) -> float:
    """片段重建文本与权威文本的非空白字符双向覆盖率（取较小方向）。"""

    a = len(re.sub(r"\s+", "", candidate))
    b = len(re.sub(r"\s+", "", reference))
    if b == 0:
        return 1.0
    if a == 0:
        return 0.0
    return min(a, b) / max(a, b)


def _render_page(page, page_text: str, output_format: str) -> str:
    """把一页的权威文本渲染为目标格式（Markdown 或纯文本）。"""

    if output_format == "text":
        return page_text

    fragments = _collect_fragments(page)
    if fragments:
        body_size = _dominant_size(fragments)
        lines = _group_into_lines(fragments, body_size)
        rebuilt = "\n".join(line.text for line in lines)
        if _coverage(rebuilt, page_text) >= 0.98:
            return _render_lines_markdown(lines, body_size)
    return _render_plain_lines(page_text)


# --------------------------------------------------------------------------
# 入口：fail-closed 校验 + 逐页提取
# --------------------------------------------------------------------------


def _open_pdf_reader(pdf_path: Path):
    """共享的 fail-closed 打开逻辑：魔数校验、依赖、解析、加密。"""

    with pdf_path.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise PaperInputError("file has a .pdf name but no valid PDF header")
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise PaperInputError(
            "PDF support requires the 'pypdf' dependency; install the project dependencies"
        ) from exc
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:  # pypdf exposes multiple parser-specific exceptions
        raise PaperInputError(f"failed to parse PDF: {exc}") from exc
    if reader.is_encrypted:
        raise PaperInputError("encrypted PDFs are not supported; provide an unlocked copy")
    return reader


def extract_pdf_page_original(
    path: str | Path,
    page_number: int,
    *,
    max_chars: int = 100_000,
) -> tuple[str, str, int]:
    """提取单个 PDF 页的"原始版面"文本，返回 ``(content, mode, total_pages)``。

    与 :func:`extract_pdf_text` 的区别：不做任何 Markdown 结构标注，
    优先使用 pypdf 的 ``layout`` 提取模式——该模式用空格近似还原文本在
    页面上的二维排布（列、表格对齐），是纯文本接口下最接近"看原 PDF"
    的形态；版面模式对该页不可用时退回默认线性文本流（``mode`` 字段
    标明实际使用的模式）。仅供模型在结构化提取结果确实无法理解时
    做单页仲裁，不适合通读整篇文档。
    """

    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise PaperInputError(f"PDF file not found: {pdf_path}")
    reader = _open_pdf_reader(pdf_path)
    total_pages = len(reader.pages)
    if not isinstance(page_number, int) or isinstance(page_number, bool):
        raise PaperInputError(
            f"page_number must be an integer between 1 and {total_pages}, got {page_number!r}"
        )
    if not 1 <= page_number <= total_pages:
        raise PaperInputError(
            f"page_number {page_number} is out of range; the PDF has {total_pages} pages"
        )

    page = reader.pages[page_number - 1]
    content = ""
    mode = "layout"
    try:
        content = page.extract_text(extraction_mode="layout") or ""
    except Exception:
        content = ""
    if not content.strip():
        mode = "text"
        try:
            content = page.extract_text() or ""
        except Exception as exc:
            raise PaperInputError(
                f"failed to extract text from PDF page {page_number}: {exc}"
            ) from exc
    if not content.strip():
        raise PaperInputError(
            f"page {page_number} contains no extractable text; "
            "provide an OCR/text version for scanned papers"
        )
    if len(content) > max_chars:
        raise PaperInputError(
            f"page {page_number} text exceeds the {max_chars} character limit"
        )
    return content, mode, total_pages


def extract_pdf_text(
    path: str | Path,
    *,
    start_page: int | None = None,
    end_page: int | None = None,
    max_pages: int = 500,
    max_chars: int = 2_000_000,
    output_format: str = "markdown",
) -> str:
    """提取 PDF 指定页范围的逐页文本，输出带页码边界标记的 Markdown。

    ``start_page``/``end_page`` 均为**原文页码**（1 起、闭区间），页码边界
    标记 ``--- Page N ---`` 始终使用原文页号，不受范围影响——下游以页码
    字段回溯证据时不会因切分而错位。``max_pages`` 作用于实际提取的页数
    （范围大小），而不是全文档页数。

    ``output_format="text"`` 保留旧行为（纯文本，不做结构标注）；
    默认 ``"markdown"`` 在同一份文本之上叠加标题层级、图表题注斜体、
    列表项与行首符号转义。内嵌图片不提取（只读文本层）。
    """

    if output_format not in ("markdown", "text"):
        raise PaperInputError(
            f"unsupported output_format: {output_format!r} (expected 'markdown' or 'text')"
        )

    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise PaperInputError(f"PDF file not found: {pdf_path}")
    reader = _open_pdf_reader(pdf_path)
    total_pages = len(reader.pages)

    first = 1 if start_page is None else int(start_page)
    last = total_pages if end_page is None else int(end_page)
    if first < 1 or last > total_pages or first > last:
        raise PaperInputError(
            f"invalid page range [{first}, {last}] for a {total_pages}-page PDF"
        )
    if last - first + 1 > max_pages:
        raise PaperInputError(
            f"requested {last - first + 1} pages, exceeding the limit of {max_pages}"
        )

    parts: list[str] = []
    size = 0
    extracted_any_text = False
    for page_number in range(first, last + 1):
        page = reader.pages[page_number - 1]
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            raise PaperInputError(
                f"failed to extract text from PDF page {page_number}: {exc}"
            ) from exc
        body = _render_page(page, page_text, output_format)
        extracted_any_text = extracted_any_text or bool(page_text.strip())
        section = f"--- Page {page_number} ---\n{body.strip()}"
        size += len(section)
        if size > max_chars:
            raise PaperInputError(
                f"extracted PDF text exceeds the {max_chars} character limit"
            )
        parts.append(section)
    text = "\n\n".join(parts).strip()
    if not extracted_any_text:
        raise PaperInputError(
            "PDF contains no extractable text; provide an OCR/text version for scanned papers"
        )
    return text


# 附录边界探测：页首标题模式（页面前几行才可能是章节标题，正文中提到
# "Appendix" 的交叉引用不算）。
_APPENDIX_HEADING_RE = re.compile(
    r"^(?:\(?[A-Z0-9.]+\)?\s+)?(?:Appendix|APPENDIX|Appendices)\b"
)
_REFERENCES_HEADING_RE = re.compile(r"^\s*References\s*$", re.IGNORECASE)
# 探测只看页面前 N 行——超过这个深度的 "Appendix" 几乎必然是正文引用。
_HEADING_SCAN_LINES = 8


@dataclass(frozen=True)
class PaperStructure:
    """正文/附录切分探测结果（纯代码启发式，不调用 LLM）。"""

    total_pages: int
    appendix_start_page: int | None
    body_pages: tuple[int, int]
    appendix_pages: tuple[int, int] | None
    appendix_char_count: int

    @property
    def has_appendix(self) -> bool:
        return self.appendix_pages is not None


def probe_paper_structure(path: str | Path) -> PaperStructure:
    """探测论文的正文/附录页边界，供规划器拆分双子任务使用。

    规则（全部基于页首文本，无 LLM 参与）：
        1. 收集“页首若干行内出现附录式标题”的候选页（如 ``A Appendix:``
           ``Appendix B`` ``APPENDIX``）；目录里的 ``Appendix`` 条目因为
           后面紧跟页码数字行、且所在页还在前部，会被第 2 步过滤。
        2. 取最后一个 ``References`` 标题页作为分界锚点，附录起点必须是
           锚点之后的第一个候选页；没有 References 锚点时退化为“后 60%
           页里的第一个候选页”。
        3. 找不到任何可靠候选 → ``appendix_pages=None``，调用方退回单任务
           全文阅读（fail-closed：宁可整体读，不可切错丢内容）。
    """

    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise PaperInputError(f"PDF file not found: {pdf_path}")
    reader = _open_pdf_reader(pdf_path)
    total_pages = len(reader.pages)

    page_lines: list[list[str]] = []
    page_char_counts: list[int] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - 单页提取失败不应拖垮整体探测
            text = ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        page_lines.append(lines)
        page_char_counts.append(len(text))

    def _is_appendix_heading_page(index: int) -> bool:
        # 附录标题只认页首几行；更深处的 "Appendix" 几乎必然是正文交叉引用。
        return any(
            _APPENDIX_HEADING_RE.match(line) for line in page_lines[index][:_HEADING_SCAN_LINES]
        )

    # "References" 独立成行时几乎必是节标题，但它常出现在页中部（例如
    # 表格之后），所以这里扫描全页行而不只是页首。
    references_page = max(
        (
            index
            for index, lines in enumerate(page_lines)
            if any(_REFERENCES_HEADING_RE.match(line) for line in lines)
        ),
        default=None,
    )

    candidates = [index for index in range(total_pages) if _is_appendix_heading_page(index)]
    if references_page is not None:
        eligible = [index for index in candidates if index >= references_page]
    else:
        threshold = int(total_pages * 0.4)
        eligible = [index for index in candidates if index >= threshold]

    if not eligible:
        return PaperStructure(
            total_pages=total_pages,
            appendix_start_page=None,
            body_pages=(1, total_pages),
            appendix_pages=None,
            appendix_char_count=0,
        )

    appendix_start = eligible[0] + 1  # 0-based -> 1-based
    appendix_chars = sum(page_char_counts[appendix_start - 1 :])
    return PaperStructure(
        total_pages=total_pages,
        appendix_start_page=appendix_start,
        body_pages=(1, appendix_start - 1),
        appendix_pages=(appendix_start, total_pages),
        appendix_char_count=appendix_chars,
    )
