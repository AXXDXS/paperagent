"""文件查找 / 文件阅读类只读工具（``ToolRiskLevel.READ_ONLY``）。

这些工具是"论文分析子智能体""代码分析子智能体""资源检查子智能体"等
只读分析类子智能体最常用的能力，全部限定在调用方提供的
``SandboxContext.resolve_readable_path`` 范围内，不能读取沙箱之外
的任何文件（呼应设计文档 §12 沙箱设计：``input/`` 只读、子智能体
禁止访问宿主机文件/其他任务目录）。

复用来源：
    路径越界校验的写法（"解析为绝对路径后检查是否仍以 workspace 为
    前缀"）直接复用自 DeepCode ``tools/code_implementation_server.py``
    的 ``validate_path`` 函数（见该文件第 88-96 行），只是把
    "WORKSPACE_DIR 全局变量"替换成了显式传入的 ``SandboxContext``，
    避免全局可变状态在多任务并发时互相污染——这是本项目相对原实现
    的一处工程改进，原因见 ``paper_agent/CHANGES_AND_DESIGN_NOTES.md``。
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any

from repro_agent.tools.base import (
    SandboxContext,
    ToolExample,
    ToolExecutionError,
    ToolOutputSpec,
    ToolParamDoc,
    ToolRiskLevel,
    ToolSpec,
)
from repro_agent.paper_input import (
    PaperInputError,
    extract_pdf_page_original,
    extract_pdf_text,
)
from repro_agent.evidence.hashing import sha256_of_directory, sha256_of_file

_MAX_READ_BYTES = 2_000_000  # 单文件最大读取字节数，避免把超大文件读进上下文
_MAX_SEARCH_RESULTS = 500


def list_directory(
    ctx: SandboxContext, path: str = ".", *, recursive: bool = False
) -> dict[str, Any]:
    """列出目录内容（文件/子目录），只能列出沙箱可读范围内的路径。"""

    resolved = Path(ctx.resolve_readable_path(path))
    if not resolved.exists():
        raise ToolExecutionError(f"path not found: {path}")
    if not resolved.is_dir():
        raise ToolExecutionError(f"not a directory: {path}")

    entries: list[dict[str, Any]] = []
    if recursive:
        for root, dirs, files in os.walk(resolved):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for name in sorted(files):
                full = Path(root) / name
                entries.append(
                    {
                        "path": str(full.relative_to(resolved.parent)),
                        "type": "file",
                        "size_bytes": full.stat().st_size,
                    }
                )
    else:
        for item in sorted(resolved.iterdir()):
            entries.append(
                {
                    "path": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size_bytes": item.stat().st_size if item.is_file() else None,
                }
            )
    return {"path": path, "entries": entries, "count": len(entries)}


def find_files(
    ctx: SandboxContext,
    pattern: str,
    *,
    root: str = ".",
    max_results: int = 200,
) -> dict[str, Any]:
    """按 glob 模式查找文件（如 ``configs/*.yaml``、``**/*.py``）。"""

    resolved_root = Path(ctx.resolve_readable_path(root))
    if not resolved_root.exists():
        raise ToolExecutionError(f"search root not found: {root}")

    max_results = min(max_results, _MAX_SEARCH_RESULTS)
    matches: list[str] = []
    for path in resolved_root.rglob("*"):
        if len(matches) >= max_results:
            break
        if path.is_file() and fnmatch.fnmatch(path.name, pattern):
            matches.append(str(path.relative_to(resolved_root)))

    return {
        "pattern": pattern,
        "root": root,
        "matches": matches,
        "truncated": len(matches) >= max_results,
    }


def grep_files(
    ctx: SandboxContext,
    query: str,
    *,
    root: str = ".",
    file_glob: str = "*",
    max_results: int = 200,
    case_sensitive: bool = False,
) -> dict[str, Any]:
    """在沙箱可读范围内做纯文本关键字检索（不依赖外部 ripgrep 二进制，
    保证在任意沙箱环境下都能工作；性能不是设计目标，正确性和隔离性是）。
    """

    resolved_root = Path(ctx.resolve_readable_path(root))
    if not resolved_root.exists():
        raise ToolExecutionError(f"search root not found: {root}")

    max_results = min(max_results, _MAX_SEARCH_RESULTS)
    needle = query if case_sensitive else query.lower()
    results: list[dict[str, Any]] = []

    for path in resolved_root.rglob(file_glob):
        if len(results) >= max_results:
            break
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        haystack = text if case_sensitive else text.lower()
        if needle not in haystack:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            check = line if case_sensitive else line.lower()
            if needle in check:
                results.append(
                    {
                        "path": str(path.relative_to(resolved_root)),
                        "line": lineno,
                        "text": line.strip()[:400],
                    }
                )
                if len(results) >= max_results:
                    break

    return {"query": query, "results": results, "truncated": len(results) >= max_results}


def read_file(
    ctx: SandboxContext,
    path: str,
    *,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict[str, Any]:
    """读取文件内容，支持指定行号范围（避免整篇论文/超大代码文件一次性
    塞进模型上下文——这与设计文档 §16.1 上下文压缩"必须保留活跃任务
    相关信息，优先压缩已完成任务详情"的精神一致：把"要读多少"的决定权
    交给调用方而不是无脑全量返回）。
    """

    resolved = Path(ctx.resolve_readable_path(path))
    if not resolved.exists():
        raise ToolExecutionError(f"file not found: {path}")
    if not resolved.is_file():
        raise ToolExecutionError(f"not a file: {path}")
    size = resolved.stat().st_size
    if size > _MAX_READ_BYTES and start_line is None and end_line is None:
        raise ToolExecutionError(
            f"file too large ({size} bytes); specify start_line/end_line to read a slice"
        )

    text = resolved.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    total_lines = len(lines)
    if start_line is not None or end_line is not None:
        s = max(1, start_line or 1)
        e = min(total_lines, end_line or total_lines)
        selected = lines[s - 1 : e]
        content = "\n".join(selected)
        return {
            "path": path,
            "start_line": s,
            "end_line": e,
            "total_lines": total_lines,
            "content": content,
        }
    return {
        "path": path,
        "start_line": 1,
        "end_line": total_lines,
        "total_lines": total_lines,
        "content": text,
    }


def get_file_stat(ctx: SandboxContext, path: str) -> dict[str, Any]:
    """获取文件元信息（大小、修改时间），不读取内容——用于资源检查等
    只需要"文件是否存在/多大"而不需要内容的场景，减少不必要的 I/O。
    """

    resolved = Path(ctx.resolve_readable_path(path))
    if not resolved.exists():
        return {"path": path, "exists": False}
    stat = resolved.stat()
    return {
        "path": path,
        "exists": True,
        "is_dir": resolved.is_dir(),
        "size_bytes": stat.st_size,
        "modified_at": stat.st_mtime,
    }


def hash_path(ctx: SandboxContext, path: str) -> dict[str, Any]:
    """Create a deterministic content digest for a staged file or directory."""

    resolved = Path(ctx.resolve_readable_path(path))
    if not resolved.exists():
        raise ToolExecutionError(f"path not found: {path}")
    digest = (
        sha256_of_directory(resolved)
        if resolved.is_dir()
        else sha256_of_file(resolved)
    )
    git_commit = ""
    repository = resolved if resolved.is_dir() else resolved.parent
    git_dir = repository / ".git"
    if git_dir.is_dir():
        try:
            head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
            if head.startswith("ref: "):
                ref = head[5:]
                ref_path = git_dir / ref
                if ref_path.is_file():
                    git_commit = ref_path.read_text(encoding="utf-8").strip()
                else:
                    packed = git_dir / "packed-refs"
                    if packed.is_file():
                        for line in packed.read_text(encoding="utf-8").splitlines():
                            if line and not line.startswith(("#", "^")):
                                value, name = line.split(" ", 1)
                                if name == ref:
                                    git_commit = value
                                    break
            else:
                git_commit = head
        except (OSError, ValueError):
            git_commit = ""
    return {
        "path": path,
        "sha256": digest,
        "kind": "directory" if resolved.is_dir() else "file",
        "git_commit": git_commit,
    }


def read_pdf_text(
    ctx: SandboxContext,
    path: str,
    *,
    start_page: int | None = None,
    end_page: int | None = None,
    max_pages: int = 500,
    max_chars: int = 2_000_000,
    output_format: str = "markdown",
) -> dict[str, Any]:
    """Extract page-delimited markdown text from a (range of a) staged PDF.

    ``start_page``/``end_page`` select an inclusive 1-based range of the
    ORIGINAL page numbering; the ``--- Page N ---`` markers keep original
    numbers so downstream evidence lookups stay accurate.  Without a range
    the whole document is read.

    ``output_format='markdown'`` (default) annotates headings, figure/table
    captions and list items on top of the exact page text; ``'text'`` keeps
    the raw linear text stream.  Embedded images are never extracted.
    """

    resolved = Path(ctx.resolve_readable_path(path))
    try:
        content = extract_pdf_text(
            resolved,
            start_page=start_page,
            end_page=end_page,
            max_pages=max_pages,
            max_chars=max_chars,
            output_format=output_format,
        )
    except PaperInputError as exc:
        raise ToolExecutionError(str(exc)) from exc
    return {"path": path, "content": content, "format": "pdf"}


# “最后手段”门槛：reason 必须达到的最小长度（字符数）。
_MIN_INSPECT_REASON_CHARS = 10


def inspect_pdf_page(
    ctx: SandboxContext,
    path: str,
    page_number: int,
    reason: str,
    *,
    max_chars: int = 100_000,
) -> dict[str, Any]:
    """Last-resort inspection of a single original PDF page.

    ``reason`` is a mandatory, non-trivial explanation of *what* in the
    extracted text could not be understood — the tool refuses to run
    without it.  Layout mode preserves the spatial arrangement of the
    page (columns, table alignment); the raw linear stream is the fallback.
    """

    reason_text = (reason or "").strip()
    if len(reason_text) < _MIN_INSPECT_REASON_CHARS:
        raise ToolExecutionError(
            "inspect_pdf_page 是最后手段（last resort）工具：reason 参数必须"
            "至少 10 个字符，并说明你在 read_pdf_text 提取文本的哪个具体"
            "位置、为什么无法理解（例如“Table 3 的数值列与表头在提取文本中"
            "错位，无法确定对应关系”）。如果只是还没读过提取文本，请先调用"
            " read_pdf_text；不要用本工具通读论文。"
        )
    resolved = Path(ctx.resolve_readable_path(path))
    try:
        content, mode, total_pages = extract_pdf_page_original(
            resolved, page_number, max_chars=max_chars
        )
    except PaperInputError as exc:
        raise ToolExecutionError(str(exc)) from exc
    return {
        "path": path,
        "page_number": page_number,
        "total_pages": total_pages,
        "content": content,
        "mode": mode,
        "reason": reason_text,
        "format": "pdf_page_inspection",
    }


def _strict_object(
    required: list[str], properties: dict[str, Any]
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


_DIRECTORY_ENTRY_SCHEMA = _strict_object(
    ["path", "type", "size_bytes"],
    {
        "path": {"type": "string"},
        "type": {"type": "string", "enum": ["file", "directory"]},
        "size_bytes": {"type": ["integer", "null"], "minimum": 0},
    },
)
_LIST_DIRECTORY_OUTPUT_SCHEMA = _strict_object(
    ["path", "entries", "count"],
    {
        "path": {"type": "string"},
        "entries": {"type": "array", "items": _DIRECTORY_ENTRY_SCHEMA},
        "count": {"type": "integer", "minimum": 0},
    },
)
_FIND_FILES_OUTPUT_SCHEMA = _strict_object(
    ["pattern", "root", "matches", "truncated"],
    {
        "pattern": {"type": "string"},
        "root": {"type": "string"},
        "matches": {"type": "array", "items": {"type": "string"}},
        "truncated": {"type": "boolean"},
    },
)
_GREP_RESULT_SCHEMA = _strict_object(
    ["path", "line", "text"],
    {
        "path": {"type": "string"},
        "line": {"type": "integer", "minimum": 1},
        "text": {"type": "string"},
    },
)
_GREP_FILES_OUTPUT_SCHEMA = _strict_object(
    ["query", "results", "truncated"],
    {
        "query": {"type": "string"},
        "results": {"type": "array", "items": _GREP_RESULT_SCHEMA},
        "truncated": {"type": "boolean"},
    },
)
_READ_FILE_OUTPUT_SCHEMA = _strict_object(
    ["path", "start_line", "end_line", "total_lines", "content"],
    {
        "path": {"type": "string"},
        "start_line": {"type": "integer", "minimum": 1},
        "end_line": {"type": "integer", "minimum": 0},
        "total_lines": {"type": "integer", "minimum": 0},
        "content": {"type": "string"},
    },
)
_FILE_STAT_OUTPUT_SCHEMA = {
    "oneOf": [
        _strict_object(
            ["path", "exists"],
            {"path": {"type": "string"}, "exists": {"const": False}},
        ),
        _strict_object(
            ["path", "exists", "is_dir", "size_bytes", "modified_at"],
            {
                "path": {"type": "string"},
                "exists": {"const": True},
                "is_dir": {"type": "boolean"},
                "size_bytes": {"type": "integer", "minimum": 0},
                "modified_at": {"type": "number"},
            },
        ),
    ]
}
_HASH_PATH_OUTPUT_SCHEMA = _strict_object(
    ["path", "sha256", "kind", "git_commit"],
    {
        "path": {"type": "string"},
        "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "kind": {"type": "string", "enum": ["file", "directory"]},
        "git_commit": {"type": "string"},
    },
)
_PDF_TEXT_OUTPUT_SCHEMA = _strict_object(
    ["path", "content", "format"],
    {
        "path": {"type": "string"},
        "content": {"type": "string"},
        "format": {"const": "pdf"},
    },
)
_PDF_PAGE_OUTPUT_SCHEMA = _strict_object(
    ["path", "page_number", "total_pages", "content", "mode", "reason", "format"],
    {
        "path": {"type": "string"},
        "page_number": {"type": "integer", "minimum": 1},
        "total_pages": {"type": "integer", "minimum": 1},
        "content": {"type": "string"},
        "mode": {"type": "string"},
        "reason": {"type": "string", "minLength": _MIN_INSPECT_REASON_CHARS},
        "format": {"const": "pdf_page_inspection"},
    },
)


TOOL_SPECS = [
    ToolSpec(
        name="list_directory",
        description="列出某个目录下一层（或递归）的文件与子目录清单。",
        risk_level=ToolRiskLevel.READ_ONLY,
        handler=list_directory,
        output=ToolOutputSpec(schema=_LIST_DIRECTORY_OUTPUT_SCHEMA),
        suggested_task_types=(
            "paper_analysis",
            "code_analysis",
            "resource_check",
        ),
        when_to_use=(
            "当你完全不知道某个目录里有什么、需要先摸清目录结构/文件命名规律"
            "再决定下一步读哪个文件时使用；例如刚拿到一个代码仓库，想先看看"
            "顶层有哪些子目录，或者确认某个 configs/ 目录下到底有多少份配置。"
            "如果你已经知道要找的文件名模式（如 *.yaml），应直接用 find_files；"
            "如果你要找的是文件内容而不是文件名，应使用 grep_files。"
        ),
        boundaries=(
            "只返回沙箱可读范围内的路径；越界路径会被拒绝，不会静默返回空列表。",
            "不会返回文件内容，只有名字/类型/大小——想看内容必须再调用 read_file。",
            "recursive=True 时不会跳过体积很大的目录（如 .git 内部对象），"
            "对超大仓库可能返回很多条目；如果只是想确认某类文件存在，"
            "优先使用 find_files 加 glob 模式做定向查找。",
        ),
        returns=(
            "{path: 查询的目录, entries: [{path, type: 'file'|'directory', "
            "size_bytes}], count: 条目数量}"
        ),
        cost_hint="单次调用是本地文件系统遍历，通常在毫秒级；recursive=True 对超大目录会明显变慢。",
        examples=(
            ToolExample(
                when="想知道代码仓库根目录下有哪些一级目录/文件",
                arguments={"path": ".", "recursive": False},
                result={"path": ".", "entries": [{"path": "src", "type": "directory", "size_bytes": None}], "count": 12},
            ),
            ToolExample(
                when="需要递归列出 configs/ 下所有文件（不含隐藏目录）",
                arguments={"path": "configs", "recursive": True},
            ),
        ),
        param_docs={
            "path": ToolParamDoc(
                description="要列出的目录路径，相对沙箱可读根目录；根目录用 '.' 表示。例如 'src/models' 或 '.'。",
                example="src/models",
            ),
            "recursive": ToolParamDoc(
                description="是否递归列出所有子目录下的文件（True）还是只列出这一层（False，默认）。",
                example=False,
            ),
        },
    ),
    ToolSpec(
        name="find_files",
        description="按文件名的 glob 模式查找文件路径。",
        risk_level=ToolRiskLevel.READ_ONLY,
        handler=find_files,
        output=ToolOutputSpec(schema=_FIND_FILES_OUTPUT_SCHEMA),
        suggested_task_types=("code_analysis", "resource_check"),
        when_to_use=(
            "当你已经知道要找的文件名特征（后缀、命名模式），但不知道它在哪个"
            "子目录时使用，例如“找出所有 *.yaml 配置文件”或“找到 train.py 在哪”。"
        ),
        boundaries=(
            "只按【文件名】做 glob 匹配（如 '*.yaml'、'train_*.py'），"
            "不会查看、也不会匹配文件内容——如果目标是“包含某个字符串/函数名的"
            "文件”，必须改用 grep_files，而不是把关键字当作 pattern 传进来。",
            "pattern 是单层 fnmatch 模式（匹配文件名本身），不支持 '**' 跨目录"
            "通配符语法；跨目录递归查找的范围由 root 决定，root 下所有子目录"
            "都会被扫描。",
            "结果按发现顺序返回，不保证按路径字母序排列；超过 max_results 会"
            "截断并在 truncated 字段标记，不代表真实匹配数量。",
        ),
        returns="{pattern, root, matches: [相对 root 的文件路径], truncated: 是否被截断}",
        cost_hint="会递归遍历 root 下所有文件做名字匹配；root 范围越大、文件数越多越慢，避免在仓库根目录反复调用，尽量缩小 root。",
        examples=(
            ToolExample(
                when="查找仓库里所有的 YAML 配置文件",
                arguments={"pattern": "*.yaml", "root": "configs"},
                result={"pattern": "*.yaml", "root": "configs", "matches": ["base.yaml", "exp1.yaml"], "truncated": False},
            ),
            ToolExample(
                when="确认某个训练脚本是否存在",
                arguments={"pattern": "train_*.py", "root": "scripts", "max_results": 20},
            ),
        ),
        param_docs={
            "pattern": ToolParamDoc(
                description="文件名 glob 模式（不是正则、不是内容关键字），例如 '*.yaml'、'train_*.py'、'*.ckpt'。",
                example="*.yaml",
            ),
            "root": ToolParamDoc(
                description="从哪个目录开始递归查找，相对沙箱可读根目录。例如 'configs' 或 '.'（表示整个可读范围）。",
                example="configs",
            ),
            "max_results": ToolParamDoc(
                description="最多返回多少条匹配结果，超过会截断（最大 500）。找单个文件时可以设小一点以更快返回。",
                example=200,
            ),
        },
    ),
    ToolSpec(
        name="grep_files",
        description="在文件内容中做纯文本关键字检索，返回命中的文件路径与行号。",
        risk_level=ToolRiskLevel.READ_ONLY,
        handler=grep_files,
        output=ToolOutputSpec(schema=_GREP_FILES_OUTPUT_SCHEMA),
        suggested_task_types=("code_analysis", "verification"),
        when_to_use=(
            "当你要找的是【出现在文件内容里的字符串】（函数名、报错信息、超参数"
            "名等），但不确定它在哪个文件、哪一行时使用；例如“哪个文件定义了"
            "learning_rate 这个超参数”“哪里抛出了这条报错信息”。"
        ),
        boundaries=(
            "只做大小写可控的纯文本子串匹配，不支持正则表达式；如果需要正则，"
            "先用简单关键字圈定候选文件，再用 read_file 读出来自行判断。",
            "不能按文件名过滤候选文件集合以外的语义（file_glob 仍是文件名"
            "glob，不是内容条件）——如果你其实想按文件名找文件，应使用"
            "find_files 而不是把文件名模式当 query 传进来。",
            "为保证在任意沙箱环境下都能工作，不依赖外部 ripgrep 二进制，"
            "因此在超大代码仓库上的检索速度慢于原生 grep/ripgrep。",
            "二进制文件、无法用 utf-8 解码的文件会被静默跳过，不会报错也不会"
            "出现在结果里。",
        ),
        returns="{query, results: [{path, line, text（命中行内容，截断至 400 字符）}], truncated}",
        cost_hint="需要逐文件读取全文再匹配，比 find_files 慢；如果只是想确认文件是否存在，优先用 find_files 或 list_directory。",
        examples=(
            ToolExample(
                when="查找定义 learning_rate 超参数的位置",
                arguments={"query": "learning_rate", "root": "src", "file_glob": "*.py"},
                result={"query": "learning_rate", "results": [{"path": "config.py", "line": 12, "text": "learning_rate: float = 1e-4"}], "truncated": False},
            ),
            ToolExample(
                when="定位某条具体报错信息在代码里的出处（区分大小写）",
                arguments={"query": "CUDA out of memory", "case_sensitive": True},
            ),
        ),
        param_docs={
            "query": ToolParamDoc(
                description="要搜索的纯文本关键字（不是正则表达式），例如 'learning_rate' 或某条报错信息片段。",
                example="learning_rate",
            ),
            "root": ToolParamDoc(
                description="从哪个目录开始递归搜索，相对沙箱可读根目录。缩小范围可以显著提速。",
                example="src",
            ),
            "file_glob": ToolParamDoc(
                description="只在文件名匹配此 glob 模式的文件里搜索内容，例如 '*.py' 只搜 Python 文件；默认 '*' 表示不限制文件名。",
                example="*.py",
            ),
            "case_sensitive": ToolParamDoc(
                description="是否区分大小写，默认 False（不区分）。查找精确的常量名/报错字符串时建议设为 True 减少误命中。",
                example=False,
            ),
        },
    ),
    ToolSpec(
        name="read_file",
        description="读取一个已知路径的文本文件内容，可指定行号范围分片读取。",
        risk_level=ToolRiskLevel.READ_ONLY,
        handler=read_file,
        output=ToolOutputSpec(schema=_READ_FILE_OUTPUT_SCHEMA),
        suggested_task_types=(
            "paper_analysis",
            "code_analysis",
            "verification",
        ),
        when_to_use=(
            "当你已经通过 find_files/grep_files/list_directory 确定了具体文件"
            "路径（以及可能的行号范围），需要查看其真实内容时使用。不要用它"
            "来“猜测式”地探索目录——如果连文件在哪都不确定，应先用 "
            "list_directory/find_files。"
        ),
        boundaries=(
            "只能读文本文件；PDF 等二进制格式必须使用 read_pdf_text，直接对"
            "PDF 调用 read_file 会读出乱码或报错。",
            "文件超过约 2MB 且未指定 start_line/end_line 时会直接拒绝执行"
            "（而不是截断返回）——遇到这种情况必须显式传入行号范围分片读取，"
            "不能假设它会自动截断。",
            "start_line/end_line 是 1-based 闭区间；超出文件总行数会被自动"
            "夹紧到有效范围，不会报错。",
        ),
        returns="{path, start_line, end_line, total_lines, content（对应区间的原始文本）}",
        cost_hint="大文件全量读取会消耗较多上下文 token；对超过几百行的文件，建议先用 grep_files 定位关键行号，再用 start_line/end_line 只读需要的片段。",
        examples=(
            ToolExample(
                when="读取一个小配置文件的全部内容",
                arguments={"path": "configs/base.yaml"},
            ),
            ToolExample(
                when="只想看某个大文件里报错所在的前后 20 行",
                arguments={"path": "src/train.py", "start_line": 100, "end_line": 140},
                result={"path": "src/train.py", "start_line": 100, "end_line": 140, "total_lines": 512, "content": "..."},
            ),
        ),
        param_docs={
            "path": ToolParamDoc(
                description="要读取的文本文件路径，相对沙箱可读根目录，例如 'src/train.py'。",
                example="src/train.py",
            ),
            "start_line": ToolParamDoc(
                description="起始行号（从 1 开始，含）。不传则从第 1 行开始。大文件必须配合 end_line 一起传，否则可能因文件过大被拒绝。",
                example=100,
            ),
            "end_line": ToolParamDoc(
                description="结束行号（含）。不传则读到文件末尾。",
                example=140,
            ),
        },
    ),
    ToolSpec(
        name="get_file_stat",
        description="只查询文件/目录的元信息（是否存在、大小、修改时间），不读取内容。",
        risk_level=ToolRiskLevel.READ_ONLY,
        handler=get_file_stat,
        output=ToolOutputSpec(schema=_FILE_STAT_OUTPUT_SCHEMA),
        suggested_task_types=("resource_check", "code_analysis"),
        when_to_use=(
            "当你只需要判断“这个路径存不存在/有多大”而不关心内容时使用，"
            "例如检查某个 checkpoint 文件是否已经下载完成、判断数据集目录"
            "大小是否合理。如果还需要看内容，应改用 read_file。"
        ),
        boundaries=(
            "路径不存在时返回 {exists: False}，不会抛异常——调用方必须显式"
            "检查 exists 字段，不能假设调用成功就代表文件存在。",
            "不会展开目录内容（不统计目录内文件数/总大小），只返回目录本身的"
            "元信息；需要统计目录内容请改用 list_directory 或"
            "check_path_resource。",
        ),
        returns="存在时：{path, exists: True, is_dir, size_bytes, modified_at}；不存在时：{path, exists: False}",
        cost_hint="一次 stat 系统调用，开销极小，可放心频繁调用。",
        examples=(
            ToolExample(
                when="确认某个 checkpoint 文件是否已生成",
                arguments={"path": "output/model.ckpt"},
                result={"path": "output/model.ckpt", "exists": True, "is_dir": False, "size_bytes": 483920, "modified_at": 1710000000.0},
            ),
        ),
        param_docs={
            "path": ToolParamDoc(
                description="要查询的文件或目录路径，相对沙箱可读根目录，例如 'output/model.ckpt'。",
                example="output/model.ckpt",
            ),
        },
    ),
    ToolSpec(
        name="read_pdf_text",
        description=(
            "提取 PDF 文件的逐页文本，输出带页码边界标记的 Markdown："
            "标题层级（#/##/###）、图表题注斜体、列表项。内嵌图片不提取。"
        ),
        risk_level=ToolRiskLevel.READ_ONLY,
        handler=read_pdf_text,
        output=ToolOutputSpec(schema=_PDF_TEXT_OUTPUT_SCHEMA),
        suggested_task_types=("paper_analysis",),
        when_to_use=(
            "当输入是 PDF 格式的论文/附录、需要提取文字内容做分析时使用——"
            "这是读取论文 PDF 的唯一正确方式，不要对 PDF 路径调用 read_file。"
        ),
        boundaries=(
            "只能提取“文本层”存在的 PDF；扫描版/纯图片 PDF 提取出的内容会是"
            "空白或乱码，本工具不做 OCR——遇到这种情况应如实报告论文无法通过"
            "文本提取解析，而不是编造内容。",
            "不解析 PDF 内嵌的表格结构、公式渲染或图片，也不提取图片内容；"
            "Markdown 结构标注（标题层级/题注/列表）基于字号与字重的启发式"
            "推断，复杂排版（多栏、表格）提取出的文字顺序可能和视觉阅读顺序"
            "不一致。",
            "页数超过 max_pages 或提取文本超过 max_chars 会直接报错（fail-closed），"
            "而不是静默截断；处理超长论文时应配合 start_page/end_page 分段读取。",
            "start_page/end_page 是原文页码（1 起、闭区间）；范围读取时页码"
            "边界标记仍使用原文页号，页号超出文档实际范围会直接报错。",
        ),
        returns=(
            "{path, content（Markdown：标题层级/题注/列表 + 页码分隔符）, "
            "format: 'pdf'}"
        ),
        cost_hint="需要解析 PDF，页数越多耗时越长；只关心部分页面时传 start_page/end_page 只提取该范围，比全量读取便宜。",
        examples=(
            ToolExample(
                when="提取论文正文用于分析实验设置",
                arguments={"path": "paper.pdf"},
            ),
            ToolExample(
                when="只读论文的附录部分（例如第 13 到 38 页）",
                arguments={"path": "paper.pdf", "start_page": 13, "end_page": 38},
            ),
            ToolExample(
                when="只要纯文本、不要 Markdown 结构标注时",
                arguments={"path": "paper.pdf", "output_format": "text"},
            ),
        ),
        param_docs={
            "path": ToolParamDoc(
                description="PDF 文件路径，相对沙箱可读根目录，例如 'paper.pdf' 或 'appendix.pdf'。",
                example="paper.pdf",
            ),
            "start_page": ToolParamDoc(
                description="起始页码（原文页码，1 起，含端点）；不传则从第 1 页开始。",
                example=13,
            ),
            "end_page": ToolParamDoc(
                description="结束页码（原文页码，含端点）；不传则读到末页。",
                example=38,
            ),
            "max_pages": ToolParamDoc(
                description="本次最多提取多少页（默认为范围大小），超出直接报错，默认 500 页。",
                example=500,
            ),
            "max_chars": ToolParamDoc(
                description="提取文本的最大字符数上限，超出直接报错（fail-closed），默认 2000000。",
                example=2000000,
            ),
            "output_format": ToolParamDoc(
                description="输出格式：'markdown'（默认，带标题层级/题注/列表结构标注）或 'text'（纯线性文本）。",
                example="markdown",
            ),
        },
    ),
    ToolSpec(
        name="inspect_pdf_page",
        description=(
            "最后手段：查看单个 PDF 页的原始版面文本（layout 模式，用空格"
            "近似还原列与表格的对齐），仅在提取文本确实无法理解时仲裁用。"
        ),
        risk_level=ToolRiskLevel.READ_ONLY,
        handler=inspect_pdf_page,
        output=ToolOutputSpec(schema=_PDF_PAGE_OUTPUT_SCHEMA),
        suggested_task_types=("paper_analysis",),
        when_to_use=(
            "仅当你已经读过 read_pdf_text 的提取结果，并且某个具体位置——"
            "典型如表格数值与表头的对应关系、公式符号断裂、多栏顺序混乱——"
            "确实无法理解时，才调用本工具查看那一页的原始版面。调用时必须在"
            " reason 里写明无法理解的具体内容与原因；这是本工具的硬性门槛。"
        ),
        boundaries=(
            "这是最后手段（last resort）：正常流程一律以 read_pdf_text 的"
            "提取结果为准。还没读过提取文本、或 reason 只写“想看看原文”"
            "这类泛泛理由，调用会被直接拒绝。",
            "每次只能查看一页（page_number 从 1 起，返回值含 total_pages 可"
            "校验范围）；不要用它逐页通读整篇论文——通读请用 read_pdf_text。",
            "返回的仍是文本渲染（layout 模式用空格近似二维排布），不是图片："
            "公式渲染、图形内容依旧不可见；该页版面模式不可用时会退回线性"
            "文本（mode 字段标明），此时信息量与提取文本基本相同，"
            "不要指望它解决所有理解问题。",
            "单页内容超过 max_chars 会直接报错（fail-closed）。",
        ),
        returns=(
            "{path, page_number, total_pages, content（该页原始版面文本）, "
            "mode: 'layout'|'text', reason（回显调用理由）, "
            "format: 'pdf_page_inspection'}"
        ),
        cost_hint="单页解析，比 read_pdf_text 便宜，但版面渲染有一定开销；调用次数应控制在个位数，仅在确有必要时使用。",
        examples=(
            ToolExample(
                when="提取文本中 Table 3 的数值列与表头错位，无法确定哪个数值对应哪个方法",
                arguments={
                    "path": "paper.pdf",
                    "page_number": 7,
                    "reason": "提取文本中 Table 3 的数值列与表头无法对应，需要原始版面确认对齐关系",
                },
            ),
        ),
        param_docs={
            "path": ToolParamDoc(
                description="PDF 文件路径，相对沙箱可读根目录，与 read_pdf_text 使用同一份文件。",
                example="paper.pdf",
            ),
            "page_number": ToolParamDoc(
                description="要查看的页码，从 1 开始；范围外会报错并提示总页数。",
                example=7,
            ),
            "reason": ToolParamDoc(
                description=(
                    "为什么必须查看原始版面：至少 10 个字符，指向提取文本中"
                    "具体无法理解的位置和原因。泛泛理由会被拒绝。"
                ),
                example="Table 3 的数值列与表头在提取文本中错位，无法确定对应关系",
            ),
            "max_chars": ToolParamDoc(
                description="单页内容字符数上限，超出直接报错（fail-closed），默认 100000。",
                example=100000,
            ),
        },
    ),
    ToolSpec(
        name="hash_path",
        description="为已隔离输入中的文件或目录计算稳定 SHA-256 摘要。",
        risk_level=ToolRiskLevel.READ_ONLY,
        handler=hash_path,
        output=ToolOutputSpec(schema=_HASH_PATH_OUTPUT_SCHEMA),
        suggested_task_types=("code_analysis", "resource_check", "experiment_execution", "verification"),
        when_to_use="在执行前锁定代码、数据、模型或配置的确切内容，用于生成可追溯运行清单。",
        boundaries=("只读取已进入当前任务沙箱的路径，不能哈希宿主机或其他任务路径。",),
        returns="{path, sha256, kind, git_commit（当目录是 Git 工作树时）}",
        param_docs={
            "path": ToolParamDoc(
                description="要锁定的沙箱虚拟路径，例如 input://repository。",
                example="input://repository",
            )
        },
    ),
]
