"""Lightweight repository index and retrieval tools for code-analysis tasks.

The repository itself stays outside the model context.  These tools build a
bounded structural index in deterministic Python code, render a token-budgeted
repository map, and retrieve exact symbol/file slices on demand.  The index is
read-only, dependency-free, and incrementally reused in-process; it never
writes into the inspected repository.
"""

from __future__ import annotations

import ast
import hashlib
import math
import os
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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

_INDEX_VERSION = 1
_MAX_INDEX_FILE_BYTES = 1_000_000
_MAX_TOTAL_INDEX_BYTES = 48_000_000
_DEFAULT_MAX_FILES = 5_000
_MAX_SEARCH_RESULTS = 50
_MAX_PREVIEW_CHARS = 2_400
_MAX_PREVIEW_LINES = 80

_IGNORED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    "site-packages",
    "dist",
    "build",
    "target",
    ".next",
    ".tox",
    ".nox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".cache",
    "coverage",
    "htmlcov",
    "wandb",
    "checkpoints",
}

_LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".pyi": "python",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".ini": "config",
    ".cfg": "config",
    ".md": "markdown",
    ".rst": "markdown",
    ".txt": "text",
    ".ipynb": "notebook",
}

_SPECIAL_FILENAMES = {
    "dockerfile": "dockerfile",
    "makefile": "makefile",
    "justfile": "makefile",
    "requirements.txt": "requirements",
    "environment.yml": "yaml",
    "environment.yaml": "yaml",
}

_SKIPPED_FILENAMES = {
    "package-lock.json",
    "npm-shrinkwrap.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
}

_GENERIC_SYMBOL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("class", re.compile(r"^\s*(?:export\s+)?(?:public\s+)?class\s+([A-Za-z_$][\w$]*)")),
    ("function", re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(")),
    ("function", re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\(")),
    ("function", re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)\s*\(")),
    ("class", re.compile(r"^\s*(?:public\s+|private\s+|protected\s+)?(?:abstract\s+)?(?:class|interface)\s+([A-Za-z_]\w*)")),
    ("heading", re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")),
)

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]*|\d+(?:\.\d+)?")


@dataclass(frozen=True)
class _Symbol:
    name: str
    qualified_name: str
    kind: str
    start_line: int
    end_line: int
    signature: str


@dataclass
class _FileRecord:
    path: str
    language: str
    size_bytes: int
    line_count: int
    digest: str
    content: str
    symbols: list[_Symbol] = field(default_factory=list)
    imports: set[str] = field(default_factory=set)
    references: set[str] = field(default_factory=set)


@dataclass
class _RepositoryIndex:
    root: Path
    records: dict[str, _FileRecord]
    stamps: dict[str, tuple[int, int]]
    digest: str
    skipped_file_count: int
    ignored_directories: int
    manifest_truncated: bool


class _PythonStructureVisitor(ast.NodeVisitor):
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.symbols: list[_Symbol] = []
        self.imports: set[str] = set()
        self.references: set[str] = set()
        self._scope: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        bases = []
        for base in node.bases:
            try:
                bases.append(ast.unparse(base))
            except Exception:  # pragma: no cover - unusual malformed AST node
                continue
        signature = f"class {node.name}"
        if bases:
            signature += f"({', '.join(bases)})"
        self._add_symbol(node, node.name, "class", signature)
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function(node, async_prefix="")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_function(node, async_prefix="async ")

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, *, async_prefix: str
    ) -> None:
        try:
            arguments = ast.unparse(node.args)
        except Exception:  # pragma: no cover - ast.unparse is available on py3.10+
            arguments = "..."
        signature = f"{async_prefix}def {node.name}({arguments})"
        self._add_symbol(node, node.name, "method" if self._scope else "function", signature)
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def _add_symbol(self, node: Any, name: str, kind: str, signature: str) -> None:
        qualified = ".".join([*self._scope, name])
        start = max(1, int(getattr(node, "lineno", 1)))
        end = max(start, int(getattr(node, "end_lineno", start)))
        self.symbols.append(
            _Symbol(name, qualified, kind, start, end, signature[:500])
        )

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            self.imports.add(alias.name)
            self.references.add(alias.asname or alias.name.rsplit(".", 1)[-1])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if node.module:
            self.imports.add(node.module)
        for alias in node.names:
            self.references.add(alias.asname or alias.name)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Load):
            self.references.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        self.references.add(node.attr)
        self.generic_visit(node)


_CACHE: dict[tuple[str, int], _RepositoryIndex] = {}
_CACHE_LOCK = threading.RLock()


def _language_for(path: Path) -> str | None:
    lower_name = path.name.lower()
    if lower_name in _SKIPPED_FILENAMES or lower_name.endswith(".min.js"):
        return None
    if lower_name in _SPECIAL_FILENAMES:
        return _SPECIAL_FILENAMES[lower_name]
    if lower_name.startswith("readme"):
        return "markdown"
    return _LANGUAGE_BY_SUFFIX.get(path.suffix.lower())


def _scan_file_stamps(
    root: Path, max_files: int
) -> tuple[dict[str, tuple[int, int]], int, int, bool]:
    stamps: dict[str, tuple[int, int]] = {}
    skipped = 0
    ignored_directories = 0
    truncated = False

    for current, dirs, files in os.walk(root, followlinks=False):
        kept_dirs = []
        for name in sorted(dirs):
            if name in _IGNORED_DIRECTORIES or (name.startswith(".") and name != ".github"):
                ignored_directories += 1
            else:
                kept_dirs.append(name)
        dirs[:] = kept_dirs
        for name in sorted(files):
            path = Path(current) / name
            if path.is_symlink() or name.startswith("."):
                skipped += 1
                continue
            if _language_for(path) is None:
                skipped += 1
                continue
            try:
                stat = path.stat()
            except OSError:
                skipped += 1
                continue
            if stat.st_size > _MAX_INDEX_FILE_BYTES:
                skipped += 1
                continue
            if len(stamps) >= max_files:
                truncated = True
                return stamps, skipped, ignored_directories, truncated
            relative = path.relative_to(root).as_posix()
            stamps[relative] = (int(stat.st_size), int(stat.st_mtime_ns))
    return stamps, skipped, ignored_directories, truncated


def _generic_symbols(lines: list[str], language: str) -> list[_Symbol]:
    symbols: list[_Symbol] = []
    for lineno, line in enumerate(lines, start=1):
        for kind, pattern in _GENERIC_SYMBOL_PATTERNS:
            if kind == "heading" and language != "markdown":
                continue
            match = pattern.match(line)
            if not match:
                continue
            name = match.group(1).strip()[:200]
            symbols.append(
                _Symbol(
                    name=name,
                    qualified_name=name,
                    kind=kind,
                    start_line=lineno,
                    end_line=lineno,
                    signature=line.strip()[:500],
                )
            )
            break
    for index, symbol in enumerate(symbols):
        next_line = symbols[index + 1].start_line - 1 if index + 1 < len(symbols) else len(lines)
        symbols[index] = _Symbol(
            symbol.name,
            symbol.qualified_name,
            symbol.kind,
            symbol.start_line,
            max(symbol.start_line, min(next_line, symbol.start_line + 120)),
            symbol.signature,
        )
    return symbols


def _parse_record(root: Path, relative: str, stamp: tuple[int, int]) -> _FileRecord | None:
    path = root / relative
    language = _language_for(path)
    if language is None:
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:8_192]:
        return None
    content = raw.decode("utf-8", errors="replace")
    lines = content.splitlines()
    symbols: list[_Symbol] = []
    imports: set[str] = set()
    references: set[str] = set()
    if language == "python":
        try:
            tree = ast.parse(content, filename=relative)
            visitor = _PythonStructureVisitor(lines)
            visitor.visit(tree)
            symbols = visitor.symbols
            imports = visitor.imports
            references = visitor.references
        except (SyntaxError, ValueError):
            symbols = _generic_symbols(lines, language)
    else:
        symbols = _generic_symbols(lines, language)
        # Lightweight reference candidates are sufficient for graph ranking;
        # exact semantic resolution remains a language-server-sized problem.
        references = set(_query_terms(content[:200_000]))
    return _FileRecord(
        path=relative,
        language=language,
        size_bytes=stamp[0],
        line_count=len(lines),
        digest=hashlib.sha256(raw).hexdigest(),
        content=content,
        symbols=symbols,
        imports=imports,
        references=references,
    )


def _repository_digest(records: dict[str, _FileRecord]) -> str:
    digest = hashlib.sha256()
    digest.update(f"code-index:v{_INDEX_VERSION}\n".encode())
    for path in sorted(records):
        record = records[path]
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _get_index(root: Path, max_files: int) -> tuple[_RepositoryIndex, bool]:
    key = (str(root), max_files)
    stamps, skipped, ignored_directories, truncated = _scan_file_stamps(root, max_files)
    with _CACHE_LOCK:
        previous = _CACHE.get(key)
        if previous is not None and previous.stamps == stamps:
            return previous, True

        records: dict[str, _FileRecord] = {}
        total_bytes = 0
        for relative in sorted(stamps):
            stamp = stamps[relative]
            if total_bytes + stamp[0] > _MAX_TOTAL_INDEX_BYTES:
                skipped += len(stamps) - len(records)
                truncated = True
                break
            record = None
            if previous is not None and previous.stamps.get(relative) == stamp:
                record = previous.records.get(relative)
            if record is None:
                record = _parse_record(root, relative, stamp)
            if record is None:
                skipped += 1
                continue
            records[relative] = record
            total_bytes += record.size_bytes

        index = _RepositoryIndex(
            root=root,
            records=records,
            stamps=stamps,
            digest=_repository_digest(records),
            skipped_file_count=skipped,
            ignored_directories=ignored_directories,
            manifest_truncated=truncated,
        )
        _CACHE[key] = index
        return index, False


def _query_terms(value: str) -> list[str]:
    terms: list[str] = []
    for raw in _TOKEN_RE.findall(str(value or "")):
        lowered = raw.lower().strip("_./-")
        if len(lowered) >= 2:
            terms.append(lowered)
        for part in re.split(r"[_./-]+", lowered):
            if len(part) >= 2:
                terms.append(part)
        camel_parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", raw)
        terms.extend(part.lower() for part in camel_parts if len(part) >= 2)
    return list(dict.fromkeys(terms))[:64]


def _role_score(path: str) -> tuple[float, list[str]]:
    lower = path.lower()
    basename = Path(lower).name
    score = max(0.0, 2.5 - path.count("/") * 0.2)
    reasons: list[str] = []
    roles = {
        "readme": 12.0,
        "pyproject": 11.0,
        "setup.py": 10.0,
        "requirements": 9.0,
        "environment": 9.0,
        "dockerfile": 8.0,
        "makefile": 8.0,
        "train": 10.0,
        "main": 8.0,
        "run": 7.0,
        "eval": 9.0,
        "test": 4.0,
        "config": 8.0,
        "model": 6.0,
        "data": 6.0,
        "infer": 7.0,
        "predict": 7.0,
        "metric": 7.0,
    }
    for token, weight in roles.items():
        if token in basename or f"/{token}" in lower:
            score += weight
            reasons.append(f"role:{token}")
    return score, reasons[:4]


def _query_score(record: _FileRecord, terms: list[str]) -> tuple[float, list[str]]:
    if not terms:
        return 0.0, []
    path_lower = record.path.lower()
    symbol_text = " ".join(
        f"{symbol.name} {symbol.qualified_name} {symbol.signature}" for symbol in record.symbols
    ).lower()
    content_lower = record.content.lower()
    score = 0.0
    reasons: list[str] = []
    for term in terms:
        if term in path_lower:
            score += 7.0
            reasons.append(f"path:{term}")
        if term in symbol_text:
            score += 5.0
            reasons.append(f"symbol:{term}")
        count = content_lower.count(term)
        if count:
            score += min(4.0, 0.6 + math.log2(count + 1))
            reasons.append(f"content:{term}")
    return score, list(dict.fromkeys(reasons))[:6]


def _rank_files(index: _RepositoryIndex, query: str) -> list[tuple[_FileRecord, float, list[str]]]:
    terms = _query_terms(query)
    owners: dict[str, set[str]] = defaultdict(set)
    for record in index.records.values():
        for symbol in record.symbols:
            if len(symbol.name) >= 3:
                owners[symbol.name].add(record.path)
    inbound: Counter[str] = Counter()
    for record in index.records.values():
        for reference in record.references:
            for owner in owners.get(reference, ()):
                if owner != record.path:
                    inbound[owner] += 1

    ranked = []
    for record in index.records.values():
        role, reasons = _role_score(record.path)
        query_value, query_reasons = _query_score(record, terms)
        centrality = min(12.0, math.log2(inbound[record.path] + 1) * 2.0)
        if centrality:
            reasons.append("referenced-symbols")
        score = role + query_value + centrality
        ranked.append((record, round(score, 4), [*query_reasons, *reasons][:8]))
    ranked.sort(key=lambda item: (-item[1], item[0].path))
    return ranked


def _render_repo_map(
    index: _RepositoryIndex,
    ranked: list[tuple[_FileRecord, float, list[str]]],
    *,
    token_budget: int,
) -> tuple[str, bool]:
    char_budget = max(1_000, token_budget * 4)
    language_counts = Counter(record.language for record in index.records.values())
    lines = [
        f"repository_digest: {index.digest}",
        f"files: {len(index.records)}; languages: "
        + ", ".join(f"{name}={count}" for name, count in sorted(language_counts.items())),
    ]
    used = sum(len(line) + 1 for line in lines)
    included = 0
    for record, score, reasons in ranked:
        header = (
            f"\n{record.path} [{record.language}; {record.line_count} lines; "
            f"score={score:g}]"
        )
        symbol_lines = [f"  {symbol.kind} {symbol.signature}" for symbol in record.symbols[:12]]
        if record.imports:
            symbol_lines.append("  imports " + ", ".join(sorted(record.imports)[:8]))
        block = "\n".join([header, *symbol_lines])
        if used + len(block) + 1 > char_budget:
            continue
        lines.append(block)
        used += len(block) + 1
        included += 1
    return "\n".join(lines), included < len(ranked)


def get_repository_map(
    ctx: SandboxContext,
    root: str,
    *,
    query: str = "",
    token_budget: int = 2_500,
    max_files: int = _DEFAULT_MAX_FILES,
) -> dict[str, Any]:
    """Build a compact, query-aware map of a repository."""

    resolved_root = Path(ctx.resolve_readable_path(root))
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise ToolExecutionError(f"repository root is not a directory: {root}")
    token_budget = min(12_000, max(500, int(token_budget)))
    max_files = min(20_000, max(100, int(max_files)))
    index, cache_hit = _get_index(resolved_root, max_files)
    ranked = _rank_files(index, query)
    repo_map, map_truncated = _render_repo_map(index, ranked, token_budget=token_budget)
    language_counts = Counter(record.language for record in index.records.values())
    ranked_files = [
        {
            "path": record.path,
            "language": record.language,
            "size_bytes": record.size_bytes,
            "line_count": record.line_count,
            "symbol_count": len(record.symbols),
            "score": score,
            "reasons": reasons,
        }
        for record, score, reasons in ranked[:40]
    ]
    return {
        "index_version": _INDEX_VERSION,
        "root": root,
        "repository_digest": index.digest,
        "file_count": len(index.stamps),
        "indexed_file_count": len(index.records),
        "skipped_file_count": index.skipped_file_count,
        "ignored_directories": index.ignored_directories,
        "languages": dict(sorted(language_counts.items())),
        "repo_map": repo_map,
        "ranked_files": ranked_files,
        "cache_hit": cache_hit,
        "truncated": bool(index.manifest_truncated or map_truncated),
    }


def _snippet(record: _FileRecord, start: int, end: int) -> tuple[int, int, str]:
    lines = record.content.splitlines()
    if not lines:
        return 1, 1, ""
    start = max(1, min(start, len(lines)))
    end = max(start, min(end, len(lines), start + _MAX_PREVIEW_LINES - 1))
    text = "\n".join(f"{line_no}: {lines[line_no - 1]}" for line_no in range(start, end + 1))
    if len(text) > _MAX_PREVIEW_CHARS:
        text = text[:_MAX_PREVIEW_CHARS] + "\n[preview truncated]"
    return start, end, text


def _first_content_match(record: _FileRecord, terms: list[str]) -> int | None:
    for lineno, line in enumerate(record.content.splitlines(), start=1):
        lowered = line.lower()
        if any(term in lowered for term in terms):
            return lineno
    return None


def search_repository_code(
    ctx: SandboxContext,
    root: str,
    query: str,
    *,
    path_prefix: str = "",
    max_results: int = 12,
    max_files: int = _DEFAULT_MAX_FILES,
) -> dict[str, Any]:
    """Search the lightweight index and return exact, line-addressed slices."""

    terms = _query_terms(query)
    if not terms:
        raise ToolExecutionError("query must contain at least one searchable term")
    resolved_root = Path(ctx.resolve_readable_path(root))
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise ToolExecutionError(f"repository root is not a directory: {root}")
    max_results = min(_MAX_SEARCH_RESULTS, max(1, int(max_results)))
    max_files = min(20_000, max(100, int(max_files)))
    index, _ = _get_index(resolved_root, max_files)
    normalized_prefix = path_prefix.strip().replace("\\", "/").lstrip("./")

    candidates: list[dict[str, Any]] = []
    for record in index.records.values():
        if normalized_prefix and not record.path.startswith(normalized_prefix):
            continue
        file_score, file_reasons = _query_score(record, terms)
        if file_score <= 0:
            continue
        matched_symbol = False
        for symbol in record.symbols:
            symbol_text = f"{symbol.name} {symbol.qualified_name} {symbol.signature}".lower()
            matched = [term for term in terms if term in symbol_text]
            if not matched:
                continue
            matched_symbol = True
            exact_bonus = sum(
                14.0 for term in terms if term in {symbol.name.lower(), symbol.qualified_name.lower()}
            )
            score = file_score + exact_bonus + len(matched) * 7.0
            start, end, preview = _snippet(record, symbol.start_line, symbol.end_line)
            candidates.append(
                {
                    "path": record.path,
                    "language": record.language,
                    "score": round(score, 4),
                    "match_type": "symbol",
                    "symbol": symbol.qualified_name,
                    "kind": symbol.kind,
                    "start_line": start,
                    "end_line": end,
                    "signature": symbol.signature,
                    "preview": preview,
                    "file_digest": record.digest,
                    "reasons": [f"symbol:{term}" for term in matched[:5]],
                }
            )
        match_line = _first_content_match(record, terms)
        if match_line is not None and (not matched_symbol or len(candidates) < max_results * 4):
            start, end, preview = _snippet(record, match_line - 5, match_line + 8)
            match_type = "path" if any(term in record.path.lower() for term in terms) else "content"
            candidates.append(
                {
                    "path": record.path,
                    "language": record.language,
                    "score": round(file_score, 4),
                    "match_type": match_type,
                    "symbol": "",
                    "kind": "file",
                    "start_line": start,
                    "end_line": end,
                    "signature": "",
                    "preview": preview,
                    "file_digest": record.digest,
                    "reasons": file_reasons,
                }
            )

    candidates.sort(
        key=lambda item: (-float(item["score"]), item["path"], int(item["start_line"]))
    )
    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for item in candidates:
        key = (item["path"], item["symbol"], item["start_line"])
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(item)
    return {
        "index_version": _INDEX_VERSION,
        "root": root,
        "repository_digest": index.digest,
        "query": query,
        "total_matches": len(deduplicated),
        "results": deduplicated[:max_results],
        "truncated": len(deduplicated) > max_results,
    }


_MAP_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "index_version", "root", "repository_digest", "file_count",
        "indexed_file_count", "skipped_file_count", "ignored_directories",
        "languages", "repo_map", "ranked_files", "cache_hit", "truncated",
    ],
    "properties": {
        "index_version": {"type": "integer"},
        "root": {"type": "string"},
        "repository_digest": {"type": "string"},
        "file_count": {"type": "integer", "minimum": 0},
        "indexed_file_count": {"type": "integer", "minimum": 0},
        "skipped_file_count": {"type": "integer", "minimum": 0},
        "ignored_directories": {"type": "integer", "minimum": 0},
        "languages": {"type": "object", "additionalProperties": {"type": "integer", "minimum": 0}},
        "repo_map": {"type": "string"},
        "ranked_files": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "language", "size_bytes", "line_count", "symbol_count", "score", "reasons"],
                "properties": {
                    "path": {"type": "string"}, "language": {"type": "string"},
                    "size_bytes": {"type": "integer", "minimum": 0},
                    "line_count": {"type": "integer", "minimum": 0},
                    "symbol_count": {"type": "integer", "minimum": 0},
                    "score": {"type": "number"},
                    "reasons": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "cache_hit": {"type": "boolean"},
        "truncated": {"type": "boolean"},
    },
}

_SEARCH_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "path", "language", "score", "match_type", "symbol", "kind",
        "start_line", "end_line", "signature", "preview", "file_digest", "reasons",
    ],
    "properties": {
        "path": {"type": "string"}, "language": {"type": "string"},
        "score": {"type": "number"},
        "match_type": {"type": "string", "enum": ["symbol", "content", "path"]},
        "symbol": {"type": "string"}, "kind": {"type": "string"},
        "start_line": {"type": "integer", "minimum": 1},
        "end_line": {"type": "integer", "minimum": 1},
        "signature": {"type": "string"}, "preview": {"type": "string"},
        "file_digest": {"type": "string"},
        "reasons": {"type": "array", "items": {"type": "string"}},
    },
}

_SEARCH_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["index_version", "root", "repository_digest", "query", "total_matches", "results", "truncated"],
    "properties": {
        "index_version": {"type": "integer"}, "root": {"type": "string"},
        "repository_digest": {"type": "string"}, "query": {"type": "string"},
        "total_matches": {"type": "integer", "minimum": 0},
        "results": {"type": "array", "items": _SEARCH_RESULT_SCHEMA},
        "truncated": {"type": "boolean"},
    },
}


TOOL_SPECS = [
    ToolSpec(
        name="get_repository_map",
        description="构建轻量代码索引并返回受 token 预算约束的仓库地图。",
        risk_level=ToolRiskLevel.READ_ONLY,
        handler=get_repository_map,
        suggested_task_types=("code_analysis",),
        when_to_use="大型仓库分析的第一步：先了解高相关文件、类、函数、语言分布和仓库内容摘要，再决定精读位置。",
        boundaries=(
            "不会把完整仓库返回给模型，只返回预算内的高相关文件与符号签名。",
            "索引完全只读；跳过依赖目录、构建产物、二进制、大于 1MB 的单文件和超过总索引预算的内容。",
            "只使用轻量静态分析，跨语言动态调用关系并非完整调用图。",
        ),
        returns="{repository_digest, 文件/语言统计, repo_map, ranked_files, cache_hit, truncated}",
        cost_hint="首次调用遍历并解析仓库；同一进程内未变化文件会复用缓存，后续查询较快。",
        examples=(ToolExample(when="为目标实验生成仓库地图", arguments={"root": "input://repository", "query": "main experiment train evaluate", "token_budget": 2500}),),
        param_docs={
            "root": ToolParamDoc("代码仓库的沙箱虚拟路径，例如 input://repository。", "input://repository"),
            "query": ToolParamDoc("当前分析目标，用于调整文件/符号排名；可包含实验名、数据集名或指标名。", "main experiment accuracy"),
            "token_budget": ToolParamDoc("repo_map 的近似 token 预算，范围 500-12000。", 2500),
            "max_files": ToolParamDoc("最多纳入索引的文本/代码文件数，默认 5000。", 5000),
        },
        output=ToolOutputSpec(schema=_MAP_OUTPUT_SCHEMA),
    ),
    ToolSpec(
        name="search_repository_code",
        description="在轻量代码索引中按路径、内容和类/函数符号混合检索，返回带真实行号的代码片段。",
        risk_level=ToolRiskLevel.READ_ONLY,
        handler=search_repository_code,
        suggested_task_types=("code_analysis",),
        when_to_use="已经看过 Repo Map，需要从文件级进一步定位到类、函数或精确代码行时使用；可以根据上一轮结果中的新符号继续检索。",
        boundaries=(
            "这是本地词法+符号检索，不是向量语义搜索；查询应包含可能出现在路径、标识符或代码中的词。",
            "返回片段有严格长度限制；需要查看完整函数或更大上下文时，再用 read_file 按 start_line/end_line 精读。",
        ),
        returns="{query, repository_digest, results: [{path, symbol, kind, start_line, end_line, preview, score, reasons}], truncated}",
        cost_hint="复用仓库索引，通常只做内存评分；首次调用可能触发索引构建。",
        examples=(ToolExample(when="查找训练入口和优化器配置", arguments={"root": "input://repository", "query": "train optimizer learning_rate", "max_results": 10}),),
        param_docs={
            "root": ToolParamDoc("代码仓库的沙箱虚拟路径。", "input://repository"),
            "query": ToolParamDoc("空格分隔的文件名、符号名、配置项或代码关键字。", "evaluate accuracy metric"),
            "path_prefix": ToolParamDoc("可选的相对仓库目录前缀，用于把检索限制在某个模块。", "src/training"),
            "max_results": ToolParamDoc("最多返回的匹配数，范围 1-50。", 12),
            "max_files": ToolParamDoc("索引文件上限，应与 get_repository_map 保持一致。", 5000),
        },
        output=ToolOutputSpec(schema=_SEARCH_OUTPUT_SCHEMA),
    ),
]
