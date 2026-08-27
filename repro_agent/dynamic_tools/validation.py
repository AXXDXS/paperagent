"""Deterministic admission checks and conservative candidate matching."""

from __future__ import annotations

import ast
import hashlib
import re
from typing import Any

from repro_agent.dynamic_tools.models import ReusableCodeCandidate, canonical_json
from repro_agent.tools.schema_validation import (
    SchemaValidationError,
    ensure_lossless_json,
    validate_json_schema,
)

MAX_CANDIDATE_CODE_CHARS = 30_000
MAX_CANDIDATE_TESTS = 20
SAFE_IMPORT_ROOTS = {
    "collections",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "json",
    "math",
    "operator",
    "re",
    "statistics",
    "string",
    "typing",
}
FORBIDDEN_CALL_NAMES = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "getattr",
    "input",
    "locals",
    "open",
    "setattr",
    "delattr",
    "vars",
    "__import__",
}
FORBIDDEN_NODES = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Nonlocal,
    ast.Raise,
    ast.With,
)


class CandidateValidationError(ValueError):
    pass


class _ShapeNormalizer(ast.NodeTransformer):
    """Erase local naming differences while retaining called APIs/constants."""

    def __init__(self) -> None:
        self.locals: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef):  # noqa: N802
        self.locals.update(arg.arg for arg in node.args.args)
        self.locals.update(arg.arg for arg in node.args.kwonlyargs)
        node.name = "function"
        for arg in [*node.args.args, *node.args.kwonlyargs]:
            arg.arg = "value"
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name):  # noqa: N802
        if isinstance(node.ctx, ast.Store):
            self.locals.add(node.id)
            node.id = "local"
        elif node.id in self.locals:
            node.id = "local"
        return node


def validate_candidate(candidate: ReusableCodeCandidate) -> str:
    if not candidate.purpose or len(candidate.purpose) > 500:
        raise CandidateValidationError("purpose must contain 1..500 characters")
    if not candidate.normalized_functional_key:
        raise CandidateValidationError("functional_key must contain an ASCII identifier")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", candidate.entry_function):
        raise CandidateValidationError("entry_function is not a safe Python identifier")
    if not candidate.code.strip() or len(candidate.code) > MAX_CANDIDATE_CODE_CHARS:
        raise CandidateValidationError(
            f"candidate code must contain 1..{MAX_CANDIDATE_CODE_CHARS} characters"
        )
    if len(candidate.tests) > MAX_CANDIDATE_TESTS:
        raise CandidateValidationError(
            f"candidate has more than {MAX_CANDIDATE_TESTS} tests"
        )
    if candidate.requires_network:
        raise CandidateValidationError("generated tools cannot require network access")
    undeclared = set(candidate.dependencies) - SAFE_IMPORT_ROOTS
    if undeclared:
        raise CandidateValidationError(
            "generated tool dependency is not allowlisted: " + ", ".join(sorted(undeclared))
        )

    _validate_schema_definition(candidate.input_schema, path="input_schema")
    _validate_schema_definition(candidate.output_schema, path="output_schema")
    if candidate.input_schema.get("type") != "object":
        raise CandidateValidationError("input_schema root type must be object")

    try:
        tree = ast.parse(candidate.code, mode="exec")
    except SyntaxError as exc:
        raise CandidateValidationError(f"candidate code has invalid syntax: {exc.msg}") from exc
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if candidate.entry_function not in functions:
        raise CandidateValidationError("entry_function is not defined at module top level")
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and _is_literal_assignment(node):
            continue
        raise CandidateValidationError(
            f"module-level {type(node).__name__} is not allowed in generated tools"
        )
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise CandidateValidationError(
                f"{type(node).__name__} is not allowed in generated tools"
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                _validate_import(alias.name)
        elif isinstance(node, ast.ImportFrom):
            _validate_import(node.module or "")
        elif isinstance(node, ast.Call):
            name = _called_name(node.func)
            if name in FORBIDDEN_CALL_NAMES:
                raise CandidateValidationError(f"call to {name} is forbidden")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise CandidateValidationError("dunder attribute access is forbidden")
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            raise CandidateValidationError("dunder name access is forbidden")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if re.search(r"(?:sk-[A-Za-z0-9_-]{16,}|AKIA[A-Z0-9]{16}|Bearer\s+[A-Za-z0-9._-]{16,})", node.value):
                raise CandidateValidationError("candidate code appears to contain a credential")

    for test in candidate.tests:
        try:
            arguments = ensure_lossless_json(test.arguments)
            expected = ensure_lossless_json(test.expected)
            validate_json_schema(arguments, candidate.input_schema)
            validate_json_schema(expected, candidate.output_schema)
        except SchemaValidationError as exc:
            raise CandidateValidationError(f"candidate test violates schema: {exc}") from exc
    return ast_fingerprint(tree)


def ast_fingerprint(tree_or_code: ast.AST | str) -> str:
    tree = ast.parse(tree_or_code, mode="exec") if isinstance(tree_or_code, str) else tree_or_code
    normalized = _ShapeNormalizer().visit(ast.fix_missing_locations(tree))
    dumped = ast.dump(normalized, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def candidates_match(record: dict[str, Any], candidate: ReusableCodeCandidate, fingerprint: str) -> bool:
    if canonical_json(record.get("input_schema")) != canonical_json(candidate.input_schema):
        return False
    if canonical_json(record.get("output_schema")) != canonical_json(candidate.output_schema):
        return False
    if record.get("code_hash") == candidate.code_hash:
        return True
    if record.get("ast_fingerprint") == fingerprint:
        return True
    if record.get("functional_key") == candidate.normalized_functional_key:
        return _text_similarity(str(record.get("purpose", "")), candidate.purpose) >= 0.55
    return _text_similarity(str(record.get("purpose", "")), candidate.purpose) >= 0.82


def _validate_import(module: str) -> None:
    root = module.split(".", 1)[0]
    if root not in SAFE_IMPORT_ROOTS:
        raise CandidateValidationError(f"import '{module}' is not allowlisted")


def _called_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_literal_assignment(node: ast.Assign | ast.AnnAssign) -> bool:
    value = node.value
    if value is None:
        return False
    try:
        ast.literal_eval(value)
        return True
    except (ValueError, TypeError):
        return False


def _text_similarity(left: str, right: str) -> float:
    def shingles(value: str) -> set[str]:
        folded = re.sub(r"\s+", "", value.casefold())
        words = set(re.findall(r"[a-z0-9_]+", folded))
        chars = {folded[index : index + 2] for index in range(max(0, len(folded) - 1))}
        return words | chars

    a, b = shingles(left), shingles(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _validate_schema_definition(schema: Any, *, path: str) -> None:
    if not isinstance(schema, dict) or not schema:
        raise CandidateValidationError(f"{path} must be a non-empty schema object")
    # Exercise the project's fail-closed keyword checker without pretending an
    # arbitrary sample must satisfy the schema. Unsupported root keywords are
    # rejected before value assertions run.
    try:
        validate_json_schema(None, schema, path=path)
    except SchemaValidationError as exc:
        if "unsupported schema keyword" in str(exc) or "schema " in str(exc):
            raise CandidateValidationError(str(exc)) from exc
    for key in ("properties",):
        children = schema.get(key, {})
        if isinstance(children, dict):
            for name, child in children.items():
                _validate_schema_definition(child, path=f"{path}.{key}.{name}")
    items = schema.get("items")
    if isinstance(items, dict):
        _validate_schema_definition(items, path=f"{path}.items")
    for key in ("additionalProperties", "contains"):
        child = schema.get(key)
        if isinstance(child, dict):
            _validate_schema_definition(child, path=f"{path}.{key}")
    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        for index, child in enumerate(schema.get(key, []) if isinstance(schema.get(key, []), list) else []):
            _validate_schema_definition(child, path=f"{path}.{key}[{index}]")
    if isinstance(schema.get("not"), dict):
        _validate_schema_definition(schema["not"], path=f"{path}.not")
