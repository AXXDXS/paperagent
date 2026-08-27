"""Structured, fail-closed parsing for model-produced JSON.

The LLM provider may support native JSON Schema responses, but provider-side
format enforcement is not sufficient: gateways can ignore the hint and models
can still return fenced, truncated, or otherwise malformed text.  This module
keeps the final contract check local to the application and provides a small,
conservative repair pass for the common presentation mistakes.
"""

from __future__ import annotations

import ast
import json
import re
import unicodedata
from typing import Any, Callable

from repro_agent.tools.schema_validation import (
    SchemaValidationError,
    validate_argument_limits,
    validate_json_schema,
)


class StructuredOutputError(ValueError):
    """The model response could not be repaired and validated."""


def parse_structured_json(
    content: str,
    schema: dict[str, Any],
    *,
    label: str = "LLM output",
    normalize: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Repair, parse and validate a model JSON response.

    Repairs are intentionally limited to syntax-only transformations:
    Markdown fences, surrounding prose, trailing commas, and Python-style
    literal quoting.  Missing fields, wrong types, unknown fields, and invalid
    enum values are never guessed; they fail closed through the local schema.

    ``normalize`` is an optional caller-supplied coercion hook applied to the
    freshly parsed value *before* validation.  It exists for deterministic
    presentation variants of a valid answer (numeric strings, percent
    suffixes, casing); the strict schema check that follows remains the final
    authority, so anything the hook cannot fix still fails closed.
    """

    raw = str(content or "").strip()
    # NFC 标准化 + 全角→半角：LLM（尤其中文模型）常输出全角竖线
    # ``｜`` (U+FF5C) 或全角引号，这些字符在 json.loads 中是非法的，
    # 但语义上与半角完全等价。在解析前统一做 Unicode 宽度标准化，
    # 避免纯排版差异导致整轮 LLM 输出被判 PARSING_ERROR。
    raw = unicodedata.normalize("NFKC", raw)
    candidates = _json_candidates(raw)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError) as exc:
            last_error = exc
            try:
                value = ast.literal_eval(candidate)
            except (SyntaxError, ValueError, TypeError) as literal_exc:
                last_error = literal_exc
                continue
        if normalize is not None:
            value = normalize(value)
        try:
            validate_argument_limits(value)
            validate_json_schema(value, schema)
        except (SchemaValidationError, TypeError, ValueError) as exc:
            last_error = exc
            # 如果是 "unexpected field" 类型错误，尝试自动剪枝多余字段后重试
            if isinstance(exc, SchemaValidationError) and "unexpected field" in str(exc):
                pruned = _prune_extra_fields(value, schema)
                if pruned is not None and pruned != value:
                    try:
                        validate_argument_limits(pruned)
                        validate_json_schema(pruned, schema)
                        if isinstance(pruned, dict):
                            return pruned
                    except (SchemaValidationError, TypeError, ValueError) as prune_exc:
                        last_error = prune_exc
            continue
        if not isinstance(value, dict):
            last_error = SchemaValidationError("$: expected object")
            continue
        return value

    detail = str(last_error) if last_error else "empty response"
    raise StructuredOutputError(f"{label} is invalid: {detail}")


def _prune_extra_fields(value: Any, schema: dict[str, Any]) -> Any:
    """递归移除 schema 中 additionalProperties=False 时不允许的多余字段。

    LLM（特别是非顶级模型）经常在结构化输出中多加字段（如 ``name``、
    ``description`` 等），这些字段在语义上无害但会触发严格验证失败。
    此函数在验证失败时作为补救措施，递归遍历 value 并移除 schema 不认识
    的字段，然后让调用者重新验证。仅在 additionalProperties=False 时剪枝；
    如果 additionalProperties=True 或缺省（允许额外字段），则原样返回。
    """

    if not isinstance(schema, dict):
        return value
    if not isinstance(value, dict):
        if isinstance(value, list) and isinstance(schema.get("items"), dict):
            return [_prune_extra_fields(item, schema["items"]) for item in value]
        return value

    additional = schema.get("additionalProperties", True)
    properties = schema.get("properties", {})

    if additional is False and isinstance(properties, dict):
        pruned = {}
        for key, item in value.items():
            if key in properties:
                pruned[key] = _prune_extra_fields(item, properties[key])
            # else: 丢弃不在 properties 中的多余字段
        return pruned

    # additionalProperties 不是 False，递归处理子字段
    result = {}
    for key, item in value.items():
        if key in properties:
            result[key] = _prune_extra_fields(item, properties[key])
        else:
            result[key] = item
    return result


def _json_candidates(raw: str) -> list[str]:
    if not raw:
        return []
    candidates: list[str] = []
    stripped = raw.strip()
    candidates.append(stripped)

    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.I | re.S).strip()
    if fenced != stripped:
        candidates.append(fenced)

    extracted = _extract_balanced_json(fenced)
    if extracted and extracted not in candidates:
        candidates.append(extracted)

    for candidate in list(candidates):
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        if repaired != candidate:
            candidates.append(repaired)
    return list(dict.fromkeys(candidates))


def _extract_balanced_json(text: str) -> str:
    """Extract the first balanced object/array while respecting JSON strings."""

    start = next((index for index, char in enumerate(text) if char in "{["), None)
    if start is None:
        return ""
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    stack = [closing]
    in_string = False
    escaped = False
    for index in range(start + 1, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or char != stack[-1]:
                return ""
            stack.pop()
            if not stack:
                return text[start : index + 1]
    return ""


# Native JSON Schema response hints for providers that support them.  The
# local validator remains authoritative for providers that ignore the hint.
#
# Slim contract (body/appendix split): the model no longer copies evidence
# quotes.  Per-parameter evidence is limited to page + confidence; richer
# provenance (scope/provenance/is_inferred) is filled deterministically by
# PaperAnalysisAgent from the task's scope, and method-level summaries live
# in ``method_summary`` (body scope only).  ``expected_results`` stays in the
# contract because the verification loop compares reproduced metrics against
# it; it is requested from the body agent only.
PAPER_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "required": ["parameters"],
    "properties": {
        "method_summary": {"type": "string"},
        "parameters": {"type": "array", "items": {"type": "object", "additionalProperties": True, "required": ["name", "value"], "properties": {
            "name": {"type": "string", "minLength": 1}, "value": {},
            "page": {"type": "string"}, "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        }}},
        "expected_results": {"type": "object", "additionalProperties": {"type": "object", "additionalProperties": True, "required": ["value", "tolerance_type", "tolerance"], "properties": {
            "value": {"type": "number"}, "tolerance_type": {"type": "string", "enum": ["absolute", "relative", "std_multiple"]},
            "tolerance": {"type": "number", "minimum": 0}, "tolerance_basis": {"type": "string"},
        }}},
        "notes": {"type": "string"},
    },
}

CODE_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": True,
    "required": ["entry_points"],
    "properties": {
        "entry_points": {"type": "array", "items": {"type": "string"}},
        "config_system": {"type": "string"}, "data_pipeline_summary": {"type": "string"},
        "model_pipeline_summary": {"type": "string"}, "training_pipeline_summary": {"type": "string"},
        "inference_pipeline_summary": {"type": "string"}, "evaluation_pipeline_summary": {"type": "string"},
        "effective_parameters": {"type": "object"}, "experiment_output_paths": {"type": "array", "items": {"type": "string"}},
        "matched_run_scripts": {"type": "object", "additionalProperties": {"type": "string"}},
        "tier_commands": {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1}},
        "required_user_configuration": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "kind", "delivery", "required", "reason", "source_ref"],
                "properties": {
                    "name": {"type": "string", "minLength": 1, "maxLength": 128},
                    "kind": {"type": "string", "enum": ["model_name", "api_base", "credential_env", "other"]},
                    "delivery": {"type": "string", "enum": ["environment", "command_argument"]},
                    "environment_variable": {"type": "string"},
                    "argument": {"type": "string"},
                    "required": {"type": "boolean"},
                    "reason": {"type": "string", "minLength": 1},
                    "source_ref": {"type": "string", "minLength": 1},
                },
            },
        },
        "repository_digest": {"type": "string"},
        "analysis_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "start_line", "end_line", "reason"],
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                    "symbol": {"type": "string"},
                    "reason": {"type": "string"},
                    "file_digest": {"type": "string"},
                },
            },
        },
        "analysis_coverage": {"type": "object"},
    },
}

REFLECTION_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["likely_source", "hypotheses", "suggested_audit_tasks"],
    "properties": {
        "likely_source": {"type": "string", "enum": ["execution_error", "config_difference", "randomness", "undisclosed_detail", "unknown"]},
        "hypotheses": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["category", "description", "priority", "confidence", "required_checks"], "properties": {
            "category": {"type": "string"}, "description": {"type": "string"}, "priority": {"type": "integer"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "required_checks": {"type": "array", "items": {"type": "string"}},
        }}},
        "suggested_audit_tasks": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["dimension", "description", "priority"], "properties": {
            "dimension": {"type": "string", "enum": ["A", "B", "C", "D", "E"]}, "description": {"type": "string"}, "priority": {"type": "integer"},
        }}},
    },
}

CODING_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["summary", "files", "unit_test"],
    "properties": {
        "summary": {"type": "string"},
        "files": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["path", "content"], "properties": {"path": {"type": "string", "minLength": 1}, "content": {"type": "string"}}}},
        "unit_test": {"type": ["object", "null"], "additionalProperties": False, "required": ["path", "content"], "properties": {"path": {"type": "string", "minLength": 1}, "content": {"type": "string"}}},
        "reusable_code_candidates": {
            "type": "array",
            "maxItems": 10,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "purpose", "functional_key", "code", "entry_function",
                    "input_schema", "output_schema", "tests",
                    "generalization_reason", "suggested_task_types",
                    "dependencies", "risk_level", "requires_network",
                ],
                "properties": {
                    "purpose": {"type": "string", "minLength": 1},
                    "functional_key": {"type": "string", "minLength": 1},
                    "code": {"type": "string", "minLength": 1},
                    "entry_function": {"type": "string", "minLength": 1},
                    "input_schema": {"type": "object"},
                    "output_schema": {"type": "object"},
                    "tests": {
                        "type": "array", "minItems": 1,
                        "items": {
                            "type": "object", "additionalProperties": False,
                            "required": ["arguments", "expected"],
                            "properties": {
                                "arguments": {"type": "object"},
                                "expected": {},
                            },
                        },
                    },
                    "generalization_reason": {"type": "string", "minLength": 1},
                    "suggested_task_types": {"type": "array", "items": {"type": "string"}},
                    "dependencies": {"type": "array", "items": {"type": "string"}},
                    "risk_level": {
                        "type": "string",
                        "enum": ["read_only", "restricted_write", "high_risk"],
                    },
                    "requires_network": {"type": "boolean"},
                },
            },
        },
    },
}

DECISION_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["decision", "reason"],
    "properties": {"decision": {"type": "string", "enum": ["retry", "split", "add_prerequisite", "ask_user", "terminal_failure"]}, "reason": {"type": "string"}},
}

DEPENDENCY_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "required": ["dependency_analysis"],
    "properties": {"dependency_analysis": {"type": "string"}},
}
