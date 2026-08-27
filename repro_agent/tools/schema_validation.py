"""Fail-closed validation for the JSON Schema subset used by tool calls.

The project deliberately avoids pulling a second validation runtime into every
worker process.  This validator covers the recursive, assertion-bearing keywords
that ToolSpec emits or accepts: object/array nesting, composition, scalar limits,
patterns, enums and additional-property policies.  Annotation/documentation-only
keywords are ignored; unknown assertion keywords fail closed instead of creating
a false sense of validation.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any


class SchemaValidationError(ValueError):
    """A value does not satisfy its declared tool JSON Schema."""


MAX_TOOL_ARGUMENT_CHARS = 2_000_000
MAX_TOOL_ARGUMENT_NODES = 10_000
MAX_TOOL_ARGUMENT_DEPTH = 20


_ANNOTATION_KEYWORDS = {
    "$schema",
    "$id",
    "title",
    "description",
    "default",
    "deprecated",
    "examples",
    "readOnly",
    "writeOnly",
}
_SUPPORTED_ASSERTION_KEYWORDS = {
    "type",
    "enum",
    "const",
    "allOf",
    "anyOf",
    "oneOf",
    "not",
    "properties",
    "required",
    "additionalProperties",
    "minProperties",
    "maxProperties",
    "items",
    "prefixItems",
    "minItems",
    "maxItems",
    "uniqueItems",
    "contains",
    "minContains",
    "maxContains",
    "minLength",
    "maxLength",
    "pattern",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
}


def validate_json_schema(value: Any, schema: dict[str, Any], *, path: str = "$") -> None:
    """Validate ``value`` or raise a path-aware, value-free error.

    Error messages intentionally never interpolate the rejected value because tool
    arguments can contain credentials.  Only schema paths and public constraints
    are reported and subsequently persisted in audit logs.
    """

    if not isinstance(schema, dict):
        raise SchemaValidationError(f"{path}: schema must be an object")
    unsupported = set(schema) - _ANNOTATION_KEYWORDS - _SUPPORTED_ASSERTION_KEYWORDS
    if unsupported:
        raise SchemaValidationError(
            f"{path}: unsupported schema keyword(s): {', '.join(sorted(unsupported))}"
        )

    _validate_composition(value, schema, path)

    if "const" in schema and value != schema["const"]:
        raise SchemaValidationError(f"{path}: value does not match const")
    if "enum" in schema:
        enum = schema["enum"]
        if not isinstance(enum, list):
            raise SchemaValidationError(f"{path}: schema enum must be an array")
        if value not in enum:
            raise SchemaValidationError(f"{path}: value is not in the allowed enum")

    declared_types = schema.get("type")
    if declared_types is not None:
        types = [declared_types] if isinstance(declared_types, str) else declared_types
        if not isinstance(types, list) or not types or not all(isinstance(t, str) for t in types):
            raise SchemaValidationError(f"{path}: schema type must be a string or string array")
        unknown_types = set(types) - {
            "null",
            "boolean",
            "object",
            "array",
            "number",
            "integer",
            "string",
        }
        if unknown_types:
            raise SchemaValidationError(
                f"{path}: unsupported schema type(s): {', '.join(sorted(unknown_types))}"
            )
        if not any(_matches_type(value, expected) for expected in types):
            raise SchemaValidationError(
                f"{path}: expected type {'|'.join(types)}"
            )

    if isinstance(value, dict):
        _validate_object(value, schema, path)
    elif isinstance(value, list):
        _validate_array(value, schema, path)
    elif isinstance(value, str):
        _validate_string(value, schema, path)
    elif _is_number(value):
        _validate_number(value, schema, path)


def validate_argument_limits(value: Any) -> None:
    """Apply schema-independent resource limits before recursive validation."""

    remaining_nodes = MAX_TOOL_ARGUMENT_NODES
    remaining_chars = MAX_TOOL_ARGUMENT_CHARS
    stack: list[tuple[Any, int, str]] = [(value, 0, "$")]
    while stack:
        current, depth, path = stack.pop()
        remaining_nodes -= 1
        if remaining_nodes < 0:
            raise SchemaValidationError(
                f"{path}: tool arguments exceed node limit={MAX_TOOL_ARGUMENT_NODES}"
            )
        if depth > MAX_TOOL_ARGUMENT_DEPTH:
            raise SchemaValidationError(
                f"{path}: tool arguments exceed depth limit={MAX_TOOL_ARGUMENT_DEPTH}"
            )
        if isinstance(current, str):
            remaining_chars -= len(current)
        elif current is None or isinstance(current, (bool, int, float)):
            pass
        elif isinstance(current, list):
            stack.extend(
                (item, depth + 1, f"{path}[{index}]")
                for index, item in enumerate(current)
            )
        elif isinstance(current, dict):
            for key, item in current.items():
                if not isinstance(key, str):
                    raise SchemaValidationError(f"{path}: object keys must be strings")
                remaining_chars -= len(key)
                stack.append((item, depth + 1, f"{path}.{key}"))
        else:
            raise SchemaValidationError(
                f"{path}: tool arguments contain non-JSON type {type(current).__name__}"
            )
        if remaining_chars < 0:
            raise SchemaValidationError(
                f"{path}: tool arguments exceed character limit={MAX_TOOL_ARGUMENT_CHARS}"
            )


def ensure_lossless_json(value: Any) -> Any:
    """Return a JSON round-tripped value or fail closed.

    ``json.dumps(..., default=str)`` is deliberately not used here: coercing a
    ``Path``, bytes, tuple, set, custom object, non-string key, or non-finite
    number would silently change the tool's result before schema validation.
    """

    try:
        _validate_lossless_json_types(value)
    except RecursionError as exc:
        raise SchemaValidationError(
            "$: tool output contains a cyclic or excessively deep structure"
        ) from exc
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise SchemaValidationError(
            f"$: tool output is not losslessly JSON representable ({type(exc).__name__})"
        ) from exc
    # Apply the same resource guard used for tool arguments after round-trip,
    # so a valid but unbounded result cannot consume the model context.
    try:
        validate_argument_limits(decoded)
    except SchemaValidationError as exc:
        raise SchemaValidationError(f"$: invalid tool output limits: {exc}") from exc
    return decoded


def _validate_lossless_json_types(
    value: Any, *, path: str = "$", seen: set[int] | None = None
) -> None:
    active = seen if seen is not None else set()
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaValidationError(f"{path}: non-finite numbers are not JSON values")
        return
    if isinstance(value, list):
        identity = id(value)
        if identity in active:
            raise SchemaValidationError(f"{path}: cyclic tool output structure")
        active.add(identity)
        try:
            for index, item in enumerate(value):
                _validate_lossless_json_types(
                    item, path=f"{path}[{index}]", seen=active
                )
        finally:
            active.remove(identity)
        return
    if isinstance(value, dict):
        identity = id(value)
        if identity in active:
            raise SchemaValidationError(f"{path}: cyclic tool output structure")
        active.add(identity)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise SchemaValidationError(f"{path}: object keys must be strings")
                _validate_lossless_json_types(
                    item, path=f"{path}.{key}", seen=active
                )
        finally:
            active.remove(identity)
        return
    raise SchemaValidationError(
        f"{path}: tool output contains non-JSON type {type(value).__name__}"
    )


def _validate_composition(value: Any, schema: dict[str, Any], path: str) -> None:
    for index, branch in enumerate(schema.get("allOf", [])):
        validate_json_schema(value, branch, path=f"{path}.allOf[{index}]")

    if "anyOf" in schema:
        branches = schema["anyOf"]
        if not isinstance(branches, list) or not branches:
            raise SchemaValidationError(f"{path}: anyOf must be a non-empty array")
        if not any(_branch_matches(value, branch, path) for branch in branches):
            raise SchemaValidationError(f"{path}: value does not match anyOf")

    if "oneOf" in schema:
        branches = schema["oneOf"]
        if not isinstance(branches, list) or not branches:
            raise SchemaValidationError(f"{path}: oneOf must be a non-empty array")
        matches = sum(_branch_matches(value, branch, path) for branch in branches)
        if matches != 1:
            raise SchemaValidationError(f"{path}: value must match exactly one oneOf branch")

    if "not" in schema and _branch_matches(value, schema["not"], path):
        raise SchemaValidationError(f"{path}: value matches forbidden schema")


def _branch_matches(value: Any, schema: Any, path: str) -> bool:
    try:
        validate_json_schema(value, schema, path=path)
        return True
    except SchemaValidationError:
        return False


def _validate_object(value: dict[Any, Any], schema: dict[str, Any], path: str) -> None:
    if any(not isinstance(key, str) for key in value):
        raise SchemaValidationError(f"{path}: object keys must be strings")
    minimum = schema.get("minProperties")
    maximum = schema.get("maxProperties")
    if minimum is not None and len(value) < minimum:
        raise SchemaValidationError(f"{path}: fewer than minProperties={minimum}")
    if maximum is not None and len(value) > maximum:
        raise SchemaValidationError(f"{path}: more than maxProperties={maximum}")

    required = schema.get("required", [])
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        raise SchemaValidationError(f"{path}: schema required must be a string array")
    missing = sorted(set(required) - set(value))
    if missing:
        raise SchemaValidationError(f"{path}: missing required field(s): {', '.join(missing)}")

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise SchemaValidationError(f"{path}: schema properties must be an object")
    additional = schema.get("additionalProperties", True)
    if not isinstance(additional, (bool, dict)):
        raise SchemaValidationError(f"{path}: additionalProperties must be boolean or schema")

    for key, item in value.items():
        child_path = f"{path}.{key}"
        if key in properties:
            validate_json_schema(item, properties[key], path=child_path)
        elif additional is False:
            raise SchemaValidationError(f"{path}: unexpected field: {key}")
        elif isinstance(additional, dict):
            validate_json_schema(item, additional, path=child_path)


def _validate_array(value: list[Any], schema: dict[str, Any], path: str) -> None:
    minimum = schema.get("minItems")
    maximum = schema.get("maxItems")
    if minimum is not None and len(value) < minimum:
        raise SchemaValidationError(f"{path}: fewer than minItems={minimum}")
    if maximum is not None and len(value) > maximum:
        raise SchemaValidationError(f"{path}: more than maxItems={maximum}")
    if schema.get("uniqueItems"):
        serialized = [json.dumps(item, sort_keys=True, ensure_ascii=False, default=str) for item in value]
        if len(serialized) != len(set(serialized)):
            raise SchemaValidationError(f"{path}: array items must be unique")

    prefix_items = schema.get("prefixItems", [])
    if not isinstance(prefix_items, list):
        raise SchemaValidationError(f"{path}: prefixItems must be an array")
    for index, item_schema in enumerate(prefix_items):
        if index >= len(value):
            break
        validate_json_schema(value[index], item_schema, path=f"{path}[{index}]")

    items = schema.get("items")
    if items is not None:
        if not isinstance(items, (bool, dict)):
            raise SchemaValidationError(f"{path}: items must be boolean or schema")
        start = len(prefix_items)
        if items is False and len(value) > start:
            raise SchemaValidationError(f"{path}: additional array items are forbidden")
        if isinstance(items, dict):
            for index in range(start, len(value)):
                validate_json_schema(value[index], items, path=f"{path}[{index}]")

    if "contains" in schema:
        contains = schema["contains"]
        count = sum(_branch_matches(item, contains, f"{path}[{index}]") for index, item in enumerate(value))
        minimum_contains = schema.get("minContains", 1)
        maximum_contains = schema.get("maxContains")
        if count < minimum_contains:
            raise SchemaValidationError(f"{path}: contains matched fewer than minContains")
        if maximum_contains is not None and count > maximum_contains:
            raise SchemaValidationError(f"{path}: contains matched more than maxContains")


def _validate_string(value: str, schema: dict[str, Any], path: str) -> None:
    minimum = schema.get("minLength")
    maximum = schema.get("maxLength")
    if minimum is not None and len(value) < minimum:
        raise SchemaValidationError(f"{path}: shorter than minLength={minimum}")
    if maximum is not None and len(value) > maximum:
        raise SchemaValidationError(f"{path}: longer than maxLength={maximum}")
    if "pattern" in schema:
        try:
            matched = re.search(schema["pattern"], value) is not None
        except (re.error, TypeError) as exc:
            raise SchemaValidationError(f"{path}: invalid schema pattern") from exc
        if not matched:
            raise SchemaValidationError(f"{path}: string does not match required pattern")


def _validate_number(value: int | float, schema: dict[str, Any], path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise SchemaValidationError(f"{path}: number must be finite")
    for keyword, predicate in (
        ("minimum", lambda current, limit: current >= limit),
        ("maximum", lambda current, limit: current <= limit),
        ("exclusiveMinimum", lambda current, limit: current > limit),
        ("exclusiveMaximum", lambda current, limit: current < limit),
    ):
        if keyword in schema and not predicate(value, schema[keyword]):
            raise SchemaValidationError(f"{path}: violates {keyword}={schema[keyword]}")
    if "multipleOf" in schema:
        multiple = schema["multipleOf"]
        if not _is_number(multiple) or multiple <= 0:
            raise SchemaValidationError(f"{path}: schema multipleOf must be positive")
        quotient = value / multiple
        if not math.isclose(quotient, round(quotient), rel_tol=1e-9, abs_tol=1e-9):
            raise SchemaValidationError(f"{path}: is not a multipleOf={multiple}")


def _matches_type(value: Any, expected: str) -> bool:
    return {
        "null": value is None,
        "boolean": isinstance(value, bool),
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "number": _is_number(value),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "string": isinstance(value, str),
    }[expected]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
