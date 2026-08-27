"""Deterministic required-resource declarations derived from experiment specs.

The paper/code agents provide semantic findings, while the specification stage
turns those findings into a small, auditable list of resources that must exist
before any experiment tier may be dispatched.  This module deliberately avoids
guessing download locations or credentials; it only normalizes explicit
resource declarations and high-confidence parameter names such as ``dataset``
or ``checkpoint_path``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


_KIND_ALIASES = {
    "data": "dataset",
    "dataset": "dataset",
    "benchmark": "dataset",
    "corpus": "dataset",
    "model": "model",
    "weights": "model",
    "checkpoint": "checkpoint",
    "ckpt": "checkpoint",
}
_DATASET_KEYS = {
    "data",
    "data_dir",
    "data_path",
    "dataset",
    "dataset_dir",
    "dataset_name",
    "dataset_id",
    "dataset_path",
    "benchmark",
    "benchmark_name",
    "corpus",
    "corpus_name",
}
_CHECKPOINT_KEY_PARTS = ("checkpoint", "ckpt", "weights_path", "model_path")
_GENERIC_VALUES = {
    "",
    "none",
    "null",
    "default",
    "custom",
    "same_as_paper",
    "same as paper",
    "not specified",
    "unknown",
}


def resource_name_key(value: Any) -> str:
    """Return a case/punctuation-insensitive key used for matching and dedupe."""

    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def normalize_required_resources(values: Any) -> list[dict[str, Any]]:
    """Normalize and deduplicate bounded resource declarations."""

    if not isinstance(values, list):
        return []
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in values[:128]:
        if not isinstance(raw, Mapping):
            continue
        name = _clean_resource_name(raw.get("name"))
        kind = _KIND_ALIASES.get(str(raw.get("kind", "")).strip().lower())
        if not name or kind is None:
            continue
        key = (kind, resource_name_key(name))
        if not key[1] or key in seen:
            continue
        seen.add(key)
        raw_aliases = raw.get("aliases") or []
        if isinstance(raw_aliases, str):
            raw_aliases = [raw_aliases]
        aliases = [
            alias
            for alias in (
                _clean_resource_name(item)
                for item in list(raw_aliases)[:16]
            )
            if alias and resource_name_key(alias) != key[1]
        ]
        normalized.append(
            {
                "resource_id": str(raw.get("resource_id") or f"{kind}:{key[1]}")[:256],
                "name": name,
                "kind": kind,
                "required": bool(raw.get("required", True)),
                "reason": str(raw.get("reason", ""))[:2000],
                "source_ref": str(raw.get("source_ref", ""))[:1000],
                "aliases": aliases,
            }
        )
    return normalized


def infer_required_resources(
    paper_findings: Mapping[str, Any] | None,
    code_findings: Mapping[str, Any] | None,
    user_overrides: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build the required-resource list consumed by ``ResourceCheckAgent``.

    Explicit declarations win, then deterministic parameter-name rules fill in
    resources already present in the paper/code findings.  Generic ``model``
    architecture parameters are intentionally not treated as local files:
    model/API selections are handled by the separate required-runtime-config
    contract unless a path/checkpoint/weights parameter proves a local resource.
    """

    paper = paper_findings or {}
    code = code_findings or {}
    overrides = user_overrides or {}
    declarations: list[dict[str, Any]] = []
    for source, source_name in (
        (paper, "paper_analysis"),
        (code, "code_analysis"),
        (overrides, "user_input"),
    ):
        explicit = source.get("required_resources")
        if isinstance(explicit, list):
            declarations.extend(_with_default_source(explicit, source_name))
        resources = source.get("resources")
        if isinstance(resources, Mapping) and isinstance(resources.get("required"), list):
            declarations.extend(_with_default_source(resources["required"], source_name))

    declarations.extend(_resources_from_parameters(paper, "paper_analysis"))
    declarations.extend(_resources_from_parameters(code, "code_analysis"))
    declarations.extend(_resources_from_parameters(overrides, "user_input"))
    return normalize_required_resources(declarations)


def _resources_from_parameters(
    findings: Mapping[str, Any], source_name: str
) -> list[dict[str, Any]]:
    declarations: list[dict[str, Any]] = []
    rich_parameters = findings.get("extracted_parameters")
    if isinstance(rich_parameters, list):
        for item in rich_parameters:
            if not isinstance(item, Mapping):
                continue
            kind = _resource_kind_for_parameter(item.get("name"))
            if kind is None:
                continue
            source_ref = source_name
            if item.get("page"):
                source_ref = f"{source_name}:page:{item['page']}"
            declarations.extend(
                _declarations_for_value(
                    item.get("value"), kind=kind, source_ref=source_ref
                )
            )

    effective = findings.get("effective_parameters")
    if isinstance(effective, Mapping):
        for name, value in effective.items():
            kind = _resource_kind_for_parameter(name)
            if kind is None:
                continue
            declarations.extend(
                _declarations_for_value(
                    value,
                    kind=kind,
                    source_ref=f"{source_name}:parameter:{name}",
                )
            )
    return declarations


def _resource_kind_for_parameter(name: Any) -> str | None:
    text = re.sub(r"[^a-z0-9]+", "_", str(name).casefold()).strip("_")
    if text in _DATASET_KEYS or text.endswith("_dataset"):
        return "dataset"
    if any(part in text for part in _CHECKPOINT_KEY_PARTS):
        return "checkpoint" if "checkpoint" in text or "ckpt" in text else "model"
    return None


def _declarations_for_value(
    value: Any, *, kind: str, source_ref: str
) -> list[dict[str, Any]]:
    names = list(_resource_names(value))
    return [
        {
            "name": name,
            "kind": kind,
            "required": True,
            "reason": f"实验规格中的 {kind} 参数要求该运行资源",
            "source_ref": source_ref,
        }
        for name in names
    ]


def _resource_names(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key in ("name", "id", "dataset", "path", "checkpoint"):
            if key in value:
                yield from _resource_names(value[key])
                return
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _resource_names(item)
        return
    if not isinstance(value, str):
        return
    text = value.strip()
    if not text or text.casefold() in _GENERIC_VALUES:
        return
    parts = re.split(r"\s*(?:,|;|\band\b|\+|\|)\s*", text, flags=re.IGNORECASE)
    for part in parts[:16]:
        cleaned = _clean_resource_name(part)
        if "/" in cleaned or "\\" in cleaned:
            path_parts = re.split(r"[/\\]+", cleaned.rstrip("/\\"))
            cleaned = path_parts[-1] if path_parts else cleaned
        if cleaned and cleaned.casefold() not in _GENERIC_VALUES:
            yield cleaned


def _clean_resource_name(value: Any) -> str:
    text = str(value or "").strip().strip("'\"")
    text = re.sub(r"\s+(?:dataset|benchmark|corpus)$", "", text, flags=re.IGNORECASE)
    return text[:256].strip()


def _with_default_source(values: list[Any], source_name: str) -> list[Any]:
    result: list[Any] = []
    for item in values:
        if not isinstance(item, Mapping):
            continue
        copied = dict(item)
        copied.setdefault("source_ref", source_name)
        result.append(copied)
    return result
