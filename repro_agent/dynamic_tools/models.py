"""Data contracts for child-reported reusable code and generated tools."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from repro_agent.domain.common import iso, new_id, utc_now
from repro_agent.tools.base import ToolRiskLevel


class DynamicToolStatus(str, Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class CandidateTestCase:
    arguments: dict[str, Any]
    expected: Any

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CandidateTestCase":
        if not isinstance(value, dict):
            raise ValueError("candidate test must be an object")
        arguments = value.get("arguments")
        if not isinstance(arguments, dict) or "expected" not in value:
            raise ValueError("candidate test requires arguments and expected")
        return cls(arguments=dict(arguments), expected=value["expected"])

    def to_dict(self) -> dict[str, Any]:
        return {"arguments": self.arguments, "expected": self.expected}


@dataclass(frozen=True)
class ReusableCodeCandidate:
    """Untrusted proposal emitted by a child agent.

    Source job/task/attempt identifiers are deliberately absent.  The main
    agent supplies those from its trusted scheduler state when ingesting the
    sidecar, so generated code cannot forge independent support.
    """

    purpose: str
    functional_key: str
    code: str
    entry_function: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    tests: tuple[CandidateTestCase, ...]
    generalization_reason: str
    suggested_task_types: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    risk_level: ToolRiskLevel = ToolRiskLevel.READ_ONLY
    requires_network: bool = False
    candidate_id: str = field(default_factory=lambda: new_id("tool_candidate"))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReusableCodeCandidate":
        if not isinstance(value, dict):
            raise ValueError("reusable code candidate must be an object")
        required = (
            "purpose",
            "functional_key",
            "code",
            "entry_function",
            "input_schema",
            "output_schema",
            "tests",
            "generalization_reason",
        )
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"candidate missing fields: {', '.join(missing)}")
        risk = ToolRiskLevel(str(value.get("risk_level", ToolRiskLevel.READ_ONLY.value)))
        tests = value.get("tests")
        if not isinstance(tests, list) or not tests:
            raise ValueError("candidate must provide at least one behavior test")
        return cls(
            purpose=str(value["purpose"]).strip(),
            functional_key=str(value["functional_key"]).strip(),
            code=str(value["code"]),
            entry_function=str(value["entry_function"]).strip(),
            input_schema=dict(value["input_schema"]),
            output_schema=dict(value["output_schema"]),
            tests=tuple(CandidateTestCase.from_dict(item) for item in tests),
            generalization_reason=str(value["generalization_reason"]).strip(),
            suggested_task_types=tuple(
                sorted({str(item) for item in value.get("suggested_task_types", []) if item})
            ),
            dependencies=tuple(
                sorted({str(item) for item in value.get("dependencies", []) if item})
            ),
            risk_level=risk,
            requires_network=bool(value.get("requires_network", False)),
            candidate_id=str(value.get("candidate_id") or new_id("tool_candidate")),
        )

    @property
    def code_hash(self) -> str:
        return hashlib.sha256(self.code.encode("utf-8")).hexdigest()

    @property
    def normalized_functional_key(self) -> str:
        return re.sub(r"[^a-z0-9]+", ".", self.functional_key.lower()).strip(".")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "purpose": self.purpose,
            "functional_key": self.functional_key,
            "code": self.code,
            "entry_function": self.entry_function,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "tests": [item.to_dict() for item in self.tests],
            "generalization_reason": self.generalization_reason,
            "suggested_task_types": list(self.suggested_task_types),
            "dependencies": list(self.dependencies),
            "risk_level": self.risk_level.value,
            "requires_network": self.requires_network,
            "code_hash": self.code_hash,
        }


def new_dynamic_tool_record(
    candidate: ReusableCodeCandidate,
    *,
    ast_fingerprint: str,
    tool_name: str,
) -> dict[str, Any]:
    now = iso(utc_now())
    return {
        "tool_id": new_id("dynamic_tool"),
        "tool_name": tool_name,
        "status": DynamicToolStatus.PENDING.value,
        "life": 10,
        "max_life": 10,
        "support_count": 0,
        "failure_count": 0,
        "purpose": candidate.purpose,
        "functional_key": candidate.normalized_functional_key,
        "code": candidate.code,
        "code_hash": candidate.code_hash,
        "ast_fingerprint": ast_fingerprint,
        "entry_function": candidate.entry_function,
        "input_schema": candidate.input_schema,
        "output_schema": candidate.output_schema,
        "tests": [item.to_dict() for item in candidate.tests],
        "generalization_reason": candidate.generalization_reason,
        "suggested_task_types": list(candidate.suggested_task_types),
        "dependencies": list(candidate.dependencies),
        "risk_level": candidate.risk_level.value,
        "requires_network": candidate.requires_network,
        "verification_error": "",
        "admission_verified_at": "",
        "admission_code_hash": "",
        "created_at": now,
        "updated_at": now,
    }


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
