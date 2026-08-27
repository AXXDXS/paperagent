"""Main-agent-owned lifecycle for generated reusable tools."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from repro_agent.domain.common import iso, new_id, utc_now
from repro_agent.dynamic_tools.execution import DynamicToolExecutor
from repro_agent.dynamic_tools.models import (
    CandidateTestCase,
    DynamicToolStatus,
    ReusableCodeCandidate,
    new_dynamic_tool_record,
)
from repro_agent.dynamic_tools.validation import (
    CandidateValidationError,
    candidates_match,
    validate_candidate,
)
from repro_agent.storage.repository import DynamicToolRepository
from repro_agent.tools.base import (
    SandboxContext,
    ToolExecutionError,
    ToolOutputSpec,
    ToolRiskLevel,
    ToolSpec,
)
from repro_agent.tools.registry import ToolRegistry
from repro_agent.tools.schema_validation import ensure_lossless_json, validate_json_schema

logger = logging.getLogger(__name__)

CANDIDATE_MAX_LIFE = 10
ACTIVE_MAX_LIFE = 30
ACTIVATION_SUPPORT = 3
MAX_SIDECAR_CANDIDATES = 20

CandidateVerifier = Callable[[ReusableCodeCandidate, SandboxContext], None]


@dataclass
class IngestOutcome:
    accepted_tool_ids: list[str] = field(default_factory=list)
    activated_tool_ids: list[str] = field(default_factory=list)
    refreshed_pending_tool_names: set[str] = field(default_factory=set)
    rejected: list[str] = field(default_factory=list)


class DynamicToolLifecycleManager:
    """Conservative propose -> repeat -> verify -> activate lifecycle.

    The manager only knows records from ``DynamicToolRepository``. Built-in
    registry entries are therefore outside its mutation domain and never age.
    """

    def __init__(
        self,
        repository: DynamicToolRepository,
        registry: ToolRegistry,
        *,
        executor: DynamicToolExecutor | None = None,
        verifier: CandidateVerifier | None = None,
    ) -> None:
        self.repository = repository
        self.registry = registry
        self.executor = executor or DynamicToolExecutor()
        self._candidate_verifier = verifier or self._verify_candidate_behavior

    def load_active_tools(self) -> None:
        """Synchronize this process registry with the shared workspace store."""

        for record in self.repository.list_all():
            name = str(record.get("tool_name", ""))
            if record.get("status") != DynamicToolStatus.ACTIVE.value:
                if (
                    name
                    and self.registry.get(name) is not None
                    and not self.registry.is_permanent(name)
                ):
                    self.registry.unregister_dynamic(name)
                continue
            try:
                self._register_record(record)
            except Exception as exc:  # noqa: BLE001 - corrupt persisted tool fails closed
                record["status"] = DynamicToolStatus.REJECTED.value
                record["verification_error"] = f"reload validation failed: {exc}"
                record["updated_at"] = iso(utc_now())
                self.repository.save(record)
                logger.warning("dynamic tool %s rejected during reload: %s", record.get("tool_id"), exc)

    def ingest_sidecar(
        self,
        path: str | Path,
        *,
        job_id: str,
        task_id: str,
        attempt_id: str,
        task_type: str,
        sandbox_ctx: SandboxContext,
    ) -> IngestOutcome:
        sidecar = Path(path)
        outcome = IngestOutcome()
        if not sidecar.is_file():
            return outcome
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            outcome.rejected.append(f"invalid reusable-code sidecar: {exc}")
            return outcome
        if not isinstance(payload, list) or len(payload) > MAX_SIDECAR_CANDIDATES:
            outcome.rejected.append(
                f"sidecar must be an array with at most {MAX_SIDECAR_CANDIDATES} candidates"
            )
            return outcome
        for raw in payload:
            try:
                candidate = ReusableCodeCandidate.from_dict(raw)
                self._ingest_one(
                    candidate,
                    job_id=job_id,
                    task_id=task_id,
                    attempt_id=attempt_id,
                    task_type=task_type,
                    sandbox_ctx=sandbox_ctx,
                    outcome=outcome,
                )
            except (CandidateValidationError, ToolExecutionError, ValueError) as exc:
                outcome.rejected.append(str(exc))
        return outcome

    def _ingest_one(
        self,
        candidate: ReusableCodeCandidate,
        *,
        job_id: str,
        task_id: str,
        attempt_id: str,
        task_type: str,
        sandbox_ctx: SandboxContext,
        outcome: IngestOutcome,
    ) -> None:
        fingerprint = validate_candidate(candidate)
        # A report only becomes evidence after the reported function itself has
        # passed its behavior examples in the current task's execution sandbox.
        self._candidate_verifier(candidate, sandbox_ctx)

        record = self._find_match(candidate, fingerprint)
        if record is None:
            tool_name = self._new_tool_name(candidate)
            record = new_dynamic_tool_record(
                candidate, ast_fingerprint=fingerprint, tool_name=tool_name
            )
            if not record["suggested_task_types"]:
                record["suggested_task_types"] = [task_type]

        evidence = {
            "evidence_id": new_id("dynamic_tool_evidence"),
            "tool_id": record["tool_id"],
            "job_id": job_id,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "task_type": task_type,
            "candidate_id": candidate.candidate_id,
            "code_hash": candidate.code_hash,
            "created_at": iso(utc_now()),
        }
        record["support_count"] = int(record.get("support_count", 0)) + 1
        record["suggested_task_types"] = sorted(
            set(record.get("suggested_task_types", []))
            | set(candidate.suggested_task_types)
            | {task_type}
        )
        record["tests"] = _merge_tests(record.get("tests", []), candidate.tests)
        record["updated_at"] = iso(utc_now())
        status = DynamicToolStatus(record["status"])
        if status in {DynamicToolStatus.PENDING, DynamicToolStatus.AWAITING_APPROVAL}:
            record["life"] = CANDIDATE_MAX_LIFE
            record["max_life"] = CANDIDATE_MAX_LIFE
            outcome.refreshed_pending_tool_names.add(record["tool_name"])

        if not self.repository.save_with_evidence(record, evidence):
            outcome.refreshed_pending_tool_names.discard(record["tool_name"])
            return

        if status == DynamicToolStatus.PENDING and record["support_count"] >= ACTIVATION_SUPPORT:
            self._attempt_activation(record, sandbox_ctx)
            if record["status"] == DynamicToolStatus.ACTIVE.value:
                outcome.activated_tool_ids.append(record["tool_id"])
        self.repository.save(record)
        outcome.accepted_tool_ids.append(record["tool_id"])

    def advance_relevant_task_event(
        self,
        task_type: str,
        *,
        refreshed_tool_names: set[str] | None = None,
        successful_tool_names: set[str] | None = None,
    ) -> list[str]:
        """Age generated tools once for one relevant, validated task."""

        refreshed = set(refreshed_tool_names or ())
        successful = set(successful_tool_names or ())
        expired: list[str] = []
        for record in self.repository.list_all():
            status = DynamicToolStatus(record["status"])
            if status not in {
                DynamicToolStatus.PENDING,
                DynamicToolStatus.ACTIVE,
                DynamicToolStatus.AWAITING_APPROVAL,
            }:
                continue
            if task_type not in set(record.get("suggested_task_types", [])):
                continue
            name = str(record["tool_name"])
            if name in refreshed or name in successful:
                continue
            record["life"] = max(0, int(record.get("life", 0)) - 1)
            record["updated_at"] = iso(utc_now())
            if record["life"] == 0:
                self._expire(record)
                expired.append(record["tool_id"])
            self.repository.save(record)
        return expired

    def observe_invocation(self, tool_name: str, succeeded: bool) -> None:
        """Reset active life only after a validated successful invocation."""

        record = self.repository.get_by_name(tool_name)
        if record is None or record.get("status") != DynamicToolStatus.ACTIVE.value:
            return
        if succeeded:
            record["life"] = ACTIVE_MAX_LIFE
            record["max_life"] = ACTIVE_MAX_LIFE
        else:
            record["failure_count"] = int(record.get("failure_count", 0)) + 1
        record["updated_at"] = iso(utc_now())
        self.repository.save(record)

    def approve(
        self, tool_id: str, sandbox_ctx: SandboxContext | None = None
    ) -> bool:
        """Explicitly activate a verified high-risk generated tool."""

        record = self.repository.get(tool_id)
        if record is None or record.get("status") != DynamicToolStatus.AWAITING_APPROVAL.value:
            return False
        if sandbox_ctx is not None:
            self._verify_record_behavior(record, sandbox_ctx)
        elif (
            record.get("admission_code_hash") != record.get("code_hash")
            or not record.get("admission_verified_at")
        ):
            raise ToolExecutionError(
                "dynamic tool has no current persisted admission verification"
            )
        validate_candidate(_candidate_from_record(record))
        record["status"] = DynamicToolStatus.ACTIVE.value
        record["life"] = ACTIVE_MAX_LIFE
        record["max_life"] = ACTIVE_MAX_LIFE
        record["updated_at"] = iso(utc_now())
        self._register_record(record)
        self.repository.save(record)
        return True

    def reject(self, tool_id: str, *, reason: str = "user rejected activation") -> bool:
        record = self.repository.get(tool_id)
        if record is None or record.get("status") != DynamicToolStatus.AWAITING_APPROVAL.value:
            return False
        record["status"] = DynamicToolStatus.REJECTED.value
        record["life"] = 0
        record["code"] = ""
        record["tests"] = []
        record["verification_error"] = reason
        record["updated_at"] = iso(utc_now())
        self.repository.save(record)
        return True

    def list_records(self, *, include_code: bool = False) -> list[dict[str, Any]]:
        records = self.repository.list_all()
        if include_code:
            return records
        return [
            {key: value for key, value in record.items() if key not in {"code", "tests"}}
            for record in records
        ]

    def _attempt_activation(self, record: dict[str, Any], sandbox_ctx: SandboxContext) -> None:
        try:
            self._verify_record_behavior(record, sandbox_ctx)
        except Exception as exc:  # noqa: BLE001 - failed admission remains pending
            record["verification_error"] = str(exc)
            return
        record["admission_verified_at"] = iso(utc_now())
        record["admission_code_hash"] = record.get("code_hash", "")
        risk = ToolRiskLevel(record["risk_level"])
        if risk != ToolRiskLevel.READ_ONLY:
            record["status"] = DynamicToolStatus.AWAITING_APPROVAL.value
            record["verification_error"] = "explicit user approval required for non-read-only tool"
            return
        record["status"] = DynamicToolStatus.ACTIVE.value
        record["life"] = ACTIVE_MAX_LIFE
        record["max_life"] = ACTIVE_MAX_LIFE
        record["verification_error"] = ""
        self._register_record(record)

    def _verify_candidate_behavior(
        self, candidate: ReusableCodeCandidate, sandbox_ctx: SandboxContext
    ) -> None:
        record = new_dynamic_tool_record(
            candidate,
            ast_fingerprint=validate_candidate(candidate),
            tool_name="verification_only",
        )
        for test in candidate.tests:
            actual = self.executor.run(record, sandbox_ctx, test.arguments)
            if ensure_lossless_json(actual) != ensure_lossless_json(test.expected):
                raise ToolExecutionError("generated candidate failed a declared behavior test")

    def _verify_record_behavior(self, record: dict[str, Any], sandbox_ctx: SandboxContext) -> None:
        candidate = _candidate_from_record(record)
        validate_candidate(candidate)
        for test in candidate.tests:
            actual = self.executor.run(record, sandbox_ctx, test.arguments)
            actual = ensure_lossless_json(actual)
            validate_json_schema(actual, record["output_schema"])
            if actual != ensure_lossless_json(test.expected):
                raise ToolExecutionError("generated tool failed its merged behavior suite")

    def _find_match(
        self, candidate: ReusableCodeCandidate, fingerprint: str
    ) -> dict[str, Any] | None:
        for record in self.repository.list_all():
            if record.get("status") in {
                DynamicToolStatus.EXPIRED.value,
                DynamicToolStatus.REJECTED.value,
            }:
                continue
            if candidates_match(record, candidate, fingerprint):
                return record
        return None

    def _register_record(self, record: dict[str, Any]) -> None:
        if self.registry.get(record["tool_name"]) is not None:
            return
        candidate = _candidate_from_record(record)
        validate_candidate(candidate)
        tool_id = str(record["tool_id"])

        def handler(ctx: SandboxContext, **kwargs: Any) -> Any:
            current = self.repository.get(tool_id)
            if current is None or current.get("status") != DynamicToolStatus.ACTIVE.value:
                raise ToolExecutionError("dynamic tool is no longer active")
            return self.executor.run(current, ctx, kwargs)

        spec = ToolSpec(
            name=str(record["tool_name"]),
            description=str(record["purpose"]),
            risk_level=ToolRiskLevel(record["risk_level"]),
            handler=handler,
            suggested_task_types=tuple(record.get("suggested_task_types", [])),
            requires_network=False,
            parameters=dict(record["input_schema"]),
            when_to_use=str(record["generalization_reason"]),
            boundaries=(
                "该工具由重复出现且经过验证的候选代码生成，只能在隔离沙箱中运行。",
                "禁止网络、GPU、任意文件 API 和非白名单依赖。",
            ),
            returns="符合动态工具声明的 output schema 的 JSON 值。",
            cost_hint="需要启动隔离执行容器，适合复用明确的计算过程。",
            output=ToolOutputSpec(schema=dict(record["output_schema"])),
        )
        self.registry.register(spec, permanent=False)

    def _expire(self, record: dict[str, Any]) -> None:
        name = str(record["tool_name"])
        if self.registry.get(name) is not None and not self.registry.is_permanent(name):
            self.registry.unregister_dynamic(name)
        record["status"] = DynamicToolStatus.EXPIRED.value
        # Keep only a tombstone and provenance hashes. Executable material is
        # removed, while evidence rows remain available for audit.
        record["code"] = ""
        record["tests"] = []
        record["verification_error"] = "life reached zero"

    def _new_tool_name(self, candidate: ReusableCodeCandidate) -> str:
        slug = re.sub(r"[^a-z0-9_]+", "_", candidate.normalized_functional_key)
        slug = slug.strip("_")[:40] or "generated"
        suffix = hashlib.sha256(
            (
                candidate.normalized_functional_key
                + candidate.code_hash
                + new_id("name")
            ).encode("utf-8")
        ).hexdigest()[:8]
        name = f"dynamic_{slug}_{suffix}"
        if self.registry.get(name) is not None:
            raise CandidateValidationError(f"generated tool name collides with {name}")
        return name


def _candidate_from_record(record: dict[str, Any]) -> ReusableCodeCandidate:
    return ReusableCodeCandidate.from_dict(
        {
            "candidate_id": record.get("tool_id"),
            "purpose": record.get("purpose", ""),
            "functional_key": record.get("functional_key", ""),
            "code": record.get("code", ""),
            "entry_function": record.get("entry_function", ""),
            "input_schema": record.get("input_schema", {}),
            "output_schema": record.get("output_schema", {}),
            "tests": record.get("tests", []),
            "generalization_reason": record.get("generalization_reason", ""),
            "suggested_task_types": record.get("suggested_task_types", []),
            "dependencies": record.get("dependencies", []),
            "risk_level": record.get("risk_level", ToolRiskLevel.READ_ONLY.value),
            "requires_network": record.get("requires_network", False),
        }
    )


def _merge_tests(
    existing: list[dict[str, Any]], new_tests: tuple[CandidateTestCase, ...]
) -> list[dict[str, Any]]:
    merged = list(existing)
    seen = {json.dumps(item, sort_keys=True, ensure_ascii=False) for item in merged}
    for test in new_tests:
        value = test.to_dict()
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=False)
        if encoded not in seen:
            merged.append(value)
            seen.add(encoded)
    return merged[:50]
