"""Independent verification of execution evidence and reported metrics."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from repro_agent.agents.base import AgentRunResult, BaseSubAgent
from repro_agent.domain.enums import ExperimentTier, FailureType, ToleranceType
from repro_agent.domain.experiment import (
    ExperimentRun,
    ExperimentSpec,
    ExpectedResult,
    MetricComparison,
    compare_metrics,
)
from repro_agent.domain.task import FailureReport
from repro_agent.evidence.anti_cheat import scan_suspicious_markers
from repro_agent.evidence.provenance import ArtifactProvenance, ProvenanceError, verify_provenance


@dataclass
class VerificationResult:
    run_actually_executed: bool = True
    is_fully_traceable: bool = False
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    all_within_tolerance: bool = False
    anti_cheat_passed: bool = True
    anti_cheat_findings: list[str] = field(default_factory=list)
    anti_cheat_input_verified: bool = False
    provenance_verified: bool = True
    provenance_errors: list[str] = field(default_factory=list)
    expected_metric_names: list[str] = field(default_factory=list)
    observed_metric_names: list[str] = field(default_factory=list)
    missing_metrics: list[str] = field(default_factory=list)
    mock: bool = False
    verification_valid: bool = False
    tier_command_verified: bool = False
    evidence_verified: bool = False
    evidence_errors: list[str] = field(default_factory=list)
    verified_artifacts: list[dict[str, Any]] = field(default_factory=list)
    verified_logs: list[str] = field(default_factory=list)
    independently_read_metrics: dict[str, float] = field(default_factory=dict)
    recomputed_metrics: dict[str, float] = field(default_factory=dict)
    recomputation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_actually_executed": self.run_actually_executed,
            "is_fully_traceable": self.is_fully_traceable,
            "comparisons": self.comparisons,
            "all_within_tolerance": self.all_within_tolerance,
            "anti_cheat_passed": self.anti_cheat_passed,
            "anti_cheat_findings": self.anti_cheat_findings,
            "anti_cheat_input_verified": self.anti_cheat_input_verified,
            "provenance_verified": self.provenance_verified,
            "provenance_errors": self.provenance_errors,
            "expected_metric_names": self.expected_metric_names,
            "observed_metric_names": self.observed_metric_names,
            "missing_metrics": self.missing_metrics,
            "mock": self.mock,
            "verification_valid": self.verification_valid,
            "tier_command_verified": self.tier_command_verified,
            "evidence_verified": self.evidence_verified,
            "evidence_errors": self.evidence_errors,
            "verified_artifacts": self.verified_artifacts,
            "verified_logs": self.verified_logs,
            "independently_read_metrics": self.independently_read_metrics,
            "recomputed_metrics": self.recomputed_metrics,
            "recomputation_errors": self.recomputation_errors,
        }


class ResultVerificationAgent(BaseSubAgent):
    task_type = "verification"
    system_prompt = (
        "你是 ReproAgent 系统的结果验证子智能体，与实验执行子智能体完全分离。"
        "你必须直接读取验证沙箱中的 metrics、日志、执行状态和产物，并重新计算"
        "可计算的指标；不能只相信执行智能体上报的 JSON，也不能重新运行实验。"
    )

    def run(self) -> AgentRunResult:
        inputs = self.task.definition.inputs
        spec_dict = inputs.get("experiment_spec", {})
        run_dict = inputs.get("experiment_run", {})
        provenance_dict = inputs.get("artifact_provenance") or run_dict.get("artifact_provenance")

        spec = self._spec_from_dict(spec_dict)
        run = self._run_from_dict(run_dict)
        result = VerificationResult(
            mock=bool(run_dict.get("mock", False)),
            tier_command_verified=bool(run_dict.get("tier_command_verified", False)),
            is_fully_traceable=run.is_fully_traceable(),
            expected_metric_names=sorted(spec.expected_results),
        )

        evidence = inputs.get("verification_evidence") or run_dict.get("verification_evidence") or []
        evidence_result = self._verify_evidence(evidence, run)
        result.evidence_verified = evidence_result["verified"]
        result.evidence_errors = evidence_result["errors"]
        result.verified_artifacts = evidence_result["artifacts"]
        result.verified_logs = evidence_result["logs"]
        result.independently_read_metrics = evidence_result["metrics"]
        if result.independently_read_metrics:
            # The execution agent's payload is not used as the source of truth.
            run.metrics = result.independently_read_metrics
        result.observed_metric_names = sorted(run.metrics)
        result.missing_metrics = sorted(set(spec.expected_results) - set(run.metrics))
        result.run_actually_executed = run.exit_code == 0 and result.evidence_verified

        recomputed, recomputation_errors = self._recompute_metrics(evidence, spec, run.metrics)
        result.recomputed_metrics = recomputed
        result.recomputation_errors = recomputation_errors
        for metric, value in recomputed.items():
            reported = run.metrics.get(metric)
            if reported is not None and not math.isclose(reported, value, rel_tol=1e-9, abs_tol=1e-9):
                result.recomputation_errors.append(
                    f"metric {metric} differs between metrics file and independent recomputation"
                )
            run.metrics[metric] = value

        comparisons = compare_metrics(spec, run)
        result.comparisons = [c.to_dict() for c in comparisons]
        result.all_within_tolerance = bool(comparisons) and all(c.within_tolerance for c in comparisons)

        anti_cheat_material, anti_cheat_errors = self._anti_cheat_material(inputs)
        result.anti_cheat_input_verified = bool(anti_cheat_material) and not anti_cheat_errors
        anti_cheat = scan_suspicious_markers(*anti_cheat_material)
        result.anti_cheat_passed = anti_cheat.passed
        result.anti_cheat_findings = [*anti_cheat_errors, *anti_cheat.reasons]
        if not result.anti_cheat_input_verified:
            result.anti_cheat_passed = False
        self._verify_manifest(result, provenance_dict)
        if result.recomputation_errors:
            result.provenance_errors.extend(result.recomputation_errors)

        result.verification_valid = bool(
            result.expected_metric_names
            and not result.missing_metrics
            and len(comparisons) == len(result.expected_metric_names)
            and result.run_actually_executed
            and result.is_fully_traceable
            and result.anti_cheat_passed
            and result.anti_cheat_input_verified
            and result.provenance_verified
            and result.evidence_verified
            and not result.recomputation_errors
            and result.tier_command_verified
            and not result.mock
        )

        result_payload = result.to_dict()
        self.write_json_output("result.json", result_payload)
        self.write_candidate_memory(self._render_candidate_memory(result, comparisons))

        if not result.verification_valid and not result.mock:
            reasons: list[str] = []
            if not result.expected_metric_names:
                reasons.append("experiment specification has no expected metrics")
            if result.missing_metrics:
                reasons.append(f"missing metrics: {result.missing_metrics}")
            if not result.is_fully_traceable:
                reasons.append("full run traceability tuple is incomplete")
            if not result.evidence_verified:
                reasons.extend(result.evidence_errors)
            if not result.provenance_verified:
                reasons.extend(result.provenance_errors)
            if not result.run_actually_executed:
                reasons.append("run did not exit successfully or execution evidence is incomplete")
            if not result.tier_command_verified:
                reasons.append("full experiment command has no explicit tier contract")
            if not result.anti_cheat_passed:
                reasons.extend(result.anti_cheat_findings)
            return AgentRunResult(
                succeeded=False,
                outputs=result_payload,
                candidate_memory_written=True,
                failure_report=FailureReport(
                    failure_type=FailureType.INVALID_OUTPUT,
                    failed_step="strict_verification",
                    error_message="; ".join(dict.fromkeys(reasons)) or "verification evidence is incomplete",
                    partial_outputs=["output/result.json", "output/candidate_memory.md"],
                    recommended_action="补齐独立指标、日志和产物证据后重新执行正式实验验证",
                ),
            )

        return AgentRunResult(succeeded=True, outputs=result_payload, candidate_memory_written=True)

    def _verify_manifest(self, result: VerificationResult, provenance_dict: Any) -> None:
        if provenance_dict and provenance_dict.get("manifest_digest"):
            manifest_digest = provenance_dict["manifest_digest"]
            manifest_body = {key: value for key, value in provenance_dict.items() if key != "manifest_digest"}
            actual_digest = hashlib.sha256(
                json.dumps(manifest_body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            required_evidence = [
                manifest_body.get("source"),
                manifest_body.get("config_digest"),
                manifest_body.get("container_digest"),
                manifest_body.get("metrics"),
            ]
            result.provenance_verified = actual_digest == manifest_digest and all(required_evidence) and result.evidence_verified
            if actual_digest != manifest_digest:
                result.provenance_errors.append("execution manifest digest mismatch")
            if not all(required_evidence):
                result.provenance_errors.append("execution manifest is incomplete")
            if not result.evidence_verified:
                result.provenance_errors.extend(result.evidence_errors)
        elif provenance_dict:
            try:
                provenance = ArtifactProvenance(**provenance_dict)
                result.provenance_verified, result.provenance_errors = verify_provenance(provenance)
            except (TypeError, ProvenanceError) as exc:
                result.provenance_verified = False
                result.provenance_errors = [str(exc)]
        else:
            result.provenance_verified = False
            result.provenance_errors = ["artifact provenance is missing"]

    def _verify_evidence(self, evidence: list[dict[str, Any]], run: ExperimentRun) -> dict[str, Any]:
        """Read and hash evidence in this verifier's own sandbox."""

        errors: list[str] = []
        artifacts: list[dict[str, Any]] = []
        logs: list[str] = []
        metrics: dict[str, float] = {}
        verified_roles: set[str] = set()
        if not isinstance(evidence, list):
            return {"verified": False, "errors": ["verification evidence is not a list"], "artifacts": [], "logs": [], "metrics": {}}
        for item in evidence:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                errors.append("malformed verification evidence entry")
                continue
            role = str(item.get("role", "artifact"))
            path = item["path"]
            try:
                declared_digest = item.get("sha256")
                if not isinstance(declared_digest, str) or re.fullmatch(
                    r"[0-9a-fA-F]{64}", declared_digest
                ) is None:
                    raise ValueError("evidence is missing a valid execution-time sha256")
                stat = self.call_tool("get_file_stat", path=path)
                if not stat.get("exists") or stat.get("is_dir"):
                    raise ValueError("evidence file is missing or is a directory")
                digest = self.call_tool("hash_path", path=path).get("sha256", "")
                if digest.lower() != declared_digest.lower():
                    raise ValueError("evidence digest mismatch")
                declared_size = item.get("size_bytes")
                if (
                    isinstance(declared_size, bool)
                    or not isinstance(declared_size, int)
                    or declared_size < 0
                ):
                    raise ValueError("evidence is missing a valid execution-time size")
                if int(stat.get("size_bytes", -1)) != declared_size:
                    raise ValueError("evidence size mismatch")
                record = {
                    "role": role,
                    "relative_path": item.get("relative_path", ""),
                    "sha256": digest,
                    "size_bytes": stat.get("size_bytes", 0),
                }
                if role in {"stdout_log", "stderr_log", "execution_state"}:
                    content = self.call_tool("read_file", path=path).get("content", "")
                    if role == "execution_state":
                        state = self._parse_json_value(content)
                        if not isinstance(state, dict) or state.get("status") not in {"COMPLETED", "TERMINATED"}:
                            errors.append("execution state does not show a completed/terminated run")
                        elif state.get("exit_code") is not None and state.get("exit_code") != run.exit_code:
                            errors.append("execution state exit code differs from run record")
                    logs.append(role)
                elif role == "metrics":
                    content = self.call_tool("read_file", path=path).get("content", "")
                    metrics = self._parse_metrics(content)
                    if not metrics:
                        errors.append("metrics artifact is empty, invalid, or contains no finite numeric values")
                artifacts.append(record)
                verified_roles.add(role)
            except Exception as exc:  # evidence failures are part of the verdict
                errors.append(f"{role}: {exc}")
        for required in ("metrics", "stdout_log", "stderr_log", "execution_state"):
            if required not in verified_roles:
                errors.append(f"missing independent evidence: {required}")
        return {
            "verified": bool(evidence) and not errors and bool(metrics),
            "errors": list(dict.fromkeys(errors)),
            "artifacts": artifacts,
            "logs": logs,
            "metrics": metrics,
        }

    def _anti_cheat_material(
        self, inputs: dict[str, Any]
    ) -> tuple[list[str], list[str]]:
        """Read bounded real source slices; never scan only an empty summary."""

        material: list[str] = []
        errors: list[str] = []
        summary = inputs.get("implementation_summary")
        if isinstance(summary, str) and summary.strip():
            material.append(summary[:100_000])

        findings = inputs.get("code_findings")
        repository_path = inputs.get("repository_path")
        if isinstance(findings, dict):
            material.append(
                json.dumps(findings, ensure_ascii=False, sort_keys=True)[:100_000]
            )
            evidence_refs = findings.get("analysis_evidence") or []
            if not evidence_refs and isinstance(findings.get("entry_points"), list):
                evidence_refs = [
                    {"path": path, "start_line": 1, "end_line": 400}
                    for path in findings["entry_points"][:12]
                    if isinstance(path, str)
                ]
            source_slices = 0
            if isinstance(repository_path, str) and isinstance(evidence_refs, list):
                for item in evidence_refs[:24]:
                    if not isinstance(item, dict):
                        continue
                    relative_path = item.get("path")
                    if not isinstance(relative_path, str):
                        continue
                    normalized = relative_path.replace("\\", "/").strip("/")
                    if not normalized or ".." in normalized.split("/") or "://" in normalized:
                        errors.append("anti-cheat source evidence contains an unsafe path")
                        continue
                    try:
                        start_line = max(1, int(item.get("start_line", 1)))
                        end_line = min(start_line + 399, int(item.get("end_line", start_line + 399)))
                        source = self.call_tool(
                            "read_file",
                            path=f"{repository_path.rstrip('/')}/{normalized}",
                            start_line=start_line,
                            end_line=max(start_line, end_line),
                        ).get("content", "")
                    except Exception as exc:
                        errors.append(f"cannot read anti-cheat source evidence: {exc}")
                        continue
                    if source:
                        material.append(str(source)[:80_000])
                        source_slices += 1
            if source_slices == 0:
                errors.append("anti-cheat could not verify any real implementation source")

        if not material:
            errors.append("anti-cheat input is missing; no implementation/code evidence was scanned")
        return material, list(dict.fromkeys(errors))

    @staticmethod
    def _parse_json_value(content: str) -> Any:
        try:
            return json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return None

    @classmethod
    def _parse_metrics(cls, content: str) -> dict[str, float]:
        value = cls._parse_json_value(content)
        if isinstance(value, dict) and isinstance(value.get("metrics"), dict):
            value = value["metrics"]
        if not isinstance(value, dict):
            return {}
        return {
            str(key): float(item)
            for key, item in value.items()
            if isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item))
        }

    def _recompute_metrics(
        self, evidence: list[dict[str, Any]], spec: ExperimentSpec, reported: dict[str, float]
    ) -> tuple[dict[str, float], list[str]]:
        """Recompute common metrics from independently staged predictions/labels."""

        by_name = {
            str(item.get("relative_path", "")).lower(): item
            for item in evidence
            if isinstance(item, dict)
        }
        prediction_item = next((item for name, item in by_name.items() if "predict" in name), None)
        label_item = next((item for name, item in by_name.items() if "label" in name or "target" in name), None)
        requested = bool(self.task.definition.inputs.get("recompute_metrics")) or bool(prediction_item and label_item)
        if not prediction_item or not label_item:
            return {}, ["requested metric recomputation evidence is missing"] if requested else []
        try:
            predictions = self._read_sequence(prediction_item["path"])
            labels = self._read_sequence(label_item["path"])
        except Exception as exc:
            return {}, [f"cannot read prediction/label artifacts for recomputation: {exc}"]
        if len(predictions) != len(labels) or not predictions:
            return {}, ["prediction and label artifacts have different or empty lengths"]
        accuracy = sum(pred == label for pred, label in zip(predictions, labels)) / len(labels)
        computed = {"accuracy": accuracy, "exact_match": accuracy}
        classes = sorted(set(labels) | set(predictions), key=str)
        if len(classes) == 2:
            positive = classes[-1]
            tp = sum(pred == positive and label == positive for pred, label in zip(predictions, labels))
            fp = sum(pred == positive and label != positive for pred, label in zip(predictions, labels))
            fn = sum(pred != positive and label == positive for pred, label in zip(predictions, labels))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            computed.update({
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            })
        return {name: value for name, value in computed.items() if name in spec.expected_results or name in reported}, []

    def _read_sequence(self, path: str) -> list[Any]:
        value = self._parse_json_value(self.call_tool("read_file", path=path).get("content", ""))
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for key in ("predictions", "labels", "targets", "values"):
                if isinstance(value.get(key), list):
                    return value[key]
        return []

    def _spec_from_dict(self, data: dict[str, Any]) -> ExperimentSpec:
        spec = ExperimentSpec(
            experiment_id=data.get("experiment_id", "unknown"),
            target_claim=data.get("target_claim", ""),
            job_id=self.task.job_id,
        )
        for metric, meta in (data.get("expected_results") or {}).items():
            try:
                tolerance_type = ToleranceType(meta.get("tolerance_type", "absolute"))
            except ValueError:
                tolerance_type = ToleranceType.ABSOLUTE
            spec.expected_results[metric] = ExpectedResult(
                metric=metric,
                value=float(meta.get("value", 0.0)),
                tolerance_type=tolerance_type,
                tolerance=float(meta.get("tolerance", 0.0)),
                tolerance_basis=meta.get("tolerance_basis", ""),
            )
        return spec

    def _run_from_dict(self, data: dict[str, Any]) -> ExperimentRun:
        try:
            tier = ExperimentTier(data.get("tier", "full_experiment"))
        except ValueError:
            tier = ExperimentTier.FULL_EXPERIMENT
        return ExperimentRun(
            experiment_id=data.get("experiment_id", "unknown"),
            job_id=self.task.job_id,
            tier=tier,
            run_id=data.get("run_id", ""),
            git_commit=data.get("git_commit", ""),
            container_digest=data.get("container_digest", ""),
            config_digest=data.get("config_digest", ""),
            dataset_digest=data.get("dataset_digest", ""),
            dataset_manifest=data.get("dataset_manifest", {}),
            model_identifier=data.get("model_identifier", ""),
            seed=data.get("seed"),
            hardware_identifier=data.get("hardware_identifier", ""),
            command=data.get("command", ""),
            exit_code=data.get("exit_code"),
            metrics=data.get("metrics", {}),
            log_path=data.get("log_path", ""),
        )

    def _render_candidate_memory(self, result: VerificationResult, comparisons: list[MetricComparison]) -> str:
        lines = [
            f"# verification.{self.task.task_id}", "", "## 摘要 (L1)",
            f"全部指标在容差范围内: {result.all_within_tolerance}; 独立证据通过: {result.evidence_verified}",
            "", "## 细节 (L2)",
        ]
        for comparison in comparisons:
            lines.append(
                f"- {comparison.metric}: paper={comparison.paper_value}, reproduced={comparison.reproduced_value}, within_tolerance={comparison.within_tolerance}"
            )
        lines.extend(["", "## 证据 (L3)"])
        for artifact in result.verified_artifacts:
            lines.append(f"- {artifact.get('role')}: sha256={artifact.get('sha256')}")
        for error in result.evidence_errors + result.provenance_errors + result.recomputation_errors:
            lines.append(f"- 校验问题: {error}")
        return "\n".join(lines) + "\n"
