"""Bind validated dependency payloads into downstream task inputs."""

from __future__ import annotations

from typing import Any, Mapping

from datetime import datetime, timezone

from repro_agent.domain.enums import TaskStatus
from repro_agent.domain.task import Task
from repro_agent.schemas.results import ResultValidationError, TaskResultEnvelope


class ArtifactResolutionError(RuntimeError):
    pass


_INPUT_KEYS = {
"paper_analysis": "paper_findings",
"code_analysis": "code_findings",
"resource_check": "resource_findings",
"specification": "experiment_spec",
"environment_build": "environment",
"experiment_execution": "experiment_run",
"verification": "verification",
}

# 同一输入键出现多个同类型依赖时，除显式合并的论文/代码发现外，
# 其余键语义上都是"当前值"型输入：环境修复（repair）任务会与原始
# 构建任务同时挂在依赖里，后完成的修复结果应覆盖先完成的原始结果，
# 而不是让任务派发直接失败。
_LATEST_WINS_KEYS = frozenset(_INPUT_KEYS.values()) - {
"paper_findings",
"code_findings",
}


def merge_paper_findings(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge body/appendix paper-analysis payloads into one findings dict.

    论文分析拆成正文 + 附录（可能多片）并行任务后，下游仍然期望单一的
    ``paper_findings``。合并规则与 ``specification._PROVENANCE_PRIORITY``
    保持一致：

    - 正文 payload 作为基底（method_summary、expected_results、目标实验
      等正文专属字段只取正文）；
    - 附录参数追加到 ``extracted_parameters``，同名参数**不覆盖**正文值
      （正文显式值优先级更高，冲突时由 specification 的冲突检测显式
      暴露，而不是在这里静默取舍）；
    - ``effective_parameters`` 同样正文优先，附录只填补缺口；
    - 各部分信息记入 ``paper_analysis_parts`` 供审计回溯（哪个任务、
      哪个页范围、贡献了几个参数）。
    """

    if not payloads:
        return {}

    def _sort_key(payload: dict[str, Any]) -> tuple[int, int]:
        # body 永远第一；附录片按页范围起始页排序，保证合并顺序稳定。
        scope = str(payload.get("scope") or "body")
        page_range = payload.get("page_range") or [0, 0]
        try:
            start = int(page_range[0])
        except (TypeError, ValueError, IndexError):
            start = 0
        return (0 if scope == "body" else 1, start)

    ordered = sorted(payloads, key=_sort_key)
    body = next((p for p in ordered if p.get("scope") != "appendix"), ordered[0])

    merged: dict[str, Any] = dict(body)
    parameters: list[dict[str, Any]] = list(body.get("extracted_parameters") or [])
    effective: dict[str, Any] = dict(body.get("effective_parameters") or {})
    notes_parts: list[str] = []
    if body.get("notes"):
        notes_parts.append(f"[body] {body['notes']}")

    parts_audit: list[dict[str, Any]] = [
        {
            "scope": body.get("scope") or "body",
            "page_range": body.get("page_range"),
            "parameters": len(body.get("extracted_parameters") or []),
        }
    ]

    for appendix in ordered:
        if appendix is body or appendix.get("scope") != "appendix":
            continue
        appendix_params = list(appendix.get("extracted_parameters") or [])
        appendix_effective = appendix.get("effective_parameters") or {}
        parameters.extend(appendix_params)
        for name, value in appendix_effective.items():
            # 正文显式值优先；同名时保留正文值，附录值仅在缺口时补充。
            if name not in effective:
                effective[name] = value
        if appendix.get("notes"):
            notes_parts.append(f"[appendix] {appendix['notes']}")
        parts_audit.append(
            {
                "scope": "appendix",
                "page_range": appendix.get("page_range"),
                "parameters": len(appendix_params),
            }
        )

    merged["extracted_parameters"] = parameters
    merged["effective_parameters"] = effective
    merged["paper_analysis_parts"] = parts_audit
    merged["notes"] = "\n".join(notes_parts)
    return merged


def merge_code_findings(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge behaviorally focused code-analysis subtasks deterministically."""

    if not payloads:
        return {}
    merged: dict[str, Any] = dict(payloads[0])
    for key in (
        "entry_points",
        "experiment_output_paths",
        "analysis_evidence",
        "required_user_configuration",
    ):
        values = []
        seen = set()
        for payload in payloads:
            for item in payload.get(key, []) or []:
                marker = repr(item)
                if marker in seen:
                    continue
                seen.add(marker)
                values.append(item)
        merged[key] = values
    conflicts: list[dict[str, Any]] = []
    for key in ("effective_parameters", "matched_run_scripts", "tier_commands"):
        combined: dict[str, Any] = {}
        for payload in payloads:
            for name, value in (payload.get(key, {}) or {}).items():
                if name in combined and combined[name] != value:
                    conflicts.append(
                        {"field": f"{key}.{name}", "values": [combined[name], value]}
                    )
                    continue
                combined[name] = value
        merged[key] = combined
    for key in (
        "data_pipeline_summary",
        "model_pipeline_summary",
        "training_pipeline_summary",
        "inference_pipeline_summary",
        "evaluation_pipeline_summary",
    ):
        parts = [str(payload.get(key, "")).strip() for payload in payloads]
        merged[key] = "\n".join(dict.fromkeys(item for item in parts if item))
    merged["analysis_merge_conflicts"] = conflicts
    merged["analysis_parts"] = len(payloads)
    return merged


class ArtifactResolver:
    def __init__(self, tasks: Mapping[str, Task]):
        self._tasks = tasks

    def resolve(self, task: Task) -> dict[str, object]:
        resolved: dict[str, object] = dict(task.definition.inputs)

        # 同一输入键可能对应多个依赖（论文分析拆为正文 + 附录任务后，
        # 两者都写 ``paper_findings``）。分组后：paper_findings 走
        # scope-aware 合并；其它键出现多依赖则是配置错误，显式报错
        # 而不是静默用最后一个覆盖前面的。
        payloads_by_key: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for dependency_id in task.dependencies:
            dependency = self._tasks.get(dependency_id)
            if dependency is None or dependency.status != TaskStatus.SUCCEEDED:
                raise ArtifactResolutionError(f"dependency {dependency_id} is not validated")
            result_path = dependency.outputs.get("result.json")
            if not result_path:
                raise ArtifactResolutionError(f"dependency {dependency_id} has no result.json")
            try:
                envelope = TaskResultEnvelope.from_file(
                    result_path,
                    expected_task_id=dependency.task_id,
                    expected_attempt_id=dependency.active_attempt_id,
                    expected_task_type=dependency.definition.task_type,
                )
            except ResultValidationError as exc:
                raise ArtifactResolutionError(
                    f"dependency {dependency_id} result is invalid: {exc}"
                ) from exc
            key = _INPUT_KEYS.get(dependency.definition.task_type)
            if key:
                payloads_by_key.setdefault(key, []).append((dependency_id, envelope.payload))

        for key, entries in payloads_by_key.items():
            if len(entries) == 1:
                resolved[key] = entries[0][1]
            elif key == "paper_findings":
                resolved[key] = merge_paper_findings([payload for _, payload in entries])
            elif key == "code_findings":
                resolved[key] = merge_code_findings([payload for _, payload in entries])
            elif key in _LATEST_WINS_KEYS:
                # 修复型前置任务与原始任务同键输出时，取完成时间最新的
                # 依赖（例如环境重建覆盖初始构建）；完成时间缺失时按
                # 依赖列表顺序取最后声明的那个（后声明 = 后规划的修复）。
                def _completed_at(entry: tuple[str, dict[str, Any]]) -> Any:
                    dependency = self._tasks.get(entry[0])
                    return (
                        getattr(dependency, "completed_at", None)
                        if dependency is not None
                        else None
                    ) or datetime.min.replace(tzinfo=timezone.utc)

                resolved[key] = max(entries, key=_completed_at)[1]
            else:
                raise ArtifactResolutionError(
                    f"multiple dependencies map to input '{key}' with no merge policy"
                )
        return resolved
