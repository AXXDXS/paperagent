"""代码分析子智能体（设计文档 §9.2）。

职责：扫描仓库、识别入口、识别配置系统、分析数据/模型/训练/推理/
评测流程、分析参数覆盖、确定最终有效值、确定实验输出路径、
识别对应论文实验的运行脚本。

风险预算：``code_analysis`` -> READ_ONLY，只能使用只读工具扫描代码
仓库（Repo Map/符号检索/read_file），不能修改
任何代码——修改代码是"代码修改子智能体"（§9.6）的职责，两者严格
分离，避免"分析的同时顺手改了"这种职责越界。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from repro_agent.agents.base import AgentRunResult, BaseSubAgent
from repro_agent.llm_output import (
    CODE_ANALYSIS_SCHEMA,
    parse_structured_json,
)
from repro_agent.orchestrator.runtime_configuration import normalize_requirements
from repro_agent.tools.base import ToolExecutionError


def _coerce_nonempty_str(value: Any) -> str | None:
    """把无歧义的标量转成非空字符串，其余返回 None（交给 strict schema 拒绝）。"""

    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def normalize_code_analysis_payload(data: Any) -> Any:
    """strict schema 校验前的确定性表现层矫正（与论文 agent 的
    normalize_paper_analysis_payload 同构）。

    真实模型常把 string 字段写成结构化变体：matched_run_scripts 的值给
    成命令数组（["python", "eval.py"]）或对象（{"script": "..."}），
    tier_commands 的值给成单条命令字符串，entry_points 给成单个字符串。
    这些都是合法答案的表现差异而非事实错误：只重写能确定性解码的形态，
    无法安全转换的直接丢弃（缺字段对 schema 合法，错类型不合法）。
    """

    if not isinstance(data, dict):
        return data

    for key in ("entry_points", "experiment_output_paths"):
        value = data.get(key)
        if isinstance(value, str):
            coerced = _coerce_nonempty_str(value)
            data[key] = [coerced] if coerced else []
        elif isinstance(value, list):
            data[key] = [
                coerced
                for coerced in (_coerce_nonempty_str(item) for item in value)
                if coerced
            ]

    scripts = data.get("matched_run_scripts")
    if isinstance(scripts, dict):
        for experiment_id, script in list(scripts.items()):
            if isinstance(script, (list, tuple)):
                parts = [
                    str(part)
                    for part in script
                    if isinstance(part, (str, int, float))
                    and not isinstance(part, bool)
                    and str(part).strip()
                ]
                scripts[experiment_id] = " ".join(parts) if parts else None
            elif isinstance(script, dict):
                candidate = next(
                    (
                        script[key]
                        for key in ("script", "path", "file", "entry", "command")
                        if isinstance(script.get(key), str) and script[key].strip()
                    ),
                    None,
                )
                scripts[experiment_id] = candidate or json.dumps(
                    script, ensure_ascii=False, sort_keys=True
                )
            else:
                scripts[experiment_id] = _coerce_nonempty_str(script)
        data["matched_run_scripts"] = {
            str(key): value for key, value in scripts.items() if value
        }

    tiers = data.get("tier_commands")
    if isinstance(tiers, dict):
        normalized_tiers: dict[str, list[str]] = {}
        for tier, command in tiers.items():
            if isinstance(command, str):
                parts = command.split()
            elif isinstance(command, (list, tuple)):
                parts = [
                    str(part)
                    for part in command
                    if isinstance(part, (str, int, float))
                    and not isinstance(part, bool)
                    and str(part).strip()
                ]
            else:
                parts = []
            if parts:
                normalized_tiers[str(tier)] = parts
        data["tier_commands"] = normalized_tiers

    # 模型常对不适用的可选字段输出显式 null（如 command_argument 条目的
    # "environment_variable": null）。这些字段在 schema 中本就可缺省，
    # 下游全部使用 .get() 消费；丢弃 null 键属于表现层矫正而非补造事实。
    requirements = data.get("required_user_configuration")
    if isinstance(requirements, list):
        cleaned_requirements: list[Any] = []
        for item in requirements:
            if isinstance(item, dict):
                cleaned = {
                    key: value for key, value in item.items() if value is not None
                }
                if cleaned:
                    cleaned_requirements.append(cleaned)
            elif item is not None:
                cleaned_requirements.append(item)
        data["required_user_configuration"] = cleaned_requirements

    return data


# 以仓库为根解析路径参数的工具：参数名 -> 路径所在的 kwarg 名。
# Repo Map、初始证据与 prompt 示例向模型展示的都是仓库相对路径
# （如 README.md、src/train.py:10），模型据此发起的工具调用按沙箱根
# 解析时会得到 file not found——这里在失败后用任务输入的仓库根重试。
_REPO_ROOTED_TOOL_ARGS = {
    "read_file": "path",
    "hash_path": "path",
    "search_repository_code": "root",
    "get_repository_map": "root",
}


@dataclass
class CodeAnalysisFinding:
    entry_points: list[str] = field(default_factory=list)
    config_system: str = ""
    data_pipeline_summary: str = ""
    model_pipeline_summary: str = ""
    training_pipeline_summary: str = ""
    inference_pipeline_summary: str = ""
    evaluation_pipeline_summary: str = ""
    effective_parameters: dict[str, Any] = field(default_factory=dict)
    experiment_output_paths: list[str] = field(default_factory=list)
    matched_run_scripts: dict[str, str] = field(default_factory=dict)  # experiment_id -> script path
    tier_commands: dict[str, list[str]] = field(default_factory=dict)
    required_user_configuration: list[dict[str, Any]] = field(default_factory=list)
    repository_digest: str = ""
    analysis_evidence: list[dict[str, Any]] = field(default_factory=list)
    analysis_coverage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_points": self.entry_points,
            "config_system": self.config_system,
            "data_pipeline_summary": self.data_pipeline_summary,
            "model_pipeline_summary": self.model_pipeline_summary,
            "training_pipeline_summary": self.training_pipeline_summary,
            "inference_pipeline_summary": self.inference_pipeline_summary,
            "evaluation_pipeline_summary": self.evaluation_pipeline_summary,
            "effective_parameters": self.effective_parameters,
            "experiment_output_paths": self.experiment_output_paths,
            "matched_run_scripts": self.matched_run_scripts,
            "tier_commands": self.tier_commands,
            "required_user_configuration": self.required_user_configuration,
            "repository_digest": self.repository_digest,
            "analysis_evidence": self.analysis_evidence,
            "analysis_coverage": self.analysis_coverage,
        }


class CodeAnalysisAgent(BaseSubAgent):
    task_type = "code_analysis"
    system_prompt = (
        "你是 ReproAgent 系统的代码分析子智能体。你的任务是扫描代码仓库，"
        "识别程序入口、配置系统，分析数据/模型/训练/推理/评测流程，"
        "确定参数的最终有效值（考虑命令行覆盖、配置文件覆盖、代码默认值"
        "的优先级），并找出与目标论文实验对应的运行脚本。你只能使用只读"
        "工具（查找文件、检索关键字、阅读文件）扫描代码，绝不能修改任何"
        "代码文件。"
    )

    def call_tool(self, tool_name: str, /, **kwargs: Any) -> Any:
        """带仓库相对路径修复的工具调用入口。

        背景：确定性后处理 ``_validate_model_evidence`` 早已对模型回传的
        相对路径做了 ``仓库根 + 相对路径`` 拼接，但交互式工具轮没有同
        等补偿——模型照抄证据里的 ``README.md`` 直接调用就会击穿任务。
        这里在首次失败后用仓库根重试一次；两次调用都会进入授权层审计
        日志（先 ERROR 后 ok），行为透明。仅当路径确实是裸相对路径
        （非 ``input://``/``workspace://`` 虚拟路径、非绝对路径）且任务
        声明了仓库根时才修复，其余错误原样重抛（由 base 工具循环回填
        给模型自纠）。
        """

        try:
            return super().call_tool(tool_name, **kwargs)
        except ToolExecutionError:
            repaired = self._repair_repository_relative_argument(tool_name, kwargs)
            if repaired is None:
                raise
            return super().call_tool(tool_name, **repaired)

    def _repair_repository_relative_argument(
        self, tool_name: str, kwargs: dict[str, Any]
    ) -> dict[str, Any] | None:
        """若失败路径是仓库相对路径，返回挂到仓库根后的新参数。"""

        arg_name = _REPO_ROOTED_TOOL_ARGS.get(tool_name)
        if arg_name is None:
            return None
        raw = kwargs.get(arg_name)
        if not isinstance(raw, str) or not raw.strip():
            return None
        path = raw.strip()
        # 虚拟路径（input://...）与绝对路径无需也不应修复。
        if "://" in path or path.startswith("/"):
            return None
        repository_root = str(
            self.task.definition.inputs.get("repository_path", "")
        ).strip()
        if not repository_root or repository_root == ".":
            return None
        repaired = dict(kwargs)
        repaired[arg_name] = f"{repository_root.rstrip('/')}/{path}"
        return repaired

    def run(self) -> AgentRunResult:
        repo_root = self.task.definition.inputs.get("repository_path", ".")
        target_experiments = self.task.definition.inputs.get("target_experiments", [])

        self.report_progress(0.05, "building lightweight repository index")
        overview = self._scan_repository(repo_root, target_experiments)
        self.report_progress(0.45, "repository map and initial evidence ready")
        prompt = self._build_prompt(repo_root, overview, target_experiments)
        # 确定性阶段先把仓库缩小到 Repo Map + 高相关证据。模型仍可基于
        # 新发现的符号继续检索/精读，但只拿到两个只读工具，并限制轮次
        # （每轮代价仅为索引检索/文件阅读，远低于一次 LLM 调用；实测
        # 中等仓库需要 5-8 轮才能覆盖入口/配置/训练/评测链路），避免
        # 大仓库探索退化为无界 Agent 循环。
        iterative_tools = [
            name
            for name in ("search_repository_code", "read_file")
            if name in self.granted_tools
        ]
        response = self.call_llm(
            prompt,
            temperature=0.2,
            tool_names=iterative_tools,
            output_schema=CODE_ANALYSIS_SCHEMA,
            output_schema_name="code_analysis",
            max_tool_rounds=8,
        )
        finding = self._parse_llm_output(response.content)
        finding.repository_digest = str(overview.get("repository_digest", ""))
        finding.analysis_coverage = dict(overview.get("coverage", {}))
        validated_evidence = self._validate_model_evidence(
            repo_root, finding.analysis_evidence, overview
        )
        finding.analysis_evidence = validated_evidence or list(
            overview.get("evidence_refs", [])
        )[:24]
        finding.required_user_configuration = self._validate_required_configuration_evidence(
            finding.required_user_configuration,
            finding.analysis_evidence,
        )

        result_payload = {"repository_path": repo_root, **finding.to_dict()}
        self.write_json_output("result.json", result_payload)
        self.write_candidate_memory(self._render_candidate_memory(finding))

        return AgentRunResult(
            succeeded=True,
            outputs=result_payload,
            candidate_memory_written=True,
            raw_llm_responses=[response.content],
        )

    def _scan_repository(
        self, repo_root: str, target_experiments: list[str]
    ) -> dict[str, Any]:
        """Build a bounded repo map and initial line-addressed evidence set."""

        return self.checkpointed(
            "repository_index_scan",
            lambda: self._scan_repository_once(repo_root, target_experiments),
        )

    def _scan_repository_once(
        self, repo_root: str, target_experiments: list[str]
    ) -> dict[str, Any]:
        """Run deterministic file -> symbol -> exact-slice localization."""

        if not {"get_repository_map", "search_repository_code"}.issubset(
            set(self.granted_tools)
        ):
            return self._legacy_scan_repository(repo_root)

        requested_tokens = self._bounded_int_input(
            "code_context_budget_tokens", default=10_000, minimum=4_000, maximum=30_000
        )
        max_files = self._bounded_int_input(
            "code_index_max_files", default=5_000, minimum=100, maximum=20_000
        )
        required_checks = [
            str(item)
            for item in self.task.definition.inputs.get("required_checks", [])
            if str(item).strip()
        ]
        target_query = " ".join(str(item) for item in target_experiments if str(item).strip())
        map_query = " ".join(
            part
            for part in (
                target_query,
                " ".join(required_checks),
                "train evaluate inference dataset model config metric output command",
            )
            if part
        )
        map_budget = min(4_000, max(1_500, requested_tokens // 4))
        repository_map_envelope = self.call_tool_for_model(
            "get_repository_map",
            root=repo_root,
            query=map_query,
            token_budget=map_budget,
            max_files=max_files,
        )
        repository_map = dict(repository_map_envelope.get("data", {}))

        queries = self._retrieval_queries(target_experiments, required_checks)
        evidence: list[dict[str, Any]] = []
        seen: set[tuple[str, str, int]] = set()
        evidence_char_budget = max(
            10_000,
            (requested_tokens - map_budget - 1_500) * 4,
        )
        evidence_chars = 0
        for query in queries:
            self.check_cancellation()
            search_envelope = self.call_tool_for_model(
                "search_repository_code",
                root=repo_root,
                query=query,
                max_results=8,
                max_files=max_files,
            )
            search = dict(search_envelope.get("data", {}))
            for item in search.get("results", []):
                key = (
                    str(item.get("path", "")),
                    str(item.get("symbol", "")),
                    int(item.get("start_line", 1)),
                )
                if key in seen:
                    continue
                preview = str(item.get("preview", ""))
                item_cost = len(preview) + 300
                if evidence_chars + item_cost > evidence_char_budget:
                    continue
                seen.add(key)
                evidence_chars += item_cost
                evidence.append({**item, "query": query})

        evidence_refs = [
            {
                "path": str(item.get("path", "")),
                "start_line": int(item.get("start_line", 1)),
                "end_line": int(item.get("end_line", item.get("start_line", 1))),
                "symbol": str(item.get("symbol", "")),
                "reason": f"initial retrieval query: {item.get('query', '')}",
                "file_digest": str(item.get("file_digest", "")),
            }
            for item in evidence
            if item.get("path")
        ]
        indexed_files = int(repository_map.get("indexed_file_count", 0))
        discovered_files = int(repository_map.get("file_count", 0))
        return {
            "scan_strategy": "lightweight_index_repo_map_layered_retrieval",
            "repository_digest": repository_map.get("repository_digest", ""),
            "repo_map": repository_map.get("repo_map", ""),
            "ranked_files": repository_map.get("ranked_files", []),
            "evidence": evidence,
            "evidence_refs": evidence_refs,
            "coverage": {
                "index_version": repository_map.get("index_version", 1),
                "discovered_file_count": discovered_files,
                "indexed_file_count": indexed_files,
                "index_coverage_ratio": (
                    round(indexed_files / discovered_files, 4) if discovered_files else 1.0
                ),
                "skipped_file_count": repository_map.get("skipped_file_count", 0),
                "ignored_directories": repository_map.get("ignored_directories", 0),
                "languages": repository_map.get("languages", {}),
                "retrieval_queries": queries,
                "retrieved_evidence_count": len(evidence),
                "context_budget_tokens": requested_tokens,
                "truncated": bool(repository_map.get("truncated", False)),
            },
        }

    def _legacy_scan_repository(self, repo_root: str) -> dict[str, Any]:
        """Keep restored pre-index tasks runnable with their historical grants."""

        if not {"find_files", "grep_files", "read_file"}.issubset(set(self.granted_tools)):
            raise RuntimeError(
                "code analysis requires get_repository_map/search_repository_code "
                "or the legacy find_files/grep_files/read_file tool set"
            )
        py_files = dict(
            self.call_tool_for_model(
                "find_files", pattern="*.py", root=repo_root, max_results=300
            ).get("data", {})
        )
        configs = []
        for suffix in ("*.yaml", "*.yml"):
            configs.extend(
                dict(
                    self.call_tool_for_model(
                        "find_files", pattern=suffix, root=repo_root, max_results=100
                    ).get("data", {})
                ).get("matches", [])
            )
        overview = {
            "py_files": py_files.get("matches", []),
            "config_files": configs,
            "readme_run_commands": dict(
                self.call_tool_for_model(
                    "grep_files",
                    query="python",
                    root=repo_root,
                    file_glob="README*",
                    max_results=50,
                ).get("data", {})
            ).get("results", []),
        }
        paths = [*configs, *overview["py_files"]]
        ranked = sorted(
            dict.fromkeys(paths),
            key=lambda path: (
                not any(
                    token in path.lower()
                    for token in ("train", "main", "run", "eval", "config", "setup")
                ),
                path.count("/"),
                path,
            ),
        )[:16]
        excerpts = {}
        for relative_path in ranked:
            try:
                result = dict(
                    self.call_tool_for_model(
                        "read_file",
                        path=f"{repo_root.rstrip('/')}/{relative_path}",
                        start_line=1,
                        end_line=300,
                    ).get("data", {})
                )
            except Exception:
                continue
            excerpts[relative_path] = str(result.get("content", ""))[:20_000]
        return {
            "scan_strategy": "legacy_bounded_scan",
            "repository_digest": "",
            "legacy_overview": overview,
            "legacy_excerpts": excerpts,
            "evidence_refs": [],
            "coverage": {"truncated": True, "legacy": True},
        }

    def _bounded_int_input(
        self, name: str, *, default: int, minimum: int, maximum: int
    ) -> int:
        try:
            value = int(self.task.definition.inputs.get(name, default))
        except (TypeError, ValueError):
            value = default
        return min(maximum, max(minimum, value))

    @staticmethod
    def _retrieval_queries(
        target_experiments: list[str], required_checks: list[str] | None = None
    ) -> list[str]:
        queries = []
        target_query = " ".join(str(item) for item in target_experiments if str(item).strip())
        if target_query:
            queries.append(target_query)
        queries.extend(
            str(item) for item in (required_checks or []) if str(item).strip()
        )
        queries.extend(
            [
                "README usage install command python train run main entrypoint",
                "dataset dataloader preprocess transform split sampler",
                "model architecture network forward backbone checkpoint",
                "train fit optimizer scheduler loss learning_rate epoch seed checkpoint",
                "evaluate evaluation metric accuracy f1 inference predict result output",
                "config argparse hydra yaml toml output_dir save_dir",
                "required model name API base API key token getenv os.environ required argument",
            ]
        )
        return list(dict.fromkeys(queries))

    def _build_prompt(
        self, repo_root: str, overview: dict[str, Any], target_experiments: list[str]
    ) -> str:
        if overview.get("scan_strategy") == "legacy_bounded_scan":
            context = json.dumps(
                {
                    "overview": overview.get("legacy_overview", {}),
                    "source_excerpts": overview.get("legacy_excerpts", {}),
                },
                ensure_ascii=False,
                indent=2,
            )[:30_000]
        else:
            evidence_blocks = []
            for index, item in enumerate(overview.get("evidence", []), start=1):
                evidence_blocks.append(
                    "\n".join(
                        [
                            f"[E{index}] {item.get('path')}:{item.get('start_line')}-{item.get('end_line')}",
                            f"query={item.get('query')}; symbol={item.get('symbol') or '-'}; digest={item.get('file_digest')}",
                            str(item.get("preview", "")),
                        ]
                    )
                )
            context = (
                "REPOSITORY MAP\n"
                f"{overview.get('repo_map', '')}\n\n"
                "INITIAL RETRIEVED EVIDENCE\n"
                + "\n\n".join(evidence_blocks)
            )
        audit_hypothesis = str(
            self.task.definition.inputs.get("audit_hypothesis", "")
        ).strip()
        required_checks = [
            str(item)
            for item in self.task.definition.inputs.get("required_checks", [])
            if str(item).strip()
        ]
        audit_note = ""
        if audit_hypothesis or required_checks:
            audit_note = (
                f"审计目标: {audit_hypothesis}\n"
                f"必须逐项核查: {json.dumps(required_checks, ensure_ascii=False)}\n"
                "相关结论必须落到 analysis_evidence 的真实文件行号。\n"
            )
        return (
            f"仓库根目录: {repo_root}\n"
            f"目标复现实验: {target_experiments}\n"
            f"{audit_note}"
            f"扫描策略: {overview.get('scan_strategy')}\n"
            f"仓库摘要: {overview.get('repository_digest', '')}\n"
            f"索引覆盖: {json.dumps(overview.get('coverage', {}), ensure_ascii=False)}\n\n"
            f"{context}\n\n"
            "请按以下分层流程分析：先依据 Repo Map 定位文件，再依据类/函数"
            "和初始证据定位实现；如果某个结论证据不足，可调用 "
            "search_repository_code，使用上一轮发现的符号形成新查询，然后用 "
            "read_file 按行精读。不要请求完整仓库或无界读取。\n"
            "所有命令、参数最终有效值、输出路径都必须来自真实代码/配置/文档。"
            "analysis_evidence 使用相对仓库路径和 1-based 行号，不要引用 Repo Map"
            "本身作为最终证据。输出 JSON:\n"
            '{"entry_points": [...], "config_system": "...", '
            '"data_pipeline_summary": "...", "model_pipeline_summary": "...", '
            '"training_pipeline_summary": "...", "inference_pipeline_summary": "...", '
            '"evaluation_pipeline_summary": "...", "effective_parameters": {...}, '
            '"experiment_output_paths": [...], "matched_run_scripts": {"exp_id": "path"}, '
            '"tier_commands": {"static_check": ["python", "-m", "compileall", "-q", "."], '
            '"unit_test": ["python", "-m", "pytest", "-q"], "smoke_test": [...], '
            '"reduced_experiment": [...], "full_experiment": [...]}, '
            '"required_user_configuration": [{"name": "MODEL_NAME", '
            '"kind": "model_name|api_base|credential_env|other", '
            '"delivery": "environment|command_argument", '
            '"environment_variable": "MODEL_NAME", "argument": "--model", '
            '"required": true, "reason": "why omission must fail", '
            '"source_ref": "src/train.py:10"}], '
            '"analysis_evidence": [{"path": "src/train.py", "start_line": 10, '
            '"end_line": 80, "symbol": "main", "reason": "训练入口与参数覆盖", '
            '"file_digest": "..."}]}\n'
            "tier_commands 必须来自仓库脚本/文档/配置的可核查依据，不要凭空编造参数。"
            "required_user_configuration 也只能列出代码能够证明为必需、当前没有"
            "有效默认值且缺失会导致实验失败的配置；不要列可选参数。credential_env"
            "只填写环境变量名，绝不能输出或猜测凭证值。source_ref 必须给出真实"
            "文件和行号。若没有此类配置，返回空数组。"
            "每个 source_ref 必须落在 analysis_evidence 的某个已引用行区间内，"
            "否则该必需配置会被确定性校验器丢弃。"
        )

    def _parse_llm_output(self, content: str) -> CodeAnalysisFinding:
        data = parse_structured_json(
            content,
            CODE_ANALYSIS_SCHEMA,
            label="code analysis output",
            normalize=normalize_code_analysis_payload,
        )
        return CodeAnalysisFinding(
            entry_points=data["entry_points"],
            config_system=data.get("config_system", ""),
            data_pipeline_summary=data.get("data_pipeline_summary", ""),
            model_pipeline_summary=data.get("model_pipeline_summary", ""),
            training_pipeline_summary=data.get("training_pipeline_summary", ""),
            inference_pipeline_summary=data.get("inference_pipeline_summary", ""),
            evaluation_pipeline_summary=data.get("evaluation_pipeline_summary", ""),
            effective_parameters=data.get("effective_parameters", {}),
            experiment_output_paths=data.get("experiment_output_paths", []),
            matched_run_scripts=data.get("matched_run_scripts", {}),
            tier_commands={
                str(key): [str(part) for part in value]
                for key, value in data.get("tier_commands", {}).items()
            },
            required_user_configuration=normalize_requirements(
                data.get("required_user_configuration", [])
            ),
            repository_digest=str(data.get("repository_digest", "")),
            analysis_evidence=[
                dict(item)
                for item in data.get("analysis_evidence", [])
                if isinstance(item, dict)
            ],
            analysis_coverage=dict(data.get("analysis_coverage", {})),
        )

    def _validate_model_evidence(
        self,
        repo_root: str,
        evidence: list[dict[str, Any]],
        overview: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Reject hallucinated/out-of-range evidence before persisting it."""

        if not evidence or "read_file" not in self.granted_tools:
            return []
        validated = []
        root_prefix = repo_root.rstrip("/") + "/"
        initial_digests = {
            str(item.get("path")): str(item.get("file_digest", ""))
            for item in overview.get("evidence", [])
        }
        for item in evidence[:24]:
            raw_path = str(item.get("path", "")).strip().replace("\\", "/")
            if raw_path.startswith(root_prefix):
                relative_path = raw_path[len(root_prefix) :]
            else:
                relative_path = raw_path[2:] if raw_path.startswith("./") else raw_path
            if not relative_path or "://" in relative_path or ".." in relative_path.split("/"):
                continue
            try:
                start_line = max(1, int(item.get("start_line", 1)))
                end_line = max(start_line, int(item.get("end_line", start_line)))
            except (TypeError, ValueError):
                continue
            end_line = min(end_line, start_line + 399)
            try:
                result = self.call_tool(
                    "read_file",
                    path=root_prefix + relative_path,
                    start_line=start_line,
                    end_line=end_line,
                )
            except Exception:
                continue
            total_lines = int(result.get("total_lines", 0))
            if start_line > total_lines:
                continue
            validated.append(
                {
                    "path": relative_path,
                    "start_line": start_line,
                    "end_line": min(end_line, total_lines),
                    "symbol": str(item.get("symbol", "")),
                    "reason": str(item.get("reason", "")),
                    "file_digest": self._trusted_file_digest(
                        root_prefix + relative_path,
                        initial_digests.get(relative_path, ""),
                    ),
                }
            )
        return validated

    def _trusted_file_digest(self, tool_path: str, indexed_digest: str) -> str:
        """Use a tool-derived digest; never persist a model-supplied digest."""

        if "hash_path" not in self.granted_tools:
            return indexed_digest
        try:
            result = self.call_tool("hash_path", path=tool_path)
        except Exception:
            return indexed_digest
        return str(result.get("sha256", "") or indexed_digest)

    @staticmethod
    def _validate_required_configuration_evidence(
        requirements: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Keep only requirements bound to a validated code line."""

        ranges: dict[str, list[tuple[int, int]]] = {}
        for item in evidence:
            path = str(item.get("path", "")).strip().removeprefix("./")
            if not path:
                continue
            try:
                start = int(item.get("start_line", 1))
                end = int(item.get("end_line", start))
            except (TypeError, ValueError):
                continue
            ranges.setdefault(path, []).append((start, end))

        validated = []
        for requirement in normalize_requirements(requirements):
            match = re.fullmatch(r"(.+):(\d+)", requirement["source_ref"].strip())
            if match is None:
                continue
            path = match.group(1).strip().removeprefix("./")
            line = int(match.group(2))
            if any(start <= line <= end for start, end in ranges.get(path, [])):
                validated.append(requirement)
        return validated

    def _render_candidate_memory(self, finding: CodeAnalysisFinding) -> str:
        lines = [
            f"# code.{self.task.task_id}",
            "",
            "## 摘要 (L1)",
            f"入口: {finding.entry_points}; 配置系统: {finding.config_system}",
            "",
            "## 细节 (L2)",
            f"- data_pipeline: {finding.data_pipeline_summary}",
            f"- model_pipeline: {finding.model_pipeline_summary}",
            f"- training_pipeline: {finding.training_pipeline_summary}",
            f"- evaluation_pipeline: {finding.evaluation_pipeline_summary}",
            f"- effective_parameters: {finding.effective_parameters}",
            f"- required_user_configuration: {finding.required_user_configuration}",
            f"- repository_digest: {finding.repository_digest}",
            f"- index_coverage: {finding.analysis_coverage}",
            "",
            "## 证据 (L3)",
        ]
        for exp_id, script in finding.matched_run_scripts.items():
            lines.append(f"- 实验 {exp_id} 对应脚本: {script}")
        for item in finding.analysis_evidence[:24]:
            lines.append(
                f"- {item.get('path')}:{item.get('start_line')}-{item.get('end_line')} "
                f"{item.get('symbol', '')} — {item.get('reason', '')}"
            )
        return "\n".join(lines) + "\n"
