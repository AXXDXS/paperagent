"""ReproAgent 命令行入口。

用法示例：

    python -m repro_agent.cli.main run \\
        --paper-path ./paper.pdf.txt \\
        --repository-path ./target_repo \\
        --target-experiment table_2_main \\
        --work-dir ./runs/my_job

    python -m repro_agent.cli.main run --mock ...   # 使用 MockLLMProvider，无需真实 API Key

设计取舍：
    只提供一个最小可用的 CLI（argparse，不引入额外依赖），
    覆盖"创建 Job → 初始规划 → 跑主循环 → 生成最终报告"这条主链路，
    足以让整个系统可以被端到端地运行和验证，具体的 Web/API 服务化
    留给未来按需扩展（设计文档 §21 推荐目录里的 ``apps/api``、
    ``apps/worker`` 属于部署形态的扩展，不影响核心库的正确性）。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from repro_agent.domain.job import JobBudget, JobInputs, ReproductionJob
from repro_agent.domain.enums import JobStatus
from repro_agent.observability.assembler import ReportAssembler
from repro_agent.observability.report import FinalReportGenerator
from repro_agent.observability.result_query import (
    JobResultIntegrityError,
    JobResultNotFoundError,
    JobResultService,
)
from repro_agent.orchestrator.main_agent import MainAgent, MainAgentConfig
from repro_agent.orchestrator.interventions import (
    InterventionService,
    InterventionValidationError,
)
from repro_agent.providers.base import LLMResponse
from repro_agent.providers.mock import MockLLMProvider
from repro_agent.providers.openai_compatible import OpenAICompatibleProvider
from repro_agent.cli.private_config import LLMSettings, PrivateConfigError, resolve_llm_settings
from repro_agent.storage.database import Database
from repro_agent.storage.repository import InterventionRepository


def _build_provider(args: argparse.Namespace):
    if args.mock:
        # Keep --mock useful after structured-output validation was made
        # fail-closed: every registered JSON-producing agent receives the same
        # deterministic, schema-compatible response envelope.
        payload = {
            "parameters": [],
            "expected_results": {},
            "notes": "mock response",
            "entry_points": ["train.py"],
            "config_system": "mock",
            "data_pipeline_summary": "mock",
            "model_pipeline_summary": "mock",
            "training_pipeline_summary": "mock",
            "inference_pipeline_summary": "mock",
            "evaluation_pipeline_summary": "mock",
            "effective_parameters": {},
            "experiment_output_paths": ["output/metrics.json"],
            "matched_run_scripts": {},
            "tier_commands": {},
            "dependency_analysis": "mock dependencies",
            "failure_type": "UNKNOWN_ERROR",
            "root_cause": "mock",
            "evidence": [],
            "recommended_actions": [],
            "requires_code_change": False,
            "requires_config_change": False,
            "confidence": 0.5,
            "likely_source": "unknown",
            "hypotheses": [],
            "suggested_audit_tasks": [],
            "summary": "mock",
            "files": [],
            "unit_test": None,
        }
        if getattr(args, "demo_profile", False):
            payload.update(
                {
                    "method_summary": "A deterministic scalar threshold classifier.",
                    "parameters": [
                        {
                            "name": "threshold",
                            "value": 0.5,
                            "page": "Method",
                            "confidence": 1.0,
                        },
                        {
                            "name": "seed",
                            "value": 7,
                            "page": "Method",
                            "confidence": 1.0,
                        },
                    ],
                    "expected_results": {
                        "accuracy": {
                            "value": 0.9,
                            "tolerance_type": "absolute",
                            "tolerance": 0.01,
                        }
                    },
                    "notes": "offline one-command demo",
                    "config_system": "command-line arguments",
                    "data_pipeline_summary": "20 fixed scalar examples bundled with the repository",
                    "model_pipeline_summary": "threshold classifier with threshold=0.5",
                    "training_pipeline_summary": "deterministic; no optimization dependency",
                    "inference_pipeline_summary": "predict 1 when x >= 0.5",
                    "evaluation_pipeline_summary": "accuracy recomputed from predictions and labels",
                    "effective_parameters": {"threshold": 0.5, "seed": 7},
                    "experiment_output_paths": [
                        "output/metrics.json",
                        "output/predictions.json",
                        "output/labels.json",
                    ],
                    "dependency_analysis": "standard-library-only deterministic demo",
                }
            )
        provider = MockLLMProvider(
            fallback_response=LLMResponse(
                content=json.dumps(
                    payload,
                    ensure_ascii=False,
                )
            )
        )
        if getattr(args, "demo_profile", False):
            provider.add_rule(
                lambda text: "候选工具目录" in text and "参考模板" in text,
                LLMResponse(
                    content=json.dumps(
                        {
                            "allowed_tools": [
                                "list_directory",
                                "find_files",
                                "grep_files",
                                "read_file",
                                "get_repository_map",
                                "search_repository_code",
                                "get_file_stat",
                                "read_pdf_text",
                                "hash_path",
                                "check_path_resource",
                                "check_disk_space",
                                "check_gpu",
                                "check_cuda",
                                "build_environment_image",
                                "build_conda_environment",
                                "write_file",
                                "write_task_output",
                                "execute_command",
                            ],
                            "reason": "deterministic demo allocation; runtime safety filters remain authoritative",
                        },
                        ensure_ascii=False,
                    )
                ),
            )
        return provider
    settings: LLMSettings = getattr(args, "_llm_settings", None) or resolve_llm_settings(
        getattr(args, "model", None)
    )
    if not settings.api_key:
        print(
            "警告: 未配置 LLM 凭证（REPRO_AGENT_API_KEY 或私密配置文件），且未使用 --mock，"
            "LLM 调用将会失败。",
            file=sys.stderr,
        )
    return OpenAICompatibleProvider(api_base=settings.api_base, api_key=settings.api_key)


def _build_config(args: argparse.Namespace) -> MainAgentConfig:
    legacy_conda_root = Path(args.work_dir) / "conda_envs"
    default_conda_root = (
        legacy_conda_root
        if getattr(args, "command", "") == "resume" and legacy_conda_root.is_dir()
        else Path.home() / ".conda" / "envs"
    )
    return MainAgentConfig(
        memory_root=f"{args.work_dir}/project_memory",
        sandbox_root=f"{args.work_dir}/sandbox",
        snapshot_root=f"{args.work_dir}/context_snapshots",
        db_path=f"{args.work_dir}/repro_agent.db",
        model=args.model,
        model_max_tokens=getattr(args, "model_max_tokens", 32768),
        llm_timeout_seconds=getattr(args, "llm_timeout_seconds", 600.0),
        mock_execution=args.mock,
        container_runtime=getattr(args, "container_runtime", "colima"),
        environment_backend=getattr(args, "environment_backend", "") or "",
        conda_executable=getattr(args, "conda_executable", "conda"),
        conda_env_root=(
            getattr(args, "conda_env_root", None)
            or str(default_conda_root)
        ),
        conda_python_version=getattr(args, "conda_python_version", "3.11"),
        mirror_policy=getattr(args, "mirror_policy", None) or "",
        pip_index_urls=tuple(getattr(args, "pip_index_url", None) or ()),
        conda_channels=tuple(getattr(args, "conda_channel", None) or ()),
        execution_image=getattr(args, "execution_image", "python:3.11-slim"),
        intervention_timeout_seconds=getattr(args, "hitl_timeout_seconds", None),
        # The bundled demo is intentionally a deterministic, non-executing
        # walkthrough.  Normal ``run`` and ``resume`` flows retain the
        # mandatory pre-execution parameter confirmation gate.
        require_execution_parameter_confirmation=not getattr(
            args, "demo_profile", False
        ),
        model_input_cost_per_million_usd=getattr(
            args, "model_input_cost_per_million_usd", 0.0
        ),
        model_output_cost_per_million_usd=getattr(
            args, "model_output_cost_per_million_usd", 0.0
        ),
    )


def _write_report_and_exit(agent: MainAgent, args: argparse.Namespace, outcome) -> int:
    """为新建运行和恢复运行生成同一份最终诊断报告。"""

    result = JobResultService(agent.db, args.work_dir).get(
        agent.job.job_id, write_legacy_reports=True
    )
    report_path = result.report_paths["markdown"]

    if outcome.paused:
        pending = agent.intervention_repo.get_pending_for_job(agent.job.job_id)
        print(f"Job {agent.job.job_id} 已暂停，状态: {agent.job.status.value}")
        if pending is not None:
            print(f"人工介入请求: {pending.request_id}")
            print(f"问题: {pending.question}")
            print(
                "查看与回答: repro_agent intervention list/respond "
                f"--work-dir {args.work_dir}"
            )
        print(f"当前诊断报告已写入: {report_path}")
        return 6

    if not outcome.completed:
        print(f"Job {agent.job.job_id} 达到迭代上限，状态仍为: {agent.job.status.value}")
        print(f"诊断报告已写入: {report_path}")
        return 2

    exit_codes = {
        JobStatus.BLOCKED_BY_MISSING_RESOURCE: 3,
        JobStatus.FAILED: 4,
        JobStatus.CANCELLED: 5,
    }
    exit_code = exit_codes.get(agent.job.status, 0)
    if exit_code:
        print(f"Job {agent.job.job_id} 终止，状态: {agent.job.status.value}")
        print(f"诊断报告已写入: {report_path}")
        return exit_code

    print(f"Job {agent.job.job_id} 已结束，状态: {agent.job.status.value}")
    print(f"报告已写入: {report_path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    inputs = JobInputs(
        paper_path=args.paper_path,
        repository_path=args.repository_path,
        target_experiments=[args.target_experiment] if args.target_experiment else [],
        appendix_paths=args.appendix_path or [],
        supplementary_paths=args.supplementary_path or [],
        dataset_paths=args.dataset_path or [],
        model_paths=args.model_path or [],
        checkpoint_paths=args.checkpoint_path or [],
        dataset_download_urls=args.dataset_download_url or [],
        user_run_commands=args.run_command or [],
        user_environment_notes=args.environment_note,
        environment_name=getattr(args, "environment_name", ""),
        cpu_cores=args.cpu_cores,
        memory_mb=args.memory_mb,
        disk_mb=args.disk_mb,
        gpu_count=args.gpu_count,
        gpu_memory_gb=args.gpu_memory_gb,
        max_runtime_seconds=args.max_runtime_seconds,
    )
    job = ReproductionJob(
        inputs=inputs,
        budget=JobBudget(
            max_gpu_hours=args.max_gpu_hours,
            max_total_runtime_seconds=args.max_total_runtime_seconds,
            max_model_call_budget_usd=args.max_model_cost_usd,
        ),
    )

    config = _build_config(args)

    provider = _build_provider(args)
    agent = MainAgent(job, config, provider)
    agent.bootstrap()
    outcome = agent.run_until_finished(max_iterations=args.max_iterations)

    return _write_report_and_exit(agent, args, outcome)


def cmd_demo(args: argparse.Namespace) -> int:
    """Run the bundled deterministic workload through the complete mock pipeline."""

    demo_root = Path(__file__).resolve().parents[2] / "examples" / "demo"
    paper_path = demo_root / "paper.md"
    repository_path = demo_root / "repository"
    if not paper_path.is_file() or not repository_path.is_dir():
        print(
            f"Demo 输入不完整，请确认目录存在: {demo_root}",
            file=sys.stderr,
        )
        return 2

    run_args = argparse.Namespace(
        paper_path=str(paper_path),
        repository_path=str(repository_path),
        target_experiment="threshold_classifier_accuracy",
        appendix_path=[],
        supplementary_path=[],
        dataset_path=[],
        model_path=[],
        checkpoint_path=[],
        dataset_download_url=[],
        run_command=[
            "python -m compileall -q .",
            "python -m unittest -q",
            "python train.py --tier smoke_test",
            "python train.py --tier reduced_experiment",
            "python train.py --tier full_experiment",
        ],
        environment_note=(
            "Bundled deterministic offline demo; standard library only; "
            "no external dataset, model, GPU, Docker, or network required."
        ),
        gpu_count=0,
        gpu_memory_gb=None,
        cpu_cores=1.0,
        memory_mb=1024,
        disk_mb=4096,
        max_runtime_seconds=60,
        max_total_runtime_seconds=300,
        max_gpu_hours=None,
        max_model_cost_usd=None,
        work_dir=args.work_dir,
        model="mock-demo-model",
        container_runtime="colima",
        execution_image="python:3.11-slim",
        model_input_cost_per_million_usd=0.0,
        model_output_cost_per_million_usd=0.0,
        max_iterations=args.max_iterations,
        hitl_timeout_seconds=None,
        mock=True,
        demo_profile=True,
    )
    print("启动 ReproAgent 离线 Demo（Mock LLM + Mock Executor）")
    print(f"论文输入: {paper_path}")
    print(f"代码输入: {repository_path}")
    print(f"输出目录: {Path(args.work_dir).resolve()}")
    exit_code = cmd_run(run_args)
    if exit_code == 0:
        print("Demo 完成：已生成任务数据库、沙箱产物和最终 Markdown/JSON 报告。")
        print("说明：该模式用于验证 Harness 全链路，结论会如实标记为 PIPELINE_ONLY。")
    return exit_code


def cmd_resume(args: argparse.Namespace) -> int:
    """从已有 work-dir 的持久化状态继续同一个 Job。"""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    provider = _build_provider(args)
    try:
        agent = MainAgent.resume_from_storage(args.job_id, _build_config(args), provider)
        recovery = agent.recover_interrupted_tasks()
    except ValueError as exc:
        print(f"无法恢复 Job {args.job_id}: {exc}", file=sys.stderr)
        return 2

    print(
        f"Job {args.job_id} 恢复完成：接纳旧输出 {len(recovery.recovered_succeeded_task_ids)} 个，"
        f"重新排队 {len(recovery.requeued_task_ids)} 个。"
    )
    outcome = agent.run_until_finished(max_iterations=args.max_iterations)
    return _write_report_and_exit(agent, args, outcome)


def cmd_result(args: argparse.Namespace) -> int:
    """按精确 job_id 校验产物并重新生成该 Job 的独立报告。"""

    database = Database(os.path.join(args.work_dir, "repro_agent.db"))
    try:
        result = JobResultService(database, args.work_dir).get(args.job_id)
    except (JobResultNotFoundError, JobResultIntegrityError, OSError, ValueError) as exc:
        print(f"无法读取 Job 结果: {exc}", file=sys.stderr)
        return 3 if isinstance(exc, JobResultIntegrityError) else 2
    finally:
        database.close()

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(f"Job {result.job_id} 状态: {result.job_status}")
    if result.reproduction_status:
        print(f"复现结论状态: {result.reproduction_status}")
    print(result.final_conclusion)
    print(f"已验证产物: {len(result.artifacts)} 个")
    print(f"Markdown 报告: {result.report_paths['markdown']}")
    print(f"JSON 报告: {result.report_paths['json']}")
    return 0


def _intervention_database(args: argparse.Namespace) -> Database:
    return Database(os.path.join(args.work_dir, "repro_agent.db"))


def cmd_intervention_list(args: argparse.Namespace) -> int:
    database = _intervention_database(args)
    try:
        requests = InterventionRepository(database).list_by_job(args.job_id)
    finally:
        database.close()
    if not requests:
        print(f"Job {args.job_id} 没有人工介入请求。")
        return 0
    for request in requests:
        print(json.dumps(request.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _response_payload(args: argparse.Namespace) -> dict:
    if getattr(args, "response_file", None):
        with open(args.response_file, encoding="utf-8") as file:
            value = json.load(file)
    else:
        value = json.loads(args.response_json)
    if not isinstance(value, dict):
        raise InterventionValidationError("response must be a JSON object")
    return value


def _run_intervention_action(args: argparse.Namespace, action: str) -> int:
    database = _intervention_database(args)
    try:
        service = InterventionService(database)
        if action == "respond":
            result = service.resolve(
                args.request_id,
                _response_payload(args),
                responded_by=args.responded_by,
            )
        elif action == "approve":
            approval_payload = {"approved": True, "reason": args.reason}
            if args.tool:
                approval_payload["approved_tools"] = args.tool
            result = service.resolve(
                args.request_id,
                approval_payload,
                responded_by=args.responded_by,
            )
        else:
            result = service.reject(
                args.request_id,
                reason=args.reason,
                responded_by=args.responded_by,
            )
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"处理人工介入请求失败: {exc}", file=sys.stderr)
        return 2
    finally:
        database.close()

    print(
        f"人工介入请求 {result.request.request_id} 已更新为 "
        f"{result.request.status.value}；Job 状态: {result.job.status.value}。"
    )
    optional_dynamic_rejection = (
        result.request.metadata.get("response_mode") == "dynamic_tool_activation"
        and result.request.status.value in {"REJECTED", "EXPIRED"}
    )
    if result.request.status.value in {"RESOLVED", "APPROVED"} or optional_dynamic_rejection:
        print(
            "可继续运行: python -m repro_agent.cli.main resume "
            f"--job-id {result.job.job_id} --work-dir {args.work_dir}"
        )
    return 0


def cmd_intervention_respond(args: argparse.Namespace) -> int:
    return _run_intervention_action(args, "respond")


def cmd_intervention_approve(args: argparse.Namespace) -> int:
    return _run_intervention_action(args, "approve")


def cmd_intervention_deny(args: argparse.Namespace) -> int:
    return _run_intervention_action(args, "deny")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="repro_agent", description="ReproAgent 论文复现多智能体系统")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="创建并运行一个复现 Job")
    run_parser.add_argument("--paper-path", required=True, help="论文文件路径")
    run_parser.add_argument("--repository-path", required=True, help="目标代码仓库路径")
    run_parser.add_argument("--appendix-path", action="append", help="论文附录路径（可重复）")
    run_parser.add_argument("--supplementary-path", action="append", help="补充材料路径（可重复）")
    run_parser.add_argument("--target-experiment", default="", help="目标复现实验 ID")
    run_parser.add_argument("--dataset-path", action="append", help="数据集路径（可重复）")
    run_parser.add_argument("--model-path", action="append", help="模型路径（可重复）")
    run_parser.add_argument("--checkpoint-path", action="append", help="checkpoint 路径（可重复）")
    run_parser.add_argument("--dataset-download-url", action="append", help="数据集下载地址（可重复）")
    run_parser.add_argument(
        "--run-command",
        action="append",
        help=(
            "分级实验命令（可重复）；如提供 5 条，依次对应 "
            "static/unit/smoke/reduced/full，可作为严格分级契约"
        ),
    )
    run_parser.add_argument("--environment-note", default="", help="用户提供的环境说明")
    run_parser.add_argument(
        "--environment-name",
        default="",
        help="Conda 环境的可读名称；默认使用代码仓库目录名",
    )
    run_parser.add_argument("--cpu-cores", type=float, help="单次实验 CPU 核数上限")
    run_parser.add_argument("--memory-mb", type=int, help="单次实验内存上限（MB）")
    run_parser.add_argument("--disk-mb", type=int, help="单次实验工作区与输出总上限（MB）")
    run_parser.add_argument("--gpu-count", type=int, help="可用 GPU 数量")
    run_parser.add_argument("--gpu-memory-gb", type=float, help="单卡显存 GB")
    run_parser.add_argument("--max-runtime-seconds", type=int, help="单次运行时间上限")
    run_parser.add_argument("--max-total-runtime-seconds", type=int, help="Job 总运行时间预算")
    run_parser.add_argument("--max-gpu-hours", type=float, help="Job GPU 小时预算")
    run_parser.add_argument("--max-model-cost-usd", type=float, help="Job 模型调用成本预算")
    run_parser.add_argument("--work-dir", default="./repro_agent_workdir", help="工作目录（数据库/沙箱/记忆/报告输出）")
    run_parser.add_argument("--model", help="LLM 模型名称；优先于私密配置和环境变量")
    run_parser.add_argument(
        "--model-max-tokens",
        type=int,
        default=32768,
        help=(
            "单次 LLM 调用的最大输出 token 数（含推理型模型的思考 token）；"
            "额度过小会导致思考阶段耗尽预算、返回空内容"
        ),
    )
    run_parser.add_argument(
        "--llm-timeout-seconds",
        type=float,
        default=600.0,
        help="单次 LLM 调用的网络超时秒数",
    )
    run_parser.add_argument(
        "--container-runtime",
        choices=("colima", "docker"),
        default="colima",
        help=(
            "正式命令执行使用的容器运行时；默认 colima，要求先运行 "
            "'colima start'，docker 用于已有 Docker daemon"
        ),
    )
    run_parser.add_argument(
        "--environment-backend",
        choices=("conda", "colima", "docker"),
        default="",
        help="环境和实验执行后端；指定 conda 可免除 Docker/Colima 依赖",
    )
    run_parser.add_argument("--conda-executable", default="conda", help="conda 或 mamba 可执行文件")
    run_parser.add_argument(
        "--conda-env-root",
        help="控制面管理的 Conda prefix 根目录（默认 ~/.conda/envs，可在 conda env list 中按名称显示）",
    )
    run_parser.add_argument("--conda-python-version", default="3.11", help="Conda 环境 Python 版本")
    run_parser.add_argument(
        "--mirror-policy",
        choices=("auto", "fixed", "offline"),
        default=None,
        help="Conda/pip 下载源策略：自动切换、固定首选源或完全离线",
    )
    run_parser.add_argument(
        "--pip-index-url",
        action="append",
        help="允许的 pip HTTPS 源，可重复指定；auto 模式会追加内置公共兜底源",
    )
    run_parser.add_argument(
        "--conda-channel",
        action="append",
        help="允许的 Conda HTTPS channel，可重复指定",
    )
    run_parser.add_argument("--execution-image", default="python:3.11-slim", help="预先拉取的基础执行镜像（建议使用 digest）")
    run_parser.add_argument("--model-input-cost-per-million-usd", type=float, default=0.0)
    run_parser.add_argument("--model-output-cost-per-million-usd", type=float, default=0.0)
    run_parser.add_argument("--max-iterations", type=int, default=1000, help="主循环最大迭代次数")
    run_parser.add_argument("--hitl-timeout-seconds", type=int, help="人工介入等待超时；默认永久等待")
    run_parser.add_argument("--mock", action="store_true", help="使用 Mock LLM Provider（无需真实 API Key，便于离线验证）")
    run_parser.set_defaults(func=cmd_run)

    demo_parser = subparsers.add_parser(
        "demo", help="一键运行内置离线 Demo（不需要 API Key 或 Docker）"
    )
    demo_parser.add_argument(
        "--work-dir",
        default="./repro_agent_demo_output",
        help="Demo 数据库、沙箱和报告输出目录",
    )
    demo_parser.add_argument(
        "--max-iterations",
        type=int,
        default=500,
        help="Demo 主循环最大迭代次数",
    )
    demo_parser.set_defaults(func=cmd_demo)

    resume_parser = subparsers.add_parser("resume", help="从 work-dir 的持久化状态继续一个中断 Job")
    resume_parser.add_argument("--job-id", required=True, help="需要恢复的 Job ID")
    resume_parser.add_argument("--work-dir", default="./repro_agent_workdir", help="原运行使用的工作目录")
    resume_parser.add_argument("--model", help="LLM 模型名称；优先于私密配置和环境变量")
    resume_parser.add_argument(
        "--model-max-tokens",
        type=int,
        default=32768,
        help="单次 LLM 调用的最大输出 token 数（含推理型模型的思考 token）",
    )
    resume_parser.add_argument(
        "--environment-backend",
        choices=("conda", "colima", "docker"),
        default="",
        help="恢复任务使用的环境和实验执行后端",
    )
    resume_parser.add_argument("--conda-executable", default="conda", help="conda 或 mamba 可执行文件")
    resume_parser.add_argument(
        "--conda-env-root", help="原运行使用的 Conda prefix 根目录"
    )
    resume_parser.add_argument("--conda-python-version", default="3.11", help="Conda 环境 Python 版本")
    resume_parser.add_argument(
        "--mirror-policy",
        choices=("auto", "fixed", "offline"),
        default=None,
        help="恢复任务使用的Conda/pip镜像切换策略",
    )
    resume_parser.add_argument("--pip-index-url", action="append")
    resume_parser.add_argument("--conda-channel", action="append")
    resume_parser.add_argument(
        "--llm-timeout-seconds",
        type=float,
        default=600.0,
        help="单次 LLM 调用的网络超时秒数",
    )
    resume_parser.add_argument(
        "--container-runtime",
        choices=("colima", "docker"),
        default="colima",
        help="恢复任务使用的容器运行时；默认 colima",
    )
    resume_parser.add_argument("--execution-image", default="python:3.11-slim", help="原运行使用的基础执行镜像")
    resume_parser.add_argument("--model-input-cost-per-million-usd", type=float, default=0.0)
    resume_parser.add_argument("--model-output-cost-per-million-usd", type=float, default=0.0)
    resume_parser.add_argument("--max-iterations", type=int, default=1000, help="恢复后主循环最大迭代次数")
    resume_parser.add_argument("--hitl-timeout-seconds", type=int, help="后续新人工介入请求的等待超时")
    resume_parser.add_argument("--mock", action="store_true", help="使用 Mock LLM/执行后端恢复离线任务")
    resume_parser.set_defaults(func=cmd_resume)

    result_parser = subparsers.add_parser(
        "result", help="按 Job ID 校验产物并显示已持久化的实验结果"
    )
    result_parser.add_argument("--job-id", required=True, help="需要查询的 Job ID")
    result_parser.add_argument(
        "--work-dir", default="./repro_agent_workdir", help="原运行使用的工作目录"
    )
    result_parser.add_argument("--json", action="store_true", help="输出结构化 JSON")
    result_parser.set_defaults(func=cmd_result)

    intervention_parser = subparsers.add_parser(
        "intervention", help="查看或处理 Human-in-the-loop 请求"
    )
    intervention_subparsers = intervention_parser.add_subparsers(
        dest="intervention_command", required=True
    )

    list_parser = intervention_subparsers.add_parser("list", help="查看 Job 的人工介入请求")
    list_parser.add_argument("--job-id", required=True, help="Job ID")
    list_parser.add_argument("--work-dir", default="./repro_agent_workdir", help="Job 工作目录")
    list_parser.set_defaults(func=cmd_intervention_list)

    respond_parser = intervention_subparsers.add_parser("respond", help="提交结构化回答")
    respond_parser.add_argument("--request-id", required=True, help="介入请求 ID")
    respond_parser.add_argument("--work-dir", default="./repro_agent_workdir", help="Job 工作目录")
    response_source = respond_parser.add_mutually_exclusive_group(required=True)
    response_source.add_argument("--response-json", help="JSON 对象；非敏感的小型回答使用")
    response_source.add_argument("--response-file", help="包含 JSON 对象的文件路径")
    respond_parser.add_argument("--responded-by", default="local-user", help="审计记录中的回答者")
    respond_parser.set_defaults(func=cmd_intervention_respond)

    approve_parser = intervention_subparsers.add_parser("approve", help="批准任务范围内的工具权限")
    approve_parser.add_argument("--request-id", required=True, help="介入请求 ID")
    approve_parser.add_argument("--work-dir", default="./repro_agent_workdir", help="Job 工作目录")
    approve_parser.add_argument("--tool", action="append", help="批准的工具名（可重复）")
    approve_parser.add_argument("--reason", default="", help="批准理由")
    approve_parser.add_argument("--responded-by", default="local-user", help="审计记录中的批准者")
    approve_parser.set_defaults(func=cmd_intervention_approve)

    deny_parser = intervention_subparsers.add_parser("deny", help="拒绝人工介入请求并终止 Job")
    deny_parser.add_argument("--request-id", required=True, help="介入请求 ID")
    deny_parser.add_argument("--work-dir", default="./repro_agent_workdir", help="Job 工作目录")
    deny_parser.add_argument("--reason", default="user rejected the intervention", help="拒绝理由")
    deny_parser.add_argument("--responded-by", default="local-user", help="审计记录中的拒绝者")
    deny_parser.set_defaults(func=cmd_intervention_deny)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"run", "resume"}:
        try:
            args._llm_settings = resolve_llm_settings(args.model)
        except PrivateConfigError as exc:
            parser.error(str(exc))
        args.model = args._llm_settings.model
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
