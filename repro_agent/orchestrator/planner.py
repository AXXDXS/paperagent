"""初始规划器（设计文档 §6 任务拆解 + §17 状态机前半段）。

负责把一个新创建的 ``ReproductionJob`` 拆解为初始任务 DAG：
    论文分析（正文 + 附录，无依赖，可并行）+ 代码分析（无依赖）
        → 资源检查（依赖三者的目标实验确定）
        → 实验规格（依赖论文分析+代码分析+资源检查）
        → 环境构建（依赖实验规格）

论文分析按用户方案拆为**两个并行子任务**：正文任务（方法概要 +
四类参数 + 主实验指标）与附录任务（仅训练相关参数）。拆分前先用
纯代码结构探测（``probe_paper_structure``）定位附录起始页；探测
不到可靠边界时退回单任务全文阅读（fail-closed：宁可整体读，
不可切错丢内容）。附录文本超长时按页继续分片（上限 4 片）。

这是 Job 生命周期中唯一一个"从零构造 DAG"的入口；后续所有任务
（分级实验执行、审计、修复、重跑）都通过 replanner/reflection_controller
在运行时动态追加，不再需要这个模块。
"""

from __future__ import annotations

import logging

from repro_agent.domain.job import ReproductionJob
from repro_agent.domain.task import Task
from repro_agent.execution.environment_naming import managed_environment_name
from repro_agent.orchestrator.task_factory import build_task_definition

logger = logging.getLogger(__name__)

# 附录文本超过该字符量时按页分片（每片一个并行子任务，上限 4 片）。
# 阈值明显低于正文 80k prompt 上限，给附录任务的输出留出充足余量。
APPENDIX_CHUNK_CHAR_LIMIT = 40_000
APPENDIX_MAX_CHUNKS = 4


class InitialPlanner:
    """§6 原则 1-3 的落地：把 Job 输入拆解为第一批任务 DAG。"""

    def __init__(
        self,
        execution_image: str = "python:3.11-slim",
        *,
        environment_backend: str = "docker",
        conda_python_version: str = "3.11",
    ):
        self.execution_image = execution_image
        self.environment_backend = environment_backend
        self.conda_python_version = conda_python_version

    def plan_initial_tasks(self, job: ReproductionJob) -> list[Task]:
        paper_tasks = self._plan_paper_tasks(job)
        paper_task_ids = [task.task_id for task in paper_tasks]

        # CodeAnalysisAgent 先用轻量结构索引生成 Repo Map，再按目标实验
        # 分层检索文件/符号/精确片段；read_file 只在模型根据检索结果需要
        # 继续精读时下发，整个过程保持只读。
        code_task = Task(
            job_id=job.job_id,
            definition=build_task_definition(
                objective="扫描代码仓库，识别入口、配置系统与各流程实现",
                task_type="code_analysis",
                inputs={
                    "repository_path": job.inputs.repository_path,
                    "target_experiments": job.inputs.target_experiments,
                    "creation_key": "initial:code_analysis",
                },
                restrict_tools=[
                    "get_repository_map",
                    "search_repository_code",
                    "read_file",
                    "hash_path",
                ],
                expected_outputs=["output/result.json", "output/candidate_memory.md"],
            ),
        )

        # ExperimentSpecificationAgent 完全靠合并 inputs 里的上游产物做
        # 确定性合并，并从论文/代码参数生成运行必需资源清单。资源检查
        # 必须消费这份规格，不能在规格生成前只检查用户显式填写的路径。
        spec_task = Task(
            job_id=job.job_id,
            definition=build_task_definition(
                objective="合并论文、代码、用户信息生成统一实验复现规格",
                task_type="specification",
                dependencies=[*paper_task_ids, code_task.task_id],
                inputs={
                    "experiment_id": (job.inputs.target_experiments or ["main_experiment"])[0],
                    "target_claim": "reproduce_main_result",
                    "creation_key": "initial:specification",
                },
                restrict_tools=[],
                expected_outputs=["output/result.json"],
            ),
        )

        # 规格生成后再检查资源。find_named_resource 会在已经显式装载的
        # repository_path 内查找 LoCoMo 等同名数据集；用户提供的路径仍由
        # check_path_resource 做存在性/非空性验证。
        resource_tools = [
            "find_named_resource",
            "check_gpu",
            "check_cuda",
            "check_disk_space",
        ]
        if job.inputs.dataset_paths or job.inputs.model_paths or job.inputs.checkpoint_paths:
            resource_tools.append("check_path_resource")
        resource_task = Task(
            job_id=job.job_id,
            definition=build_task_definition(
                objective="按实验规格检查数据、模型、checkpoint、GPU、CUDA、磁盘等运行资源",
                task_type="resource_check",
                dependencies=[spec_task.task_id],
                inputs={
                    "repository_path": job.inputs.repository_path,
                    "dataset_paths": job.inputs.dataset_paths,
                    "model_paths": job.inputs.model_paths,
                    "checkpoint_paths": job.inputs.checkpoint_paths,
                    "requested_cpu_cores": job.inputs.cpu_cores,
                    "requested_memory_mb": job.inputs.memory_mb,
                    "requested_disk_mb": job.inputs.disk_mb,
                    "requested_gpu_count": job.inputs.gpu_count,
                    "requested_gpu_memory_gb": job.inputs.gpu_memory_gb,
                    "creation_key": "initial:resource_check",
                },
                restrict_tools=resource_tools,
                expected_outputs=["output/result.json"],
            ),
        )

        # EnvironmentBuildAgent 用 find_files 定位依赖声明文件、
        # write_file 写 Dockerfile/锁定依赖（受 _guarded_write_file
        # 白名单二次约束）、execute_command 跑 pip/import 自检；不遍历
        # 目录也不读取任意文件全文。
        env_task = Task(
            job_id=job.job_id,
            definition=build_task_definition(
                objective="构建可复现的运行环境（Dockerfile、依赖锁定、import 自检）",
                task_type="environment_build",
                dependencies=[spec_task.task_id, resource_task.task_id],
                inputs={
                    "repository_path": job.inputs.repository_path,
                    "dependencies_hint": job.inputs.user_environment_notes,
                    "base_image": self.execution_image,
                    "environment_backend": self.environment_backend,
                    "environment_name": managed_environment_name(
                        job.inputs.environment_name, job.inputs.repository_path
                    ),
                    "python_version": self.conda_python_version,
                    "cpu_cores": job.inputs.cpu_cores or 1.0,
                    "memory_mb": job.inputs.memory_mb or 1024,
                    "disk_mb": job.inputs.disk_mb or 4096,
                    "creation_key": "initial:environment_build",
                },
                restrict_tools=[
                    "find_files",
                    "read_file",
                    "write_file",
                    "execute_command",
                    (
                        "build_conda_environment"
                        if self.environment_backend == "conda"
                        else "build_environment_image"
                    ),
                ],
                expected_outputs=["output/result.json"],
                # 环境构建是典型的重型任务：初始报备预估由 task_factory
                # 设为 1800s；这里再为依赖分析、下载和缓存重建保留独立
                # 的软观测窗口与绝对硬上限。
                soft_timeout_seconds=3600,
                hard_timeout_seconds=5400,
            ),
        )

        return [*paper_tasks, code_task, spec_task, resource_task, env_task]

    # ---- 论文分析任务拆分（正文 + 附录并行，用户方案） ----

    def _plan_paper_tasks(self, job: ReproductionJob) -> list[Task]:
        """正文 + 附录双子任务（或探测失败时的单任务退回）。

        拆分决策完全基于纯代码结构探测，不消耗 LLM 调用：
            - 探测到附录页边界 → 正文任务（页 1..k-1）+ 附录任务
              （页 k..N；超长时按页分片）；独立的附录/补充文件归入附录
              任务的输入；
            - 探测不到边界（无附录、非 PDF、探测异常）→ 单任务读全文
              （scope=body，无页范围），行为与拆分前一致。
        """

        paper_path = job.inputs.paper_path
        target_experiments = job.inputs.target_experiments
        appendix_files = [*job.inputs.appendix_paths, *job.inputs.supplementary_paths]

        structure = None
        if paper_path and str(paper_path).lower().endswith(".pdf"):
            try:
                from repro_agent.paper_input import probe_paper_structure

                structure = probe_paper_structure(paper_path)
            except Exception:  # noqa: BLE001 - 探测失败退回单任务全文阅读
                logger.warning(
                    "job %s: paper structure probe failed for %s; falling back "
                    "to a single whole-document paper_analysis task",
                    job.job_id,
                    paper_path,
                    exc_info=True,
                )

        tools = ["read_file", "read_pdf_text", "inspect_pdf_page"]
        expected = ["output/result.json", "output/candidate_memory.md"]

        if structure is None or not structure.has_appendix:
            return [
                Task(
                    job_id=job.job_id,
                    definition=build_task_definition(
                        objective=(
                            "解析论文，提取方法概要、目标实验、训练/模型/数据/评测"
                            "参数与主实验指标"
                        ),
                        task_type="paper_analysis",
                        inputs={
                            "paper_path": paper_path,
                            "target_experiments": target_experiments,
                            "scope": "body",
                            "files": [paper_path, *appendix_files],
                            "creation_key": "initial:paper_analysis:body",
                        },
                        restrict_tools=tools,
                        expected_outputs=expected,
                    ),
                )
            ]

        body_start, body_end = structure.body_pages
        body_task = Task(
            job_id=job.job_id,
            definition=build_task_definition(
                objective=(
                    f"解析论文正文（第 {body_start}-{body_end} 页）：方法概要、"
                    "四类复现参数与主实验指标"
                ),
                task_type="paper_analysis",
                inputs={
                    "paper_path": paper_path,
                    "target_experiments": target_experiments,
                    "scope": "body",
                    "page_range": [body_start, body_end],
                    "files": [paper_path],
                    "creation_key": "initial:paper_analysis:body",
                },
                restrict_tools=tools,
                expected_outputs=expected,
            ),
        )

        tasks = [body_task]
        # 独立附录/补充文件交给第一片附录任务（无独立文件时该列表为空）。
        for part, (chunk_start, chunk_end) in enumerate(
            _split_appendix_pages(
                structure,
                char_limit=APPENDIX_CHUNK_CHAR_LIMIT,
                max_chunks=APPENDIX_MAX_CHUNKS,
            )
        ):
            inputs = {
                "paper_path": paper_path,
                "target_experiments": target_experiments,
                "scope": "appendix",
                "page_range": [chunk_start, chunk_end],
                "files": [paper_path] + (appendix_files if part == 0 else []),
                "creation_key": f"initial:paper_analysis:appendix:{part}",
            }
            tasks.append(
                Task(
                    job_id=job.job_id,
                    definition=build_task_definition(
                        objective=(
                            f"解析论文附录（第 {chunk_start}-{chunk_end} 页）："
                            "仅提取与复现相关的训练参数"
                        ),
                        task_type="paper_analysis",
                        inputs=inputs,
                        restrict_tools=tools,
                        expected_outputs=expected,
                    ),
                )
            )
        return tasks


def _split_appendix_pages(
    structure,
    *,
    char_limit: int,
    max_chunks: int,
) -> list[tuple[int, int]]:
    """把附录页范围切成字符量均衡的页片段（不少于 1 片，不超过 max_chunks）。

    分片按页均分而不是精确按字符——页是 PDF 提取的最小干净边界，
    按字符硬切会把跨页表格拦腰截断。"""

    if structure.appendix_pages is None:
        return []
    start, end = structure.appendix_pages
    total_pages = end - start + 1
    total_chars = max(1, structure.appendix_char_count)
    chunks = min(max_chunks, max(1, -(-total_chars // char_limit)))
    if chunks <= 1:
        return [(start, end)]
    per_chunk = -(-total_pages // chunks)
    ranges = []
    current = start
    while current <= end:
        chunk_end = min(end, current + per_chunk - 1)
        ranges.append((current, chunk_end))
        current = chunk_end + 1
    return ranges
