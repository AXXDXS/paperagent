"""任务工厂：为不同任务类型生成带有正确 ``allowed_tools`` 的任务定义。

这是"主智能体按任务下发受限工具集"这一需求在**任务创建阶段**的体现
（对应"工具授权"需求的第一步，第二步是运行时的 ``ToolAuthorizer``，
见 orchestrator/dispatcher.py）：主智能体在规划阶段就应该为每个任务
声明清晰、最小化的 ``allowed_tools``，而不是所有任务都给同一份大而
全的工具列表。这里为每种任务类型提供一份"标准工具模板"，
体现设计文档 §9 对每类子智能体职责边界的要求（例如实验执行子智能体
不能拿到 write_file）。

主智能体在生成具体任务时应当优先使用这些模板，只有在有明确理由时才
临时增减（例如某个论文分析任务额外需要访问用户上传的补充材料目录，
可以在 ``inputs.files`` 里声明，但 allowed_tools 通常不需要变化）。
"""

from __future__ import annotations

import logging

from repro_agent.domain.task import TaskDefinition

logger = logging.getLogger(__name__)

# 每种任务类型的标准工具模板（与 tools/authorization.py 的
# TASK_TYPE_RISK_BUDGET 一一对应，风险等级不会超过该任务类型的预算）。
STANDARD_TOOL_TEMPLATES: dict[str, list[str]] = {
    "paper_analysis": [
        "list_directory",
        "find_files",
        "grep_files",
        "read_file",
        "read_pdf_text",
        "inspect_pdf_page",
        "get_file_stat",
        "hash_path",
        "write_task_output",
    ],
    "code_analysis": [
        "list_directory",
        "find_files",
        "grep_files",
        "read_file",
        "get_repository_map",
        "search_repository_code",
        "get_file_stat",
        "hash_path",
        "write_task_output",
    ],
    "resource_check": [
        "find_named_resource",
        "check_path_resource",
        "check_disk_space",
        "check_gpu",
        "check_cuda",
        "get_file_stat",
        "hash_path",
        "write_task_output",
    ],
    "specification": [
        "read_file",
        "write_task_output",
    ],
    "environment_build": [
        "list_directory",
        "find_files",
        "read_file",
        "write_file",
        "write_task_output",
        "execute_command",
        "build_environment_image",
        "build_conda_environment",
    ],
    "coding": [
        "list_directory",
        "find_files",
        "grep_files",
        "read_file",
        "hash_path",
        "write_file",
        "write_task_output",
        "execute_command",
    ],
    "experiment_execution": [
        # 刻意不包含任何写文件/修改代码的工具（§9.7 硬约束）
        "read_file",
        "get_file_stat",
        "hash_path",
        "write_task_output",
        "execute_command",
    ],
    "verification": [
        "read_file",
        "get_file_stat",
        "hash_path",
        "write_task_output",
    ],
    "reflection": [
        "read_file",
        "write_task_output",
    ],
}

# Initial report promises are role-aware.  Heavy environment/execution tasks
# no longer inherit the same five-minute expectation as lightweight analysis.
DEFAULT_EXPECTED_DURATION_SECONDS: dict[str, int] = {
    "paper_analysis": 600,
    "code_analysis": 900,
    "resource_check": 300,
    "specification": 300,
    "environment_build": 1800,
    "coding": 900,
    "experiment_execution": 3600,
    "verification": 600,
    "reflection": 600,
}


_ALWAYS_KEEP_TOOLS = {"write_task_output"}
"""即使调用方传入了很窄的 ``restrict_tools``，也永远保留的工具。

``write_task_output`` 是所有任务类型写自己 ``result.json``/
``candidate_memory.md`` 的唯一途径（§15.2），如果因为调用方一时疏忽
漏写了它导致任务连结果都写不出去，会产生一种很隐蔽、很难排查的失败
（子智能体跑完全部业务逻辑后，最后一步才因为权限不足而失败）。这里
用一条显式规则兜底，而不是要求每一处调用 ``build_task_definition``
的地方都要记得手写这一个工具名。
"""


def build_task_definition(
    *,
    objective: str,
    task_type: str,
    dependencies: list[str] | None = None,
    inputs: dict | None = None,
    extra_allowed_tools: list[str] | None = None,
    restrict_tools: list[str] | None = None,
    expected_outputs: list[str] | None = None,
    completion_criteria: list[str] | None = None,
    **kwargs,
) -> TaskDefinition:
    """按任务类型套用标准工具模板创建任务定义，并支持按具体任务实例收窄。

    对应用户需求"主智能体只传入它认为分给子智能体的任务需要的工具，
    而不是让子智能体自己去判断需要哪些工具"：``STANDARD_TOOL_TEMPLATES``
    仍然是每种 ``task_type`` 的"能力上限"（一种任务类型理论上可能用到
    的全部工具，用于对齐 ``TASK_TYPE_RISK_BUDGET`` 的风险预算），但主
    智能体在为**某一个具体任务实例**规划 ``allowed_tools`` 时，应当
    结合这个任务的 ``objective``/``inputs`` 判断它实际用得到哪些工具，
    通过 ``restrict_tools`` 传入这份精确子集——最终授权范围是
    "类型模板 ∩ restrict_tools"，而不是类型模板全集。

    两个参数的关系（收窄 vs. 追加，不能互相替代）：
        - ``restrict_tools``：在模板范围内做**交集收窄**，只能让权限
          变得更小，不能用它加出模板之外的工具（写错/传超集时会被
          静默忽略并记录警告，而不是意外提权）；
        - ``extra_allowed_tools``：在模板基础上做**追加**，用于模板
          没有覆盖、但确有正当理由的极少数场景；是否真的放行仍然由
          运行时 ``ToolAuthorizer`` 的风险预算校验兜底。

    不传 ``restrict_tools``（``None``）时保持旧行为——使用完整类型
    模板，适用于确实每个工具都可能用到、或者调用方还没来得及做精细化
    梳理的任务类型（比如 ``coding``/``environment_build`` 这类多阶段
    任务，往往每个模板工具在某个阶段都会被用到）。
    """

    template = STANDARD_TOOL_TEMPLATES.get(task_type, ["write_task_output"])

    if restrict_tools is not None:
        requested = set(restrict_tools) | _ALWAYS_KEEP_TOOLS
        ignored = requested - set(template)
        if ignored:
            logger.warning(
                "build_task_definition(task_type=%s): restrict_tools 中的 %s "
                "不在标准工具模板 %s 内，已忽略（restrict_tools 只能收窄，"
                "不能扩权；如确需新增工具请使用 extra_allowed_tools）",
                task_type,
                sorted(ignored),
                template,
            )
        base_tools = [name for name in template if name in requested]
    else:
        base_tools = template

    allowed_tools = list(dict.fromkeys(base_tools + (extra_allowed_tools or [])))
    kwargs.setdefault(
        "expected_duration_seconds",
        DEFAULT_EXPECTED_DURATION_SECONDS.get(task_type, 300),
    )

    return TaskDefinition(
        objective=objective,
        task_type=task_type,
        dependencies=dependencies or [],
        inputs=inputs or {},
        allowed_tools=allowed_tools,
        expected_outputs=expected_outputs or ["output/result.json"],
        completion_criteria=completion_criteria or [],
        **kwargs,
    )
