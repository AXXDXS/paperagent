"""资源检查类只读工具（设计文档 §9.3 资源检查子智能体）。

覆盖：数据/模型/checkpoint 存在性检查、GPU/显存/CUDA/驱动检查、
磁盘空间检查。全部是只读探测，不修改任何系统状态，因此风险等级
统一为 ``READ_ONLY``——即便如此依然要求通过 ``SandboxContext``
访问文件系统，保证"资源检查子智能体"也遵守"子智能体不能访问其他
任务目录/宿主机任意路径"的沙箱约束（§3 原则 8-11），只是它的
可读范围通常会被主智能体配置为用户提供的 dataset/model 路径
（在任务 ``inputs.files`` 中显式声明，而不是隐式访问整个宿主机）。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from repro_agent.domain.enums import ResourceStatus
from repro_agent.tools.base import (
    SandboxContext,
    ToolExample,
    ToolOutputSpec,
    ToolParamDoc,
    ToolRiskLevel,
    ToolSpec,
)


def check_path_resource(ctx: SandboxContext, path: str, kind: str = "data") -> dict[str, Any]:
    """检查数据集/模型/checkpoint 路径是否存在及基本形态（§9.3）。

    返回结果里的 ``status`` 直接采用设计文档 §9.3 定义的
    ``ResourceStatus`` 枚举值，供上层资源检查子智能体直接引用，
    不需要再做一次字符串到枚举的映射。
    """

    try:
        resolved = Path(ctx.resolve_readable_path(path))
    except Exception:
        return {"path": path, "kind": kind, "status": ResourceStatus.MISSING.value}

    if not resolved.exists():
        return {"path": path, "kind": kind, "status": ResourceStatus.MISSING.value}

    if resolved.is_dir():
        file_count = sum(1 for _ in resolved.rglob("*") if _.is_file())
        total_size = sum(f.stat().st_size for f in resolved.rglob("*") if f.is_file())
        status = (
            ResourceStatus.AVAILABLE_BUT_UNVERIFIED
            if file_count > 0
            else ResourceStatus.MISSING
        )
        return {
            "path": path,
            "kind": kind,
            "status": status.value,
            "is_dir": True,
            "file_count": file_count,
            "total_size_bytes": total_size,
        }

    return {
        "path": path,
        "kind": kind,
        "status": ResourceStatus.AVAILABLE_BUT_UNVERIFIED.value,
        "is_dir": False,
        "size_bytes": resolved.stat().st_size,
    }


def find_named_resource(
    ctx: SandboxContext,
    root: str,
    name: str,
    kind: str = "dataset",
    *,
    max_results: int = 20,
    max_entries: int = 20_000,
) -> dict[str, Any]:
    """Find a paper/spec-declared resource inside one explicitly staged root.

    Matching is case- and punctuation-insensitive, so ``LoCoMo``, ``locomo``
    and ``lo-como`` identify the same directory.  The traversal is bounded and
    never follows symlinks; this is a discovery probe, not content validation.
    """

    try:
        resolved_root = Path(ctx.resolve_readable_path(root))
    except Exception:
        return {
            "root": root,
            "name": name,
            "kind": kind,
            "status": ResourceStatus.MISSING.value,
            "candidates": [],
            "scanned_entries": 0,
            "truncated": False,
        }
    if not resolved_root.is_dir():
        return {
            "root": root,
            "name": name,
            "kind": kind,
            "status": ResourceStatus.MISSING.value,
            "candidates": [],
            "scanned_entries": 0,
            "truncated": False,
        }

    target = _resource_name_key(name)
    if not target:
        return {
            "root": root,
            "name": name,
            "kind": kind,
            "status": ResourceStatus.MISSING.value,
            "candidates": [],
            "scanned_entries": 0,
            "truncated": False,
        }

    result_limit = min(max(1, int(max_results)), 50)
    entry_limit = min(max(1, int(max_entries)), 100_000)
    candidates: list[dict[str, Any]] = []
    scanned = 0
    truncated = False

    for current, dirs, files in os.walk(resolved_root, followlinks=False):
        dirs[:] = sorted(
            item
            for item in dirs
            if not item.startswith(".") and not (Path(current) / item).is_symlink()
        )
        entries = [(item, True) for item in dirs]
        entries.extend((item, False) for item in sorted(files))
        for item_name, is_dir in entries:
            scanned += 1
            if scanned > entry_limit:
                truncated = True
                break
            candidate = Path(current) / item_name
            if candidate.is_symlink() or not _resource_name_matches(target, item_name):
                continue
            if is_dir:
                try:
                    non_empty = any(
                        child.is_file() and not child.is_symlink()
                        for child in candidate.rglob("*")
                    )
                except OSError:
                    non_empty = False
                size_bytes = 0
            else:
                try:
                    size_bytes = candidate.stat().st_size
                except OSError:
                    size_bytes = 0
                non_empty = size_bytes > 0
            candidates.append(
                {
                    "relative_path": str(candidate.relative_to(resolved_root)),
                    "is_dir": is_dir,
                    "non_empty": non_empty,
                    "size_bytes": size_bytes,
                }
            )
            if len(candidates) >= result_limit:
                truncated = True
                break
        if truncated:
            break

    available = any(item["non_empty"] for item in candidates)
    return {
        "root": root,
        "name": name,
        "kind": kind,
        "status": (
            ResourceStatus.AVAILABLE_BUT_UNVERIFIED.value
            if available
            else ResourceStatus.MISSING.value
        ),
        "candidates": candidates,
        "scanned_entries": min(scanned, entry_limit),
        "truncated": truncated,
    }


def _resource_name_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _resource_name_matches(target: str, candidate_name: str) -> bool:
    candidate = _resource_name_key(candidate_name)
    if not candidate:
        return False
    return target == candidate or target in candidate or (
        len(candidate) >= 4 and candidate in target
    )


def check_disk_space(ctx: SandboxContext, path: str = ".") -> dict[str, Any]:
    """检查磁盘剩余空间（§9.3 检查磁盘）。"""

    resolved = Path(ctx.resolve_readable_path(path))
    total, used, free = shutil.disk_usage(resolved if resolved.exists() else Path.cwd())
    return {
        "path": path,
        "total_bytes": total,
        "used_bytes": used,
        "free_bytes": free,
        "free_gb": round(free / (1024**3), 2),
    }


def _run_probe_command(args: list[str]) -> tuple[bool, str]:
    """运行一次只读的系统探测命令（如 nvidia-smi/nvcc），失败不抛异常，
    因为"探测不到 GPU"本身就是一种合法的结果（§9.3 ResourceStatus.UNKNOWN），
    不应该让整个资源检查任务因为没有 GPU 环境而崩溃退出。
    """

    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=15
        )
        return result.returncode == 0, (result.stdout or result.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)


def check_gpu(ctx: SandboxContext | None = None) -> dict[str, Any]:
    """检查 GPU/显存/驱动情况（§9.3 检查 GPU、显存、驱动）。

    注意：GPU 探测是宿主机级别的只读命令，不依赖沙箱路径，
    因此 ``ctx`` 参数仅用于保持工具签名的一致性（便于统一的
    ``ToolAuthorization`` 调用约定），可以为 ``None``。
    """

    ok, output = _run_probe_command(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"]
    )
    if not ok:
        return {"available": False, "detail": output or "nvidia-smi not found"}

    gpus = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            memory_total_mb = _parse_memory_mib(parts[1])
            memory_used_mb = _parse_memory_mib(parts[2])
            gpus.append(
                {
                    "name": parts[0],
                    "memory_total": parts[1],
                    "memory_used": parts[2],
                    "memory_total_mb": memory_total_mb,
                    "memory_used_mb": memory_used_mb,
                    "driver_version": parts[3],
                }
            )
    return {"available": True, "gpu_count": len(gpus), "gpus": gpus}


def _parse_memory_mib(value: str) -> int:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(value))
    return int(float(match.group(1))) if match else 0


def check_cuda(ctx: SandboxContext | None = None) -> dict[str, Any]:
    """检查 CUDA 版本（§9.3 检查 CUDA）。"""

    ok, output = _run_probe_command(["nvcc", "--version"])
    if not ok:
        return {"available": False, "detail": output or "nvcc not found"}
    return {"available": True, "raw_output": output}


def _strict_object(
    required: list[str], properties: dict[str, Any]
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


_PATH_RESOURCE_OUTPUT_SCHEMA = {
    "oneOf": [
        _strict_object(
            ["path", "kind", "status"],
            {
                "path": {"type": "string"},
                "kind": {"type": "string"},
                "status": {"const": ResourceStatus.MISSING.value},
            },
        ),
        _strict_object(
            ["path", "kind", "status", "is_dir", "file_count", "total_size_bytes"],
            {
                "path": {"type": "string"},
                "kind": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": [
                        ResourceStatus.MISSING.value,
                        ResourceStatus.AVAILABLE_BUT_UNVERIFIED.value,
                    ],
                },
                "is_dir": {"const": True},
                "file_count": {"type": "integer", "minimum": 0},
                "total_size_bytes": {"type": "integer", "minimum": 0},
            },
        ),
        _strict_object(
            ["path", "kind", "status", "is_dir", "size_bytes"],
            {
                "path": {"type": "string"},
                "kind": {"type": "string"},
                "status": {"const": ResourceStatus.AVAILABLE_BUT_UNVERIFIED.value},
                "is_dir": {"const": False},
                "size_bytes": {"type": "integer", "minimum": 0},
            },
        ),
    ]
}
_NAMED_RESOURCE_CANDIDATE_SCHEMA = _strict_object(
    ["relative_path", "is_dir", "non_empty", "size_bytes"],
    {
        "relative_path": {"type": "string"},
        "is_dir": {"type": "boolean"},
        "non_empty": {"type": "boolean"},
        "size_bytes": {"type": "integer", "minimum": 0},
    },
)
_NAMED_RESOURCE_OUTPUT_SCHEMA = _strict_object(
    [
        "root",
        "name",
        "kind",
        "status",
        "candidates",
        "scanned_entries",
        "truncated",
    ],
    {
        "root": {"type": "string"},
        "name": {"type": "string"},
        "kind": {"type": "string"},
        "status": {
            "type": "string",
            "enum": [
                ResourceStatus.MISSING.value,
                ResourceStatus.AVAILABLE_BUT_UNVERIFIED.value,
            ],
        },
        "candidates": {
            "type": "array",
            "items": _NAMED_RESOURCE_CANDIDATE_SCHEMA,
            "maxItems": 50,
        },
        "scanned_entries": {"type": "integer", "minimum": 0},
        "truncated": {"type": "boolean"},
    },
)
_DISK_OUTPUT_SCHEMA = _strict_object(
    ["path", "total_bytes", "used_bytes", "free_bytes", "free_gb"],
    {
        "path": {"type": "string"},
        "total_bytes": {"type": "integer", "minimum": 0},
        "used_bytes": {"type": "integer", "minimum": 0},
        "free_bytes": {"type": "integer", "minimum": 0},
        "free_gb": {"type": "number", "minimum": 0},
    },
)
_PROBE_UNAVAILABLE_SCHEMA = _strict_object(
    ["available", "detail"],
    {"available": {"const": False}, "detail": {"type": "string"}},
)
_GPU_ITEM_SCHEMA = _strict_object(
    [
        "name",
        "memory_total",
        "memory_used",
        "memory_total_mb",
        "memory_used_mb",
        "driver_version",
    ],
    {
        "name": {"type": "string"},
        "memory_total": {"type": "string"},
        "memory_used": {"type": "string"},
        "memory_total_mb": {"type": "integer", "minimum": 0},
        "memory_used_mb": {"type": "integer", "minimum": 0},
        "driver_version": {"type": "string"},
    },
)
_GPU_OUTPUT_SCHEMA = {
    "oneOf": [
        _PROBE_UNAVAILABLE_SCHEMA,
        _strict_object(
            ["available", "gpu_count", "gpus"],
            {
                "available": {"const": True},
                "gpu_count": {"type": "integer", "minimum": 0},
                "gpus": {"type": "array", "items": _GPU_ITEM_SCHEMA},
            },
        ),
    ]
}
_CUDA_OUTPUT_SCHEMA = {
    "oneOf": [
        _PROBE_UNAVAILABLE_SCHEMA,
        _strict_object(
            ["available", "raw_output"],
            {"available": {"const": True}, "raw_output": {"type": "string"}},
        ),
    ]
}


TOOL_SPECS = [
    ToolSpec(
        name="find_named_resource",
        description="在已显式装载的仓库目录中按资源名称查找数据集、模型或 checkpoint 候选。",
        risk_level=ToolRiskLevel.READ_ONLY,
        handler=find_named_resource,
        output=ToolOutputSpec(schema=_NAMED_RESOURCE_OUTPUT_SCHEMA),
        suggested_task_types=("resource_check",),
        when_to_use=(
            "实验规格声明了必需资源但用户没有显式提供路径时，在已装载的"
            "代码仓库内查找同名目录或文件；例如查找 LoCoMo/locomo 数据集。"
        ),
        boundaries=(
            "只在 root 指定的沙箱可读目录内搜索，不访问宿主机其他目录。",
            "只按名称和非空性发现候选，不验证数据格式、版本或 split。",
            "搜索有条目数上限，truncated=True 时代表结果可能不完整。",
        ),
        returns=(
            "{root, name, kind, status, candidates: "
            "[{relative_path, is_dir, non_empty, size_bytes}], scanned_entries, truncated}"
        ),
        cost_hint="只读有界目录遍历；默认最多检查 2 万个目录项。",
        examples=(
            ToolExample(
                when="论文规格要求 LoCoMo 数据集且用户未填写 dataset_paths",
                arguments={
                    "root": "input://repository",
                    "name": "LoCoMo",
                    "kind": "dataset",
                },
                result={
                    "root": "input://repository",
                    "name": "LoCoMo",
                    "kind": "dataset",
                    "status": "AVAILABLE_BUT_UNVERIFIED",
                    "candidates": [
                        {
                            "relative_path": "data/locomo",
                            "is_dir": True,
                            "non_empty": True,
                            "size_bytes": 0,
                        }
                    ],
                    "scanned_entries": 12,
                    "truncated": False,
                },
            ),
        ),
        param_docs={
            "root": ToolParamDoc(
                description="已由主智能体装载进当前沙箱的搜索根目录。",
                example="input://repository",
            ),
            "name": ToolParamDoc(
                description="实验规格声明的资源名称；匹配时忽略大小写和标点。",
                example="LoCoMo",
            ),
            "kind": ToolParamDoc(
                description="dataset/model/checkpoint 分类标签。",
                example="dataset",
            ),
        },
    ),
    ToolSpec(
        name="check_path_resource",
        description="检查数据集/模型/checkpoint 路径是否存在，并给出粗略的可用性状态。",
        risk_level=ToolRiskLevel.READ_ONLY,
        handler=check_path_resource,
        output=ToolOutputSpec(schema=_PATH_RESOURCE_OUTPUT_SCHEMA),
        suggested_task_types=("resource_check",),
        when_to_use=(
            "在资源检查阶段，需要判断论文所需的数据集/预训练模型/checkpoint"
            "是否已经就绪时使用；例如确认 inputs.dataset_paths 里声明的路径"
            "是不是真的存在、里面有没有文件。"
        ),
        boundaries=(
            "只检查路径是否存在、是文件还是目录、文件数量与总大小，不校验"
            "内容格式是否正确（例如不会验证数据集内部字段是否与代码期望的"
            "schema 匹配）——格式校验属于更细致的人工/后续任务范畴。",
            "status 字段最乐观也只会给到 AVAILABLE_BUT_UNVERIFIED（存在但"
            "未验证内容），不会给出 AVAILABLE（完全确认可用）；不要把"
            "'路径存在' 误判为 '数据一定完整可用'。",
            "找不到路径或路径解析失败时返回 status=MISSING，不会抛异常。",
        ),
        returns=(
            "{path, kind, status: ResourceStatus 枚举值; 若为目录还会带 "
            "is_dir=True, file_count, total_size_bytes；若为文件则带 "
            "is_dir=False, size_bytes}"
        ),
        cost_hint="目录会递归统计文件数与总大小，对超大数据集目录（数十万文件）可能耗时数秒。",
        examples=(
            ToolExample(
                when="确认预训练模型 checkpoint 目录是否已下载",
                arguments={"path": "input/checkpoints/resnet50", "kind": "model"},
                result={"path": "input/checkpoints/resnet50", "kind": "model", "status": "AVAILABLE_BUT_UNVERIFIED", "is_dir": True, "file_count": 3, "total_size_bytes": 102400000},
            ),
        ),
        param_docs={
            "path": ToolParamDoc(
                description="要检查的数据集/模型/checkpoint 路径，相对沙箱可读根目录，例如 'input/dataset' 或 'input/checkpoints/model.pt'。",
                example="input/dataset",
            ),
            "kind": ToolParamDoc(
                description="资源种类标签，仅用于在返回结果里标注、便于下游归类，例如 'data'、'model'、'checkpoint'；默认 'data'。",
                example="model",
            ),
        },
    ),
    ToolSpec(
        name="check_disk_space",
        description="检查指定路径所在磁盘分区的剩余空间。",
        risk_level=ToolRiskLevel.READ_ONLY,
        handler=check_disk_space,
        output=ToolOutputSpec(schema=_DISK_OUTPUT_SCHEMA),
        suggested_task_types=("resource_check", "environment_build"),
        when_to_use=(
            "在准备下载数据集/构建镜像/运行实验前，需要确认磁盘空间是否够用"
            "时使用；例如判断剩余空间是否足以容纳一份 50GB 的数据集。"
        ),
        boundaries=(
            "返回的是路径所在【分区】的整体剩余空间，不是该路径已用了多少"
            "空间——如果目标是查看某个目录本身占用了多大空间，应使用"
            "check_path_resource 或 get_file_stat。",
            "路径不存在时会退化为检查当前工作目录所在分区，不会报错，但"
            "结果可能不代表你真正关心的那个分区。",
        ),
        returns="{path, total_bytes, used_bytes, free_bytes, free_gb（保留两位小数，便于直接展示）}",
        cost_hint="一次系统调用，开销极小。",
        examples=(
            ToolExample(
                when="下载数据集前确认磁盘是否至少还有 50GB 剩余",
                arguments={"path": "."},
                result={"path": ".", "total_bytes": 500000000000, "used_bytes": 300000000000, "free_bytes": 200000000000, "free_gb": 186.26},
            ),
        ),
        param_docs={
            "path": ToolParamDoc(
                description="要检查所在分区剩余空间的路径，默认 '.'（沙箱工作目录所在分区）。",
                example=".",
            ),
        },
    ),
    ToolSpec(
        name="check_gpu",
        description="通过 nvidia-smi 探测宿主机 GPU 数量、显存占用与驱动版本。",
        risk_level=ToolRiskLevel.READ_ONLY,
        handler=check_gpu,
        output=ToolOutputSpec(schema=_GPU_OUTPUT_SCHEMA),
        suggested_task_types=("resource_check", "environment_build"),
        when_to_use=(
            "需要判断当前环境是否有可用 GPU、显存是否够跑目标实验时使用；"
            "环境构建/资源检查阶段的标准第一步探测。"
        ),
        boundaries=(
            "依赖宿主机是否安装了 nvidia-smi；探测不到 GPU（available=False）"
            "是完全合法的结果（可能是纯 CPU 环境），不代表工具调用失败，"
            "不需要重试。",
            "只反映调用时刻的静态快照（当前显存占用），不会持续监控；如果"
            "要在实验运行过程中追踪显存变化，需要多次调用而不是期望一次"
            "调用给出趋势。",
            "不会返回 CUDA 版本，只有驱动版本；CUDA 版本需要单独调用"
            "check_cuda。",
        ),
        returns=(
            "available=False 时：{available: False, detail: 探测失败原因}；"
            "available=True 时：{available: True, gpu_count, gpus: "
            "[{name, memory_total, memory_used, driver_version}]}"
        ),
        cost_hint="子进程调用 nvidia-smi，超时上限 15 秒，通常在 1 秒内返回。",
        examples=(
            ToolExample(
                when="确认环境是否有 GPU 以及显存是否足够",
                arguments={},
                result={"available": True, "gpu_count": 1, "gpus": [{"name": "A100", "memory_total": "40960 MiB", "memory_used": "1024 MiB", "driver_version": "535.104"}]},
            ),
        ),
    ),
    ToolSpec(
        name="check_cuda",
        description="通过 nvcc --version 探测宿主机 CUDA 编译器版本。",
        risk_level=ToolRiskLevel.READ_ONLY,
        handler=check_cuda,
        output=ToolOutputSpec(schema=_CUDA_OUTPUT_SCHEMA),
        suggested_task_types=("resource_check", "environment_build"),
        when_to_use=(
            "需要确认当前环境的 CUDA 版本是否满足论文/代码仓库要求的版本"
            "约束时使用（例如某些框架要求 CUDA >= 11.8）。"
        ),
        boundaries=(
            "只探测 nvcc（CUDA 编译工具链）版本，不代表 PyTorch/TensorFlow"
            "实际链接的运行时 CUDA 版本——两者可能不一致，如果需要确认"
            "深度学习框架实际使用的 CUDA 版本，应改用 execute_command 运行"
            "框架自带的版本查询接口（如 python -c 'import torch; print(torch.version.cuda)'）。",
            "找不到 nvcc（available=False）是合法结果，不代表环境没有 GPU——"
            "有些环境只装了驱动没装完整 CUDA Toolkit。",
        ),
        returns="available=False 时：{available: False, detail}；available=True 时：{available: True, raw_output: nvcc 原始输出}",
        cost_hint="子进程调用，超时上限 15 秒，通常在 1 秒内返回。",
        examples=(
            ToolExample(
                when="确认 CUDA 版本是否满足代码仓库 README 里声明的最低要求",
                arguments={},
            ),
        ),
    ),
]
