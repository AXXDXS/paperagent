"""沙箱策略定义（设计文档 §12 沙箱设计）。

每个任务的沙箱资源上限 + 网络策略 + 目录权限，序列化后写入
``policy.json``（§12 目录结构中要求存在的只读策略文件），
既作为运行时校验依据，也作为事后审计的证据之一。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SandboxResourceLimits:
    """资源限制（§12：CPU/内存/磁盘/GPU/进程数/文件数/最大日志/工具调用）。"""

    cpu_cores: float | None = 1.0
    memory_mb: int | None = 1024
    disk_mb: int | None = 4096
    gpu_count: int = 0
    # Docker/NVIDIA does not expose a portable per-container VRAM hard cap.
    # This value is therefore an audited admission requirement: the resource
    # check must prove that every requested GPU has at least this much memory
    # before dispatch.  It is still carried through the execution policy so a
    # run cannot silently forget what the user approved.
    gpu_memory_mb: int | None = None
    max_processes: int = 32
    max_open_files: int = 256
    max_log_bytes: int = 20_000_000
    max_tool_calls: int = 500

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "disk_mb": self.disk_mb,
            "gpu_count": self.gpu_count,
            "gpu_memory_mb": self.gpu_memory_mb,
            "max_processes": self.max_processes,
            "max_open_files": self.max_open_files,
            "max_log_bytes": self.max_log_bytes,
            "max_tool_calls": self.max_tool_calls,
        }


@dataclass
class SandboxPolicy:
    """单个任务沙箱的完整策略（落盘为 ``policy.json``，只读）。"""

    task_id: str
    allow_network: bool = False
    readable_extra_roots: list[str] = field(default_factory=list)
    resource_limits: SandboxResourceLimits = field(default_factory=SandboxResourceLimits)
    soft_timeout_seconds: int = 600
    hard_timeout_seconds: int = 1200
    attempt_number: int = 0
    approved_destructive_command_fingerprints: list[str] = field(default_factory=list)

    # 设计文档 §12 明确列出的禁止访问项，这里作为策略的静态声明，
    # 实际拦截逻辑分布在 workspace.py（路径越界）和
    # write_tools.execute_command（网络策略）中。
    forbidden_resources: list[str] = field(
        default_factory=lambda: [
            "global_memory",
            "other_task_directories",
            "host_filesystem",
            "ssh_key",
            "api_key",
            "git_credentials",
            "database",
            "docker_socket",
            "internal_network",
            "unauthorized_external_network",
        ]
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "allow_network": self.allow_network,
            "readable_extra_roots": self.readable_extra_roots,
            "resource_limits": self.resource_limits.to_dict(),
            "soft_timeout_seconds": self.soft_timeout_seconds,
            "hard_timeout_seconds": self.hard_timeout_seconds,
            "attempt_number": self.attempt_number,
            "approved_destructive_command_fingerprints": list(
                self.approved_destructive_command_fingerprints
            ),
            "forbidden_resources": self.forbidden_resources,
        }

    def destructive_command_is_approved(self, fingerprint: str) -> bool:
        return fingerprint in self.approved_destructive_command_fingerprints
