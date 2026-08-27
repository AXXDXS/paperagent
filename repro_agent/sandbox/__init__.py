"""任务级沙箱系统（设计文档 §12）。"""

from repro_agent.sandbox.manager import SandboxManager
from repro_agent.sandbox.paths import PathEscapeError, validate_within_roots
from repro_agent.sandbox.policy import SandboxPolicy, SandboxResourceLimits
from repro_agent.sandbox.workspace import TaskSandbox

__all__ = [
    "PathEscapeError",
    "SandboxManager",
    "SandboxPolicy",
    "SandboxResourceLimits",
    "TaskSandbox",
    "validate_within_roots",
]
