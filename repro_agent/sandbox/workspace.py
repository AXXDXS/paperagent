"""任务级沙箱工作区（设计文档 §12 沙箱设计）。

目录结构严格对齐设计文档 §12：

    /sandbox/task_<task_id>/
        ├── input/      只读
        ├── workspace/  可读写
        ├── output/     可写
        ├── logs/       执行器记录
        ├── tmp/        临时文件
        └── policy.json 只读

``TaskSandbox`` 实现了 ``repro_agent.tools.base.SandboxContext``
协议，是连接"工具系统"与"物理文件系统"的唯一桥梁——工具的
handler 函数只调用 ``ctx.resolve_readable_path``/
``ctx.resolve_writable_path``，从不直接拼接路径字符串，这样路径
越界校验（``sandbox/paths.py``）就能对所有工具生效，不用在每个
工具里重复写校验代码。

复用来源：
    整体目录结构与权限划分直接来自设计文档 §12 原文；
    "input 只读、workspace 可读写、output 可写"的三段式布局与
    DeerFlow 沙箱系统的 ``/mnt/user-data/{uploads,workspace,outputs}``
    虚拟路径映射（``doc/DeerFlow_架构分析.md`` 第 3.2 节）理念一致，
    这里选择更贴合设计文档原文的目录命名（input/workspace/output）
    而不是照搬 DeerFlow 的命名，因为设计文档已经给出了明确的目录
    结构规范，应当优先遵循用户自己的设计文档。
"""

from __future__ import annotations

import json
import hashlib
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from repro_agent.sandbox.paths import (
    PathEscapeErrorForScheme,
    split_virtual_path,
    to_virtual_path,
    validate_within_roots,
)
from repro_agent.sandbox.policy import SandboxPolicy


@dataclass
class TaskSandbox:
    """单个任务的物理沙箱目录 + 权限边界。实现 ``SandboxContext`` 协议。"""

    task_id: str
    attempt_id: str
    root: Path
    policy: SandboxPolicy
    extra_readable_roots: list[Path] = field(default_factory=list)
    execution_backend: object | None = None
    execution_image: str = "python:3.11-slim"
    cancellation_event: threading.Event | None = None

    def __post_init__(self) -> None:
        self.input_dir = self.root / "input"
        self.workspace_dir = self.root / "workspace"
        self.output_dir = self.root / "output"
        self.logs_dir = self.root / "logs"
        self.tmp_dir = self.root / "tmp"
        for d in (self.input_dir, self.workspace_dir, self.output_dir, self.logs_dir, self.tmp_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._write_policy_file()

    def _write_policy_file(self) -> None:
        policy_path = self.root / "policy.json"
        policy_path.write_text(
            json.dumps(self.policy.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # policy.json 只读：写完后去掉写权限，防止子智能体篡改自己的策略。
        try:
            policy_path.chmod(0o444)
        except OSError:
            pass  # 部分平台/权限环境下 chmod 可能受限，不影响主流程

    # ---- SandboxContext 协议实现 ----
    #
    # 三个 resolve_* 方法是子智能体/工具接触文件系统的唯一入口，也是
    # "虚拟路径映射"（借鉴 DeerFlow ``replace_virtual_path``）落地的
    # 唯一位置：子智能体传入的路径字符串应当是 ``input://``、
    # ``workspace://``、``output://``、``tmp://`` 四种虚拟协议头之一，
    # 这里先把协议头映射到对应的物理子目录，再复用
    # ``validate_within_roots`` 做越界校验兜底。为了不破坏任何历史
    # 调用（早期实现里传的是不带协议头的裸相对路径），未识别出协议头
    # 的输入会回退到旧的"多根猜测"逻辑，但新代码一律应使用带协议头的
    # 虚拟路径。

    def _resolve_scheme(self, path: str, scheme_roots: dict[str, Path], fallback_roots: list[Path]) -> Path:
        parsed = split_virtual_path(path)
        if parsed is not None:
            scheme, rest = parsed
            root = scheme_roots.get(scheme)
            if root is None:
                raise PathEscapeErrorForScheme(scheme, path)
            return validate_within_roots(rest, [root])
        # 未带协议头：回退兼容旧的相对/绝对路径解析方式。
        return validate_within_roots(path, fallback_roots)

    def resolve_readable_path(self, relative_path: str) -> str:
        """input/、workspace/、output/ 以及显式授权的额外只读根目录均可读。

        推荐调用方式是带虚拟协议头（``input://xxx``），此时直接锚定
        到对应子目录，不存在"多根猜测"歧义；未带协议头的输入按
        ``[input_dir, workspace_dir, output_dir, tmp_dir, *extra]``
        顺序兼容解析（历史行为，见 ``validate_within_roots`` 的消歧
        说明）。
        """

        scheme_roots = {
            "input": self.input_dir,
            "workspace": self.workspace_dir,
            "output": self.output_dir,
            "tmp": self.tmp_dir,
        }
        fallback_roots = [self.input_dir, self.workspace_dir, self.output_dir, self.tmp_dir]
        fallback_roots.extend(self.extra_readable_roots)
        resolved = self._resolve_scheme(relative_path, scheme_roots, fallback_roots)
        return str(resolved)

    def resolve_writable_path(self, relative_path: str) -> str:
        """只有 workspace/、output/、tmp/ 可写；input/ 与额外只读根永远不可写。

        与 ``resolve_readable_path`` 一样，推荐使用带协议头的虚拟路径
        （``workspace://xxx``/``output://xxx``/``tmp://xxx``），此时直接
        锚定到对应子目录。``input://`` 协议头在这里会被拒绝（只读区
        不可写）。未带协议头的裸相对路径按
        ``[workspace_dir, output_dir, tmp_dir]`` 顺序兼容解析——这是
        历史行为，容易把 ``"output/x"`` 误判为 workspace 下的路径，
        凡是明确要写任务产物目录的场景应使用 ``resolve_output_path``
        或 ``output://`` 虚拟路径，不要依赖这里的兼容猜测。
        """

        parsed = split_virtual_path(relative_path)
        if parsed is not None and parsed[0] == "input":
            raise PathEscapeErrorForScheme("input", relative_path, reason="input/ is read-only")
        scheme_roots = {
            "workspace": self.workspace_dir,
            "output": self.output_dir,
            "tmp": self.tmp_dir,
        }
        fallback_roots = [self.workspace_dir, self.output_dir, self.tmp_dir]
        resolved = self._resolve_scheme(relative_path, scheme_roots, fallback_roots)
        return str(resolved)

    def resolve_output_path(self, relative_path: str) -> str:
        """显式锚定到 ``output/`` 目录，不做多根猜测（供
        ``write_task_output`` 等"明确写入任务产物目录"的工具使用）。

        接受裸文件名、``output://xxx`` 虚拟路径两种写法，语义等价。
        """

        parsed = split_virtual_path(relative_path)
        rest = parsed[1] if parsed is not None and parsed[0] == "output" else relative_path
        resolved = validate_within_roots(rest, [self.output_dir])
        return str(resolved)

    def network_allowed(self) -> bool:
        return self.policy.allow_network

    # ---- 输入文件预置 ----

    def stage_input_file(self, source_path: str, dest_relative_path: str | None = None) -> Path:
        """把宿主机上的输入文件拷贝进沙箱 ``input/``（只读区）。

        之所以要求"拷贝"而不是"挂载/软链接"到原路径，是为了保证
        §12 "子智能体禁止访问宿主机文件"这一条即便在文件系统层面
        实现有疏漏时，也有一层物理隔离兜底（拷贝后子智能体只能看到
        沙箱内的副本，即使某个工具实现有 bug 允许读取符号链接目标，
        目标也只是沙箱内文件而非宿主机原始路径）。

        返回值是宿主机绝对路径（供 ``SandboxManager`` 计算相对路径
        并转换为虚拟路径写回 ``task.definition.inputs``），调用方不应
        把这个返回值直接交给子智能体——子智能体应该只看到
        ``stage_input_file_as_virtual_path`` 返回的虚拟路径形式。
        """

        src = Path(source_path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"input source not found: {source_path}")
        if dest_relative_path:
            dest_name = dest_relative_path
        else:
            # Never use only ``src.name``: two repositories/data directories
            # commonly share names such as ``repository`` or ``data``.
            digest = hashlib.sha256(str(src).encode("utf-8")).hexdigest()[:16]
            basename = re.sub(r"[^A-Za-z0-9_.-]", "-", src.name or "resource")
            dest_name = f"_staged/{digest}-{basename}"
        dest = self.input_dir / dest_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            import shutil

            # Preserve repository symlinks instead of following them while
            # staging.  Following a malicious in-tree link could copy arbitrary
            # host files into the sandbox before path validation ever runs.
            shutil.copytree(
                src,
                dest,
                dirs_exist_ok=True,
                symlinks=True,
                ignore_dangling_symlinks=True,
            )
        else:
            import shutil

            shutil.copy2(src, dest)
        return dest

    def stage_input_file_as_virtual_path(
        self, source_path: str, dest_relative_path: str | None = None
    ) -> str:
        """``stage_input_file`` 的虚拟路径版本：拷贝文件后直接返回
        ``input://xxx`` 形式的虚拟路径，供 orchestrator 写回任务输入，
        这是子智能体应该看到的唯一形式（不暴露宿主机绝对路径）。
        """

        dest = self.stage_input_file(source_path, dest_relative_path)
        return to_virtual_path("input", str(dest.relative_to(self.input_dir)))

    def collect_outputs(self) -> dict[str, str]:
        """收集 ``output/`` 目录下所有产物文件的相对路径 -> 绝对路径映射。

        键仍然是不带协议头的相对路径（供
        ``OutputValidator``/``FinalReportGenerator`` 等主智能体侧代码
        直接与 ``expected_outputs``（如 ``"output/result.json"``）做
        字符串匹配，这些校验逻辑运行在主智能体一侧，不受"子智能体只能
        看到虚拟路径"约束的限制）。如需要虚拟路径形式，使用
        ``to_virtual_path("output", rel)`` 自行包装。
        """

        result = {}
        if not self.output_dir.exists():
            return result
        for path in self.output_dir.rglob("*"):
            # Container commands may create symlinks.  Never let a symlinked
            # artifact make host-side validators or evidence hashing follow a
            # target outside the output tree.
            if path.is_file() and not path.is_symlink():
                result[str(path.relative_to(self.output_dir))] = str(path)
        return result
