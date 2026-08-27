"""沙箱路径越界校验 + 虚拟路径映射。

复用来源：
    校验逻辑（"拼接后 resolve 成绝对路径，再检查是否仍以允许的根
    目录为前缀"）直接复用自 DeepCode
    ``tools/code_implementation_server.py::validate_path``（见该文件
    88-96 行）。这是一个被反复验证过的简单有效模式：只要严格使用
    ``Path.resolve()``（会处理 ``..``、符号链接展开为真实路径）
    再做前缀比较，就能可靠地阻止路径穿越攻击（例如
    ``../../etc/passwd`` 或恶意符号链接指向沙箱外）。

    虚拟路径前缀（``input://``、``workspace://``、``output://``）的设计
    直接借鉴 DeerFlow 的虚拟路径映射机制
    （``deerflow/sandbox/tools.py::replace_virtual_path``，见
    ``doc/DeerFlow_架构分析.md``）：DeerFlow 让 Agent 始终只看到固定的
    虚拟前缀 ``/mnt/user-data/{workspace,uploads,outputs}``，真正的宿主机
    路径只在工具执行层通过 ``replace_virtual_path`` 换算并二次校验
    （``_validate_resolved_user_data_path``），Agent 的 Prompt/输出里
    永远不会出现、也不需要知道宿主机真实路径。这里采用同样的思路，
    但结合本项目"每个任务一个独立沙箱根"的特点，把虚拟前缀简化为
    ``input://``/``workspace://``/``output://``/``tmp://`` 四个协议头，
    分别对应 ``TaskSandbox`` 的四个物理子目录：

        input://paper.txt        -> <sandbox_root>/input/paper.txt
        workspace://scratch.json -> <sandbox_root>/workspace/scratch.json
        output://result.json     -> <sandbox_root>/output/result.json
        tmp://cache.bin          -> <sandbox_root>/tmp/cache.bin

    子智能体（``agents/*/agent.py``）与工具描述（Prompt）里出现的路径
    永远是这种虚拟形式；只有 ``TaskSandbox.resolve_*`` 方法在真正访问
    文件系统前才会把它换算成宿主机绝对路径，且换算结果依然要经过
    ``validate_within_roots`` 的越界校验兜底——即便虚拟路径解析逻辑
    本身有疏漏，也不会真正突破沙箱边界。
"""

from __future__ import annotations

from pathlib import Path

# 虚拟路径协议头 -> 对应的沙箱子目录名。子智能体和工具 Prompt 描述里
# 只应该出现这四种协议头前缀的路径，不应该出现裸的相对路径或宿主机
# 绝对路径（后两者依然被兼容解析，见 ``resolve_virtual_or_relative``，
# 是为了不破坏未显式加协议头的历史调用，但不再是推荐用法）。
VIRTUAL_SCHEMES = {
    "input": "input",
    "workspace": "workspace",
    "output": "output",
    "tmp": "tmp",
}


class PathEscapeError(PermissionError):
    """尝试访问沙箱允许范围之外的路径。"""


class PathEscapeErrorForScheme(PathEscapeError):
    """虚拟路径协议头本身不被当前操作允许（例如对 ``input://`` 写入，
    或引用了未知协议头）。"""

    def __init__(self, scheme: str, path: str, reason: str = ""):
        self.scheme = scheme
        self.path = path
        message = reason or f"unknown or disallowed virtual path scheme '{scheme}://' in {path}"
        super().__init__(message)


def split_virtual_path(path: str) -> tuple[str, str] | None:
    """把 ``"input://a/b.txt"`` 拆成 ``("input", "a/b.txt")``；
    不是虚拟路径格式则返回 ``None``（调用方应回退到旧的相对路径解析，
    保持向后兼容）。
    """

    for scheme in VIRTUAL_SCHEMES:
        prefix = f"{scheme}://"
        if path.startswith(prefix):
            return scheme, path[len(prefix):]
    return None


def to_virtual_path(scheme: str, relative_path: str) -> str:
    """构造一个虚拟路径字符串，供 ``collect_outputs``/审计日志等场景
    把内部相对路径重新包装成子智能体应该看到的虚拟形式。
    """

    return f"{scheme}://{relative_path}"


def validate_within_roots(path: str, roots: list[Path]) -> Path:
    """把 ``path``（可能是相对路径）解析为绝对路径，并校验其位于
    ``roots`` 任一根目录之内；否则抛出 ``PathEscapeError``。

    对每个候选根目录都尝试一次"相对路径拼接"和"绝对路径直接解析"，
    因为调用方（LLM 生成的工具调用参数）既可能给相对路径也可能给
    绝对路径，两种都要支持，但都必须落在允许范围内。

    相对路径的多根消歧规则：
        纯粹按 ``roots`` 列表顺序"命中第一个就返回"是有陷阱的——对
        一个相对路径而言，把它拼到任意一个 root 下几乎总是"合法"
        （只要不含 ``..`` 试图跳出该 root），所以永远会命中列表里的
        第一个 root，即使调用方的意图明显是另一个 root（例如
        ``"output/result.json"`` 拼到排在最前面的 ``input_dir`` 下
        依然合法，但那根本不是调用方想要的位置）。
        因此这里优先选择"拼接后实际存在于文件系统"的候选路径；
        如果所有候选都不存在（比如要新建的输出文件），才退回列表
        顺序的第一个 root 作为默认落点。这样既不破坏"必须落在某个
        root 内"的安全校验，又能让"该文件已经在哪个 root 下"这一
        客观事实来消歧，而不是誰排在列表前面就选谁。
    """

    candidate = Path(path)
    resolved_roots = [r.resolve() for r in roots]

    if candidate.is_absolute():
        resolved = candidate.resolve()
        for root in resolved_roots:
            if _is_within(resolved, root):
                return resolved
        raise PathEscapeError(f"absolute path {path} is outside sandbox roots {roots}")

    # 相对路径：先找已存在的候选，找不到再退回第一个合法候选。
    candidates: list[Path] = []
    for root in resolved_roots:
        resolved = (root / candidate).resolve()
        if _is_within(resolved, root):
            candidates.append(resolved)

    if not candidates:
        raise PathEscapeError(f"path {path} could not be resolved within sandbox roots {roots}")

    for resolved in candidates:
        if resolved.exists():
            return resolved

    return candidates[0]


def _is_within(resolved: Path, root: Path) -> bool:
    try:
        resolved.relative_to(root)
        return True
    except ValueError:
        return False
