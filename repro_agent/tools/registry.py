"""全局工具注册表。

只有主智能体（严格来说，是 orchestrator 层的 ``ToolAuthorizer``）能够
访问这个全局注册表；子智能体运行时代码里永远不会 import 这个模块，
只会持有 ``authorization.ToolAuthorization`` 对象暴露出来的、
已经过滤好的工具子集。这是"子智能体不能访问全局资源"（设计文档
§3 原则 9）在工具层面的直接体现。
"""

from __future__ import annotations

from typing import Iterable

from repro_agent.tools.base import ToolRiskLevel, ToolSpec


class ToolRegistry:
    """进程内单例风格的工具注册表（不强制单例，测试时可以创建独立实例）。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._permanent_names: set[str] = set()

    def register(self, spec: ToolSpec, *, permanent: bool = True) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool '{spec.name}' already registered")
        if not isinstance(spec.output.schema, dict) or not spec.output.schema:
            raise ValueError(
                f"tool '{spec.name}' must declare a non-empty output schema"
            )
        self._tools[spec.name] = spec
        if permanent:
            self._permanent_names.add(spec.name)

    def unregister_dynamic(self, name: str) -> bool:
        """Remove only a generated tool; fixed built-ins are immutable."""

        if name in self._permanent_names:
            raise ValueError(f"built-in tool '{name}' is permanent")
        return self._tools.pop(name, None) is not None

    def is_permanent(self, name: str) -> bool:
        return name in self._permanent_names

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def all_specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def by_risk_level(self, level: ToolRiskLevel) -> list[ToolSpec]:
        return [s for s in self._tools.values() if s.risk_level == level]

    def filter_names(self, names: Iterable[str]) -> list[ToolSpec]:
        """按名称列表取出已注册的工具，静默忽略不存在的名字（由调用方决定
        是否要对"引用了不存在的工具"这件事报错——授权层会报错，
        纯查询场景不报错）。
        """

        return [self._tools[n] for n in names if n in self._tools]


_default_registry: ToolRegistry | None = None


def default_registry() -> ToolRegistry:
    """返回进程级默认注册表，首次调用时完成内置工具注册。"""

    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
        _register_builtin_tools(_default_registry)
    return _default_registry


def create_builtin_registry() -> ToolRegistry:
    """Return a fresh registry containing only permanent built-in tools.

    MainAgent uses a workspace-local instance so generated tools from one
    database cannot leak through the process-global compatibility registry into
    another workspace.
    """

    registry = ToolRegistry()
    _register_builtin_tools(registry)
    return registry


def _register_builtin_tools(registry: ToolRegistry) -> None:
    # 延迟导入，避免模块级循环依赖（filesystem_tools/resource_tools 会
    # import 本模块用于类型标注但不会在导入期触发注册表构造）。
    from repro_agent.tools import (
        code_index_tools,
        filesystem_tools,
        resource_tools,
        write_tools,
    )

    for spec in filesystem_tools.TOOL_SPECS:
        registry.register(spec)
    for spec in code_index_tools.TOOL_SPECS:
        registry.register(spec)
    for spec in resource_tools.TOOL_SPECS:
        registry.register(spec)
    for spec in write_tools.TOOL_SPECS:
        registry.register(spec)
