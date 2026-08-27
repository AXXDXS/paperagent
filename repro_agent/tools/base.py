"""工具（Tool）抽象基类与风险分级模型。

设计动机（对应设计文档 §7 任务定义协议 ``allowed_tools``/
``forbidden_actions`` 字段，以及 §3 原则 8-11 子智能体沙箱隔离）：

    设计文档把"允许的工具"作为任务定义的一部分下发给子智能体，但
    没有规定工具本身应该如何分级、如何被主智能体安全地"只下发一部分
    工具实例"。这里补充一套显式的**风险分级 + 授权令牌**机制：

    1. 每个工具在注册时声明一个 ``ToolRiskLevel``（只读/受限写/高危）；
    2. 子智能体从来**拿不到**全局工具注册表，只能拿到主智能体针对
       某个具体任务构造的 ``ToolAuthorization``（见 authorization.py）
       所包含的工具实例子集；
    3. 即使某个工具名字出现在任务的 ``allowed_tools`` 里，
       ``ToolAuthorization`` 也会二次校验：该工具的风险等级是否被
       任务类型允许（例如"论文分析子智能体"这类只读分析任务，
       即使手误在 allowed_tools 里写了 "execute_bash"，
       授权层也会拒绝下发）——这是纵深防御（defense in depth），
       不完全信任任务定义本身没有配置错误。

复用来源：
    风险分级 + "宁可拒绝也不要静默放行"的思路参考了 DeerFlow 的
    Fail-Closed 设计哲学（见 ``doc/DeerFlow_架构分析.md`` 第 14 节
    "Fail-Closed 优先于灵活性"）；工具执行的路径沙箱校验复用了
    DeepCode ``tools/code_implementation_server.py`` 中
    ``validate_path``/``log_operation`` 的模式（见该文件 88-103 行），
    在本项目中体现为 ``sandbox.paths.validate_within_sandbox`` +
    本模块的 ``ToolInvocationLog``。

工具描述设计原则（面向 LLM 的"决策文档"，而不是面向人的"API 文档"）：

    子智能体选错工具、传错参数，绝大多数根因不是模型能力不够，而是
    工具描述本身没有说清楚。因此 ``ToolSpec`` 在"名字 + 一句话描述"
    之外，显式建模了以下几段，全部会被渲染进最终发给 LLM 的
    ``function.description``（OpenAI function-calling 协议本身只有
    name/description/parameters 三个槽位，没有专门的结构化字段，因此
    约定把下面这些信息序列化进 description 富文本，而不是发明协议不
    支持的自定义字段）：

    1. ``when_to_use``：**决策导向**而非能力罗列。回答"什么情况下应该
       调用我"，而不是"我能做什么"——例如不写"检索文件"，而写"当你
       需要按文件名定位文件、但还不知道内容在哪一行时使用"。
    2. ``boundaries``：**边界与反例**，明确"做不到什么、不接受什么
       输入"。这一段往往比能力描述本身更重要，因为模型不会主动
       意识到自己在犯"过度泛化"的错误（例如以为"文件查找"工具
       也能搜索文件内容）。
    3. ``returns``：返回值结构说明，减少下游解析阶段的臆测。
    4. ``cost_hint``：执行代价与替代建议（耗时/额度消耗），帮助模型
       规划调用顺序、避免用重工具做轻量的事。
    5. ``examples``：1-5 个真实调用示例（参数取值 + 典型返回），
       JSON Schema 只能声明类型，无法表达"时间戳到底是秒还是毫秒"
       这类隐式约定，例子是传达这些约定最省 token、最不会被误读的
       方式。

    参数级别的文档同理：``ToolParamDoc`` 允许为每个参数单独提供
    ``description``（用带具体例子的自然语言，而不是抽象规范）和
    ``example``，覆盖掉从函数签名反射出的、只有类型信息的默认 schema。
"""

from __future__ import annotations

import inspect
import json
import types
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Protocol, get_args, get_origin, get_type_hints

from repro_agent.domain.common import iso, new_id, utc_now
from repro_agent.providers.base import ContentBlock


class ToolRiskLevel(str, Enum):
    """工具风险等级，等级越高，默认越不容易被下发给子智能体。"""

    READ_ONLY = "read_only"  # 文件查找/阅读/资源探测等，无副作用
    RESTRICTED_WRITE = "restricted_write"  # 仅限沙箱 workspace/output 内写入
    HIGH_RISK = "high_risk"  # 执行命令、网络访问、修改公共代码仓库等


@dataclass
class ToolInvocationLog:
    """单次工具调用的审计记录（呼应设计文档 §12 沙箱设计"日志审计"）。"""

    tool_name: str
    task_id: str
    arguments: dict[str, Any]
    result_summary: str
    succeeded: bool
    # 下面字段为断点恢复和逐次审计补充。保留默认值，兼容现有工具授权
    # 测试和使用旧构造方式的调用方。
    attempt_id: str = ""
    sequence: int = 0
    result: Any = None
    error: str = ""
    replayed: bool = False
    invocation_id: str = field(default_factory=lambda: new_id("tool"))
    timestamp: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "task_id": self.task_id,
            "arguments": self.arguments,
            "result_summary": self.result_summary,
            "succeeded": self.succeeded,
            "attempt_id": self.attempt_id,
            "sequence": self.sequence,
            "result": self.result,
            "error": self.error,
            "replayed": self.replayed,
            "invocation_id": self.invocation_id,
            "timestamp": iso(self.timestamp),
        }


class ToolExecutionError(RuntimeError):
    """工具执行失败（区别于"未授权调用"，后者是 PermissionError）。"""


INVALID_TOOL_OUTPUT = "INVALID_TOOL_OUTPUT"


class ToolInputValidationError(ToolExecutionError):
    """模型提供的工具参数没有通过 Schema/资源边界校验。"""


class InvalidToolOutputError(ToolExecutionError):
    """工具返回值无法安全地交给模型。"""

    code = INVALID_TOOL_OUTPUT

    def __init__(self, message: str):
        super().__init__(f"{self.code}: {message}")


# Keep a descriptive name for callers that prefer the validation terminology.
ToolOutputValidationError = InvalidToolOutputError


class ToolPermissionError(PermissionError):
    """子智能体尝试调用未被授权的工具或触碰 forbidden_actions。"""


class ToolGrantDeniedError(ToolPermissionError):
    """主智能体（或人工）已对"补授缺失工具"做出明确拒绝裁决。

    与普通 ``ToolPermissionError`` 的区别：普通权限错误意味着"还没人
    裁决过能不能给这个工具"；本异常携带主智能体裁决的理由，表明
    "已经裁决过且结论是不给"——上游（dispatcher）据此在
    ``FailureReport.metadata`` 里写入 ``tool_grant_adjudicated`` 标记，
    避免主循环对同一请求反复升级。
    """

    def __init__(self, tool_name: str, reason: str):
        self.tool_name = tool_name
        self.grant_reason = reason
        super().__init__(
            f"main agent adjudicated and DENIED tool '{tool_name}': {reason}"
        )


ToolHandler = Callable[..., Any]


def _default_tool_output_render(value: Any) -> ContentBlock:
    """Render a validated tool value as a JSON content block."""

    return ContentBlock(type="json", data=value)


@dataclass(frozen=True)
class ToolOutputSpec:
    """Machine contract for a tool's return value and model rendering."""

    schema: dict[str, Any] = field(
        # Keep construction backwards compatible so an incomplete ToolSpec can
        # be assembled and inspected, but ToolRegistry.register rejects this
        # empty sentinel. Every callable tool must opt into an explicit schema.
        default_factory=dict
    )
    render: Callable[[Any], ContentBlock] = _default_tool_output_render


@dataclass
class ToolParamDoc:
    """单个参数的"决策级"文档，覆盖掉从函数签名反射出的空描述。

    ``description`` 应该用带具体例子的自然语言写格式约定（例如
    "RFC3339 格式，例如 2024-03-15T14:30:00Z"），而不是只写规范名称
    （"RFC3339 格式"）——模型在同时处理多个工具、多轮历史时，
    对抽象规范的注意力预算很有限，具体例子可以直接套用、不需要
    额外的换算/推理步骤。``example`` 会被合并进 JSON Schema 的
    ``examples`` 字段，供支持该字段的客户端展示。
    """

    description: str
    example: Any = None


@dataclass
class ToolExample:
    """一次真实的调用示例：参数取值 + 典型返回，用于传达 JSON Schema
    无法表达的隐式约定（参数如何组合、字段单位是什么等）。

    ``when`` 用一句话说明这个示例对应的使用场景，帮助模型把"当前
    诉求"和"该用哪组参数"对上号，而不是只看一堆孤立的 key-value。
    """

    when: str
    arguments: dict[str, Any]
    result: Any = None


@dataclass
class ToolSpec:
    """工具的静态元数据（注册到全局注册表 ``ToolRegistry`` 中的单元）。

    面向 LLM 的描述被拆成多段（见模块顶部"工具描述设计原则"），
    渲染顺序固定为：一句话能力 → 何时使用 → 边界与反例 → 返回值 →
    执行代价 → 调用示例。拆分成独立字段而不是让每个工具作者自己在
    一整段 ``description`` 里手写这些内容，是为了：
        1. 强制每个工具都过一遍"边界/示例是否写了"的检查清单；
        2. 保证跨工具的呈现顺序、格式一致，模型不需要为不同工具
           重新适应不同的描述风格。
    """

    name: str
    description: str
    risk_level: ToolRiskLevel
    handler: ToolHandler
    # 该工具允许被哪些任务类型使用的白名单建议（供 orchestrator 在生成
    # 任务定义时参考，不是强制约束——真正强制的是 ToolAuthorization）。
    suggested_task_types: tuple[str, ...] = field(default_factory=tuple)
    requires_network: bool = False
    parameters: dict[str, Any] | None = None
    # ---- 决策导向的补充描述（见模块文档顶部原则 1-5） ----
    when_to_use: str = ""
    boundaries: tuple[str, ...] = field(default_factory=tuple)
    returns: str = ""
    cost_hint: str = ""
    examples: tuple[ToolExample, ...] = field(default_factory=tuple)
    # 参数名 -> 参数级文档；未覆盖的参数仍会从函数签名反射出裸类型。
    param_docs: dict[str, ToolParamDoc] = field(default_factory=dict)
    # 工具执行结果的机器契约。注册表拒绝空 schema；所有可调用工具都
    # 必须声明结果结构，renderer 只会在 JSON/Schema 校验通过后执行。
    output: ToolOutputSpec = field(default_factory=ToolOutputSpec)

    def describe(self) -> dict[str, Any]:
        return self.to_openai_tool()

    def argument_schema(self) -> dict[str, Any]:
        """返回严格 JSON Schema；未显式配置时从 handler 签名 + ``param_docs`` 生成。

        即使显式提供了 ``parameters``，也会用 ``param_docs`` 补齐/覆盖
        其中的 ``description``/``examples``——两者不是互斥关系：
        ``parameters`` 负责类型/约束的机器可读部分，``param_docs``
        负责"给模型看的例子和用法说明"这部分人类可读但机器难以从
        类型签名推断出的信息。
        """

        schema = (
            dict(self.parameters)
            if self.parameters is not None
            else self._reflect_schema()
        )
        if self.param_docs and "properties" in schema:
            properties = dict(schema["properties"])
            for name, doc in self.param_docs.items():
                prop = dict(properties.get(name, {}))
                if doc.description:
                    prop["description"] = doc.description
                if doc.example is not None:
                    prop["examples"] = [doc.example]
                properties[name] = prop
            schema = {**schema, "properties": properties}
        return schema

    def _reflect_schema(self) -> dict[str, Any]:
        signature = inspect.signature(self.handler)
        try:
            hints = get_type_hints(self.handler)
        except (NameError, TypeError):
            hints = {}
        properties: dict[str, Any] = {}
        required: list[str] = []
        for index, (name, parameter) in enumerate(signature.parameters.items()):
            if index == 0:  # SandboxContext 由授权层注入，不暴露给模型。
                continue
            annotation = hints.get(name, parameter.annotation)
            properties[name] = _annotation_schema(annotation)
            if parameter.default is inspect.Parameter.empty:
                required.append(name)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def _render_description(self) -> str:
        """把"一句话能力 + 何时用 + 边界 + 返回值 + 代价 + 示例"拼成
        一段结构化的自然语言，作为最终发给 LLM 的 ``function.description``。

        各段落只在非空时才渲染，保证旧工具（尚未补充这些字段）依然
        能正常工作、只是退化为"仅有一句话描述"的最小可用状态。
        """

        parts = [self.description.strip()]

        if self.when_to_use:
            parts.append(f"何时使用：{self.when_to_use.strip()}")

        if self.boundaries:
            bullet_lines = "\n".join(f"  - {b}" for b in self.boundaries)
            parts.append(f"边界（做不到什么/不接受什么输入）：\n{bullet_lines}")

        if self.returns:
            parts.append(f"返回值：{self.returns.strip()}")

        if self.cost_hint:
            parts.append(f"执行代价：{self.cost_hint.strip()}")

        if self.examples:
            example_lines = []
            for ex in self.examples:
                line = f"  - 场景：{ex.when}\n    参数：{json.dumps(ex.arguments, ensure_ascii=False)}"
                if ex.result is not None:
                    line += f"\n    典型返回：{json.dumps(ex.result, ensure_ascii=False)}"
                example_lines.append(line)
            parts.append("调用示例：\n" + "\n".join(example_lines))

        return "\n\n".join(parts)

    def to_openai_tool(self) -> dict[str, Any]:
        """只发送 OpenAI function-calling 支持的字段，隐藏内部风险元数据。"""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self._render_description(),
                "parameters": self.argument_schema(),
            },
        }


def _annotation_schema(annotation: Any) -> dict[str, Any]:
    if annotation in (inspect.Parameter.empty, Any):
        return {}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (types.UnionType, __import__("typing").Union):
        non_null = [item for item in args if item is not type(None)]
        if len(non_null) == 1:
            return _annotation_schema(non_null[0])
        return {"anyOf": [_annotation_schema(item) for item in non_null]}
    if origin is list or annotation is list:
        item_type = args[0] if args else Any
        return {"type": "array", "items": _annotation_schema(item_type)}
    if origin is dict or annotation is dict:
        return {"type": "object"}
    if annotation is str:
        return {"type": "string"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        return {"type": "string", "enum": [item.value for item in annotation]}
    return {}


class SandboxContext(Protocol):
    """工具执行时可以感知的沙箱上下文（避免工具直接依赖具体沙箱实现）。

    真正的实现见 ``repro_agent.sandbox.workspace.TaskSandbox``，这里
    用 Protocol 做接口隔离，方便未来替换沙箱实现或在单元测试中打桩。
    """

    task_id: str

    def resolve_readable_path(self, relative_path: str) -> str:
        ...

    def resolve_writable_path(self, relative_path: str) -> str:
        ...

    def resolve_output_path(self, relative_path: str) -> str:
        ...

    def network_allowed(self) -> bool:
        ...
