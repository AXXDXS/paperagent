"""工具授权：主智能体按任务 ``allowed_tools`` 下发受限工具集给子智能体。

这是对用户新增需求的核心实现：
    "文件查找、文件阅读、资源阅读等这些封装为工具，子 agent 只能使用
    主 agent 传给他的、低风险的工具"

设计要点：
    1. **子智能体永远拿不到全局 ``ToolRegistry``**。子智能体运行时
       代码（``repro_agent/agents/*``）只接受一个
       ``ToolAuthorization`` 对象，通过它调用工具；``ToolAuthorization``
       内部持有的是"已经过滤好的工具处理函数"，而不是注册表引用，
       因此子智能体代码即使拿到了 ``ToolAuthorization`` 对象也无法
       "越权"访问未被列入的工具（没有暴露任何取全集的方法）。

    2. **两层授权校验（纵深防御）**：
       a. 显式白名单：任务定义 ``TaskDefinition.allowed_tools`` 中
          列出的工具名称；
       b. 风险预算：每种任务类型（``task_type``）在
          ``TASK_TYPE_RISK_BUDGET`` 中声明了"最高允许的风险等级"，
          即使 ``allowed_tools`` 里出现了超出该任务类型风险预算的
          工具名，也会被拒绝并记录审计日志——防止任务定义本身配置
          错误（比如 Prompt 生成任务定义时的疏漏）导致子智能体
          意外获得高危能力。
       这与设计文档 §7 任务协议中 ``forbidden_actions`` 字段的精神
       一致："访问全局记忆""读取其他任务目录""修改代码文件""访问
       外部网络"这类禁止项，本质上就是风险预算的自然语言表达，这里
       把它转成了机器可校验的枚举比较。

    3. **调用即审计**：每次工具调用都会通过
       ``SandboxContext``/``ToolInvocationLog`` 留痕，写入任务的
       output 目录，供主智能体验证输出、事后审计（呼应 §12"日志
       审计"与 §11.3 反思智能体检查维度 B"是否存在条件分支没有进入"
       这类需要复盘工具调用轨迹的场景）。

复用来源：
    "运行时把可调用工具面收窄到一个能力子集"这一思路参考了 DeepCode
    Paper2Code 索引增强模式的设计（``workflows/code_implementation_workflow.py``
    第 7.1 节：``enable_indexing=True`` 时只暴露 ``write_file`` 和
    ``search_code_references`` 两个工具），以及 DeerFlow Skills 系统
    的"``allowed-tools`` 仅在技能被激活时才生效、且不能放宽已有限制"
    的策略（``doc/DeerFlow_架构分析.md`` 第 9.1 节）。
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from repro_agent.providers.base import ContentBlock
from repro_agent.tools.base import (
    InvalidToolOutputError,
    SandboxContext,
    ToolInvocationLog,
    ToolInputValidationError,
    ToolPermissionError,
    ToolRiskLevel,
    ToolSpec,
)
from repro_agent.tools.registry import ToolRegistry, default_registry
from repro_agent.tools.result_sanitization import (
    ToolResultSanitizationConfig,
    redact_sensitive_text,
    sanitize_tool_result_for_model,
)
from repro_agent.tools.schema_validation import (
    SchemaValidationError,
    ensure_lossless_json,
    validate_argument_limits,
    validate_json_schema,
)

logger = logging.getLogger(__name__)


# 任务类型 -> 该类型任务默认允许的最高工具风险等级。
# 只读分析类任务永远拿不到 RESTRICTED_WRITE / HIGH_RISK 工具，
# 即使任务定义的 allowed_tools 里意外写了这些工具名。
_RISK_ORDER = {
    ToolRiskLevel.READ_ONLY: 0,
    ToolRiskLevel.RESTRICTED_WRITE: 1,
    ToolRiskLevel.HIGH_RISK: 2,
}

TASK_TYPE_RISK_BUDGET: dict[str, ToolRiskLevel] = {
    # 纯分析/只读类子智能体：论文分析、代码分析、资源检查、结果验证、反思
    "paper_analysis": ToolRiskLevel.READ_ONLY,
    "code_analysis": ToolRiskLevel.READ_ONLY,
    "resource_check": ToolRiskLevel.READ_ONLY,
    "verification": ToolRiskLevel.READ_ONLY,
    "reflection": ToolRiskLevel.READ_ONLY,
    # 实验规格汇总只读取信息、写自己的输出文件
    "specification": ToolRiskLevel.RESTRICTED_WRITE,
    # 需要真正写文件/跑命令的子智能体
    "environment_build": ToolRiskLevel.HIGH_RISK,
    "coding": ToolRiskLevel.HIGH_RISK,
    "experiment_execution": ToolRiskLevel.HIGH_RISK,
}
_DEFAULT_TASK_TYPE_BUDGET = ToolRiskLevel.READ_ONLY

# 少数工具即使风险等级标注为 RESTRICTED_WRITE，也应被视为"每个任务
# 都必需的最小能力"而不是"额外提权"——因为它们的写入范围被严格限定
# 在该任务自己的 output/ 目录内（§15.2 候选记忆产出协议要求所有
# 子智能体，包括纯只读分析类，都必须能写 result.json/
# candidate_memory.md）。这里用一份显式的豁免名单而不是简单地把
# write_task_output 的 risk_level 直接标成 READ_ONLY，是为了在
# ToolSpec 层面依然如实反映"这是一次真实的文件写入"这一事实
# （便于审计/展示），只在风险预算校验这一步单独放行。
_ALWAYS_ALLOWED_TOOLS = {"write_task_output"}


def risk_allowed(task_type: str, risk_level: ToolRiskLevel, tool_name: str = "") -> bool:
    if tool_name in _ALWAYS_ALLOWED_TOOLS:
        return True
    budget = TASK_TYPE_RISK_BUDGET.get(task_type, _DEFAULT_TASK_TYPE_BUDGET)
    return _RISK_ORDER[risk_level] <= _RISK_ORDER[budget]


@dataclass
class ToolDenial:
    tool_name: str
    reason: str


class ToolAuthorization:
    """子智能体持有的、已裁剪好的工具调用句柄集合。

    子智能体代码只应该通过 ``call(tool_name, **kwargs)`` 或
    ``bound_tools()`` 拿到的闭包来调用工具，永远不接触
    ``ToolRegistry``/``ToolSpec.handler`` 本身。

    运行期动态授权（工具分配权上收到主智能体后的新增能力）：
        ``grant_additional`` 允许主智能体在任务执行过程中向本对象追加
        新的 ``ToolSpec``；``escalation_handler`` 则是"子智能体调用了一个
        存在于全局注册表、但未分配给自己的工具"时的升级通道——在
        **不注入 handler** 的一切旧场景（单测、旧调用方）中，行为与
        改造前完全一致：直接抛 ``ToolPermissionError``。handler 由
        ``AgentDispatcher`` 在派发时注入，它代表主智能体对这次缺工具
        请求的裁决（补授 / 拒绝 / 转人工），子智能体线程在 handler 内
        阻塞等待裁决，裁决为补授后 ``call`` 直接继续执行原工具——
        这就是"追加工具后子智能体原地继续、不重启"的机制核心。
        handler 永远不会由子智能体侧代码设置，也不会暴露任何能绕过
        风险预算校验的接口：补授的 spec 同样来自
        ``ToolAuthorizer.authorize`` 的完整校验。

    并发模型：``_granted`` 会被子智能体线程读、主智能体线程（人工
    批准后注入）写，因此所有读写都在 ``_granted_lock`` 保护下进行；
    handler 的调用本身发生在锁外（handler 内部可能再调
    ``grant_additional``，RLock 可重入，不会死锁）。
    """

    def __init__(
        self,
        task_id: str,
        granted_specs: list[ToolSpec],
        sandbox_ctx: SandboxContext,
        denials: list[ToolDenial] | None = None,
        *,
        attempt_id: str = "",
        invocation_recorder: Callable[[ToolInvocationLog], None] | None = None,
        invocation_observer: Callable[[str, bool], None] | None = None,
        escalation_handler: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        self.task_id = task_id
        self._sandbox_ctx = sandbox_ctx
        self._granted: dict[str, ToolSpec] = {s.name: s for s in granted_specs}
        self._granted_lock = threading.RLock()
        self.denials = denials or []
        self.invocation_log: list[ToolInvocationLog] = []
        self.attempt_id = attempt_id
        self._invocation_recorder = invocation_recorder
        self._invocation_observer = invocation_observer
        self._invocation_sequence = 0
        self._escalation_handler = escalation_handler

    def set_escalation_handler(
        self, handler: Callable[[str, dict[str, Any]], None] | None
    ) -> None:
        """由派发方（AgentDispatcher）注入缺工具升级通道。

        只应该在子智能体启动前调用；handler 本身就是主智能体侧的
        裁决入口，不存在被子智能体伪造授权的路径。
        """

        self._escalation_handler = handler

    def grant_additional(self, spec: ToolSpec) -> None:
        """运行期向本授权对象追加一个工具（线程安全）。

        调用方（dispatcher / 人工批准路径）必须已经过
        ``ToolAuthorizer`` 的完整校验——本方法不做风险预算判断，
        因为合法 spec 的唯一生产路径就是 ``ToolAuthorizer.authorize``
        （主智能体侧），子智能体代码接触不到本对象之外的任何 spec
        来源，不存在越权注入面。
        """

        with self._granted_lock:
            self._granted[spec.name] = spec
            self.denials = [d for d in self.denials if d.tool_name != spec.name]

    def _get_granted(self, tool_name: str) -> Optional[ToolSpec]:
        with self._granted_lock:
            return self._granted.get(tool_name)

    def get_granted_spec(self, tool_name: str) -> Optional[ToolSpec]:
        """公开只读访问：取出某个已授权工具的 spec（无则 None）。"""

        return self._get_granted(tool_name)

    @property
    def granted_tool_names(self) -> list[str]:
        with self._granted_lock:
            return sorted(self._granted)

    @property
    def remaining_tool_calls(self) -> int:
        max_calls = getattr(
            getattr(getattr(self._sandbox_ctx, "policy", None), "resource_limits", None),
            "max_tool_calls",
            0,
        )
        return max(0, max_calls - len(self.invocation_log))

    def describe_granted(self, tool_names: list[str] | None = None) -> list[dict[str, Any]]:
        """返回可下发给 LLM 的工具描述（供 Prompt 中列出可用工具）。

        ``tool_names`` 为 ``None`` 时返回该任务被授权的**全部**工具，
        仅作为向后兼容的兜底；调用方（``BaseSubAgent.call_llm``）应当
        始终显式传入"这一次 LLM 调用真正需要哪些工具"，实现"每次工具
        暴露面都收窄到当次任务步骤所需的最小子集"，而不是把整个任务
        的 granted 权限一次性全部亮给模型，让模型自己去猜该用哪个。

        请求了未被授权的工具名时直接抛错，而不是静默忽略——这通常
        意味着子智能体代码写错了工具名，或者该任务类型的
        ``allowed_tools`` 模板漏配了这个工具，两种情况都应该在开发/
        联调阶段尽早暴露，而不是被吞掉之后表现为"LLM 一直调不到工具"
        这种更难排查的现象。
        """

        if tool_names is None:
            with self._granted_lock:
                return [
                    self._granted[name].to_openai_tool()
                    for name in sorted(self._granted)
                ]

        specs = []
        missing = []
        with self._granted_lock:
            for name in sorted(set(tool_names)):
                spec = self._granted.get(name)
                if spec is None:
                    missing.append(name)
                else:
                    specs.append(spec)
        if missing:
            raise ToolPermissionError(
                f"task {self.task_id} requested tool description(s) not granted: "
                f"{', '.join(missing)} (granted={self.granted_tool_names})"
            )
        return [spec.to_openai_tool() for spec in specs]

    def call(self, tool_name: str, /, **kwargs: Any) -> Any:
        """子智能体调用工具的唯一入口。

        未授权分支的两种行为：
            - 注入了 ``escalation_handler``（正常运行路径）：先尝试升级
              给主智能体裁决。handler 要么把工具补授进本对象（裁决为
              GRANT，可能经过人工批准的等待）后正常返回，要么抛出
              ``ToolGrantDeniedError``/``ToolPermissionError``（裁决为
              DENY）——升级失败与未升级一样走 DENIED 审计 + 抛错。
            - 未注入 handler（单测/旧路径）：行为与改造前完全一致，
              直接记录 DENIED 审计并抛 ``ToolPermissionError``。
        """

        spec = self._get_granted(tool_name)
        if spec is None and self._escalation_handler is not None:
            # 工具可能已注册但未分配：先请求主智能体裁决。handler
            # 在子智能体线程内阻塞执行（GRANT 立即返回；ASK_USER 会
            # 挂起等待人工），裁决成功后工具已并入 _granted。
            self._escalation_handler(tool_name, _redact_arguments(kwargs))
            spec = self._get_granted(tool_name)
            if spec is None:
                # handler 声称补授但没有真正补授（实现 bug），fail closed。
                logger.error(
                    "task %s: escalation handler returned without granting "
                    "tool '%s'; failing closed",
                    self.task_id,
                    tool_name,
                )
        if spec is None:
            error = ToolPermissionError(
                f"task {self.task_id} is not authorized to call tool '{tool_name}'"
            )
            self._append_invocation(
                ToolInvocationLog(
                    tool_name=tool_name,
                    task_id=self.task_id,
                    arguments=_redact_arguments(kwargs),
                    result_summary=f"DENIED: {error}",
                    succeeded=False,
                    error=str(error),
                )
            )
            raise error
        max_calls = getattr(
            getattr(getattr(self._sandbox_ctx, "policy", None), "resource_limits", None),
            "max_tool_calls",
            0,
        )
        if max_calls <= 0 or len(self.invocation_log) >= max_calls:
            error = ToolPermissionError(
                f"task {self.task_id} tool-call budget exhausted ({max_calls})"
            )
            self._append_invocation(
                ToolInvocationLog(
                    tool_name=tool_name,
                    task_id=self.task_id,
                    arguments=_redact_arguments(kwargs),
                    result_summary=f"DENIED: {error}",
                    succeeded=False,
                    error=str(error),
                )
            )
            raise error
        redacted_arguments = _redact_arguments(kwargs)
        try:
            _validate_arguments(spec.argument_schema(), kwargs)
        except ToolInputValidationError as exc:
            self._append_invocation(
                ToolInvocationLog(
                    tool_name=tool_name,
                    task_id=self.task_id,
                    arguments=redacted_arguments,
                    result_summary=f"VALIDATION_REJECTED: {exc}",
                    succeeded=False,
                    error=str(exc),
                )
            )
            raise
        try:
            result = spec.handler(self._sandbox_ctx, **kwargs)
        except Exception as exc:  # noqa: BLE001 - 记录后重新抛出，保留原始异常类型
            self._append_invocation(
                ToolInvocationLog(
                    tool_name=tool_name,
                    task_id=self.task_id,
                    arguments=redacted_arguments,
                    result_summary=f"ERROR: {exc}",
                    succeeded=False,
                    error=str(exc),
                )
            )
            raise
        try:
            _validate_tool_output(spec, result)
        except InvalidToolOutputError as exc:
            self._append_invocation(
                ToolInvocationLog(
                    tool_name=tool_name,
                    task_id=self.task_id,
                    arguments=redacted_arguments,
                    result_summary=str(exc),
                    succeeded=False,
                    error=str(exc),
                )
            )
            raise
        self._append_invocation(
            ToolInvocationLog(
                tool_name=tool_name,
                task_id=self.task_id,
                arguments=redacted_arguments,
                result_summary=_summarize(result),
                succeeded=True,
                result=_audit_result(result),
            )
        )
        return result

    def sanitize_result_for_model(
        self,
        tool_name: str,
        result: Any,
        *,
        config: ToolResultSanitizationConfig | None = None,
    ) -> dict[str, Any]:
        """Build the only tool-result representation allowed into model context.

        Deterministic Agent code may keep using the raw in-process return value from
        :meth:`call`.  Whenever the consumer is an LLM, callers must pass it through
        this method so secrets, binary/unsupported values and unbounded collections
        cannot cross the model boundary.
        """

        if tool_name not in self._granted:
            raise ToolPermissionError(
                f"task {self.task_id} cannot sanitize result for ungranted tool '{tool_name}'"
            )
        return self.render_result_for_model(
            tool_name, result, config=config
        ).data

    def render_result_for_model(
        self,
        tool_name: str,
        result: Any,
        *,
        config: ToolResultSanitizationConfig | None = None,
    ) -> ContentBlock:
        """Validate, render and sanitize a tool result for model context.

        The returned ``ContentBlock`` is the only representation that should
        be appended to an LLM conversation.  ``sanitize_result_for_model``
        remains as a compatibility helper for deterministic callers that still
        expect the historical dictionary envelope.
        """

        spec = self._get_granted(tool_name)
        if spec is None:
            raise ToolPermissionError(
                f"task {self.task_id} cannot render result for ungranted tool '{tool_name}'"
            )
        validated = _validate_tool_output(spec, result)
        try:
            block = spec.output.render(validated)
        except Exception as exc:  # noqa: BLE001 - renderer is a tool boundary
            raise InvalidToolOutputError(
                f"output renderer failed: {type(exc).__name__}"
            ) from exc
        if not isinstance(block, ContentBlock):
            raise InvalidToolOutputError(
                "output renderer must return a ContentBlock"
            )
        if not isinstance(block.type, str) or not block.type.strip():
            raise InvalidToolOutputError(
                "ContentBlock.type must be a non-empty string"
            )
        try:
            # A custom renderer is another boundary: do not allow it to smuggle
            # an unsupported value into the provider after the tool result was
            # validated.
            rendered_data = ensure_lossless_json(block.data)
            if not isinstance(block.metadata, dict):
                raise SchemaValidationError("$: ContentBlock.metadata must be an object")
            rendered_metadata = ensure_lossless_json(block.metadata)
            if not isinstance(rendered_metadata, dict):
                raise SchemaValidationError("$: ContentBlock.metadata must be an object")
            safe_data = sanitize_tool_result_for_model(
                tool_name, rendered_data, config=config
            )
        except SchemaValidationError as exc:
            raise InvalidToolOutputError(
                f"rendered ContentBlock data is not losslessly JSON representable: {exc}"
            ) from exc
        return ContentBlock(
            type=block.type,
            data=safe_data,
            metadata=rendered_metadata,
        )

    def _append_invocation(self, log: ToolInvocationLog) -> None:
        """追加内存日志并立即持久化到审计存储。

        持久化回调运行在工具调用所在的子 Agent 线程。SQLite Repository
        自身提供事务和线程锁，因此一次成功/失败调用在 Agent 尚未结束前
        就已经可被恢复器与审计页面看到。
        """

        self._invocation_sequence += 1
        log.attempt_id = self.attempt_id
        log.sequence = self._invocation_sequence
        self.invocation_log.append(log)
        if self._invocation_recorder is not None:
            try:
                self._invocation_recorder(log)
            except Exception:  # noqa: BLE001 - 审计失败不能导致工具语义改变
                logger.exception(
                    "failed to persist tool invocation for task %s", self.task_id
                )
        if self._invocation_observer is not None:
            try:
                self._invocation_observer(log.tool_name, log.succeeded)
            except Exception:  # noqa: BLE001 - lifecycle accounting is observational
                logger.exception(
                    "failed to update lifecycle for tool %s", log.tool_name
                )

    def bound_tools(self) -> dict[str, Any]:
        """返回 {工具名: 可直接调用的闭包} 字典，方便以"函数调用"风格
        暴露给 LLM 工具调用框架（不同 LLM SDK 的 function-calling 接入
        方式不同，这里只做最小公分母封装）。
        """

        with self._granted_lock:
            names = list(self._granted)
        return {name: (lambda _n=name, **kw: self.call(_n, **kw)) for name in names}


def _summarize(result: Any) -> str:
    text = str(result)
    return text if len(text) <= 300 else text[:300] + "...(truncated)"


_SENSITIVE_KEY = re.compile(r"(?:secret|token|password|api[_-]?key|credential)", re.I)
_AUDIT_MAX_STRING = 8_192
_AUDIT_MAX_COLLECTION = 100


def _redact_arguments(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if _SENSITIVE_KEY.search(str(key))
                else "[OMITTED_WRITE_CONTENT]"
                if str(key) == "content"
                else _redact_arguments(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_arguments(item) for item in value]
    return _audit_value(value)


def _audit_result(value: Any) -> Any:
    """将工具结果转换成可安全写入 SQLite 的有界审计值。"""

    if isinstance(value, dict):
        items = list(value.items())[:_AUDIT_MAX_COLLECTION]
        result = {}
        for key, item in items:
            result[str(key)] = (
                "[REDACTED]"
                if _SENSITIVE_KEY.search(str(key))
                else _audit_result(item)
            )
        if len(value) > _AUDIT_MAX_COLLECTION:
            result["__truncated_items__"] = len(value) - _AUDIT_MAX_COLLECTION
        return result
    if isinstance(value, (list, tuple)):
        result = [_audit_result(item) for item in value[:_AUDIT_MAX_COLLECTION]]
        if len(value) > _AUDIT_MAX_COLLECTION:
            result.append({"__truncated_items__": len(value) - _AUDIT_MAX_COLLECTION})
        return result
    return _audit_value(value)


def _audit_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text, _ = redact_sensitive_text(str(value))
    if len(text) <= _AUDIT_MAX_STRING:
        return text
    return text[:_AUDIT_MAX_STRING] + f"...(truncated {len(text) - _AUDIT_MAX_STRING} chars)"


def _validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
    if not isinstance(arguments, dict):
        raise ToolInputValidationError("tool arguments must be a JSON object")
    try:
        validate_argument_limits(arguments)
        validate_json_schema(arguments, schema)
    except SchemaValidationError as exc:
        raise ToolInputValidationError(f"invalid tool arguments: {exc}") from exc


def _validate_tool_output(spec: ToolSpec, result: Any) -> Any:
    """Enforce the result -> lossless JSON -> declared schema sequence."""

    try:
        json_result = ensure_lossless_json(result)
        validate_json_schema(json_result, spec.output.schema)
    except SchemaValidationError as exc:
        raise InvalidToolOutputError(str(exc)) from exc
    return json_result


class ToolAuthorizer:
    """主智能体侧的授权决策者：给定任务定义，生成 ``ToolAuthorization``。"""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        *,
        invocation_observer: Callable[[str, bool], None] | None = None,
    ):
        self.registry = registry or default_registry()
        self.invocation_observer = invocation_observer

    def authorize(
        self,
        *,
        task_id: str,
        task_type: str,
        allowed_tools: list[str],
        forbidden_actions: list[str] | None = None,
        sandbox_ctx: SandboxContext,
        attempt_id: str = "",
        invocation_recorder: Callable[[ToolInvocationLog], None] | None = None,
    ) -> ToolAuthorization:
        """核心授权逻辑：显式白名单 ∩ 风险预算 ∩ 已注册工具。"""

        granted: list[ToolSpec] = []
        denials: list[ToolDenial] = []
        forbidden = set(forbidden_actions or [])

        for name in allowed_tools:
            spec = self.registry.get(name)
            if spec is None:
                denials.append(ToolDenial(name, "tool not registered"))
                continue
            if name in forbidden:
                denials.append(ToolDenial(name, "explicitly forbidden"))
                continue
            if spec.requires_network and {
                "network",
                "external_network",
                "unauthorized_external_network",
            }.intersection(forbidden):
                denials.append(ToolDenial(name, "network access explicitly forbidden"))
                continue
            if not risk_allowed(task_type, spec.risk_level, tool_name=name):
                denials.append(
                    ToolDenial(
                        name,
                        f"risk level {spec.risk_level.value} exceeds budget for "
                        f"task_type={task_type}",
                    )
                )
                logger.warning(
                    "denying tool '%s' for task %s: risk %s exceeds budget for task_type %s",
                    name,
                    task_id,
                    spec.risk_level.value,
                    task_type,
                )
                continue
            granted.append(spec)

        return ToolAuthorization(
            task_id=task_id,
            granted_specs=granted,
            sandbox_ctx=sandbox_ctx,
            denials=denials,
            attempt_id=attempt_id,
            invocation_recorder=invocation_recorder,
            invocation_observer=self.invocation_observer,
        )

    def validate_human_approval(
        self,
        *,
        task_type: str,
        tool_names: list[str],
        forbidden_actions: list[str] | None = None,
    ) -> list[ToolDenial]:
        """校验人工批准能否安全地下发到一个具体任务。

        人工批准只能补齐任务实例的工具白名单，不能突破任务类型风险预算、
        显式 forbidden_actions 或网络隔离红线。返回空列表才表示这些工具
        可以安全加入该任务；这使 HITL 成为受控授权，而不是安全旁路。
        """

        denials: list[ToolDenial] = []
        forbidden = set(forbidden_actions or [])
        for name in tool_names:
            spec = self.registry.get(name)
            if spec is None:
                denials.append(ToolDenial(name, "tool not registered"))
                continue
            if name in forbidden:
                denials.append(ToolDenial(name, "explicitly forbidden"))
                continue
            if spec.requires_network and {
                "network",
                "external_network",
                "unauthorized_external_network",
            }.intersection(forbidden):
                denials.append(ToolDenial(name, "network access explicitly forbidden"))
                continue
            if not risk_allowed(task_type, spec.risk_level, tool_name=name):
                denials.append(
                    ToolDenial(
                        name,
                        f"risk level {spec.risk_level.value} exceeds budget for "
                        f"task_type={task_type}",
                    )
                )
        return denials

    def default_allowed_tools_for(self, task_type: str) -> list[str]:
        """当任务定义没有显式指定 ``allowed_tools`` 时，按任务类型给出
        一组合理的默认只读工具集合（永远不含写入/高危工具，除非任务
        类型本身的风险预算本来就更高——即便如此这里也只给出"最小够用"
        的默认集合，具体的写入/执行类工具仍需 orchestrator 显式授予）。
        """

        return [
            spec.name
            for spec in self.registry.all_specs()
            if risk_allowed(task_type, spec.risk_level, tool_name=spec.name)
        ]
