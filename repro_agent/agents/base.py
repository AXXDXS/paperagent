"""子智能体基类（设计文档 §9 十个子智能体的公共骨架）。

核心约束（响应用户需求，也是本次改动的重点）：
    子智能体**只能**通过构造函数注入的 ``ToolAuthorization`` 调用
    工具，绝不导入 ``repro_agent.tools.registry``。为了在代码层面
    尽量"物理"杜绝违规，本基类：

    1. 构造函数只接受 ``ToolAuthorization``，不接受 ``ToolRegistry``
       或 ``SandboxContext``（沙箱上下文已经被授权对象封装在内部，
       子智能体代码拿不到裸的 ``SandboxContext``，因此也无法绕过
       ``ToolAuthorization.call`` 的权限检查直接操作文件系统）；
    2. 提供的 ``self.call_tool(name, **kwargs)`` 是子智能体唯一可用
       的"动手"入口，内部直接委托给 ``ToolAuthorization.call``，
       未授权工具调用会抛出 ``ToolPermissionError``；
    3. 也不持有 ``MemoryManager``/``MainAgentCapability``——正式记忆
       读取权限同样只属于主智能体（见 memory/manager.py），子智能体
       只能通过 ``write_task_output`` 工具产出"候选记忆"文件
       （§15.2），由主智能体事后决定是否转正。

每个具体子智能体（paper/code/resource/... 目录下）都继承
``BaseSubAgent`` 并实现 ``run()``，返回结构化的 ``AgentRunResult``，
交给 orchestrator 解析、写回任务状态。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterator, Optional

from repro_agent.domain.task import FailureReport, Task
from repro_agent.providers.base import (
    ContentBlock,
    LLMMessage,
    LLMProvider,
    LLMRequestParams,
    LLMResponse,
)
from repro_agent.providers.prompt_cache import (
    build_stable_system_prompt,
    canonicalize_tools,
    prompt_cache_key_for_tools,
)
from repro_agent.providers.retry import call_with_retry
from repro_agent.schemas.results import TaskResultEnvelope
from repro_agent.tools.authorization import ToolAuthorization
from repro_agent.tools.result_sanitization import MODEL_TOOL_RESULT_POLICY
from repro_agent.tools.base import ToolInputValidationError, ToolPermissionError
from repro_agent.tools.base import ToolExecutionError

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str, "Optional[float]"], None]
CheckpointReader = Callable[[str], dict[str, Any] | None]
CheckpointWriter = Callable[[str, dict[str, Any]], None]
"""子智能体主动上报的窄通道：``(progress, current_step, eta_seconds)``。

由 ``AgentDispatcher`` 在启动子智能体线程时注入（见
``orchestrator/dispatcher.py::SubAgentHandle``），子智能体自身不直接
持有调度器或任务仓库的引用——这与"子智能体只能通过
``ToolAuthorization`` 接触外部世界"的隔离原则是同一思路的延伸：
业务报备和底层活动都经过这条独立于工具调用的窄通道，只能把
``(进度, 当前步骤, 预计剩余秒数)`` 这三个值向外传递，不能反向获得
任何调度器状态。``current_step`` 以 ``activity:`` 开头时只作为底层
活动证据，不刷新动态报备截止时间。
"""


class CancellationRequested(RuntimeError):
    """主智能体已请求（优雅）取消当前任务，子智能体应尽快停止并返回。

    对应用户需求"子agent对优雅信号无响应时才强制终止"——
    ``BaseSubAgent.check_cancellation()`` 就是这里说的"优雅信号"：
    在关键检查点（每次工具调用/LLM 调用前后）主动探测取消标志，
    一旦发现，立即抛出本异常，由 ``run()`` 的调用方（子智能体线程
    包装器）捕获并标记为"已优雅终止"，不需要主智能体真正杀掉线程。
    """


@dataclass
class AgentRunResult:
    """子智能体一次运行的结构化结果，供 orchestrator 写回任务状态。"""

    succeeded: bool
    outputs: dict[str, Any] = field(default_factory=dict)
    candidate_memory_written: bool = False
    failure_report: Optional[FailureReport] = None
    raw_llm_responses: list[str] = field(default_factory=list)
    reusable_code_candidates: list[dict[str, Any]] = field(default_factory=list)


def task_checkpoint_scope_hash(task: Task) -> str:
    """为 task 的可恢复步骤生成稳定输入作用域。

    同一个 task 只有在任务类型、目标和已解析 inputs 都相同的情况下才
    能复用 checkpoint。这样恢复旧 Job 时可避免把不同任务配置下的 LLM
    或只读工具结果混用。
    """

    value = {
        "task_type": task.definition.task_type,
        "objective": task.definition.objective,
        "inputs": task.definition.inputs,
    }
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _llm_checkpoint_key(messages: list[LLMMessage], params: LLMRequestParams) -> str:
    """为一次无工具 LLM 请求生成 task 内稳定检查点键。"""

    value = {
        "messages": [
            {
                "role": item.role,
                "content": item.content,
                "tool_call_id": item.tool_call_id,
                "name": item.name,
                "tool_calls": [
                    {
                        "tool_name": call.tool_name,
                        "arguments": call.arguments,
                        "call_id": call.call_id,
                        "arguments_valid": call.arguments_valid,
                    }
                    for call in item.tool_calls
                ],
            }
            for item in messages
        ],
        "params": {
            "model": params.model,
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
            "tools": params.tools,
            "response_schema": params.response_schema,
            "response_schema_name": params.response_schema_name,
        },
    }
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return "llm:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _llm_response_to_checkpoint(response: LLMResponse) -> dict[str, Any]:
    return {
        "content": response.content,
        "tool_calls": [
            {
                "tool_name": call.tool_name,
                "arguments": call.arguments,
                "call_id": call.call_id,
                "arguments_valid": call.arguments_valid,
            }
            for call in response.tool_calls
        ],
        "finish_reason": response.finish_reason,
        "usage": response.usage,
    }


def _llm_response_from_checkpoint(value: dict[str, Any]) -> LLMResponse:
    from repro_agent.providers.base import ToolCallRequest

    return LLMResponse(
        content=str(value.get("content", "")),
        tool_calls=[
            ToolCallRequest(
                tool_name=str(call.get("tool_name", "")),
                arguments=dict(call.get("arguments", {})),
                call_id=str(call.get("call_id", "")),
                arguments_valid=bool(call.get("arguments_valid", True)),
            )
            for call in value.get("tool_calls", [])
            if isinstance(call, dict)
        ],
        finish_reason=str(value.get("finish_reason", "stop")),
        usage=dict(value.get("usage", {})),
    )


class BaseSubAgent:
    """所有子智能体的公共基类。

    ``task_type`` 类属性用于和 ``tools.authorization.TASK_TYPE_RISK_BUDGET``
    对齐——具体子类应该覆盖它，声明自己在设计文档 §9 中对应的任务类型。
    """

    task_type: str = "generic"
    system_prompt: str = "你是 ReproAgent 系统中的一个只读分析型子智能体。"

    def __init__(
        self,
        task: Task,
        tool_authorization: ToolAuthorization,
        llm_provider: LLMProvider,
        *,
        model: str = "gpt-4o-mini",
        max_tokens: int = 4096,
        max_llm_retries: int = 3,
        llm_timeout_seconds: float = 120.0,
        progress_callback: Optional[ProgressCallback] = None,
        cancellation_event: Optional[threading.Event] = None,
        attempt_id: str | None = None,
        checkpoint_reader: CheckpointReader | None = None,
        checkpoint_writer: CheckpointWriter | None = None,
    ):
        if task.definition.task_type != self.task_type and self.task_type != "generic":
            logger.warning(
                "task %s has task_type=%s but is being handled by agent for task_type=%s",
                task.task_id,
                task.definition.task_type,
                self.task_type,
            )
        self.task = task
        self._tools = tool_authorization
        self._llm = llm_provider
        self._model = model
        self._max_tokens = max_tokens
        self._max_llm_retries = max_llm_retries
        # 单次 LLM 调用的超时：长输入+长输出的任务（如整本论文参数
        # 提取）需要比默认更宽的超时，否则网络层会在模型还在生成时
        # 提前掐断连接。
        self._llm_timeout_seconds = llm_timeout_seconds
        # push 报备通道：由 SubAgentHandle 注入，子智能体主动调用
        # report_progress()；activity 前缀只表示底层活动。
        self._progress_callback = progress_callback
        # 优雅取消信号：由 SubAgentHandle 在判定需要终止时 set()，
        # 子智能体应在长耗时循环/多步骤 run() 中调用
        # check_cancellation() 主动探测。
        self._cancellation_event = cancellation_event
        # 固定在构造时的尝试标识，不能在写结果时再读取可能已被重试
        # 流程修改的 task.active_attempt_id。
        self._attempt_id = attempt_id or task.active_attempt_id or f"attempt_{task.attempt}"
        self._checkpoint_reader = checkpoint_reader
        self._checkpoint_writer = checkpoint_writer
        # Narrow, write-only reporting channel for code the child believes may
        # generalize.  The main agent treats these values as untrusted proposals;
        # children never receive the dynamic tool store or registry.
        self._reusable_code_candidates: list[dict[str, Any]] = []

    # ---- 业务报备与活动上报（见 orchestrator/dispatcher.py） ----

    def report_progress(
        self,
        progress: float,
        current_step: str,
        *,
        eta_seconds: Optional[float] = None,
    ) -> None:
        """子智能体主动向主智能体汇报当前进度与预计剩余时间。

        对应用户需求"子agent若达到主agent要求的时间没有完成，必须
        向主agent汇报状态，并汇报大概还需要多久完成"：子智能体应该
        在自己的 ``run()`` 实现中，每完成一个可识别的阶段性步骤就
        调用一次本方法（而不是等主智能体来问），这是"push"语义的
        具体体现——必须由子智能体先主动汇报，主智能体的
        ``get_subagent_status``（pull）只是汇报缺失时的兜底探测手段，
        不是常规的信息获取路径。

        ``progress`` 取值 ``[0.0, 1.0]``；``eta_seconds`` 是子智能体
        自己估计的"大概还需要多久完成"，允许为 ``None``（表示暂时
        无法估计），由主智能体在 UI/日志中原样展示，不做二次推断。
        """

        self.check_cancellation()
        if self._progress_callback is not None:
            try:
                self._progress_callback(progress, current_step, eta_seconds)
            except Exception:  # noqa: BLE001 - 上报失败不应打断任务本身
                logger.exception(
                    "task %s failed to push progress report", self.task.task_id
                )

    def report_activity(self, current_step: str) -> None:
        """发送一次底层活动信号，不续期业务报备合同。"""

        if self._progress_callback is None:
            return
        heartbeat = getattr(self.task, "heartbeat", None)
        progress = heartbeat.progress if heartbeat is not None else 0.0
        try:
            self._progress_callback(
                progress, f"activity:{current_step}", None
            )
        except Exception:  # noqa: BLE001 - 诊断信号失败不应打断任务
            logger.exception(
                "task %s failed to push activity signal", self.task.task_id
            )

    def check_cancellation(self) -> None:
        """探测主智能体是否已经发出（优雅）取消信号。

        子智能体基类在 ``call_tool``/``call_llm``/``report_progress``
        这些天然的"阶段性检查点"上都会调用本方法，因此绝大多数子
        智能体不需要显式调用它也能及时响应取消——只有那些会长时间
        阻塞在单次工具调用内部（例如未来接入真实训练脚本执行）的
        场景，才需要子智能体自己在循环体内额外调用。
        """

        if self._cancellation_event is not None and self._cancellation_event.is_set():
            raise CancellationRequested(
                f"task {self.task.task_id} received graceful cancellation signal"
            )

    # ---- 工具调用：唯一的"动手"入口 ----

    def call_tool(self, tool_name: str, /, **kwargs: Any) -> Any:
        """调用一个已被主智能体授权的工具；未授权则抛出 ``ToolPermissionError``。"""

        self.check_cancellation()
        try:
            with self._tool_activity(tool_name):
                result = self._tools.call(tool_name, **kwargs)
        except ToolPermissionError:
            logger.error(
                "task %s (%s) attempted to call unauthorized tool '%s'; granted=%s",
                self.task.task_id,
                self.task_type,
                tool_name,
                self._tools.granted_tool_names,
            )
            raise
        self.check_cancellation()
        return result

    @contextmanager
    def _tool_activity(self, tool_name: str) -> Iterator[None]:
        """Emit event-based tool activity without a fixed keepalive thread."""

        self.report_activity(f"tool_started:{tool_name}")
        try:
            yield
        finally:
            self.report_activity(f"tool_finished:{tool_name}")

    def call_tool_checkpointed(
        self, checkpoint_key: str, tool_name: str, /, **kwargs: Any
    ) -> Any:
        """执行并保存一个已明确声明为只读的工具步骤。

        该方法不会自动判断工具是否安全可重放；调用方只能用于无副作用
        工具。写文件、执行命令等操作必须在具体 Agent 中实现幂等恢复，
        不能借由通用 checkpoint 被静默跳过。
        """

        return self.checkpointed(
            f"tool:{checkpoint_key}", lambda: self.call_tool(tool_name, **kwargs)
        )

    def call_tool_for_model(self, tool_name: str, /, **kwargs: Any) -> dict[str, Any]:
        """Call a tool and return its validated, rendered model-facing envelope.

        Deterministic sub-agent stages sometimes gather context before the LLM
        call instead of letting the model issue the tool call itself.  Such
        results cross the same security boundary and therefore must follow the
        identical sequence used by the interactive tool loop: execute ->
        lossless JSON -> ``tool.output.schema`` -> ``output.render`` -> bounded
        and redacted ContentBlock.  Returning the envelope (rather than raw
        handler data) makes accidental raw insertion into a prompt harder.
        """

        result = self.call_tool(tool_name, **kwargs)
        block = self._tools.render_result_for_model(tool_name, result)
        if not isinstance(block.data, dict):
            raise ToolExecutionError("model-facing tool ContentBlock must contain an object")
        return block.data

    def checkpointed(self, checkpoint_key: str, operation: Callable[[], Any]) -> Any:
        """复用或持久化一个逻辑步骤的 JSON 结果。"""

        if self._checkpoint_reader is not None:
            saved = self._checkpoint_reader(checkpoint_key)
            if saved is not None and "value" in saved:
                return saved["value"]
        value = operation()
        if self._checkpoint_writer is None:
            return value
        try:
            # 检查点必须可序列化；归一化后返回，确保首次运行与恢复运行
            # 在值类型上保持一致。
            normalized = json.loads(json.dumps(value, ensure_ascii=False, default=str))
            self._checkpoint_writer(checkpoint_key, {"value": normalized})
            return normalized
        except Exception:  # noqa: BLE001 - 检查点写入故障不应遮蔽原操作
            logger.exception(
                "task %s failed to persist checkpoint %s",
                self.task.task_id,
                checkpoint_key,
            )
            return value

    @property
    def granted_tools(self) -> list[str]:
        """子智能体可以查询自己当前被授予了哪些工具（用于在 Prompt 中
        告知模型"你只能使用以下工具"），但无法反向获得工具本身的能力。
        """

        return self._tools.granted_tool_names

    # ---- LLM 调用封装（统一走 token 递减重试） ----

    def call_llm(
        self,
        user_prompt: str,
        *,
        extra_system_prompt: str = "",
        temperature: float = 0.3,
        tool_names: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        output_schema_name: str = "structured_output",
        max_tool_rounds: int | None = None,
    ) -> LLMResponse:
        """发起一次 LLM 调用。

        ``tool_names`` 是本次改动的核心参数：显式声明"这一次 LLM 调用
        真正需要哪些工具"，而不是像过去那样把 ``ToolAuthorization``
        授予本任务的**全部**工具（``describe_granted()`` 的旧默认行为）
        无差别地暴露给模型。

        对应用户需求"主 agent 只传入它认为分给子 agent 的任务需要的
        工具，而不是让子 agent 自己去判断需要哪些工具"——这里把"判断
        当前这一步该用哪些工具"的责任，从"扔一堆工具描述给 LLM 让它自
        己选"，收回到子智能体的 ``run()`` 实现代码里：每次调用
        ``call_llm`` 时由调用方（子智能体作者，代表主智能体对任务的
        理解）用 Python 代码显式写死这一步该给模型看哪些工具，工具选
        择变成确定性的代码逻辑而不是 LLM 的自由裁量。

        ``tool_names`` 默认为 ``None``，此时**不携带任何工具**
        （等价于 ``tool_names=[]``）——多数子智能体的多数 LLM 调用其实
        只是"计划/生成/分析文本"，本来就不需要模型自己再发起工具调用
        （例如 ``CodingAgent._plan_changes`` 只是要一段 JSON 变更计划，
        文件读写已经由 ``run()`` 中的确定性代码完成）。只有确实需要模型
        在对话中动态决定"要不要再读一个文件/再查一次"的步骤，才应该
        显式传入一个精确的工具名子集（必须是本任务已被
        ``ToolAuthorization`` 授权的工具，否则 ``describe_granted``
        会直接抛出 ``ToolPermissionError``，在联调阶段尽早暴露配置
        错误）。
        ``max_tool_rounds`` 对模型驱动的迭代检索设置硬上限。一次模型响应
        中的一组并行工具调用算一轮；达到上限后如果模型仍继续请求工具，
        调用会失败关闭，避免大仓库探索退化为无界循环。
        """

        requested_tools = canonicalize_tools(
            self._tools.describe_granted(tool_names or [])
        )
        system_suffixes = [self.system_prompt, extra_system_prompt]
        if tool_names:
            system_suffixes.append(
                "TOOL RESULT SECURITY POLICY:\n" + MODEL_TOOL_RESULT_POLICY
            )
        retry_guidance = str(
            self.task.definition.inputs.get("retry_guidance", "") or ""
        ).strip()
        effective_user_prompt = user_prompt
        if retry_guidance:
            effective_user_prompt = (
                "MAIN AGENT RETRY GUIDANCE (one instruction):\n"
                f"{retry_guidance}\n\n"
                "CURRENT TASK INPUT:\n"
                f"{user_prompt}"
            )

        messages = [
            LLMMessage(
                role="system",
                content=build_stable_system_prompt(*system_suffixes),
            ),
            LLMMessage(role="user", content=effective_user_prompt),
        ]
        params = LLMRequestParams(
            model=self._model,
            temperature=temperature,
            max_tokens=self._max_tokens,
            tools=requested_tools,
            timeout_seconds=self._llm_timeout_seconds,
            prompt_cache_key=prompt_cache_key_for_tools(requested_tools),
            response_schema=output_schema,
            response_schema_name=output_schema_name,
        )
        tool_rounds = 0
        while True:
            self.check_cancellation()
            checkpoint_key = _llm_checkpoint_key(messages, params)
            saved = (
                self._checkpoint_reader(checkpoint_key)
                if self._checkpoint_reader is not None
                else None
            )
            if saved is not None and "response" in saved:
                response = _llm_response_from_checkpoint(saved["response"])
            else:
                response = call_with_retry(
                    self._llm.complete,
                    messages,
                    params,
                    max_retries=self._max_llm_retries,
                )
                # 有 tool calls 的模型响应不能通用重放：恢复后可能再次
                # 执行写文件或命令。只有纯 LLM 响应会自动形成检查点。
                if not response.tool_calls and self._checkpoint_writer is not None:
                    self._checkpoint_writer(
                        checkpoint_key,
                        {"response": _llm_response_to_checkpoint(response)},
                    )
            self.check_cancellation()
            if not response.tool_calls:
                return response
            tool_rounds += 1
            exhausted_reason: str | None = None
            if max_tool_rounds is not None and tool_rounds > max(0, max_tool_rounds):
                exhausted_reason = (
                    f"model tool-call rounds exceeded limit ({max_tool_rounds})"
                )
            elif len(response.tool_calls) > self._tools.remaining_tool_calls:
                exhausted_reason = (
                    "model requested more tool calls than the task budget allows"
                )
            if exhausted_reason is not None:
                # 预算耗尽不再直接击穿整个 attempt：已经收集到的信息可能
                # 足够产出可用结果，重跑一次 attempt 的代价（重新索引、
                # 重新检索、全部 LLM 轮次）远高于一次强制收尾。这里显式
                # 告知模型工具预算已用完，并去掉工具描述再取最后一次
                # 回答——超过预算的工具调用一个都不会被执行，硬上限
                # 语义保持不变；仅当供应商在无工具请求下仍返回 tool_calls
                # （协议违约）时才回退 fatal。
                logger.warning(
                    "task %s (%s): %s; forcing a no-tool final turn",
                    self.task.task_id,
                    self.task_type,
                    exhausted_reason,
                )
                messages.append(
                    LLMMessage(
                        role="user",
                        content=(
                            "TOOL BUDGET EXHAUSTED: you have used all allowed "
                            "tool rounds/calls. Do not call any more tools. "
                            "Produce your final answer now using the "
                            "information already gathered; follow the output "
                            "JSON contract from the original instructions."
                        ),
                    )
                )
                final_params = replace(params, tools=None)
                final_response = call_with_retry(
                    self._llm.complete,
                    messages,
                    final_params,
                    max_retries=self._max_llm_retries,
                )
                if final_response.tool_calls:
                    raise ToolExecutionError(
                        f"{exhausted_reason}; no-tool final turn still returned "
                        "tool calls"
                    )
                if self._checkpoint_writer is not None:
                    self._checkpoint_writer(
                        _llm_checkpoint_key(messages, final_params),
                        {"response": _llm_response_to_checkpoint(final_response)},
                    )
                return final_response

            messages.append(
                LLMMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )
            for tool_call in response.tool_calls:
                self.check_cancellation()
                if not tool_call.arguments_valid or not isinstance(tool_call.arguments, dict):
                    raise ToolInputValidationError("tool arguments must be a JSON object")
                try:
                    result = self.call_tool(tool_call.tool_name, **tool_call.arguments)
                except ToolExecutionError as exc:
                    # 工具执行失败（file not found、参数越界等）不终止
                    # 任务：把错误文本作为 tool 结果回填，模型可以在剩余
                    # 轮次/预算内自纠（换路径、换参数、改用其它工具）。
                    # 错误已在 ToolAuthorization.call 中写入审计日志，这里
                    # 只改变它对模型 conversation 的可见性。
                    # 边界：ToolPermissionError（未授权/预算耗尽）不在捕获
                    # 范围，仍保持致命——权限边界不能靠模型自纠绕过。
                    # 轮次/预算超限的 raise 在本 try 之外，fatal 语义不变。
                    messages.append(
                        LLMMessage(
                            role="tool",
                            content=ContentBlock(
                                type="json",
                                data={
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "hint": "tool call failed; correct the arguments or use a different approach",
                                },
                            ).to_model_content(),
                            tool_call_id=tool_call.call_id,
                            name=tool_call.tool_name,
                        )
                    )
                    continue
                result_block = self._tools.render_result_for_model(
                    tool_call.tool_name, result
                )
                messages.append(
                    LLMMessage(
                        role="tool",
                        content=result_block.to_model_content(),
                        tool_call_id=tool_call.call_id,
                        name=tool_call.tool_name,
                    )
                )

    # ---- 输出落盘（统一走 write_task_output 工具，不直接碰文件系统）----

    def write_output(self, filename: str, content: str) -> None:
        self.call_tool("write_task_output", filename=filename, content=content)

    def write_json_output(self, filename: str, data: dict[str, Any]) -> None:
        payload: dict[str, Any] = data
        if filename == "result.json":
            payload = TaskResultEnvelope.succeeded(
                task_id=self.task.task_id,
                attempt_id=self._attempt_id,
                task_type=self.task.definition.task_type,
                payload=data,
            ).to_dict()
        self.write_output(filename, json.dumps(payload, indent=2, ensure_ascii=False))

    def write_candidate_memory(self, markdown: str) -> None:
        """写入候选记忆（§15.2：子智能体只能写候选记忆，不能直接写正式记忆）。"""

        self.write_output("candidate_memory.md", markdown)

    def report_reusable_code_candidate(self, candidate: dict[str, Any]) -> None:
        """Propose reusable code without registering or retaining it in context."""

        if not isinstance(candidate, dict):
            raise TypeError("reusable code candidate must be an object")
        self._reusable_code_candidates.append(dict(candidate))

    def persist_reusable_code_candidates(self) -> list[dict[str, Any]]:
        """Write proposals to a sidecar consumed only after output validation."""

        candidates = list(self._reusable_code_candidates)
        if candidates:
            self.write_output(
                "reusable_code_candidates.json",
                json.dumps(candidates, indent=2, ensure_ascii=False),
            )
        # Returning copies lets the dispatcher expose short-lived diagnostic
        # state while the durable source of truth is the sandbox sidecar.
        return candidates

    # ---- 子类必须实现 ----

    def run(self) -> AgentRunResult:  # pragma: no cover - 抽象方法
        raise NotImplementedError
