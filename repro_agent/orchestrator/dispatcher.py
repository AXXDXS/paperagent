"""子智能体派发器：沙箱创建 + 工具授权 + 子智能体运行的完整闭环。

这是本次改动响应用户需求的核心整合点。一次任务派发的完整链路：

    1. ``SandboxManager.create_sandbox``：为任务创建物理隔离目录，
       返回实现了 ``SandboxContext`` 协议的 ``TaskSandbox``；
    2. ``ToolAuthorizer.authorize``：依据任务定义的
       ``allowed_tools`` 白名单 + 任务类型的风险预算，从全局
       ``ToolRegistry`` 中筛出被授权的工具子集，绑定到上一步的
       沙箱上下文，产出 ``ToolAuthorization``；
    3. ``get_agent_class(task_type)``：查表得到对应的子智能体实现类；
    4. 用 ``ToolAuthorization``（而不是原始 registry/沙箱）构造子
       智能体实例——子智能体从此只持有这一个"能力句柄"，物理上
       不接触全局工具注册表、不接触裸的沙箱路径；
    5. 在**独立后台线程**中运行子智能体（``SubAgentHandle``），主线程
       立即返回句柄而不阻塞——这是响应用户需求"子智能体运行期间必须
       能被主智能体轮询/强制查询/必要时强制终止"的前提：如果还是
       同步阻塞调用，主智能体在子智能体运行期间根本没有机会去做
       "2 分钟未汇报则强制查询状态""判定死亡后强制终止"这些动作；
    6. 捕获子智能体线程内的异常（含 ``ToolPermissionError``，即子
       智能体代码试图调用未授权工具的情况），转换为标准的
       ``AgentRunResult``/``FailureReport``，写回任务状态；
    7. 记录本次任务的工具调用审计日志（``ToolAuthorization.invocation_log``）
       到任务事件表，供反思智能体和最终报告回溯。

``AgentDispatcher`` 是唯一允许同时接触
"全局工具注册表 + 沙箱管理器 + LLM Provider"的模块——即整个系统里
"高权限"和"子智能体运行时"之间的唯一网关，也是**唯一允许构造子
智能体实例的地方**：子智能体类（``agents/*/agent.py``）的构造函数
需要 ``ToolAuthorization``，而 ``ToolAuthorization`` 只能通过
``ToolAuthorizer.authorize`` 产出，``ToolAuthorizer`` 只被
``MainAgent`` 持有并传给本类——因此除了主智能体（经由本
dispatcher）之外，系统中没有任何其它代码路径能够实例化一个子智能体，
满足"只有主 Agent 能生成 subagent"的红线要求。
"""

from __future__ import annotations

import logging
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

from repro_agent.agents.base import (
    AgentRunResult,
    CancellationRequested,
    task_checkpoint_scope_hash,
)
from repro_agent.agents.registry import get_agent_class
from repro_agent.llm_output import StructuredOutputError
from repro_agent.domain.common import utc_now
from repro_agent.domain.enums import FailureType, TaskStatus, ToolGrantDecision
from repro_agent.domain.task import FailureReport, Heartbeat, Task
from repro_agent.orchestrator.tool_grant import ToolGrantOutcome
from repro_agent.providers.base import LLMProvider
from repro_agent.sandbox.manager import SandboxManager
from repro_agent.storage.repository import (
    TaskCheckpointRepository,
    TaskRepository,
    ToolInvocationRepository,
)
from repro_agent.tools.authorization import ToolAuthorizer
from repro_agent.tools.base import (
    InvalidToolOutputError,
    ToolExecutionError,
    ToolGrantDeniedError,
    ToolInputValidationError,
    ToolPermissionError,
)
from repro_agent.tools.destructive_actions import (
    DestructiveActionConfirmationRequired,
)

logger = logging.getLogger(__name__)


def _bounded_text(text: str, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


@dataclass
class DispatchOutcome:
    task_id: str
    result: AgentRunResult
    tool_calls_made: int
    denied_tools_requested: list[str]


@dataclass
class ToolEscalationRequest:
    """一次运行期缺工具升级请求的登记项。

    生命周期：子智能体线程在 ``_handle_tool_escalation`` 里创建并登记
    它，裁决（主智能体 GRANT/DENY，或人工批准/拒绝）到达后通过
    ``resolved`` 事件唤醒；等待期间子智能体线程的完整运行时状态
    （LLM 对话上下文、中间变量、已写文件）都原封不动地保留在
    线程栈上——这就是"追加工具后不重启"的物质基础。
    """

    task_id: str
    attempt_id: str
    tool_name: str
    rationale: str = ""
    resolved: threading.Event = field(default_factory=threading.Event)
    granted: bool = False
    outcome: Optional[ToolGrantOutcome] = None
    # 关联的人工介入请求 id（仅 ASK_USER 分支会填充）。
    request_id: Optional[str] = None
    # 供主智能体线程注入裁决时使用的闭包引用（不参与序列化）。
    task: Optional[Task] = None
    authorization: Optional[object] = None


class SubAgentHandle:
    """一次子智能体运行的句柄：封装后台线程 + push/pull 双通道状态。

    生命周期（对应用户需求描述的"拉取(pull) / 推送(push)"双机制）：

        1. ``start()``：在守护线程中运行 ``agent.run()``；
        2. 运行期间，子智能体通过 ``BaseSubAgent.report_progress()``
           调用本句柄的 ``_on_progress_push`` 回调——这是 **push**
           通道，必须由子智能体主动发起，本句柄从不替它编造心跳；
        3. 主智能体可以随时调用 ``pull_status()`` 主动查询当前状态
           （**pull** 通道）：正常情况下这只是"顺手看一眼"，只有当
           ``LivenessPolicy`` 判定"超过宽限期没有 push"时，主循环
           才会把这次 pull 的结果当作是否存活的关键证据；
        4. ``request_graceful_cancel()``：设置取消信号，子智能体在
           下一个检查点（``check_cancellation()``）会自行抛出
           ``CancellationRequested`` 并退出线程——这是"优雅终止"；
        5. ``force_kill()``：优雅信号在宽限期内无响应时的兜底——
           Python 线程无法被外部真正抢占式杀死，这里如实地把这个
           限制体现在实现里：将线程标记为 ``daemon=True``（进程退出
           时不会阻塞主进程退出）、放弃对它的 ``join()`` 等待、把
           任务标记为"强制终止，可能有悬挂资源/未完成写入"，不假装
           能做到操作系统级别的强杀。这与用户描述的"仅在子 Agent
           对优雅信号无响应时使用，代价是可能遗留悬挂资源与未完成
           写入"完全一致——本实现如实呈现这个代价，而不是掩盖它。
    """

    def __init__(
        self,
        task: Task,
        run_fn,
        *,
        attempt_id: str,
        on_progress_push,
    ):
        self.task = task
        self.attempt_id = attempt_id
        self._run_fn = run_fn
        self._on_progress_push = on_progress_push
        self._cancellation_event = threading.Event()
        self._result_lock = threading.Lock()
        self._result: Optional[AgentRunResult] = None
        self._exception: Optional[BaseException] = None
        self._started_at = utc_now()
        self._finished_at: Optional[float] = None
        self._forced_killed = False
        self._gracefully_cancelled = False
        self._thread = threading.Thread(
            target=self._run_wrapper,
            name=f"subagent-{task.task_id}",
            daemon=True,
        )

    @property
    def cancellation_event(self) -> threading.Event:
        return self._cancellation_event

    def start(self) -> None:
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def is_finished(self) -> bool:
        if self._thread.is_alive():
            return False
        with self._result_lock:
            return self._result is not None or self._exception is not None

    def _run_wrapper(self) -> None:
        try:
            result = self._run_fn()
            with self._result_lock:
                self._result = result
        except CancellationRequested as exc:
            logger.info("task %s sub-agent thread exited via graceful cancellation: %s", self.task.task_id, exc)
            with self._result_lock:
                self._gracefully_cancelled = True
                self._result = AgentRunResult(
                    succeeded=False,
                    failure_report=FailureReport(
                        failure_type=FailureType.AGENT_STALLED,
                        failed_step="graceful_cancellation",
                        error_message=str(exc),
                        likely_causes=["主智能体判定超时/需要重规划，发出了优雅取消信号"],
                        recommended_action="按照 replanner 的分类结果重试或拆分",
                    ),
                )
        except BaseException as exc:  # noqa: BLE001 - 线程内必须兜底，否则异常静默丢失
            logger.exception("task %s sub-agent thread raised unhandled exception", self.task.task_id)
            with self._result_lock:
                self._exception = exc

    def push_progress(self, progress: float, current_step: str, eta_seconds: Optional[float]) -> None:
        """由 ``BaseSubAgent.report_progress()`` 触发的 push 回调实现。"""

        if self.task.active_attempt_id != self.attempt_id:
            return
        self._on_progress_push(self.task, progress, current_step, eta_seconds)

    def pull_status(self) -> Heartbeat:
        """主智能体主动查询（pull）子智能体当前状态的实现。

        本实现是单进程内的线程模型，"查询"即"读取线程是否存活 +
        读取任务当前已知的最新心跳快照"；如果子智能体运行在独立
        进程/沙箱容器中（真实生产部署形态），这里应该替换为一次
        真正的 IPC/RPC 调用，但对上层（``MainAgent``）暴露的接口
        签名不变，因此替换执行后端不需要改动主循环代码。
        """

        alive = self.is_alive()
        current = self.task.heartbeat
        progress = current.progress if current else 0.0
        current_step = current.current_step if current else "unknown"
        eta = current.eta_seconds if current else None
        return Heartbeat(
            progress=progress,
            current_step=f"{current_step} ({'alive' if alive else 'not_alive'})",
            last_completed_step=current.last_completed_step if current else "",
            last_log_position=current.last_log_position if current else 0,
            updated_at=utc_now(),
            eta_seconds=eta,
            reported_by="pull",
        )

    def request_graceful_cancel(self) -> None:
        self._cancellation_event.set()

    def force_kill(self) -> None:
        """强制终止的兜底实现：放弃等待线程退出，如实标记"forced"。

        不调用 ``self._thread.join()``——线程可能永远不会响应（例如
        卡在一次没有超时保护的第三方阻塞调用里），继续等待只会让
        主智能体自己也被拖死。放弃等待意味着该线程可能在后台继续
        残留一段时间直到进程退出，这正是用户描述的"可能遗留悬挂
        资源与未完成写入"的代价，本实现选择诚实地承担这个代价，
        而不是伪造一个"已终止"的假象。
        """

        self._forced_killed = True
        self._cancellation_event.set()

    @property
    def forced_killed(self) -> bool:
        return self._forced_killed

    @property
    def gracefully_cancelled(self) -> bool:
        return self._gracefully_cancelled

    def collect_result(self) -> AgentRunResult:
        """线程已结束后调用，取出最终结果；线程仍在跑则抛异常。"""

        with self._result_lock:
            if self._exception is not None:
                raise self._exception
            if self._result is None:
                raise RuntimeError(f"task {self.task.task_id} sub-agent thread has not produced a result yet")
            return self._result


class AgentDispatcher:
    """把 READY→DISPATCHED 的任务真正运行起来的执行网关。

    调度器（``scheduler.TaskScheduler``）只负责"选出该跑哪个任务"，
    不知道如何运行；``AgentDispatcher`` 负责"怎么跑"，两者职责分离，
    调度逻辑可以完全脱离 LLM/沙箱进行单元测试。
    """

    def __init__(
        self,
        sandbox_manager: SandboxManager,
        tool_authorizer: ToolAuthorizer,
        llm_provider: LLMProvider,
        task_repo: TaskRepository,
        *,
        model: str = "gpt-4o-mini",
        max_tokens: int = 32768,
        llm_timeout_seconds: float = 600.0,
        on_progress_push=None,
        tool_grant_decision_maker=None,
        request_human_tool_grant=None,
        escalation_wait_poll_seconds: float = 0.2,
    ):
        self.sandbox_manager = sandbox_manager
        self.tool_authorizer = tool_authorizer
        self.llm_provider = llm_provider
        self.task_repo = task_repo
        self.checkpoint_repo = TaskCheckpointRepository(task_repo.db)
        self.tool_invocation_repo = ToolInvocationRepository(task_repo.db)
        self.model = model
        # 子智能体单次 LLM 调用的输出上限/超时：推理系模型的思考 token
        # 计入 max_tokens，必须从配置层透传，避免各调用点硬编码小额度。
        self.max_tokens = max_tokens
        self.llm_timeout_seconds = llm_timeout_seconds
        # push 心跳落库回调：默认写 task.heartbeat（内存态）+ 落库，
        # 由 MainAgent 在构造本类时注入真正的 scheduler.report_heartbeat，
        # 这里给一个保底默认实现，保证单测/独立使用 dispatcher 时
        # push 通道也不会因为回调缺失而报错。
        self._on_progress_push = on_progress_push or self._default_progress_push
        self._handles: dict[str, SubAgentHandle] = {}
        # ---- 运行期缺工具升级通道（工具分配权上收后新增） ----
        # 主智能体的补授裁决器；为 None 时升级通道关闭，子智能体调用
        # 未授权工具的行为与改造前完全一致（直接抛 ToolPermissionError）。
        self.tool_grant_decision_maker = tool_grant_decision_maker
        # 人工介入请求创建回调：(task, tool_name, reason) -> request_id，
        # 由 MainAgent 注入（封装 InterventionService）。返回 None 表示
        # 无法创建（如已有其他介入挂起），升级按拒绝处理（fail closed）。
        self.request_human_tool_grant = request_human_tool_grant
        # 挂起等待人工裁决时的轮询间隔：既用于响应优雅取消信号，也让
        # 等待期间能周期性推心跳，避免被存活检测误判。
        self.escalation_wait_poll_seconds = escalation_wait_poll_seconds
        self._pending_escalations: dict[str, ToolEscalationRequest] = {}
        self._escalations_lock = threading.Lock()

    def _default_progress_push(self, task: Task, progress: float, current_step: str, eta_seconds) -> None:
        task.heartbeat = Heartbeat(
            progress=progress,
            current_step=current_step,
            last_completed_step=task.heartbeat.last_completed_step if task.heartbeat else "",
            updated_at=utc_now(),
            eta_seconds=eta_seconds,
            reported_by="push",
        )

    # ---- 异步派发：唯一真正构造子智能体实例、启动其执行线程的入口 ----

    def start_async(
        self, task: Task, *, resolved_inputs: dict | None = None
    ) -> SubAgentHandle:
        """在后台线程中启动子智能体运行，立即返回句柄，不阻塞调用方。

        这是"只有主 Agent 能生成 subagent"的具体落地：本方法内部
        才会调用 ``agent_cls(...)`` 构造子智能体实例，而本方法只应
        该被 ``MainAgent``（通过其持有的唯一 ``AgentDispatcher``
        实例）调用——参见类文档顶部关于唯一网关的说明。
        """

        attempt_id = task.active_attempt_id or f"attempt_{task.attempt}"
        agent_task = deepcopy(task)
        if resolved_inputs is not None:
            agent_task.definition.inputs = deepcopy(resolved_inputs)
        sandbox = self.sandbox_manager.get(task.task_id)
        if (
            sandbox is None
            or sandbox.attempt_id != attempt_id
            or resolved_inputs is not None
        ):
            sandbox = self.sandbox_manager.create_sandbox(agent_task)

        checkpoint_scope = task_checkpoint_scope_hash(agent_task)
        authorization = self.tool_authorizer.authorize(
            task_id=task.task_id,
            task_type=task.definition.task_type,
            allowed_tools=task.definition.allowed_tools
            or self.tool_authorizer.default_allowed_tools_for(task.definition.task_type),
            forbidden_actions=task.definition.forbidden_actions,
            sandbox_ctx=sandbox,
            attempt_id=attempt_id,
            invocation_recorder=lambda log: self.tool_invocation_repo.record(
                task.job_id, log
            ),
        )

        if authorization.denials:
            logger.warning(
                "task %s (%s): %d tool(s) denied at authorization time: %s",
                task.task_id,
                task.definition.task_type,
                len(authorization.denials),
                [d.tool_name for d in authorization.denials],
            )
            self.task_repo.record_event(
                task.job_id,
                task.task_id,
                "tool_authorization_denials",
                {"denials": [{"tool_name": d.tool_name, "reason": d.reason} for d in authorization.denials]},
            )

        handle = SubAgentHandle(
            task,
            lambda: None,
            attempt_id=attempt_id,
            on_progress_push=self._on_progress_push,
        )
        sandbox.cancellation_event = handle.cancellation_event

        # 注入运行期缺工具升级通道：授权对象从此具备"遇到已注册但未
        # 分配的工具时，先升级给主智能体裁决"的能力。未配置裁决器时
        # 不注入，行为与改造前完全一致。
        if self.tool_grant_decision_maker is not None:
            authorization.set_escalation_handler(
                lambda tool_name, arguments: self._handle_tool_escalation(
                    task, authorization, handle, sandbox, tool_name, arguments
                )
            )

        try:
            agent_cls = get_agent_class(task.definition.task_type)
        except ValueError as exc:
            failure = FailureReport(
                failure_type=FailureType.INVALID_OUTPUT,
                failed_step="resolve_agent_class",
                error_message=str(exc),
                likely_causes=["任务类型拼写错误或尚未实现对应子智能体"],
                recommended_action="检查任务定义的 task_type 字段",
            )
            handle._result = AgentRunResult(succeeded=False, failure_report=failure)  # noqa: SLF001
            # Keep the immediately-finished handle discoverable so the normal
            # main-loop collection path can convert the failure into task state.
            self._handles[task.task_id] = handle
            return handle

        agent = agent_cls(
            agent_task,
            authorization,
            self.llm_provider,
            model=self.model,
            max_tokens=self.max_tokens,
            llm_timeout_seconds=self.llm_timeout_seconds,
            progress_callback=handle.push_progress,
            cancellation_event=handle.cancellation_event,
            attempt_id=attempt_id,
            checkpoint_reader=lambda key: self.checkpoint_repo.get(
                task_id=task.task_id,
                checkpoint_key=key,
                scope_hash=checkpoint_scope,
            ),
            checkpoint_writer=lambda key, payload: self.checkpoint_repo.save(
                task_id=task.task_id,
                checkpoint_key=key,
                scope_hash=checkpoint_scope,
                attempt_id=attempt_id,
                payload=payload,
            ),
        )

        def _run() -> AgentRunResult:
            # 自动的首尾 push 心跳：保证即便具体子智能体实现没有在
            # run() 内部显式调用 report_progress()，也至少有"开始"和
            # "结束"两次由子智能体自己（通过其 report_progress 方法）
            # 发出的汇报，满足"必须保证一定是子智能体先主动汇报"的
            # 约束——这两次调用是 agent 实例自己触发的，不是主智能体
            # 代替它汇报，只是把"至少汇报一次"的兜底责任下沉到基类，
            # 不强制要求每个具体子智能体都手写汇报语句。子智能体如果
            # 想要更细粒度的中途进度，可以在自己的 run() 里多次调用
            # self.report_progress(...)，与这里的首尾汇报并不冲突。
            agent.report_progress(0.0, "started")
            try:
                result = agent.run()
                if result.succeeded:
                    result.reusable_code_candidates = (
                        agent.persist_reusable_code_candidates()
                    )
                agent.report_progress(
                    1.0 if result.succeeded else 0.0,
                    "completed" if result.succeeded else "failed",
                )
            except DestructiveActionConfirmationRequired as exc:
                logger.warning(
                    "task %s paused before destructive command %s",
                    task.task_id,
                    exc.fingerprint,
                )
                result = AgentRunResult(
                    succeeded=False,
                    failure_report=FailureReport(
                        failure_type=FailureType.PERMISSION_ERROR,
                        failed_step="destructive_action_confirmation",
                        error_message=str(exc),
                        likely_causes=["命令包含删除操作，执行前必须取得人工确认"],
                        recommended_action="展示精确命令与风险，等待用户明确批准或拒绝",
                        metadata={
                            "response_mode": "destructive_action",
                            "command": exc.command,
                            "command_fingerprint": exc.fingerprint,
                            "detection_reasons": exc.reasons,
                        },
                    ),
                )
            except ToolPermissionError as exc:
                logger.error("task %s: unauthorized tool call attempted: %s", task.task_id, exc)
                # 主智能体（或人工）已明确裁决拒绝的请求会以
                # ToolGrantDeniedError 到达：在失败报告里带上裁决标记，
                # 避免主循环对同一请求反复升级、重复弹人工请求。
                grant_metadata: dict = {}
                if isinstance(exc, ToolGrantDeniedError):
                    grant_metadata = {
                        "tool_grant_adjudicated": True,
                        "tool_grant_tool": exc.tool_name,
                        "tool_grant_reason": exc.grant_reason,
                    }
                result = AgentRunResult(
                    succeeded=False,
                    failure_report=FailureReport(
                        failure_type=FailureType.PERMISSION_ERROR,
                        failed_step="tool_call",
                        error_message=str(exc),
                        likely_causes=["子智能体尝试调用未被主智能体授权的工具"],
                        recommended_action="检查任务定义 allowed_tools 是否遗漏必要工具，"
                        "或子智能体逻辑是否越权",
                        metadata=grant_metadata,
                    ),
                )
            except ToolInputValidationError as exc:
                logger.warning("task %s: invalid model tool arguments: %s", task.task_id, exc)
                result = AgentRunResult(
                    succeeded=False,
                    failure_report=FailureReport(
                        failure_type=FailureType.INVALID_OUTPUT,
                        failed_step="tool_argument_validation",
                        error_message=str(exc),
                        likely_causes=["模型生成的工具参数不符合 JSON Schema 或资源边界"],
                        recommended_action="保留授权边界并让模型修正参数后重试",
                    ),
                )
            except InvalidToolOutputError as exc:
                logger.warning("task %s: invalid tool output: %s", task.task_id, exc)
                result = AgentRunResult(
                    succeeded=False,
                    failure_report=FailureReport(
                        failure_type=FailureType.INVALID_OUTPUT,
                        failed_step="tool_output_validation",
                        error_message=str(exc),
                        likely_causes=[
                            "工具返回值无法无损表示为 JSON、未通过 output schema，或 renderer 未生成合法 ContentBlock"
                        ],
                        recommended_action="修复工具返回契约或 renderer 后重试；不要把未经校验的结果交给模型",
                    ),
                )
            except StructuredOutputError as exc:
                logger.warning("task %s: invalid structured LLM output: %s", task.task_id, exc)
                result = AgentRunResult(
                    succeeded=False,
                    failure_report=FailureReport(
                        failure_type=FailureType.PARSING_ERROR,
                        failed_step="structured_llm_output_validation",
                        error_message=str(exc),
                        likely_causes=["模型输出无法通过 JSON 修复或字段 Schema 校验"],
                        recommended_action="重试当前任务或更换支持结构化输出的模型",
                    ),
                )
            except ToolExecutionError as exc:
                result = AgentRunResult(
                    succeeded=False,
                    failure_report=FailureReport(
                        failure_type=FailureType.TOOL_ERROR,
                        failed_step="tool_call",
                        error_message=str(exc),
                        likely_causes=["工具调用参数不合法或运行时环境异常"],
                        recommended_action="检查工具调用参数，必要时重试",
                    ),
                )
            finally:
                self._record_tool_invocations(task, authorization)
            return result

        handle._run_fn = _run  # noqa: SLF001 - 闭包需要访问上面刚构造的 authorization/agent
        handle.start()
        self._handles[task.task_id] = handle
        return handle

    def get_handle(self, task_id: str) -> Optional[SubAgentHandle]:
        return self._handles.get(task_id)

    def discard_handle(self, task_id: str, *, attempt_id: str | None = None) -> None:
        """任务已经过验证并被主智能体最终确认关闭后，释放句柄引用。

        只应该在 ``MainAgent`` 确认任务输出验证通过（或已按失败流程
        处理完毕）之后调用——呼应用户需求"主agent对子agent返回的结果
        必须要验证，验证通过后，才可以关闭子agent"：句柄本身在验证
        完成前一直保留，方便中途还需要再次 ``pull_status``。
        """

        handle = self._handles.get(task_id)
        if handle is not None and (attempt_id is None or handle.attempt_id == attempt_id):
            self._handles.pop(task_id, None)

    def dispatch_and_run(self, task: Task) -> DispatchOutcome:
        """同步派发的兼容封装：内部调用 ``start_async`` 后原地等待线程
        结束再取结果。保留这个方法是为了不破坏现有直接同步调用
        dispatcher 的测试/脚本；``MainAgent`` 正常主循环走的是
        ``start_async`` + 轮询，不会阻塞在这里。
        """

        handle = self.start_async(task)
        handle._thread.join()  # noqa: SLF001 - 同步兼容路径，明确需要等待完成
        result = handle.collect_result()
        self.discard_handle(task.task_id)
        return DispatchOutcome(
            task_id=task.task_id,
            result=result,
            tool_calls_made=0,
            denied_tools_requested=[],
        )

    # ---- 运行期缺工具升级通道（工具分配权上收后新增） ----

    def _handle_tool_escalation(
        self,
        task: Task,
        authorization,
        handle: SubAgentHandle,
        sandbox,
        tool_name: str,
        arguments_summary: dict | None,
    ) -> None:
        """缺工具升级请求的裁决入口（在**子智能体线程**内执行）。

        裁决流程（对应需求"子 agent 没有分配到已存在的工具时交给主
        agent 处理，主 agent 判断能否补授，拿不定主意再申请人工介入"）：
            1. 主智能体裁决器给出 GRANT / DENY / ASK_USER；
            2. GRANT → 增量授权并入当前 ``ToolAuthorization``，调用方
               （``ToolAuthorization.call``）直接继续执行原工具，
               子智能体**原地继续、不重启**；
            3. DENY → 抛出 ``ToolGrantDeniedError``（携带裁决理由），
               子智能体线程按权限错误失败，失败报告会带上
               ``tool_grant_adjudicated`` 标记防止反复升级；
            4. ASK_USER → 创建人工介入请求后在本线程挂起等待，
               任务状态切到 WAITING_FOR_PERMISSION（存活检测/超时
               检查都不再触碰它），人工批准后由主智能体调用
               ``resume_escalation`` 唤醒本线程原地继续。
        """

        import json as _json

        rationale = (
            f"子智能体在执行任务时尝试调用工具 '{tool_name}'"
            f"（调用参数摘要: "
            f"{_bounded_text(_json.dumps(arguments_summary or {}, ensure_ascii=False, default=str), 300)}）"
        )

        # 让主循环知道线程正在等裁决而不是卡死：推一条 push 心跳。
        self._push_escalation_heartbeat(handle, task, f"waiting_for_tool_grant:{tool_name}")
        self.task_repo.record_event(
            task.job_id,
            task.task_id,
            "tool_grant_escalated",
            {
                "attempt_id": handle.attempt_id,
                "tool_name": tool_name,
                "rationale": rationale,
            },
        )

        maker = self.tool_grant_decision_maker
        if maker is None:
            # 升级通道未启用（旧配置/单测），保持改造前行为。
            raise ToolPermissionError(
                f"task {task.task_id} is not authorized to call tool '{tool_name}'"
            )

        outcome = maker.adjudicate(
            task_id=task.task_id,
            task_type=task.definition.task_type,
            objective=task.definition.objective,
            inputs=task.definition.inputs,
            allowed_tools=authorization.granted_tool_names,
            forbidden_actions=task.definition.forbidden_actions,
            tool_name=tool_name,
            rationale=rationale,
        )
        self.task_repo.record_event(
            task.job_id,
            task.task_id,
            "tool_grant_adjudicated",
            {
                "attempt_id": handle.attempt_id,
                "tool_name": tool_name,
                "decision": outcome.decision.value,
                "reason": outcome.reason,
                "source": outcome.source,
            },
        )

        if outcome.decision == ToolGrantDecision.GRANT:
            if not self._grant_tool(task, authorization, sandbox, tool_name):
                raise ToolGrantDeniedError(
                    tool_name,
                    "增量授权未通过 ToolAuthorizer 硬校验（不应发生，fail closed）",
                )
            return

        if outcome.decision == ToolGrantDecision.DENY:
            raise ToolGrantDeniedError(tool_name, outcome.reason)

        # ---- ASK_USER：登记挂起请求，创建人工介入，阻塞等待裁决 ----
        escalation = ToolEscalationRequest(
            task_id=task.task_id,
            attempt_id=handle.attempt_id,
            tool_name=tool_name,
            rationale=rationale,
            task=task,
            authorization=authorization,
        )
        with self._escalations_lock:
            self._pending_escalations[task.task_id] = escalation

        request_id = None
        if self.request_human_tool_grant is not None:
            try:
                request_id = self.request_human_tool_grant(task, tool_name, outcome.reason)
            except Exception:  # noqa: BLE001 - 创建人工请求失败不应静默吞掉升级
                logger.exception(
                    "task %s: failed to create human intervention for tool grant "
                    "request '%s'",
                    task.task_id,
                    tool_name,
                )
                request_id = None
        escalation.request_id = request_id

        if request_id is None:
            # 无法创建人工请求（如已有其他介入挂起）：fail closed，
            # 按主智能体拒绝处理，绝不能无限挂起子智能体线程。
            with self._escalations_lock:
                self._pending_escalations.pop(task.task_id, None)
            raise ToolGrantDeniedError(
                tool_name,
                "无法创建人工审批请求（可能已有其他人工介入在进行），本次补授按拒绝处理",
            )

        try:
            while not escalation.resolved.wait(timeout=self.escalation_wait_poll_seconds):
                # 等待期间响应主智能体的优雅取消信号：被终止的任务不能
                # 继续挂在人工审批上。
                if handle.cancellation_event.is_set():
                    raise CancellationRequested(
                        f"task {task.task_id} cancelled while waiting for "
                        f"tool grant approval of '{tool_name}'"
                    )
                self._push_escalation_heartbeat(
                    handle, task, f"waiting_for_user_tool_approval:{tool_name}"
                )
        finally:
            with self._escalations_lock:
                self._pending_escalations.pop(task.task_id, None)

        if escalation.granted:
            # 人工批准的工具已由 resume_escalation 并入授权对象。
            return
        reason = escalation.outcome.reason if escalation.outcome else "人工拒绝了补授请求"
        raise ToolGrantDeniedError(tool_name, reason)

    def _grant_tool(
        self,
        task: Task,
        authorization,
        sandbox,
        tool_name: str,
    ) -> bool:
        """对单个工具执行增量授权并并入当前 ``ToolAuthorization``。

        补授 spec 仍从 ``ToolAuthorizer.authorize`` 的完整校验（注册表 /
        风险预算 / forbidden_actions）产出，主智能体的裁量没有任何
        绕过硬边界的路径；同时把工具名持久化进任务定义的
        ``allowed_tools``，保证重试/恢复后授权仍然生效。
        """

        incremental = self.tool_authorizer.authorize(
            task_id=task.task_id,
            task_type=task.definition.task_type,
            allowed_tools=[tool_name],
            forbidden_actions=task.definition.forbidden_actions,
            sandbox_ctx=sandbox,
            attempt_id=authorization.attempt_id,
            invocation_recorder=lambda log: self.tool_invocation_repo.record(
                task.job_id, log
            ),
        )
        if incremental.denials or not incremental.granted_tool_names:
            logger.error(
                "task %s: incremental authorization for tool '%s' was denied: %s",
                task.task_id,
                tool_name,
                [d.reason for d in incremental.denials],
            )
            return False
        for spec_name in incremental.granted_tool_names:
            spec = incremental.get_granted_spec(spec_name)
            if spec is not None:
                authorization.grant_additional(spec)
        if tool_name not in task.definition.allowed_tools:
            task.definition.allowed_tools.append(tool_name)
            self.task_repo.save(task)
        self.task_repo.record_event(
            task.job_id,
            task.task_id,
            "tool_granted_at_runtime",
            {"tool_name": tool_name, "attempt_id": authorization.attempt_id},
        )
        return True

    def _push_escalation_heartbeat(self, handle: SubAgentHandle, task: Task, step: str) -> None:
        """升级等待期间的心跳：证明线程活着且在等裁决。"""

        try:
            current = task.heartbeat
            handle.push_progress(
                current.progress if current else 0.0,
                step,
                None,
            )
        except Exception:  # noqa: BLE001 - 心跳失败不能阻断裁决流程
            logger.exception(
                "task %s failed to push escalation heartbeat", task.task_id
            )

    def resume_escalation(
        self,
        task_id: str,
        *,
        approved_tools: list[str],
        reason: str = "",
    ) -> bool:
        """人工批准后唤醒挂起中的升级请求，子智能体原地继续。

        由 MainAgent 在人工介入 resolve 之后调用：若批准的工具正是
        挂起请求的工具，则执行增量授权并把任务状态从
        WAITING_FOR_PERMISSION 恢复为 RUNNING，再唤醒子智能体线程——
        线程从阻塞点继续执行原工具调用，运行时状态零丢失。批准的
        工具不匹配（或未被批准）时同样唤醒，但携带拒绝裁决。
        """

        with self._escalations_lock:
            escalation = self._pending_escalations.get(task_id)
        if escalation is None:
            return False

        if (
            escalation.tool_name in approved_tools
            and escalation.task is not None
            and escalation.authorization is not None
        ):
            sandbox = self.sandbox_manager.get(task_id)
            granted = False
            if sandbox is not None:
                granted = self._grant_tool(
                    escalation.task, escalation.authorization, sandbox, escalation.tool_name
                )
            if granted:
                escalation.granted = True
                escalation.outcome = ToolGrantOutcome(
                    decision=ToolGrantDecision.GRANT,
                    tool_name=escalation.tool_name,
                    reason=reason or "人工批准补授",
                    source="human",
                )
                # 任务状态恢复 RUNNING：线程即将从阻塞点继续执行，
                # 主循环的 liveness/timeout/结果收集都会重新接管它。
                escalation.task.status = TaskStatus.RUNNING
                self.task_repo.save(escalation.task)
                self.task_repo.record_event(
                    escalation.task.job_id,
                    task_id,
                    "tool_grant_resumed",
                    {
                        "tool_name": escalation.tool_name,
                        "request_id": escalation.request_id,
                        "reason": reason,
                    },
                )
        if not escalation.granted:
            escalation.outcome = ToolGrantOutcome(
                decision=ToolGrantDecision.DENY,
                tool_name=escalation.tool_name,
                reason=reason or "人工拒绝了补授请求",
                source="human",
            )
        escalation.resolved.set()
        return True

    def fail_escalation(self, task_id: str, reason: str = "") -> bool:
        """以拒绝裁决唤醒挂起中的升级请求（介入被拒绝/超时/过期时）。"""

        with self._escalations_lock:
            escalation = self._pending_escalations.get(task_id)
        if escalation is None:
            return False
        escalation.granted = False
        escalation.outcome = ToolGrantOutcome(
            decision=ToolGrantDecision.DENY,
            tool_name=escalation.tool_name,
            reason=reason or "人工介入被拒绝或已过期",
            source="human",
        )
        escalation.resolved.set()
        return True

    def get_pending_escalation(self, task_id: str) -> Optional[ToolEscalationRequest]:
        with self._escalations_lock:
            return self._pending_escalations.get(task_id)

    def get_pending_escalation_by_request(self, request_id: str) -> Optional[ToolEscalationRequest]:
        with self._escalations_lock:
            for escalation in self._pending_escalations.values():
                if escalation.request_id == request_id:
                    return escalation
        return None

    def _record_tool_invocations(self, task: Task, authorization) -> None:
        """记录本 attempt 的审计批次完成事件。

        每条工具调用已在 ToolAuthorization.call() 返回时写进
        ``tool_invocations``；这里仅保留一个轻量事件，避免把完整结果
        再复制一遍到 task_events。
        """

        if not authorization.invocation_log:
            return
        self.task_repo.record_event(
            task.job_id,
            task.task_id,
            "tool_invocation_batch_finished",
            {
                "attempt_id": task.active_attempt_id,
                "call_count": len(authorization.invocation_log),
            },
        )
