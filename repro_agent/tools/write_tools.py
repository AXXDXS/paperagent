"""受限写入类与高危类工具。

风险分级说明：
    - ``write_task_output`` / ``write_file``：``RESTRICTED_WRITE``，
      只能写入沙箱 ``workspace/`` 或 ``output/`` 目录（§12 沙箱设计：
      workspace 可读写、output 可写），不允许写到 ``input/`` 或沙箱外。
    - ``execute_command``：``HIGH_RISK``，会真正 fork 子进程执行 shell
      命令（环境构建、代码修改后跑单测、实验执行子智能体都需要它）。
      默认关闭网络（§12："默认关闭网络"），命令执行严格限定
      ``cwd`` 在沙箱 workspace 内，并设置超时。
    - ``git_worktree_apply``：保留给旧任务定义的禁用兼容入口；调用会
      fail closed。代码修改现在使用控制面创建的 attempt 级仓库副本，
      不在宿主机执行 Git 或仓库 hook。

这几类工具默认不会出现在"论文分析""代码分析""资源检查"等只读
任务的 ``allowed_tools`` 里；即便任务定义里误写了，
``authorization.ToolAuthorization`` 也会按任务类型的风险预算二次拦截
（见 authorization.py 的 ``TASK_TYPE_RISK_BUDGET``）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from repro_agent.tools.base import (
    SandboxContext,
    ToolExample,
    ToolExecutionError,
    ToolOutputSpec,
    ToolParamDoc,
    ToolRiskLevel,
    ToolSpec,
)
from repro_agent.execution.backend import (
    CondaEnvironmentBuildRequest,
    ExecutionRequest,
    ExecutionResourcePolicy,
    ImageBuildRequest,
)
from repro_agent.evidence.hashing import sha256_of_file
from repro_agent.tools.destructive_actions import (
    DestructiveActionConfirmationRequired,
    inspect_destructive_command,
)

_MAX_COMMAND_TIMEOUT_S = 1800


def write_file(ctx: SandboxContext, path: str, content: str) -> dict[str, Any]:
    """写入文件到沙箱可写范围（workspace/output）。"""

    resolved = Path(ctx.resolve_writable_path(path))
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return {"path": path, "bytes_written": len(content.encode("utf-8"))}


def write_task_output(ctx: SandboxContext, filename: str, content: str) -> dict[str, Any]:
    """把结果写入任务的 ``output/`` 目录（§15.2 候选记忆的标准落盘位置：
    result.json / report.md / candidate_memory.md / evidence.json /
    error_report.json 等文件都通过这个工具产出）。

    这里必须使用 ``resolve_output_path`` 而不是复用
    ``write_file(ctx, f"output/{filename}", ...)``：后者会把
    ``"output/xxx"`` 交给 ``resolve_writable_path`` 做多根路径猜测，
    而 ``resolve_writable_path`` 对相对路径是按
    ``[workspace_dir, output_dir, tmp_dir]`` 顺序依次尝试拼接、命中
    第一个就返回——`"output/xxx"` 会被错误地解析为
    ``workspace_dir/output/xxx``，而不是真正的 ``output_dir/xxx``。
    直接锚定到 ``output_dir`` 才能保证任务产物真的落在
    ``TaskSandbox.collect_outputs()``/``OutputValidator`` 扫描的目录里。
    """

    resolved = Path(ctx.resolve_output_path(filename))
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return {"path": f"output/{filename}", "bytes_written": len(content.encode("utf-8"))}


def execute_command(
    ctx: SandboxContext,
    command: list[str],
    *,
    timeout_seconds: int = 600,
    allow_network: bool = False,
    working_dir: str = ".",
    environment: dict[str, str] | None = None,
    passthrough_environment: list[str] | None = None,
    gpu_count: int = 0,
    workspace_read_only: bool = False,
) -> dict[str, Any]:
    """在沙箱 workspace 内执行 shell 命令（§12：默认关闭网络）。

    ``allow_network`` 参数是显式声明，即使调用方传 True，如果
    ``SandboxContext.network_allowed()`` 返回 False（沙箱策略层面
    禁网），依然会被拒绝——策略优先级高于工具调用方的意愿，这是
    典型的 fail-closed 设计（借鉴 DeerFlow 的安全默认取向）。
    """

    if allow_network and not ctx.network_allowed():
        raise ToolExecutionError(
            "experiment command requested network access outside its approved sandbox policy"
        )
    if not command:
        raise ToolExecutionError("command must be a non-empty list of arguments")

    destructive = inspect_destructive_command(command)
    if destructive is not None and not ctx.policy.destructive_command_is_approved(
        destructive.fingerprint
    ):
        raise DestructiveActionConfirmationRequired(destructive)

    timeout = min(timeout_seconds, _MAX_COMMAND_TIMEOUT_S)
    limits = ctx.policy.resource_limits
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 0:
        raise ToolExecutionError("gpu_count must be a non-negative integer")
    if gpu_count > limits.gpu_count:
        raise ToolExecutionError(
            f"gpu_count={gpu_count} exceeds sandbox allowance={limits.gpu_count}"
        )
    backend = getattr(ctx, "execution_backend", None)
    if backend is None:
        raise ToolExecutionError("no isolated execution backend is configured")
    workspace = Path(ctx.resolve_writable_path("."))
    resolved_working_dir = Path(ctx.resolve_writable_path(working_dir))
    try:
        relative_working_dir = str(resolved_working_dir.relative_to(workspace)) or "."
    except ValueError as exc:
        raise ToolExecutionError("working_dir must be inside workspace") from exc
    if not resolved_working_dir.is_dir():
        raise ToolExecutionError(f"working_dir does not exist: {working_dir}")
    try:
        result = backend.execute(
            ExecutionRequest(
                task_id=ctx.task_id,
                attempt_id=getattr(ctx, "attempt_id", ctx.task_id),
                command=command,
                image=getattr(ctx, "execution_image", ""),
                input_dir=ctx.input_dir,
                workspace_dir=workspace,
                output_dir=ctx.output_dir,
                timeout_seconds=timeout,
                resources=ExecutionResourcePolicy(
                    cpu_cores=limits.cpu_cores or 1.0,
                    memory_mb=limits.memory_mb or 1024,
                    disk_mb=limits.disk_mb or 4096,
                    max_processes=limits.max_processes,
                    max_open_files=limits.max_open_files,
                    max_log_bytes=limits.max_log_bytes,
                    gpu_memory_mb=limits.gpu_memory_mb or 0,
                ),
                environment=environment or {},
                passthrough_environment=passthrough_environment or [],
                working_dir=relative_working_dir,
                workspace_read_only=workspace_read_only,
                gpu_count=gpu_count,
                network_enabled=allow_network,
                cancellation_event=getattr(ctx, "cancellation_event", None),
                state_path=ctx.logs_dir
                / f"{getattr(ctx, 'attempt_id', ctx.task_id)}.execution.json",
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ToolExecutionError(f"failed to execute command: {exc}") from exc

    stdout_log = ctx.logs_dir / f"{getattr(ctx, 'attempt_id', ctx.task_id)}.stdout.log"
    stderr_log = ctx.logs_dir / f"{getattr(ctx, 'attempt_id', ctx.task_id)}.stderr.log"
    stdout_log.write_text(result.stdout, encoding="utf-8")
    stderr_log.write_text(result.stderr, encoding="utf-8")
    stdout_log_sha256 = sha256_of_file(stdout_log)
    stderr_log_sha256 = sha256_of_file(stderr_log)
    execution_state = Path(result.execution_state_path) if result.execution_state_path else None
    execution_state_sha256 = (
        sha256_of_file(execution_state)
        if execution_state is not None and execution_state.is_file()
        else ""
    )
    output_artifacts = []
    if ctx.output_dir.exists():
        for artifact in sorted(ctx.output_dir.rglob("*")):
            if artifact.is_file() and not artifact.is_symlink():
                output_artifacts.append(
                    {
                        "path": str(artifact),
                        "relative_path": str(artifact.relative_to(ctx.output_dir)),
                        "size_bytes": artifact.stat().st_size,
                        "sha256": sha256_of_file(artifact),
                    }
                )
    return {
        "command": command,
        "exit_code": result.exit_code,
        "stdout": result.stdout[-20000:],
        "stderr": result.stderr[-20000:],
        "container_name": result.container_name,
        "container_digest": result.image_digest,
        "termination_reason": result.termination_reason,
        "mock": result.mock,
        "started_at": result.started_at,
        "completed_at": result.completed_at,
        "duration_seconds": result.duration_seconds,
        "stdout_log_path": str(stdout_log),
        "stdout_log_sha256": stdout_log_sha256,
        "stdout_log_size_bytes": stdout_log.stat().st_size,
        "stderr_log_path": str(stderr_log),
        "stderr_log_sha256": stderr_log_sha256,
        "stderr_log_size_bytes": stderr_log.stat().st_size,
        "execution_state_path": result.execution_state_path,
        "execution_state_sha256": execution_state_sha256,
        "execution_state_size_bytes": (
            execution_state.stat().st_size
            if execution_state is not None and execution_state.is_file()
            else 0
        ),
        "output_artifacts": output_artifacts,
    }


def git_worktree_apply(
    ctx: SandboxContext,
    repository_path: str,
    worktree_name: str,
    branch_name: str,
) -> dict[str, Any]:
    """Fail closed until Git mutation has its own isolated execution backend.

    Running host ``git`` can execute repository hooks and mutate metadata outside
    the container boundary. Keeping the name registered produces a clear,
    auditable denial for older task definitions instead of silently falling back
    to host execution.
    """

    raise ToolExecutionError(
        "git_worktree_apply is disabled: host Git execution is forbidden; "
        "use an isolated coding backend"
    )


def build_environment_image(
    ctx: SandboxContext,
    dockerfile: str,
    *,
    image_tag: str,
    timeout_seconds: int = 900,
    force_rebuild: bool = False,
    network_enabled: bool = False,
) -> dict[str, Any]:
    """Build or reuse a content-addressed image through the control plane.

    ``network_enabled=True`` opts the build into "Route A" build-time
    networking: the Docker build runs on the bridge network so pip can
    resolve dependencies online, and a missing base image may be pulled.
    Experiment runtime isolation is decided elsewhere and is unaffected.
    """

    backend = getattr(ctx, "execution_backend", None)
    if backend is None or not hasattr(backend, "build_image"):
        raise ToolExecutionError("execution backend does not support image builds")
    dockerfile_path = Path(ctx.resolve_writable_path(dockerfile))
    if not dockerfile_path.is_file():
        raise ToolExecutionError(f"Dockerfile does not exist: {dockerfile}")
    try:
        result = backend.build_image(
            ImageBuildRequest(
                task_id=ctx.task_id,
                attempt_id=getattr(ctx, "attempt_id", ctx.task_id),
                context_dir=ctx.workspace_dir,
                dockerfile=dockerfile_path,
                image_tag=image_tag,
                timeout_seconds=min(timeout_seconds, _MAX_COMMAND_TIMEOUT_S),
                max_log_bytes=ctx.policy.resource_limits.max_log_bytes,
                cancellation_event=getattr(ctx, "cancellation_event", None),
                log_dir=ctx.logs_dir,
                force_rebuild=force_rebuild,
                network_enabled=network_enabled,
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ToolExecutionError(f"failed to build environment image: {exc}") from exc
    if result.exit_code == 0:
        ctx.execution_image = result.image_ref
    return {
        "image_ref": result.image_ref,
        "image_digest": result.image_digest,
        "exit_code": result.exit_code,
        "stdout": result.stdout[-20_000:],
        "stderr": result.stderr[-20_000:],
        "mock": result.mock,
        "termination_reason": result.termination_reason,
        "cache_hit": result.cache_hit,
        "environment_fingerprint": result.environment_fingerprint,
        "cache_ref": result.cache_ref,
    }


def build_conda_environment(
    ctx: SandboxContext,
    requirements_file: str,
    *,
    environment_name: str = "",
    python_version: str = "3.11",
    wheel_dirs: list[str] | None = None,
    timeout_seconds: int = 1800,
    force_rebuild: bool = False,
    network_enabled: bool = False,
) -> dict[str, Any]:
    """Create or reuse a controller-managed Conda prefix.

    The prefix itself is never exposed as an arbitrary host path.  Consumers
    receive an opaque ``conda://<sha256>`` reference which the configured Conda
    execution backend resolves inside its managed environment root.
    """

    backend = getattr(ctx, "execution_backend", None)
    if backend is None or not hasattr(backend, "build_conda_environment"):
        raise ToolExecutionError("execution backend does not support Conda environments")
    requirements_path = Path(ctx.resolve_writable_path(requirements_file))
    if not requirements_path.is_file():
        raise ToolExecutionError(f"requirements file does not exist: {requirements_file}")
    resolved_wheels: list[Path] = []
    for value in wheel_dirs or []:
        path = Path(ctx.resolve_writable_path(value))
        if not path.is_dir():
            raise ToolExecutionError(f"wheel directory does not exist: {value}")
        resolved_wheels.append(path)
    try:
        result = backend.build_conda_environment(
            CondaEnvironmentBuildRequest(
                task_id=ctx.task_id,
                attempt_id=getattr(ctx, "attempt_id", ctx.task_id),
                requirements_file=requirements_path,
                environment_name=environment_name,
                python_version=python_version,
                timeout_seconds=min(timeout_seconds, _MAX_COMMAND_TIMEOUT_S),
                max_log_bytes=ctx.policy.resource_limits.max_log_bytes,
                force_rebuild=force_rebuild,
                network_enabled=network_enabled,
                wheel_dirs=tuple(resolved_wheels),
                cancellation_event=getattr(ctx, "cancellation_event", None),
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ToolExecutionError(f"failed to build Conda environment: {exc}") from exc
    if result.exit_code == 0:
        ctx.execution_image = result.environment_ref
    return {
        "environment_ref": result.environment_ref,
        "environment_digest": result.environment_digest,
        "exit_code": result.exit_code,
        "stdout": result.stdout[-20_000:],
        "stderr": result.stderr[-20_000:],
        "mock": result.mock,
        "termination_reason": result.termination_reason,
        "cache_hit": result.cache_hit,
        "environment_fingerprint": result.environment_fingerprint,
        "cache_ref": result.cache_ref,
        "package_manifest_digest": result.package_manifest_digest,
        "environment_name": result.environment_name,
    }


def _strict_object(
    required: list[str], properties: dict[str, Any]
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


_WRITE_OUTPUT_SCHEMA = _strict_object(
    ["path", "bytes_written"],
    {
        "path": {"type": "string"},
        "bytes_written": {"type": "integer", "minimum": 0},
    },
)
_SHA256_SCHEMA = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
_OPTIONAL_SHA256_SCHEMA = {
    "anyOf": [_SHA256_SCHEMA, {"const": ""}],
}
_OUTPUT_ARTIFACT_SCHEMA = _strict_object(
    ["path", "relative_path", "size_bytes", "sha256"],
    {
        "path": {"type": "string"},
        "relative_path": {"type": "string"},
        "size_bytes": {"type": "integer", "minimum": 0},
        "sha256": _SHA256_SCHEMA,
    },
)
_EXECUTE_OUTPUT_SCHEMA = _strict_object(
    [
        "command",
        "exit_code",
        "stdout",
        "stderr",
        "container_name",
        "container_digest",
        "termination_reason",
        "mock",
        "started_at",
        "completed_at",
        "duration_seconds",
        "stdout_log_path",
        "stdout_log_sha256",
        "stdout_log_size_bytes",
        "stderr_log_path",
        "stderr_log_sha256",
        "stderr_log_size_bytes",
        "execution_state_path",
        "execution_state_sha256",
        "execution_state_size_bytes",
        "output_artifacts",
    ],
    {
        "command": {"type": "array", "items": {"type": "string"}},
        "exit_code": {"type": "integer"},
        "stdout": {"type": "string"},
        "stderr": {"type": "string"},
        "container_name": {"type": "string"},
        "container_digest": {"type": "string"},
        "termination_reason": {"type": "string"},
        "mock": {"type": "boolean"},
        "started_at": {"type": "string"},
        "completed_at": {"type": "string"},
        "duration_seconds": {"type": "number", "minimum": 0},
        "stdout_log_path": {"type": "string"},
        "stdout_log_sha256": _SHA256_SCHEMA,
        "stdout_log_size_bytes": {"type": "integer", "minimum": 0},
        "stderr_log_path": {"type": "string"},
        "stderr_log_sha256": _SHA256_SCHEMA,
        "stderr_log_size_bytes": {"type": "integer", "minimum": 0},
        "execution_state_path": {"type": "string"},
        "execution_state_sha256": _OPTIONAL_SHA256_SCHEMA,
        "execution_state_size_bytes": {"type": "integer", "minimum": 0},
        "output_artifacts": {"type": "array", "items": _OUTPUT_ARTIFACT_SCHEMA},
    },
)
_IMAGE_BUILD_OUTPUT_SCHEMA = _strict_object(
    [
        "image_ref",
        "image_digest",
        "exit_code",
        "stdout",
        "stderr",
        "mock",
        "termination_reason",
        "cache_hit",
        "environment_fingerprint",
        "cache_ref",
    ],
    {
        "image_ref": {"type": "string"},
        "image_digest": {"type": "string"},
        "exit_code": {"type": "integer"},
        "stdout": {"type": "string"},
        "stderr": {"type": "string"},
        "mock": {"type": "boolean"},
        "termination_reason": {"type": "string"},
        "cache_hit": {"type": "boolean"},
        "environment_fingerprint": {"type": "string"},
        "cache_ref": {"type": "string"},
    },
)
_CONDA_BUILD_OUTPUT_SCHEMA = _strict_object(
    [
        "environment_ref",
        "environment_digest",
        "exit_code",
        "stdout",
        "stderr",
        "mock",
        "termination_reason",
        "cache_hit",
        "environment_fingerprint",
        "cache_ref",
        "package_manifest_digest",
        "environment_name",
    ],
    {
        "environment_ref": {"type": "string"},
        "environment_digest": {"type": "string"},
        "exit_code": {"type": "integer"},
        "stdout": {"type": "string"},
        "stderr": {"type": "string"},
        "mock": {"type": "boolean"},
        "termination_reason": {"type": "string"},
        "cache_hit": {"type": "boolean"},
        "environment_fingerprint": {"type": "string"},
        "cache_ref": {"type": "string"},
        "package_manifest_digest": {"type": "string"},
        "environment_name": {"type": "string"},
    },
)
_DISABLED_TOOL_OUTPUT_SCHEMA = _strict_object([], {})


TOOL_SPECS = [
    ToolSpec(
        name="build_conda_environment",
        description="按锁定依赖创建或复用控制面管理的 Conda 环境。",
        risk_level=ToolRiskLevel.HIGH_RISK,
        handler=build_conda_environment,
        output=ToolOutputSpec(schema=_CONDA_BUILD_OUTPUT_SCHEMA),
        suggested_task_types=("environment_build",),
        when_to_use="环境后端明确选择 conda，且锁定依赖和 import 自检脚本已经准备完成时使用。",
        boundaries=(
            "仅能读取当前任务 workspace 中的依赖文件和 wheel 目录。",
            "返回 opaque conda:// 引用，不把任意宿主机环境路径暴露给子 Agent。",
            "Conda 是可信本地运行模式，不提供 Docker 等价的网络和文件系统隔离。",
        ),
        returns=(
            "{environment_ref, environment_digest, exit_code, stdout, stderr, "
            "termination_reason, cache_hit, environment_fingerprint, cache_ref, "
            "package_manifest_digest, environment_name}"
        ),
    ),
    ToolSpec(
        name="build_environment_image",
        description="按构建内容复用或构建隔离容器镜像，并返回不可变摘要。",
        risk_level=ToolRiskLevel.HIGH_RISK,
        handler=build_environment_image,
        output=ToolOutputSpec(schema=_IMAGE_BUILD_OUTPUT_SCHEMA),
        suggested_task_types=("environment_build",),
        when_to_use="环境文件准备完成后，复用或构建后续实验真正使用的镜像。",
        boundaries=(
            "默认构建网络为 none（离线）；仅当显式传 network_enabled=true "
            "时允许构建期联网在线安装依赖（Route A），实验运行期仍保持隔离。",
            "Dockerfile 必须位于当前任务 workspace 中。",
            "force_rebuild 仅用于缓存镜像未通过 import 自检时的强制修复。",
        ),
        returns=(
            "{image_ref, image_digest, exit_code, stdout, stderr, mock, "
            "termination_reason, cache_hit, environment_fingerprint, cache_ref}"
        ),
    ),
    ToolSpec(
        name="write_file",
        description="把代码/配置等工作产物写入沙箱的可写工作区（workspace）。",
        risk_level=ToolRiskLevel.RESTRICTED_WRITE,
        handler=write_file,
        output=ToolOutputSpec(schema=_WRITE_OUTPUT_SCHEMA),
        suggested_task_types=("coding", "environment_build"),
        when_to_use=(
            "当你需要新增/修改一份代码、配置或脚本文件，作为后续步骤"
            "（如 execute_command 运行）的输入时使用；例如修复代码里的 bug、"
            "写一份新的训练配置文件。如果目的是产出【任务的最终交付结果】"
            "（result.json/report.md 等），应改用 write_task_output，不要"
            "把最终结果写进 workspace，否则不会被主智能体的输出校验收集到。"
        ),
        boundaries=(
            "只能写入沙箱的 workspace/output 范围内，不能写到 input/（只读）"
            "或沙箱之外的任意宿主机路径——即使 path 里写了 '../' 或绝对路径"
            "试图逃逸，也会被拒绝。",
            "会覆盖同名已有文件且不会先备份；如果需要保留旧版本，请先自行"
            "读取原内容另存一份。",
            "不会自动执行写入的脚本，写完之后如果需要运行，必须再显式调用"
            "execute_command。",
        ),
        returns="{path, bytes_written}",
        cost_hint="本地磁盘写入，开销很小；单次调用请一次性写入完整内容，不要为了绕过长度限制多次调用做拼接（拼接语义未定义，后写入会整体覆盖前一次）。",
        examples=(
            ToolExample(
                when="修复训练脚本里的一处 bug 后写回文件",
                arguments={"path": "src/train.py", "content": "# fixed content ..."},
                result={"path": "src/train.py", "bytes_written": 4096},
            ),
        ),
        param_docs={
            "path": ToolParamDoc(
                description="要写入的文件路径，相对沙箱可写工作区（workspace），例如 'src/train.py'。不支持写到 input/ 下。",
                example="src/train.py",
            ),
            "content": ToolParamDoc(
                description="要写入的完整文件内容（会整体覆盖已有文件，不是追加）。",
                example="print('hello world')\n",
            ),
        },
    ),
    ToolSpec(
        name="write_task_output",
        description="把任务的最终交付产物写入沙箱 output/ 目录，供主智能体收集与校验。",
        risk_level=ToolRiskLevel.RESTRICTED_WRITE,
        handler=write_task_output,
        output=ToolOutputSpec(schema=_WRITE_OUTPUT_SCHEMA),
        suggested_task_types=(
            "paper_analysis",
            "code_analysis",
            "resource_check",
            "specification",
            "environment_build",
            "coding",
            "experiment_execution",
            "verification",
            "reflection",
        ),
        when_to_use=(
            "每个任务在结束前【必须】用它写出至少一份结果文件（通常是"
            "result.json），这是主智能体判断任务是否完成、能否进入下一阶段"
            "的唯一依据。中间过程产物（临时脚本、调试用的中间文件）应该用"
            "write_file 写到 workspace，不要和最终结果混在一起写进 output/。"
        ),
        boundaries=(
            "只能写到 output/ 目录（filename 不需要、也不应该自己拼接"
            "'output/' 前缀，直接传文件名本身，例如 'result.json'），不能"
            "通过 filename 里的路径穿越写到 workspace/input/ 或沙箱之外。",
            "本工具不会替你校验 JSON 是否合法——如果要写结构化结果，必须"
            "自己先用 json.dumps 等方式序列化成字符串再传入 content，传入"
            "非法 JSON 字符串本工具依然会 '成功' 写入，但下游解析会失败。",
            "写入 result.json 时约定的字段结构由具体任务类型的 schema 决定"
            "（见各子智能体的 payload 约定），本工具本身不做字段级校验。",
        ),
        returns="{path: 'output/<filename>', bytes_written}",
        cost_hint="本地磁盘写入，开销很小；建议每个任务只在确定所有分析/执行都完成后调用一次，避免中途产生半成品结果文件误导校验。",
        examples=(
            ToolExample(
                when="任务完成后写出标准结果文件",
                arguments={"filename": "result.json", "content": '{"status": "ok"}'},
                result={"path": "output/result.json", "bytes_written": 17},
            ),
            ToolExample(
                when="额外写出一份供主智能体转正的候选记忆摘要",
                arguments={"filename": "candidate_memory.md", "content": "# 关键发现\n..."},
            ),
        ),
        param_docs={
            "filename": ToolParamDoc(
                description="output/ 目录下的文件名（不要带 'output/' 前缀，也不要带上级目录路径），例如 'result.json'、'report.md'、'candidate_memory.md'。",
                example="result.json",
            ),
            "content": ToolParamDoc(
                description="要写入的完整文本内容；写结构化结果时应传入已序列化好的 JSON 字符串。",
                example='{"metric": "accuracy", "value": 0.91}',
            ),
        },
    ),
    ToolSpec(
        name="execute_command",
        description="在沙箱工作区内以隔离容器方式执行一条 shell 命令，并返回退出码与输出。",
        risk_level=ToolRiskLevel.HIGH_RISK,
        handler=execute_command,
        output=ToolOutputSpec(schema=_EXECUTE_OUTPUT_SCHEMA),
        suggested_task_types=("environment_build", "experiment_execution", "coding"),
        when_to_use=(
            "当你需要真正运行一个程序才能得到结果时使用——安装依赖、跑单元"
            "测试、执行训练/推理脚本等。只有确实需要【执行】而不是【读取/"
            "查找】时才应该用它；如果目的只是查看文件内容或确认文件是否"
            "存在，不要为此启动一次容器，应改用 read_file/get_file_stat。"
        ),
        boundaries=(
            "默认关闭网络。只有任务由已确认的必需 API Base 派生出联网策略，"
            "且调用显式传 allow_network=True 时才启用；任何一侧缺失都会"
            "fail closed。环境构建使用独立的联网开关，不会继承实验网络设置。",
            "command 必须是参数数组（如 ['python', 'train.py', '--seed', '1']），"
            "不是一整条 shell 字符串——不支持管道符 '|'、重定向 '>'、'&&' 等"
            "shell 语法糖，这些需要拆成多次调用或封装进一个脚本文件后用"
            "['bash', 'script.sh'] 调用。",
            "超时时间最长 1800 秒（30 分钟），超过会被强制终止并计为失败，"
            "不适合用它直接跑数十小时的完整训练——长任务应该拆分为"
            "可断点恢复的多次调用，或降级为缩减规模的实验层级。",
            "stdout/stderr 只保留最后 20000 字符，早期输出会被截断——如果"
            "需要完整日志，应让脚本自己把日志写到文件（写到 output/ 或"
            "workspace 下），再用 read_file 读取。",
            "工作目录固定在沙箱 workspace 内，不能通过 cd/绝对路径切换到"
            "沙箱之外执行。",
        ),
        returns=(
            "{command, exit_code, stdout（末尾截断至 2 万字符）, stderr（同）, "
            "container_name, container_digest, termination_reason, mock}"
        ),
        cost_hint="需要启动一次隔离容器，比纯文件操作重得多，通常是秒级到分钟级；只需要查看/查找文件时不要用它，改用只读文件工具。",
        examples=(
            ToolExample(
                when="运行训练脚本的缩减规模版本",
                arguments={"command": ["python", "train.py", "--config", "configs/smoke.yaml"], "timeout_seconds": 300},
                result={"command": ["python", "train.py", "--config", "configs/smoke.yaml"], "exit_code": 0, "stdout": "...", "stderr": "", "mock": False},
            ),
            ToolExample(
                when="安装本地已缓存好的依赖（不需要联网）",
                arguments={"command": ["pip", "install", "--no-index", "--find-links", "vendor/wheels", "-r", "requirements.txt"]},
            ),
        ),
        param_docs={
            "command": ToolParamDoc(
                description="要执行的命令，必须是参数数组形式（不是 shell 字符串），例如 ['python', 'train.py', '--epochs', '1']。不支持管道/重定向等 shell 语法。",
                example=["python", "train.py", "--epochs", "1"],
            ),
            "timeout_seconds": ToolParamDoc(
                description="命令执行超时时间（秒），超过会被强制终止；上限 1800 秒，默认 600 秒。",
                example=600,
            ),
            "allow_network": ToolParamDoc(
                description="是否请求联网执行；即使传 True，只要沙箱策略禁网依然会被拒绝，实际效果始终等价于 False，不要依赖它绕过网络限制。",
                example=False,
            ),
        },
    ),
    ToolSpec(
        name="git_worktree_apply",
        description="（当前已禁用）原用于为代码修改任务创建独立 Git worktree，现在调用会直接返回错误。",
        risk_level=ToolRiskLevel.HIGH_RISK,
        handler=git_worktree_apply,
        output=ToolOutputSpec(schema=_DISABLED_TOOL_OUTPUT_SCHEMA),
        suggested_task_types=("coding",),
        when_to_use=(
            "不要主动调用此工具——它目前处于禁用状态，任何调用都会失败。"
            "如果任务描述里提到需要隔离的代码修改环境，应改用 execute_command"
            "在沙箱工作区内直接操作（工作区本身已经是隔离的），不需要再"
            "额外创建 Git worktree。"
        ),
        boundaries=(
            "调用必定失败并抛出 ToolExecutionError：宿主机 git 命令可能触发"
            "仓库 hooks、修改容器边界之外的元数据，在没有独立隔离执行后端"
            "之前被整体下线（fail-closed），不存在任何参数组合能让它成功执行。",
            "保留这个工具名只是为了让引用了它的旧任务定义得到一个明确、"
            "可审计的拒绝，而不是静默回退到不安全的宿主机执行——不要尝试"
            "'重试'或'换参数'来绕过这个拒绝。",
        ),
        returns="始终抛出异常，不会有正常返回值。",
        cost_hint="调用会立即失败，不产生实际开销，但也不会带来任何效果——不要调用。",
    ),
]
