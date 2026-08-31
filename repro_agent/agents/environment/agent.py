"""环境构建子智能体（设计文档 §9.5）。

职责：分析依赖、生成锁定依赖、按选择的 Docker/Colima/Conda 后端
创建可复用环境、执行 Import 测试、保存不可变环境摘要、修复环境问题。

**硬约束（§9.5 最后一句）："环境智能体只能修改环境配置，不能擅自
修改算法代码。"** 本实现在工具层面强制这一点：即使任务定义的
``allowed_tools`` 里出现了 ``write_file``，本子智能体也只调用
``write_file`` 写入以下白名单文件模式（Dockerfile、requirements、
environment.yml、导入自检脚本），任何试图写入其它路径（尤其是
``*.py`` 源码文件）的行为都会在 ``_guard_environment_only_write``
里被拒绝并记录，不依赖 LLM 自觉遵守 Prompt 里的约束。

风险预算：``environment_build`` -> HIGH_RISK（需要 execute_command
来跑 ``pip install``/``docker build``/import 自检等）。
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from dataclasses import dataclass, field
from typing import Any

from repro_agent.agents.base import AgentRunResult, BaseSubAgent
from repro_agent.domain.enums import FailureType
from repro_agent.domain.task import FailureReport
from repro_agent.execution.environment_naming import managed_environment_name
from repro_agent.llm_output import (
    DEPENDENCY_ANALYSIS_SCHEMA,
    parse_structured_json,
)

_ALLOWED_ENV_FILE_PATTERNS = (
    re.compile(r"^Dockerfile(\..*)?$"),
    re.compile(r".*requirements.*\.txt$"),
    re.compile(r".*environment.*\.ya?ml$"),
    re.compile(r".*poetry\.lock$"),
    re.compile(r".*import_smoke_test\.py$"),  # 唯一允许的"代码"文件：自检脚本，非算法代码
)


class EnvironmentGuardError(RuntimeError):
    """尝试通过环境构建子智能体修改算法代码（违反 §9.5 硬约束）。"""


@dataclass
class EnvironmentBuildResult:
    environment_backend: str = "docker"
    environment_name: str = ""
    environment_ref: str = ""
    environment_digest: str = ""
    dependency_analysis: str = ""
    dockerfile_path: str = ""
    lockfile_path: str = ""
    cuda_compatible: bool = True
    pytorch_compatible: bool = True
    install_log_tail: str = ""
    import_test_passed: bool = False
    image_digest: str = ""
    image_ref: str = ""
    build_succeeded: bool = False
    cache_hit: bool = False
    cache_rebuilt: bool = False
    environment_fingerprint: str = ""
    cache_ref: str = ""
    python_version: str = ""
    package_manifest_digest: str = ""
    selected_conda_source: str = ""
    selected_pip_source: str = ""
    source_attempts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_backend": self.environment_backend,
            "environment_name": self.environment_name,
            "environment_ref": self.environment_ref,
            "environment_digest": self.environment_digest,
            "dependency_analysis": self.dependency_analysis,
            "dockerfile_path": self.dockerfile_path,
            "lockfile_path": self.lockfile_path,
            "cuda_compatible": self.cuda_compatible,
            "pytorch_compatible": self.pytorch_compatible,
            "install_log_tail": self.install_log_tail,
            "import_test_passed": self.import_test_passed,
            "image_digest": self.image_digest,
            "image_ref": self.image_ref,
            "build_succeeded": self.build_succeeded,
            "cache_hit": self.cache_hit,
            "cache_rebuilt": self.cache_rebuilt,
            "environment_fingerprint": self.environment_fingerprint,
            "cache_ref": self.cache_ref,
            "python_version": self.python_version,
            "package_manifest_digest": self.package_manifest_digest,
            "selected_conda_source": self.selected_conda_source,
            "selected_pip_source": self.selected_pip_source,
            "source_attempts": self.source_attempts,
        }


class EnvironmentBuildAgent(BaseSubAgent):
    task_type = "environment_build"
    system_prompt = (
        "你是 ReproAgent 系统的环境构建子智能体。你的任务是分析项目依赖，"
        "生成锁定依赖文件，并按任务指定的 Docker/Colima/Conda 后端创建"
        "环境，检查 CUDA/PyTorch 兼容性并执行 import 自检。你只能修改环境相关配置文件（Dockerfile、"
        "requirements.txt、environment.yml 等），绝对不能修改任何算法"
        "代码文件（.py 源码，除自检脚本外）。"
    )

    def _guarded_write_file(self, path: str, content: str) -> None:
        """在写文件前先校验路径命中环境文件白名单（§9.5 硬约束的代码兜底）。"""

        filename = path.rsplit("/", 1)[-1]
        if not any(p.match(filename) for p in _ALLOWED_ENV_FILE_PATTERNS):
            raise EnvironmentGuardError(
                f"环境构建子智能体禁止写入非环境配置文件: {path} "
                "(只允许 Dockerfile/requirements/environment.yml/import 自检脚本)"
            )
        self.call_tool("write_file", path=path, content=content)

    def run(self) -> AgentRunResult:
        repo_root = self.task.definition.inputs.get("repository_path", ".")
        requested_deps = self.task.definition.inputs.get("dependencies_hint", "")
        repair_dependencies = self._validated_repair_dependencies(
            self.task.definition.inputs.get("repair_dependencies", [])
        )
        base_image = self.task.definition.inputs.get("base_image", "python:3.11-slim")
        environment_backend = str(
            self.task.definition.inputs.get("environment_backend", "docker")
        ).strip().lower() or "docker"
        python_version = str(
            self.task.definition.inputs.get("python_version", "3.11")
        ).strip()
        environment_name = managed_environment_name(
            self.task.definition.inputs.get("environment_name"), repo_root
        )
        if environment_backend not in {"docker", "colima", "conda"}:
            return AgentRunResult(
                succeeded=False,
                failure_report=FailureReport(
                    failure_type=FailureType.ENVIRONMENT_ERROR,
                    failed_step="select_environment_backend",
                    error_message=f"unsupported environment backend: {environment_backend}",
                    recommended_action="选择 docker、colima 或 conda 环境后端",
                ),
            )

        try:
            dependency_files, wheel_dirs = self._read_dependency_files(repo_root)
            analysis = self._analyze_dependencies(
                repo_root, requested_deps, dependency_files
            )
            result = EnvironmentBuildResult(
                environment_backend=environment_backend,
                environment_name=environment_name,
                dependency_analysis=analysis,
                python_version=python_version,
            )

            lockfile_content = (
                self._generate_lockfile(
                    dependency_files, repair_dependencies=repair_dependencies
                )
                if repair_dependencies
                else self._generate_lockfile(dependency_files)
            )
            self._guarded_write_file(
                "workspace://requirements.lock.txt", lockfile_content
            )
            result.lockfile_path = "workspace://requirements.lock.txt"

            import_test_code = self._generate_import_smoke_test(lockfile_content)
            self._guarded_write_file(
                "workspace://import_smoke_test.py", import_test_code
            )

            # Runtime-discovered packages may not exist in a vendored wheel
            # directory. Keep local wheels as candidates while allowing an
            # incremental repair to reach the configured package index.
            online_build = bool(repair_dependencies) or not wheel_dirs
            force_rebuild = bool(
                self.task.definition.inputs.get("force_rebuild", False)
            )
            if environment_backend == "conda":
                build_result = self._build_conda_environment(
                    environment_name=environment_name,
                    python_version=python_version,
                    wheel_dirs=wheel_dirs,
                    force_rebuild=force_rebuild,
                    repair_existing=bool(
                        self.task.definition.inputs.get(
                            "repair_existing_environment", False
                        )
                    ),
                    base_environment_ref=str(
                        self.task.definition.inputs.get("base_environment_ref", "")
                    ),
                    network_enabled=online_build,
                )
                self._record_conda_build_result(result, build_result)
                self._require_successful_conda_build(build_result)
            else:
                dockerfile_content = self._generate_dockerfile(
                    base_image, analysis, wheel_dirs
                )
                self._guarded_write_file("workspace://Dockerfile", dockerfile_content)
                result.dockerfile_path = "workspace://Dockerfile"
                safe_task_id = re.sub(
                    r"[^a-zA-Z0-9_.-]", "-", self.task.task_id
                ).lower()
                image_tag = f"repro-agent/{safe_task_id}:{self._attempt_id[-12:]}"
                build_result = self._build_environment_image(
                    image_tag=image_tag,
                    force_rebuild=force_rebuild,
                    network_enabled=online_build,
                )
                self._record_build_result(result, build_result)
                self._require_successful_build(build_result)

            result.import_test_passed = self._run_import_smoke_test()
            if result.cache_hit and not result.import_test_passed:
                # A local tag alone is not sufficient evidence that an image is
                # usable.  Rebuild once without Docker layer cache, then validate
                # the replacement before allowing downstream experiment tasks to
                # inherit it.
                result.cache_rebuilt = True
                if environment_backend == "conda":
                    build_result = self._build_conda_environment(
                        environment_name=environment_name,
                        python_version=python_version,
                        wheel_dirs=wheel_dirs,
                        force_rebuild=True,
                        network_enabled=online_build,
                    )
                    self._record_conda_build_result(result, build_result)
                    self._require_successful_conda_build(build_result)
                else:
                    build_result = self._build_environment_image(
                        image_tag=image_tag,
                        force_rebuild=True,
                        network_enabled=online_build,
                    )
                    self._record_build_result(result, build_result)
                    self._require_successful_build(build_result)
                result.import_test_passed = self._run_import_smoke_test()

        except EnvironmentGuardError as exc:
            failure = FailureReport(
                failure_type=FailureType.PERMISSION_ERROR,
                failed_step="write_environment_file",
                error_message=str(exc),
                likely_causes=["LLM 试图修改算法代码而非环境配置"],
                recommended_action="检查任务指令，确保环境构建任务不涉及代码修改",
            )
            return AgentRunResult(succeeded=False, failure_report=failure)
        except RuntimeError as exc:
            failure = FailureReport(
                failure_type=FailureType.ENVIRONMENT_ERROR,
                failed_step="build_or_validate_environment_image",
                error_message=str(exc),
                likely_causes=[
                    "依赖未锁定或在线拉取失败（无本地 wheel 时需构建期联网）",
                    "基础镜像/Conda 不可用，或 import 自检失败",
                ],
                recommended_action="提供锁定依赖与本地 wheel，或允许构建期联网在线安装",
            )
            return AgentRunResult(succeeded=False, failure_report=failure)

        if not result.import_test_passed:
            return AgentRunResult(
                succeeded=False,
                failure_report=FailureReport(
                    failure_type=FailureType.ENVIRONMENT_ERROR,
                    failed_step="import_smoke_test",
                    error_message="built environment failed dependency import validation",
                    recommended_action="修正依赖锁定或运行环境后重建",
                ),
            )

        result_payload = result.to_dict()
        self.write_json_output("result.json", result_payload)
        self.write_candidate_memory(self._render_candidate_memory(result))

        return AgentRunResult(succeeded=True, outputs=result_payload, candidate_memory_written=True)

    def _build_environment_image(
        self,
        *,
        image_tag: str,
        force_rebuild: bool = False,
        network_enabled: bool = False,
    ) -> dict[str, Any]:
        return self.call_tool(
            "build_environment_image",
            dockerfile="workspace://Dockerfile",
            image_tag=image_tag,
            # Online builds must download multi-GB pinned wheels (torch +
            # CUDA libraries), so allow the tool's maximum timeout.
            timeout_seconds=1800,
            force_rebuild=force_rebuild,
            network_enabled=network_enabled,
        )

    def _build_conda_environment(
        self,
        *,
        environment_name: str,
        python_version: str,
        wheel_dirs: list[str],
        force_rebuild: bool = False,
        repair_existing: bool = False,
        base_environment_ref: str = "",
        network_enabled: bool = False,
    ) -> dict[str, Any]:
        staged_wheel_dirs = [
            "workspace://repository"
            if directory == "."
            else f"workspace://repository/{directory}"
            for directory in wheel_dirs
        ]
        return self.call_tool(
            "build_conda_environment",
            requirements_file="workspace://requirements.lock.txt",
            environment_name=environment_name,
            python_version=python_version,
            wheel_dirs=staged_wheel_dirs,
            timeout_seconds=1800,
            force_rebuild=force_rebuild,
            repair_existing=repair_existing,
            base_environment_ref=base_environment_ref,
            network_enabled=network_enabled,
        )

    @staticmethod
    def _record_build_result(
        result: EnvironmentBuildResult, build_result: dict[str, Any]
    ) -> None:
        result.build_succeeded = build_result.get("exit_code") == 0
        result.image_digest = build_result.get("image_digest", "")
        result.image_ref = build_result.get("image_ref", "")
        result.cache_hit = bool(build_result.get("cache_hit", False))
        result.environment_fingerprint = build_result.get(
            "environment_fingerprint", ""
        )
        result.cache_ref = build_result.get("cache_ref", "")
        result.install_log_tail = (
            build_result.get("stdout", "") + "\n" + build_result.get("stderr", "")
        )[-2000:]
        result.environment_ref = result.image_ref
        result.environment_digest = result.image_digest

    @staticmethod
    def _record_conda_build_result(
        result: EnvironmentBuildResult, build_result: dict[str, Any]
    ) -> None:
        result.build_succeeded = build_result.get("exit_code") == 0
        result.environment_ref = build_result.get("environment_ref", "")
        result.environment_digest = build_result.get("environment_digest", "")
        # Preserve the legacy environment payload fields while downstream
        # consumers migrate to the runtime-neutral names.
        result.image_ref = result.environment_ref
        result.image_digest = result.environment_digest
        result.cache_hit = bool(build_result.get("cache_hit", False))
        result.environment_fingerprint = build_result.get(
            "environment_fingerprint", ""
        )
        result.cache_ref = build_result.get("cache_ref", "")
        result.package_manifest_digest = build_result.get(
            "package_manifest_digest", ""
        )
        result.environment_name = build_result.get(
            "environment_name", result.environment_name
        )
        result.selected_conda_source = build_result.get(
            "selected_conda_source", ""
        )
        result.selected_pip_source = build_result.get("selected_pip_source", "")
        result.source_attempts = list(build_result.get("source_attempts", []))
        result.install_log_tail = (
            build_result.get("stdout", "") + "\n" + build_result.get("stderr", "")
        )[-2000:]

    @staticmethod
    def _require_successful_build(build_result: dict[str, Any]) -> None:
        if build_result.get("exit_code") != 0 or not build_result.get("image_digest"):
            raise RuntimeError(
                "environment image build failed or did not produce an immutable digest: "
                + build_result.get("stderr", "")[-1000:]
            )

    @staticmethod
    def _require_successful_conda_build(build_result: dict[str, Any]) -> None:
        if (
            build_result.get("exit_code") != 0
            or not build_result.get("environment_digest")
            or not str(build_result.get("environment_ref", "")).startswith("conda://")
        ):
            raise RuntimeError(
                "Conda environment build failed or did not produce a verified reference: "
                + build_result.get("stderr", "")[-1000:]
            )

    def _run_import_smoke_test(self) -> bool:
        test_result = self.call_tool(
            "execute_command",
            command=["python", "import_smoke_test.py"],
            working_dir=".",
            timeout_seconds=120,
        )
        return test_result.get("exit_code") == 0

    def _read_dependency_files(
        self, repo_root: str
    ) -> tuple[dict[str, str], list[str]]:
        req_files = self.call_tool("find_files", pattern="requirements*.txt", root=repo_root)
        lock_files = self.call_tool("find_files", pattern="*.lock", root=repo_root)
        pyproject_files = self.call_tool("find_files", pattern="pyproject.toml", root=repo_root)
        wheel_files = self.call_tool("find_files", pattern="*.whl", root=repo_root)
        contents: dict[str, str] = {}
        for path in [
            *req_files.get("matches", []),
            *lock_files.get("matches", []),
            *pyproject_files.get("matches", []),
        ][:12]:
            try:
                read = self.call_tool(
                    "read_file", path=f"{repo_root.rstrip('/')}/{path}",
                    start_line=1, end_line=2000,
                )
            except Exception:
                continue
            contents[path] = read.get("content", "")
        wheel_dirs = sorted(
            {
                str(path.rsplit("/", 1)[0]) if "/" in path else "."
                for path in wheel_files.get("matches", [])
                if ".." not in path.split("/")
            }
        )
        return contents, wheel_dirs

    def _analyze_dependencies(
        self, repo_root: str, hint: str, dependency_files: dict[str, str]
    ) -> str:
        setup_files = self.call_tool("find_files", pattern="setup.py", root=repo_root)
        prompt = (
            f"仓库依赖文件: {list(dependency_files)}, setup.py: {setup_files.get('matches')}\n"
            f"依赖声明摘要: {json.dumps(dependency_files, ensure_ascii=False)[:12000]}\n"
            f"额外依赖提示: {hint}\n"
            "请分析该项目所需依赖，特别注意 CUDA/PyTorch 版本兼容性，"
            '并严格输出 JSON：{"dependency_analysis": "..."}。'
        )
        # 依赖文件列表已经通过 find_files 收集并写进 prompt，这里只需要
        # 模型输出一段分析文本，不需要再自己调用工具（Dockerfile/锁定
        # 依赖文件的写入由 run() 中的 _guarded_write_file 确定性完成）。
        response = self.call_llm(
            prompt,
            temperature=0.2,
            tool_names=[],
            output_schema=DEPENDENCY_ANALYSIS_SCHEMA,
            output_schema_name="dependency_analysis",
        )
        return parse_structured_json(
            response.content,
            DEPENDENCY_ANALYSIS_SCHEMA,
            label="dependency analysis output",
        )["dependency_analysis"]

    def _generate_dockerfile(
        self, base_image: str, analysis: str, wheel_dirs: list[str]
    ) -> str:
        if not base_image or any(char.isspace() for char in base_image):
            raise RuntimeError("invalid base image")
        if wheel_dirs:
            # Offline mode: every dependency must come from wheels vendored
            # inside the build context (--no-index blocks any index access).
            find_links = " ".join(
                shlex.quote(f"--find-links=/source/{directory}")
                for directory in wheel_dirs
            )
            install_command = (
                f"python -m pip install --no-index --no-cache-dir {find_links} "
                "-r requirements.lock.txt"
            )
        else:
            # Route A: no vendored wheels exist, so the pinned dependencies are
            # resolved from the package index during the build.  Build-time
            # networking only; the experiment runtime stays isolated.
            install_command = (
                "python -m pip install --no-cache-dir -r requirements.lock.txt"
            )
        return (
            f"FROM {base_image}\n"
            "WORKDIR /workspace\n"
            "COPY requirements.lock.txt .\n"
            "COPY repository /source\n"
            f"RUN if [ -s requirements.lock.txt ]; then {install_command}; fi\n"
            "COPY import_smoke_test.py .\n"
            # LLM output is evidence, never Dockerfile syntax.  Keeping it out
            # of the build file prevents newline-based instruction injection.
            "# Dependency analysis is stored in the task result metadata.\n"
        )

    @staticmethod
    def _validated_repair_dependencies(values: Any) -> list[str]:
        """Accept only plain package names extracted by the controller.

        Experiment stderr is untrusted input.  In particular it must never be
        able to inject pip options, URLs or local paths into a repair command.
        """

        if not isinstance(values, list):
            raise RuntimeError("repair_dependencies must be a list")
        dependencies: list[str] = []
        for value in values:
            package = str(value).strip()
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", package):
                raise RuntimeError(f"invalid repair dependency: {package!r}")
            if package.lower() not in {item.lower() for item in dependencies}:
                dependencies.append(package)
        return dependencies

    def _generate_lockfile(
        self,
        dependency_files: dict[str, str],
        *,
        repair_dependencies: list[str] | None = None,
    ) -> str:
        candidates = [
            (path, content)
            for path, content in dependency_files.items()
            if "requirements" in path.lower()
        ]
        pinned = [item for item in candidates if "lock" in item[0].lower()]
        selected = (pinned or candidates)
        if not selected:
            if dependency_files:
                raise RuntimeError(
                    "project declares dependencies but has no requirements-style "
                    "offline lock file"
                )
            lines: list[str] = []
        else:
            content = selected[0][1]
            lines = [line.rstrip() for line in content.splitlines() if line.strip()]
            unpinned = [
                line for line in lines
                if not line.lstrip().startswith(("#", "-")) and "==" not in line
            ]
            if unpinned:
                raise RuntimeError(
                    "dependency declaration is not fully pinned: " + ", ".join(unpinned[:5])
                )
            lines = self._filter_platform_incompatible(lines)
        declared_names = {
            re.split(r"[=<>!~\[\s]", line.lstrip(), maxsplit=1)[0].lower()
            for line in lines
            if line.strip() and not line.lstrip().startswith(("#", "-"))
        }
        for package in repair_dependencies or []:
            if package.lower() not in declared_names:
                lines.append(package)
                declared_names.add(package.lower())
        return "\n".join(lines) + ("\n" if lines else "")

    @staticmethod
    def _filter_platform_incompatible(lines: list[str]) -> list[str]:
        """剔除与当前平台不可能兼容的锁定项。

        Linux+CUDA 机器上 ``pip freeze`` 出的 requirements 会锁定
        ``nvidia-*``/``triton`` 等 torch 的 CUDA 传递依赖；这些 wheel 只在
        Linux 平台发布，在 macOS/Windows 上 pip 直接报"no matching
        distribution"。torch 本体在非 Linux 平台安装的是不依赖它们的
        CPU/默认构建，剔除后语义不变。无法确定的包一律保留。
        """

        if sys.platform.startswith("linux"):
            return lines
        dropped: list[str] = []
        kept: list[str] = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith(("#", "-")):
                kept.append(line)
                continue
            name = re.split(r"[=<>!~\[\s]", stripped, maxsplit=1)[0].strip().lower()
            if name.startswith("nvidia-") or name in {"triton", "pytorch-triton"}:
                dropped.append(stripped)
                continue
            kept.append(line)
        if dropped:
            # 记录进锁文件注释，保证可追溯（不影响 pip 解析）。
            kept.insert(
                0,
                "# filtered for non-Linux build host (Linux-only CUDA wheels): "
                + ", ".join(dropped[:32]),
            )
        return kept

    def _generate_import_smoke_test(self, lockfile: str) -> str:
        packages = []
        for line in lockfile.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "-")):
                continue
            packages.append(re.split(r"[=<>!~\[]", stripped, maxsplit=1)[0])
        return (
            "from importlib import import_module\n"
            "from importlib.metadata import packages_distributions, version\n"
            f"packages = {packages!r}\n"
            "mapping = packages_distributions()\n"
            "for package in packages:\n"
            "    print(package, version(package))\n"
            "    modules = [name for name, dists in mapping.items() if package in dists]\n"
            "    imported = False\n"
            "    for module in modules:\n"
            "        try:\n"
            "            import_module(module)\n"
            "            imported = True\n"
            "            break\n"
            "        except ImportError:\n"
            "            continue\n"
            "    if modules and not imported:\n"
            "        raise SystemExit(f'no importable module for {package}: {modules}')\n"
            "print('environment import metadata check passed')\n"
        )

    def _render_candidate_memory(self, result: EnvironmentBuildResult) -> str:
        return (
            f"# environment.{self.task.task_id}\n\n"
            "## 摘要 (L1)\n"
            f"后端: {result.environment_backend}, 环境引用: {result.environment_ref}, "
            f"import 测试通过: {result.import_test_passed}\n\n"
            "## 细节 (L2)\n"
            f"- dependency_analysis: {result.dependency_analysis[:300]}\n"
            f"- cuda_compatible: {result.cuda_compatible}\n"
            f"- pytorch_compatible: {result.pytorch_compatible}\n"
            f"- environment_fingerprint: {result.environment_fingerprint}\n"
            f"- cache_ref: {result.cache_ref}\n"
            f"- cache_hit: {result.cache_hit}\n"
            f"- cache_rebuilt: {result.cache_rebuilt}\n"
            f"- selected_conda_source: {result.selected_conda_source}\n"
            f"- selected_pip_source: {result.selected_pip_source}\n"
            f"- source_attempts: {len(result.source_attempts)}\n\n"
            "## 证据 (L3)\n"
            f"- install_log_tail: {result.install_log_tail[-500:]}\n"
        )
