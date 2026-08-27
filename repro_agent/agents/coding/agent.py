"""代码修改子智能体（设计文档 §9.6）。

职责：补全缺失代码、修复路径/参数传递/数据接口/模型接口/训练逻辑/
评测逻辑、添加单元测试、提交代码 Diff。

**硬约束（§9.6）**：多个子智能体不能并行修改同一个公共目录。本实现
不在宿主机调用 Git（仓库 hook 也属于不可信代码），而是由控制面为每个
task attempt 复制一份独立的可写仓库到沙箱 ``workspace/repository``；
后续写入和回归测试都只发生在该副本中。旧 attempt 与其他任务看不到
这份副本，也不会修改用户的原仓库。

风险预算：``coding`` -> HIGH_RISK（write_file/execute_command 跑单元
测试都是高危操作）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from repro_agent.agents.base import AgentRunResult, BaseSubAgent
from repro_agent.domain.enums import FailureType
from repro_agent.domain.task import FailureReport
from repro_agent.tools.base import ToolInputValidationError
from repro_agent.llm_output import CODING_PLAN_SCHEMA, parse_structured_json


@dataclass
class CodeChangeResult:
    worktree_path: str = ""
    branch_name: str = ""
    changed_files: list[str] = field(default_factory=list)
    diff_summary: str = ""
    unit_test_added: bool = False
    unit_test_passed: bool = False
    base_repository_digest: str = ""
    modified_repository_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "worktree_path": self.worktree_path,
            "branch_name": self.branch_name,
            "changed_files": self.changed_files,
            "diff_summary": self.diff_summary,
            "unit_test_added": self.unit_test_added,
            "unit_test_passed": self.unit_test_passed,
            "base_repository_digest": self.base_repository_digest,
            "modified_repository_digest": self.modified_repository_digest,
        }


class CodingAgent(BaseSubAgent):
    task_type = "coding"
    system_prompt = (
        "你是 ReproAgent 系统的代码修改子智能体。你的任务是补全缺失代码、"
        "修复路径/参数传递/数据接口/模型接口/训练逻辑/评测逻辑问题，并为"
        "修改添加单元测试。你只能在当前 attempt 的独立沙箱仓库副本中工作，"
        "绝不能直接修改主代码目录，也不能影响其他并行任务。"
    )

    def run(self) -> AgentRunResult:
        inputs = self.task.definition.inputs
        repository_path = inputs.get("repository_path", "")
        repository_workdir = inputs.get("repository_workdir", "")
        branch_name = inputs.get("branch_name", f"isolated/{self._attempt_id}")
        fix_instructions = inputs.get("fix_instructions", "")
        failure_context = str(inputs.get("failure_context", ""))
        failing_command = inputs.get("failing_command", [])
        failing_tier = str(inputs.get("failing_tier", ""))

        if not repository_path or not repository_workdir:
            failure = FailureReport(
                failure_type=FailureType.INPUT_MISSING,
                failed_step="setup_worktree",
                error_message="coding task has no staged writable repository",
                likely_causes=["任务未声明 repository_path，或沙箱输入分期失败"],
                recommended_action="由主智能体补充 repository_path 并重新创建 attempt 沙箱",
            )
            return AgentRunResult(succeeded=False, failure_report=failure)

        result = CodeChangeResult(
            worktree_path=repository_workdir, branch_name=branch_name
        )
        base_hash = self.call_tool("hash_path", path=repository_path)
        result.base_repository_digest = base_hash.get("sha256", "")

        change_plan = self._plan_changes(
            repository_path,
            fix_instructions,
            failure_context=failure_context,
            failing_command=failing_command,
            failing_tier=failing_tier,
        )
        for file_change in change_plan.get("files", []):
            rel_path = self._validated_relative_path(file_change["path"])
            full_path = f"{repository_workdir.rstrip('/')}/{rel_path}"
            self.call_tool("write_file", path=full_path, content=file_change["content"])
            result.changed_files.append(rel_path)

        result.diff_summary = change_plan.get("summary", "")

        if change_plan.get("unit_test"):
            test_rel_path = self._validated_relative_path(change_plan["unit_test"]["path"])
            test_path = f"{repository_workdir.rstrip('/')}/{test_rel_path}"
            self.call_tool("write_file", path=test_path, content=change_plan["unit_test"]["content"])
            result.unit_test_added = True
            test_run = self.call_tool(
                "execute_command",
                command=["python", "-m", "pytest", test_rel_path, "-q"],
                working_dir=repository_workdir,
                timeout_seconds=300,
            )
            result.unit_test_passed = test_run.get("exit_code") == 0

        modified_hash = self.call_tool("hash_path", path=repository_workdir)
        result.modified_repository_digest = modified_hash.get("sha256", "")

        if (
            not result.changed_files
            or not result.unit_test_added
            or not result.unit_test_passed
        ):
            return AgentRunResult(
                succeeded=False,
                outputs=result.to_dict(),
                failure_report=FailureReport(
                    failure_type=FailureType.CODE_ERROR,
                    failed_step="apply_or_validate_patch",
                    error_message=(
                        "repair plan produced no file changes"
                        if not result.changed_files
                        else "repair requires a passing generated regression test"
                    ),
                    recommended_action="重新生成最小补丁并在独立 attempt 工作区验证",
                ),
            )

        result_payload = result.to_dict()
        self.write_json_output("result.json", result_payload)
        self.write_candidate_memory(self._render_candidate_memory(result))
        for candidate in change_plan.get("reusable_code_candidates", []):
            self.report_reusable_code_candidate(candidate)

        return AgentRunResult(succeeded=True, outputs=result_payload, candidate_memory_written=True)

    def _plan_changes(
        self,
        repository_path: str,
        fix_instructions: str,
        *,
        failure_context: str = "",
        failing_command: Any = None,
        failing_tier: str = "",
    ) -> dict[str, Any]:
        source_files = self.call_tool(
            "find_files", pattern="*.py", root=repository_path, max_results=100
        ).get("matches", [])
        traceback_filenames = {
            PurePosixPath(path.replace("\\", "/")).name
            for path in re.findall(
                r"File\s+[\"']([^\"']+\.py)[\"']",
                failure_context,
            )
        }
        source_files = sorted(
            source_files,
            key=lambda path: (
                PurePosixPath(str(path)).name not in traceback_filenames,
                str(path),
            ),
        )
        excerpts: dict[str, str] = {}
        for path in source_files[:12]:
            try:
                read = self.call_tool(
                    "read_file", path=f"{repository_path.rstrip('/')}/{path}",
                    start_line=1, end_line=400,
                )
            except Exception:
                continue
            excerpts[path] = read.get("content", "")[:20_000]
        prompt = (
            f"修复指令: {fix_instructions}\n\n"
            f"失败层级: {failing_tier}\n"
            f"失败命令: {failing_command or []}\n"
            f"原始运行诊断:\n{failure_context[-8000:]}\n\n"
            f"相关源码摘录: {excerpts}\n\n"
            "请规划需要修改/新增的文件，输出 JSON:\n"
            '{"summary": "...", "files": [{"path": "relative/path.py", "content": "..."}], '
            '"unit_test": {"path": "tests/test_x.py", "content": "..."}, '
            '"reusable_code_candidates": [...]}\n'
            "unit_test 必须是针对上述运行错误的最小回归测试；补丁必须修复根因，"
            "不能吞掉异常、硬编码期望结果或修改测试来掩盖错误。\n"
            "如果本次确实编写了与当前仓库解耦、以后可能重复使用的纯函数，"
            "将它作为 reusable_code_candidates 上报；否则返回空数组。候选必须"
            "是自包含 Python 代码，声明 functional_key、入口函数、输入/输出"
            "JSON Schema、至少一个 arguments/expected 行为测试、适用任务类型"
            "和依赖。禁止文件写入、网络、GPU、动态执行和项目私有 import；"
            "不要为了凑候选而上报一次性胶水代码。"
        )
        # 这一步只是让模型规划"要改哪些文件、改成什么内容"，返回纯 JSON
        # 文本；真正的写文件/跑测试动作都在 run() 里由确定性代码调用
        # write_file/execute_command 完成，所以这里不需要把
        # write_file/execute_command 等工具暴露给
        # 模型，避免它在规划阶段"顺手"发起工具调用。
        response = self.call_llm(
            prompt,
            temperature=0.2,
            tool_names=[],
            output_schema=CODING_PLAN_SCHEMA,
            output_schema_name="coding_plan",
        )
        plan = parse_structured_json(
            response.content, CODING_PLAN_SCHEMA, label="coding plan output"
        )
        for item in plan.get("files", []):
            # The shared schema has already checked the fields.  Keep this
            # local assertion close to the write boundary as defense in depth.
            if not isinstance(item["path"], str) or not isinstance(item["content"], str):
                raise ToolInputValidationError("repair files must contain string path and content fields")
        unit_test = plan.get("unit_test")
        if unit_test is not None and (not isinstance(unit_test["path"], str) or not isinstance(unit_test["content"], str)):
            raise ToolInputValidationError(
                "unit_test must contain string path and content fields"
            )
        return plan

    @staticmethod
    def _validated_relative_path(path: str) -> str:
        candidate = PurePosixPath(str(path))
        if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
            raise ToolInputValidationError(f"unsafe patch path: {path}")
        return str(candidate)

    def _render_candidate_memory(self, result: CodeChangeResult) -> str:
        return (
            f"# coding.{self.task.task_id}\n\n"
            "## 摘要 (L1)\n"
            f"{result.diff_summary}\n\n"
            "## 细节 (L2)\n"
            f"- worktree: {result.worktree_path} (branch={result.branch_name})\n"
            f"- changed_files: {result.changed_files}\n"
            f"- unit_test_added: {result.unit_test_added}, passed: {result.unit_test_passed}\n\n"
            "## 证据 (L3)\n"
            f"- worktree_path: {result.worktree_path}\n"
        )
