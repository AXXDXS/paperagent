"""Execute generated pure-function tools inside the existing container sandbox."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from repro_agent.domain.common import new_id
from repro_agent.tools.base import SandboxContext, ToolExecutionError
from repro_agent.tools.write_tools import execute_command

_RESULT_MARKER = "__REPRO_DYNAMIC_TOOL_RESULT__="
_RUNNER = r'''from __future__ import annotations
import importlib.util
import json
import sys

module_path, entry_function, arguments_path = sys.argv[1:4]
spec = importlib.util.spec_from_file_location("generated_tool_impl", module_path)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load generated tool module")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
arguments = json.loads(open(arguments_path, "r", encoding="utf-8").read())
result = getattr(module, entry_function)(**arguments)
encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
print("__REPRO_DYNAMIC_TOOL_RESULT__=" + encoded)
'''


class DynamicToolExecutor:
    """Container-backed executor used by both admission tests and live calls."""

    def run(
        self,
        record: dict[str, Any],
        sandbox_ctx: SandboxContext,
        arguments: dict[str, Any],
        *,
        timeout_seconds: int = 30,
    ) -> Any:
        code = str(record.get("code", ""))
        expected_hash = str(record.get("code_hash", ""))
        if not code or hashlib.sha256(code.encode("utf-8")).hexdigest() != expected_hash:
            raise ToolExecutionError("dynamic tool code is missing or fails its stored digest")

        invocation = new_id("dynamic_invocation")
        tool_id = str(record.get("tool_id", "unknown"))
        relative_root = f".repro_dynamic_tools/{tool_id}/{invocation}"
        host_root = Path(sandbox_ctx.resolve_writable_path(f"workspace://{relative_root}"))
        host_root.mkdir(parents=True, exist_ok=False)
        implementation = host_root / "tool_impl.py"
        runner = host_root / "runner.py"
        argument_file = host_root / "arguments.json"
        implementation.write_text(code, encoding="utf-8")
        runner.write_text(_RUNNER, encoding="utf-8")
        argument_file.write_text(
            json.dumps(arguments, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )

        result = execute_command(
            sandbox_ctx,
            [
                "python",
                f"{relative_root}/runner.py",
                f"{relative_root}/tool_impl.py",
                str(record["entry_function"]),
                f"{relative_root}/arguments.json",
            ],
            timeout_seconds=max(1, min(int(timeout_seconds), 60)),
            allow_network=False,
            working_dir="workspace://.",
            environment={"PYTHONDONTWRITEBYTECODE": "1"},
            gpu_count=0,
            workspace_read_only=True,
        )
        if result.get("mock"):
            raise ToolExecutionError(
                "mock execution cannot verify or execute a generated tool"
            )
        if result.get("exit_code") != 0:
            raise ToolExecutionError(
                "generated tool process failed: " + str(result.get("stderr", ""))[-1000:]
            )
        for line in reversed(str(result.get("stdout", "")).splitlines()):
            if line.startswith(_RESULT_MARKER):
                try:
                    return json.loads(line[len(_RESULT_MARKER) :])
                except json.JSONDecodeError as exc:
                    raise ToolExecutionError("generated tool returned invalid JSON") from exc
        raise ToolExecutionError("generated tool produced no structured result marker")
