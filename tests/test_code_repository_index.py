from __future__ import annotations

import json
from pathlib import Path

from repro_agent.agents.code.agent import (
    CodeAnalysisAgent,
    normalize_code_analysis_payload,
)
from repro_agent.agents.base import BaseSubAgent
from repro_agent.domain.task import Task
from repro_agent.orchestrator.task_factory import build_task_definition
from repro_agent.providers.base import LLMResponse, ToolCallRequest
from repro_agent.providers.mock import MockLLMProvider
from repro_agent.sandbox.manager import SandboxManager
from repro_agent.tools.authorization import ToolAuthorizer

def _code_task(repo: Path, *, target: str = "rare_experiment") -> Task:
    return Task(
        job_id="job-code-index",
        definition=build_task_definition(
            objective="analyze a large repository",
            task_type="code_analysis",
            inputs={
                "repository_path": str(repo),
                "target_experiments": [target],
                "code_context_budget_tokens": 6_000,
            },
            restrict_tools=[
                "get_repository_map",
                "search_repository_code",
                "read_file",
            ],
        ),
        active_attempt_id="attempt-1",
    )


def _authorization(task: Task, tmp_path: Path):
    sandbox = SandboxManager(tmp_path / "sandboxes").create_sandbox(task)
    authorization = ToolAuthorizer().authorize(
        task_id=task.task_id,
        task_type=task.definition.task_type,
        allowed_tools=task.definition.allowed_tools,
        sandbox_ctx=sandbox,
    )
    return sandbox, authorization


def _make_large_repo(root: Path, *, distractors: int = 120) -> tuple[Path, int]:
    root.mkdir()
    (root / "README.md").write_text(
        "Run the rare experiment with python experiments/rare_driver.py\n",
        encoding="utf-8",
    )
    (root / "configs").mkdir()
    (root / "configs" / "rare.yaml").write_text(
        "learning_rate: 0.0003\noutput_dir: artifacts/rare\n",
        encoding="utf-8",
    )
    distractor_dir = root / "src" / "distractors"
    distractor_dir.mkdir(parents=True)
    for index in range(distractors):
        (distractor_dir / f"module_{index:04d}.py").write_text(
            f"def helper_{index}(value):\n    return value + {index}\n",
            encoding="utf-8",
        )
    ignored = root / "node_modules" / "noise"
    ignored.mkdir(parents=True)
    (ignored / "fake_train.py").write_text(
        "def misleading_training_entry():\n    pass\n", encoding="utf-8"
    )
    experiments = root / "experiments"
    experiments.mkdir()
    filler = "\n".join(f"FILLER_{index} = {index}" for index in range(650))
    target_line = 651
    (experiments / "rare_driver.py").write_text(
        filler
        + "\n"
        + "def run_rare_experiment(config_path='configs/rare.yaml'):\n"
        + "    metric_name = 'rare_validation_score'\n"
        + "    return {'metric': metric_name, 'output': 'artifacts/rare'}\n\n"
        + "if __name__ == '__main__':\n"
        + "    run_rare_experiment()\n",
        encoding="utf-8",
    )
    return root, target_line


def test_lightweight_index_finds_symbol_beyond_old_300_line_limit(tmp_path: Path) -> None:
    repo, target_line = _make_large_repo(tmp_path / "repo")
    task = _code_task(repo)
    sandbox, authorization = _authorization(task, tmp_path)
    repo_root = task.definition.inputs["repository_path"]

    first_map = authorization.call(
        "get_repository_map",
        root=repo_root,
        query="rare_experiment train metric",
        token_budget=1800,
    )
    rendered = authorization.render_result_for_model("get_repository_map", first_map)
    second_map = authorization.call(
        "get_repository_map",
        root=repo_root,
        query="rare_experiment train metric",
        token_budget=1800,
    )
    search = authorization.call(
        "search_repository_code",
        root=repo_root,
        query="run_rare_experiment rare_validation_score",
        max_results=10,
    )

    assert first_map["indexed_file_count"] >= 123
    assert first_map["languages"]["python"] >= 121
    assert "node_modules" not in first_map["repo_map"]
    assert "experiments/rare_driver.py" in first_map["repo_map"]
    assert rendered.data["_tool_result_meta"]["tool_name"] == "get_repository_map"
    assert rendered.data["data"]["repository_digest"] == first_map["repository_digest"]
    assert second_map["cache_hit"] is True
    target_matches = [
        item
        for item in search["results"]
        if item["path"] == "experiments/rare_driver.py"
        and item["symbol"] == "run_rare_experiment"
    ]
    assert target_matches
    assert target_matches[0]["start_line"] == target_line
    assert "rare_validation_score" in target_matches[0]["preview"]
    assert len(target_matches[0]["file_digest"]) == 64

    # A changed file invalidates the manifest and is reparsed; unchanged files
    # remain reusable inside the incremental cache.
    staged_target = Path(sandbox.resolve_readable_path(repo_root)) / "experiments/rare_driver.py"
    staged_target.write_text(
        staged_target.read_text(encoding="utf-8")
        + "\ndef newly_added_metric():\n    return 'incremental_metric'\n",
        encoding="utf-8",
    )
    refreshed_map = authorization.call(
        "get_repository_map", root=repo_root, query="incremental_metric", token_budget=1800
    )
    refreshed_search = authorization.call(
        "search_repository_code", root=repo_root, query="newly_added_metric"
    )
    assert refreshed_map["cache_hit"] is False
    assert refreshed_map["repository_digest"] != first_map["repository_digest"]
    assert any(item["symbol"] == "newly_added_metric" for item in refreshed_search["results"])


def test_code_analysis_agent_uses_repo_map_layered_retrieval_and_evidence(
    tmp_path: Path,
) -> None:
    repo, target_line = _make_large_repo(tmp_path / "repo")
    task = _code_task(repo)
    _, authorization = _authorization(task, tmp_path)
    provider = MockLLMProvider(
        fallback_response=LLMResponse(
            content=json.dumps(
                {
                    "entry_points": ["experiments/rare_driver.py"],
                    "config_system": "YAML",
                    "data_pipeline_summary": "not present in fixture",
                    "model_pipeline_summary": "not present in fixture",
                    "training_pipeline_summary": "run_rare_experiment",
                    "inference_pipeline_summary": "not present in fixture",
                    "evaluation_pipeline_summary": "rare_validation_score",
                    "effective_parameters": {"learning_rate": 0.0003},
                    "experiment_output_paths": ["artifacts/rare"],
                    "matched_run_scripts": {
                        "rare_experiment": "experiments/rare_driver.py"
                    },
                    "tier_commands": {
                        "smoke_test": ["python", "experiments/rare_driver.py"]
                    },
                    "analysis_evidence": [
                        {
                            "path": "experiments/rare_driver.py",
                            "start_line": target_line,
                            "end_line": target_line + 3,
                            "symbol": "run_rare_experiment",
                            "reason": "experiment entry and output metric",
                        }
                    ],
                }
            )
        )
    )

    result = CodeAnalysisAgent(task, authorization, provider).run()

    assert result.succeeded is True
    assert len(result.outputs["repository_digest"]) == 64
    assert result.outputs["analysis_coverage"]["indexed_file_count"] >= 123
    assert result.outputs["analysis_coverage"]["retrieved_evidence_count"] > 0
    assert result.outputs["analysis_evidence"][0]["path"] == "experiments/rare_driver.py"
    assert result.outputs["analysis_evidence"][0]["start_line"] == target_line
    prompt = provider.call_log[0][-1].content
    assert "REPOSITORY MAP" in prompt
    assert "run_rare_experiment" in prompt
    assert len(prompt) < 60_000
    tool_names = {tool["function"]["name"] for tool in provider.params_log[0].tools}
    assert tool_names == {"search_repository_code", "read_file"}


def test_code_analysis_allows_bounded_model_driven_followup_search(tmp_path: Path) -> None:
    repo, _ = _make_large_repo(tmp_path / "repo", distractors=10)
    task = _code_task(repo)
    _, authorization = _authorization(task, tmp_path)
    final_payload = {
        "entry_points": ["experiments/rare_driver.py"],
        "analysis_evidence": [],
    }
    provider = MockLLMProvider(
        scripted_responses=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        tool_name="search_repository_code",
                        arguments={
                            "root": task.definition.inputs["repository_path"],
                            "query": "run_rare_experiment",
                            "max_results": 3,
                        },
                        call_id="search-1",
                    )
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(content=json.dumps(final_payload)),
        ]
    )

    result = CodeAnalysisAgent(task, authorization, provider).run()

    assert result.succeeded is True
    assert provider.call_count == 2
    tool_messages = [message for message in provider.call_log[-1] if message.role == "tool"]
    assert tool_messages
    model_tool_result = json.loads(tool_messages[0].content)
    assert model_tool_result["_tool_result_meta"]["tool_name"] == "search_repository_code"
    assert model_tool_result["data"]["results"][0]["path"] == "experiments/rare_driver.py"


def test_normalize_code_analysis_payload_coerces_presentation_variants() -> None:
    """真实模型常见变体：脚本值给数组/对象、tier 值给命令串、入口给单串。"""

    payload = {
        "entry_points": "evaluation/locomo/eval_locomo.py",
        "experiment_output_paths": ["runs/output", 3],
        "matched_run_scripts": {
            "main": ["python", "evaluation/locomo/eval_locomo.py"],
            "ablation": {"script": "scripts/eval_hotpotqa.sh"},
            "empty": [],
        },
        "tier_commands": {
            "unit_test": "python -m pytest -q",
            "static_check": ["python", "-m", "compileall", "-q", "."],
            "broken": 42,
        },
    }

    normalize_code_analysis_payload(payload)

    assert payload["entry_points"] == ["evaluation/locomo/eval_locomo.py"]
    assert payload["experiment_output_paths"] == ["runs/output", "3"]
    assert payload["matched_run_scripts"] == {
        "main": "python evaluation/locomo/eval_locomo.py",
        "ablation": "scripts/eval_hotpotqa.sh",
    }
    assert payload["tier_commands"] == {
        "unit_test": ["python", "-m", "pytest", "-q"],
        "static_check": ["python", "-m", "compileall", "-q", "."],
    }


def test_normalize_code_analysis_payload_drops_null_optional_fields() -> None:
    """模型对不适用字段输出显式 null：丢弃后 strict schema 应放行。"""

    payload = {
        "required_user_configuration": [
            {
                "name": "MODEL_NAME",
                "kind": "model_name",
                "delivery": "command_argument",
                "environment_variable": None,
                "argument": "--model",
                "required": True,
                "reason": "选择评测模型",
                "source_ref": "config.py:20",
            },
            None,
        ]
    }

    normalize_code_analysis_payload(payload)

    items = payload["required_user_configuration"]
    assert items == [
        {
            "name": "MODEL_NAME",
            "kind": "model_name",
            "delivery": "command_argument",
            "argument": "--model",
            "required": True,
            "reason": "选择评测模型",
            "source_ref": "config.py:20",
        }
    ]


def test_model_driven_retrieval_rounds_are_hard_bounded(tmp_path: Path) -> None:
    """超过轮次上限不再击穿任务：改为强制收尾，且超限轮的工具不执行。"""

    repo, _ = _make_large_repo(tmp_path / "repo", distractors=1)
    task = _code_task(repo)
    _, authorization = _authorization(task, tmp_path)
    repo_root = task.definition.inputs["repository_path"]
    final_answer = LLMResponse(content='{"entry_points": []}')
    provider = MockLLMProvider(
        scripted_responses=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        tool_name="search_repository_code",
                        arguments={"root": repo_root, "query": "rare experiment"},
                        call_id="round-1",
                    )
                ]
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        tool_name="search_repository_code",
                        arguments={"root": repo_root, "query": "rare metric"},
                        call_id="round-2",
                    )
                ]
            ),
        ],
        fallback_response=final_answer,
    )

    response = BaseSubAgent(task, authorization, provider).call_llm(
        "bounded search",
        tool_names=["search_repository_code"],
        max_tool_rounds=1,
    )

    # 轮次上限仍 是硬上限：第 2 轮的检索没有被执行。
    executed = [log for log in authorization.invocation_log if log.tool_name == "search_repository_code"]
    assert len(executed) == 1
    assert executed[0].arguments["query"] == "rare experiment"
    # 超限后追加了一次无工具的强制收尾调用并返回其结果。
    assert response is final_answer
    assert provider.call_count == 3
    assert provider.params_log[-1].tools is None
    budget_notice = provider.call_log[-1][-1]
    assert budget_notice.role == "user"
    assert "TOOL BUDGET EXHAUSTED" in budget_notice.content
