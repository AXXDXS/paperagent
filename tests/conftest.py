"""端到端回归测试的共享 fixture。

这里搭建的是一个最小但真实的 ``MainAgent`` 环境（真实 SQLite、真实
沙箱目录、真实 ``AgentDispatcher`` 后台线程），只把 LLM 换成
``MockLLMProvider``——回归测试要验证的是本轮改动引入的编排/并发/
授权机制本身是否正确，而不是某个具体子智能体的业务逻辑，因此不需要
真实的论文/代码仓库输入。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from repro_agent.domain.job import JobInputs, ReproductionJob
from repro_agent.orchestrator.main_agent import MainAgent, MainAgentConfig
from repro_agent.providers.base import LLMResponse
from repro_agent.providers.mock import MockLLMProvider


@pytest.fixture()
def work_dir(tmp_path: Path) -> Path:
    return tmp_path / "job_work_dir"


@pytest.fixture()
def sample_paper(tmp_path: Path) -> Path:
    paper_path = tmp_path / "paper.txt"
    paper_path.write_text(
        "Title: A Toy Paper\n\nWe train a model with learning_rate=0.001, "
        "batch_size=32, epochs=10 and report accuracy=0.95 on the test set.",
        encoding="utf-8",
    )
    return paper_path


@pytest.fixture()
def sample_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "target_repo"
    repo_dir.mkdir()
    (repo_dir / "train.py").write_text("print('training entry point')\n", encoding="utf-8")
    return repo_dir


def _mock_llm_json(payload: dict) -> LLMResponse:
    return LLMResponse(content=json.dumps(payload, ensure_ascii=False))


@pytest.fixture()
def mock_provider() -> MockLLMProvider:
    """返回一个对所有已知子智能体 Prompt 都能给出合法 JSON 响应的 mock。

    不区分具体任务类型，直接给一个通用的、包含各任务解析器所需最小
    字段的 JSON——各子智能体的 ``_parse_llm_output`` 对缺失字段基本都
    有默认值兜底（见各 agent.py 实现），所以一份通用响应足以让整条
    初始任务链（paper_analysis -> code_analysis -> resource_check ->
    specification -> environment_build）都能正常跑通并产出
    ``output/result.json``。
    """

    provider = MockLLMProvider(
        fallback_response=_mock_llm_json(
            {
                "parameters": [
                    {
                        "name": "learning_rate",
                        "value": 0.001,
                        "experiment_scope": "main",
                        "provenance": "PAPER_EXPLICIT",
                        "page": "3",
                        "section": "4.1",
                        "original_text": "learning_rate=0.001",
                        "confidence": 0.95,
                        "is_inferred": False,
                    }
                ],
                "notes": "mock extraction",
                "entry_points": ["train.py"],
                "data_pipeline_summary": "mock data pipeline",
                "model_pipeline_summary": "mock model pipeline",
                "training_pipeline_summary": "mock training pipeline",
                "evaluation_pipeline_summary": "mock evaluation pipeline",
                "experiment_output_paths": ["output/result.json"],
                "dataset_status": {},
                "model_status": {},
                "gpu_info": {},
                "blocking_issues": [],
                "fields": {},
                "expected_results": {
                    "accuracy": {
                        "value": 0.9,
                        "tolerance_type": "absolute",
                        "tolerance": 0.01,
                    }
                },
                "resources": {},
                "import_test_passed": True,
                "dockerfile": "FROM python:3.10\n",
                "dependency_analysis": "mock dependency analysis",
            }
        )
    )
    return provider


@pytest.fixture()
def job(sample_paper: Path, sample_repo: Path) -> ReproductionJob:
    inputs = JobInputs(
        paper_path=str(sample_paper),
        repository_path=str(sample_repo),
        target_experiments=["main_experiment"],
    )
    return ReproductionJob(inputs=inputs)


@pytest.fixture()
def main_agent(job: ReproductionJob, work_dir: Path, mock_provider: MockLLMProvider) -> MainAgent:
    config = MainAgentConfig(
        memory_root=str(work_dir / "project_memory"),
        sandbox_root=str(work_dir / "sandbox"),
        snapshot_root=str(work_dir / "context_snapshots"),
        db_path=str(work_dir / "repro_agent.db"),
        model="mock-model",
    )
    agent = MainAgent(job, config, mock_provider)
    return agent
