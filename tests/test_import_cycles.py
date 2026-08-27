"""Import-boundary regressions must be tested in a fresh Python interpreter."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_agents_can_import_runtime_configuration_without_a_cycle() -> None:
    root = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "from repro_agent.agents.paper.agent import PaperAnalysisAgent; "
            "from repro_agent.agents.code.agent import CodeAnalysisAgent; "
            "from repro_agent.orchestrator.runtime_configuration import normalize_requirements; "
            "assert PaperAnalysisAgent and CodeAnalysisAgent and normalize_requirements",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert process.returncode == 0, process.stderr


def test_orchestrator_public_exports_remain_available() -> None:
    root = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "from repro_agent.orchestrator import MainAgent, build_task_definition; "
            "assert MainAgent and build_task_definition",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert process.returncode == 0, process.stderr
