from __future__ import annotations

import json

from repro_agent.agents.base import BaseSubAgent
from repro_agent.domain.task import Task
from repro_agent.orchestrator.task_factory import build_task_definition
from repro_agent.providers.base import LLMMessage, LLMRequestParams, LLMResponse
from repro_agent.providers.mock import MockLLMProvider
from repro_agent.providers.openai_compatible import OpenAICompatibleProvider
from repro_agent.providers.prompt_cache import (
    STABLE_PROMPT_PREFIX,
    STABLE_PROMPT_PREFIX_VERSION,
    build_stable_system_prompt,
    canonicalize_tools,
    prompt_cache_key_for_tools,
)
from repro_agent.sandbox.manager import SandboxManager
from repro_agent.tools.authorization import ToolAuthorizer


def test_shared_prefix_is_long_versioned_and_byte_stable() -> None:
    assert len(STABLE_PROMPT_PREFIX) >= 5_000

    first = build_stable_system_prompt("role A", "dynamic suffix A")
    second = build_stable_system_prompt("role B", "dynamic suffix B")

    assert first[: len(STABLE_PROMPT_PREFIX)] == STABLE_PROMPT_PREFIX
    assert second[: len(STABLE_PROMPT_PREFIX)] == STABLE_PROMPT_PREFIX
    assert first[len(STABLE_PROMPT_PREFIX) :] != second[len(STABLE_PROMPT_PREFIX) :]


def test_tool_schema_and_cache_key_are_independent_of_input_order() -> None:
    read_tool = {
        "function": {
            "parameters": {
                "properties": {"path": {"type": "string"}},
                "type": "object",
            },
            "name": "read_file",
            "description": "read",
        },
        "type": "function",
    }
    stat_tool = {
        "type": "function",
        "function": {
            "description": "stat",
            "name": "get_file_stat",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    forward = canonicalize_tools([read_tool, stat_tool])
    reverse = canonicalize_tools([stat_tool, read_tool])

    assert forward == reverse
    assert [tool["function"]["name"] for tool in forward] == [
        "get_file_stat",
        "read_file",
    ]
    assert prompt_cache_key_for_tools(forward) == prompt_cache_key_for_tools(reverse)
    assert prompt_cache_key_for_tools(forward).startswith(
        f"{STABLE_PROMPT_PREFIX_VERSION}:"
    )


def test_subagent_keeps_prefix_and_cache_route_stable_across_user_inputs(tmp_path) -> None:
    task = Task(
        job_id="job",
        definition=build_task_definition(
            objective="inspect",
            task_type="paper_analysis",
            restrict_tools=["read_file", "get_file_stat"],
        ),
    )
    authorization = ToolAuthorizer().authorize(
        task_id=task.task_id,
        task_type=task.definition.task_type,
        allowed_tools=task.definition.allowed_tools,
        sandbox_ctx=SandboxManager(tmp_path / "sandboxes").create_sandbox(task),
    )
    provider = MockLLMProvider(
        scripted_responses=[LLMResponse(content="{}"), LLMResponse(content="{}")]
    )
    agent = BaseSubAgent(task, authorization, provider)

    # Deliberately vary both the user input and caller-provided tool order.
    agent.call_llm(
        "dynamic user input A",
        tool_names=["read_file", "get_file_stat"],
    )
    agent.call_llm(
        "dynamic user input B",
        tool_names=["get_file_stat", "read_file"],
    )

    first_messages, second_messages = provider.call_log
    assert first_messages[0].content == second_messages[0].content
    assert first_messages[0].content.startswith(STABLE_PROMPT_PREFIX)
    assert first_messages[1].content != second_messages[1].content
    assert provider.params_log[0].tools == provider.params_log[1].tools
    assert (
        provider.params_log[0].prompt_cache_key
        == provider.params_log[1].prompt_cache_key
    )


def test_openai_compatible_payload_carries_cache_key(monkeypatch) -> None:
    captured: dict = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        @staticmethod
        def read() -> bytes:
            return b'{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}'

    def _urlopen(request, timeout):
        captured.update(json.loads(request.data.decode("utf-8")))
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    provider = OpenAICompatibleProvider(
        api_base="https://example.invalid/v1",
        api_key="test-key",
    )
    params = LLMRequestParams(
        model="test-model",
        prompt_cache_key="repro-agent-runtime-v1:test-profile",
    )

    provider.complete([LLMMessage(role="system", content="stable")], params)

    assert captured["prompt_cache_key"] == params.prompt_cache_key
