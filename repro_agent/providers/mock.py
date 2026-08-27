"""Mock Provider：用于单元测试和无网络环境下的本地开发/演示。

按注册的规则（关键词匹配 → 固定响应）返回结果，不产生任何真实网络
调用，方便在 CI 或无 API Key 的环境中运行整个主循环的集成测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from repro_agent.providers.base import LLMMessage, LLMRequestParams, LLMResponse


@dataclass
class MockLLMProvider:
    """按顺序返回预设响应，或按 messages 内容匹配规则返回响应。"""

    scripted_responses: list[LLMResponse] = field(default_factory=list)
    fallback_response: LLMResponse = field(
        default_factory=lambda: LLMResponse(content="{}")
    )
    rules: list[tuple[Callable[[str], bool], LLMResponse]] = field(default_factory=list)
    call_count: int = 0
    call_log: list[list[LLMMessage]] = field(default_factory=list)
    params_log: list[LLMRequestParams] = field(default_factory=list)
    """每次 ``complete`` 调用时收到的完整 ``LLMRequestParams``，主要
    供测试断言"这一次调用到底带了哪些工具描述"（``params.tools``），
    验证"按需下发工具子集"而非"总是暴露全部授权工具"这一行为。
    """

    def add_rule(self, predicate: Callable[[str], bool], response: LLMResponse) -> None:
        self.rules.append((predicate, response))

    def complete(self, messages: list[LLMMessage], params: LLMRequestParams) -> LLMResponse:
        self.call_log.append(messages)
        self.params_log.append(params)
        last_user_content = ""
        for m in reversed(messages):
            if m.role == "user":
                last_user_content = m.content
                break

        for predicate, response in self.rules:
            if predicate(last_user_content):
                self.call_count += 1
                return response

        if self.call_count < len(self.scripted_responses):
            response = self.scripted_responses[self.call_count]
            self.call_count += 1
            return response

        self.call_count += 1
        return self.fallback_response
