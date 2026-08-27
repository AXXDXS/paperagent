"""Provider decorator that emits usage for every non-replayed model call."""

from __future__ import annotations

from collections.abc import Callable

from repro_agent.providers.base import (
    LLMMessage,
    LLMProvider,
    LLMRequestParams,
    LLMResponse,
)


class MeteredLLMProvider:
    def __init__(
        self,
        provider: LLMProvider,
        callback: Callable[[LLMRequestParams, LLMResponse], None],
    ):
        self._provider = provider
        self._callback = callback

    def complete(
        self, messages: list[LLMMessage], params: LLMRequestParams
    ) -> LLMResponse:
        response = self._provider.complete(messages, params)
        self._callback(params, response)
        return response
