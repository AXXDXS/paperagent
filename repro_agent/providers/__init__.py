"""LLM Provider 抽象层：统一接口 + Token 递减重试 + 指数退避。

复用来源见 ``retry.py`` 顶部说明（DeepCode token 递减策略 +
标准指数退避）。
"""

from repro_agent.providers.base import (
    ContentBlock,
    LLMMessage,
    LLMProvider,
    LLMProviderError,
    LLMRequestParams,
    LLMResponse,
    ToolCallRequest,
)
from repro_agent.providers.mock import MockLLMProvider
from repro_agent.providers.openai_compatible import OpenAICompatibleProvider
from repro_agent.providers.retry import (
    adjust_params_for_retry,
    call_with_retry,
    exponential_backoff_delay,
)

__all__ = [
    "ContentBlock",
    "LLMMessage",
    "LLMProvider",
    "LLMProviderError",
    "LLMRequestParams",
    "LLMResponse",
    "MockLLMProvider",
    "OpenAICompatibleProvider",
    "ToolCallRequest",
    "adjust_params_for_retry",
    "call_with_retry",
    "exponential_backoff_delay",
]
