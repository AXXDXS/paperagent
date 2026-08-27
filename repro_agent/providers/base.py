"""LLM Provider 抽象接口。

抽象出这一层的原因：设计文档全篇不绑定具体模型供应商，主智能体和
十个子智能体都只依赖"给定 messages + 工具描述，返回结构化结果"这
一最小接口，具体接哪个模型（OpenAI/Anthropic/内部网关）由
``providers`` 下的具体实现决定，替换供应商不需要改动任何智能体代码。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass(frozen=True)
class ContentBlock:
    """A validated block of content produced for an LLM message.

    Tool execution remains an in-process Python call, but the value crossing
    into a model conversation is represented explicitly as a content block.
    The current provider abstraction still uses string tool messages, so
    :meth:`to_model_content` is the compatibility adapter at that boundary.
    """

    type: str
    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_model_content(self) -> str:
        """Serialize the block payload for providers using string content."""

        if self.type == "text" and isinstance(self.data, str):
            return self.data
        return json.dumps(self.data, ensure_ascii=False, allow_nan=False)


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    tool_calls: list["ToolCallRequest"] = field(default_factory=list)


@dataclass
class ToolCallRequest:
    """LLM 请求调用某个工具（由子智能体运行时循环解析并通过
    ``ToolAuthorization.call`` 执行，见 tools/authorization.py）。
    """

    tool_name: str
    arguments: dict[str, Any]
    call_id: str
    # Provider parsing must preserve whether the model supplied a valid JSON
    # object.  Replacing malformed JSON with {} would let no-argument tools run.
    arguments_valid: bool = True


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None


@dataclass
class LLMRequestParams:
    """一次请求的可调参数（供重试策略动态调整，见 retry.py）。"""

    model: str
    temperature: float = 0.3
    max_tokens: int = 4096
    tools: list[dict[str, Any]] = field(default_factory=list)
    timeout_seconds: float = 120.0
    prompt_cache_key: Optional[str] = None
    # Optional native JSON Schema hint.  The application still validates the
    # response locally because compatible gateways are allowed to ignore it.
    response_schema: Optional[dict[str, Any]] = None
    response_schema_name: str = "structured_output"


class LLMProviderError(RuntimeError):
    """LLM 调用失败的统一异常类型（供重试策略识别是否可重试）。"""

    def __init__(self, message: str, *, is_context_length_error: bool = False, is_retryable: bool = True):
        super().__init__(message)
        self.is_context_length_error = is_context_length_error
        self.is_retryable = is_retryable


class LLMProvider(Protocol):
    """所有具体 Provider 实现（OpenAI/Anthropic/Mock）都遵循此接口。"""

    def complete(
        self, messages: list[LLMMessage], params: LLMRequestParams
    ) -> LLMResponse:
        ...
