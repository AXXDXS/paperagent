"""OpenAI 兼容协议的 LLM Provider 实现。

绝大多数国内外模型网关（含企业内部网关）都提供 OpenAI 兼容的
``/chat/completions`` 接口，因此只实现这一种协议即可覆盖最广泛的
部署场景；需要接入协议不同的供应商时，只需新增一个实现
``providers.base.LLMProvider`` 协议的类，不影响其余代码。

网络请求使用标准库 ``urllib``，不引入额外的 HTTP 客户端依赖，
保持 ``paper_agent`` 的依赖面最小化（是否要换成 ``httpx`` 等库
留给未来根据实际部署环境决定）。
"""

from __future__ import annotations

import http.client
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass

from repro_agent.providers.base import (
    LLMMessage,
    LLMProviderError,
    LLMRequestParams,
    LLMResponse,
    ToolCallRequest,
)

logger = logging.getLogger(__name__)

_CONTEXT_LENGTH_ERROR_MARKERS = (
    "maximum context length",
    "context_length_exceeded",
    "too many tokens",
    "reduce the length",
)


@dataclass
class OpenAICompatibleProvider:
    api_base: str
    api_key: str
    default_model: str = "gpt-4o-mini"

    def complete(
        self, messages: list[LLMMessage], params: LLMRequestParams
    ) -> LLMResponse:
        payload = self._build_payload(messages, params)
        try:
            body = self._post_chat_completion(payload, params.timeout_seconds)
        except LLMProviderError as exc:
            # 部分 OpenAI 兼容网关（例如 AIGC 网关的 deepseek-v4-pro）不支持
            # ``response_format=json_schema`` 提示并返回 HTTP 400。本地 JSON
            # Schema 校验（``llm_output.parse_structured_json``）才是权威，
            # 该提示只是尽力引导——因此去掉提示后原样重发一次，不放宽任何
            # 本地校验，fail-closed 语义不变。
            if "response_format" not in payload or "response_format" not in str(exc):
                raise
            logger.warning(
                "LLM gateway rejected the response_format hint (%s); "
                "retrying once without it",
                exc,
            )
            payload = {key: value for key, value in payload.items() if key != "response_format"}
            # fallback 请求也可能遇到瞬时网络异常（RemoteDisconnected
            # 等），_post_chat_completion 已将其归一化为可重试的
            # LLMProviderError；这里直接 re-raise 交给上层
            # retry.call_with_retry 的指数退避重试处理，避免在
            # complete() 内部死循环重试同一个 fallback。
            body = self._post_chat_completion(payload, params.timeout_seconds)
        return self._parse_response(body)

    def _build_payload(
        self, messages: list[LLMMessage], params: LLMRequestParams
    ) -> dict:
        payload = {
            "model": params.model or self.default_model,
            "messages": [self._to_openai_message(m) for m in messages],
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
        }
        if params.tools:
            payload["tools"] = params.tools
        if params.prompt_cache_key:
            payload["prompt_cache_key"] = params.prompt_cache_key
        if params.response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": params.response_schema_name,
                    # Local validation is authoritative.  ``strict=false``
                    # keeps this hint compatible with gateways that implement
                    # JSON Schema response hints but do not implement the
                    # provider-specific strict subset (for example dynamic
                    # metric names under expected_results).
                    "strict": False,
                    "schema": params.response_schema,
                },
            }
        return payload

    def _post_chat_completion(self, payload: dict, timeout_seconds: float) -> dict:
        url = self.api_base.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            normalized_error = error_body.lower()
            is_context_error = any(
                marker in normalized_error for marker in _CONTEXT_LENGTH_ERROR_MARKERS
            )
            rejected_response_format = "response_format" in normalized_error
            # 4xx 中除限流(429)外一般不可重试；5xx 可重试。
            is_retryable = exc.code == 429 or exc.code >= 500 or is_context_error
            safe_detail = (
                "; gateway rejected response_format"
                if rejected_response_format
                else "; response body omitted to protect sensitive data"
            )
            raise LLMProviderError(
                f"HTTP {exc.code} calling LLM API{safe_detail}",
                is_context_length_error=is_context_error,
                is_retryable=is_retryable,
            ) from exc
        except urllib.error.URLError as exc:
            raise LLMProviderError(
                f"network error calling LLM API: {exc}", is_retryable=True
            ) from exc
        except TimeoutError as exc:
            raise LLMProviderError(
                f"timeout calling LLM API after {timeout_seconds}s",
                is_retryable=True,
            ) from exc
        except http.client.HTTPException as exc:
            # http.client.RemoteDisconnected / BadStatusLine /
            # IncompleteRead 等连接级异常是 urllib.error.URLError 的
            # 兄弟分支而非子类，原来的 except 链漏掉了它们；一旦网关
            # 瞬时断连，异常会越过 retry.call_with_retry 的
            # ``except LLMProviderError`` 直接冒泡到子代理线程顶层，
            # 触发"未处理异常 → 心跳超时 → 任务被取消"的连锁反应。
            # 归一化为可重试的 LLMProviderError，让重试层正常工作。
            raise LLMProviderError(
                f"connection error calling LLM API: {exc}",
                is_retryable=True,
            ) from exc

    @staticmethod
    def _to_openai_message(message: LLMMessage) -> dict:
        payload = {"role": message.role, "content": message.content}
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.name:
            payload["name"] = message.name
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.tool_name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]
        return payload

    @staticmethod
    def _parse_response(body: dict) -> LLMResponse:
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        tool_calls = []
        for tc in message.get("tool_calls") or []:
            function = tc.get("function", {})
            arguments_valid = True
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
                arguments_valid = False
            if not isinstance(arguments, dict):
                arguments = {}
                arguments_valid = False
            tool_calls.append(
                ToolCallRequest(
                    tool_name=function.get("name", ""),
                    arguments=arguments,
                    call_id=tc.get("id", ""),
                    arguments_valid=arguments_valid,
                )
            )
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
            usage=body.get("usage", {}),
            raw=body,
        )
