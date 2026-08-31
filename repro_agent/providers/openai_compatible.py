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
import time
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
        deadline = time.monotonic() + max(0.0, params.timeout_seconds)
        request_trace: dict = {"request_count": 0, "usages": []}
        body = self._post_with_format_fallback(
            payload,
            params,
            deadline=deadline,
            request_trace=request_trace,
        )
        # 推理型模型护栏：若模型把全部输出预算耗在思考 token 上（content
        # 为空、finish_reason=length、reasoning_tokens == completion_tokens），
        # 在同一个总超时预算内重发一次并请求关闭思考。兼容网关如果不支持
        # 该参数会返回协议错误，此时保留原响应交给上层结构化重试处理。
        if self._is_reasoning_budget_exhausted(body):
            logger.warning(
                "LLM burned the entire completion budget on reasoning tokens "
                "(empty content, finish_reason=length); retrying with "
                "reasoning_effort=none"
            )
            no_think_payload = dict(payload)
            no_think_payload["reasoning_effort"] = "none"
            try:
                no_think_body = self._post_with_format_fallback(
                    no_think_payload,
                    params,
                    deadline=deadline,
                    request_trace=request_trace,
                )
                if self._has_usable_output(no_think_body):
                    body = no_think_body
            except LLMProviderError as exc:
                logger.warning(
                    "reasoning-disabled retry failed (%s); keeping original response", exc
                )
        body = self._attach_aggregate_usage(body, request_trace)
        return self._parse_response(body)

    def _post_with_format_fallback(
        self,
        payload: dict,
        params: LLMRequestParams,
        *,
        deadline: float,
        request_trace: dict,
    ) -> dict:
        try:
            return self._post_once(payload, deadline, request_trace)
        except LLMProviderError as exc:
            # 部分 OpenAI 兼容网关不支持
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
            stripped = {key: value for key, value in payload.items() if key != "response_format"}
            # fallback 请求也可能遇到瞬时网络异常（RemoteDisconnected
            # 等），_post_chat_completion 已将其归一化为可重试的
            # LLMProviderError；这里直接 re-raise 交给上层
            # retry.call_with_retry 的指数退避重试处理，避免在
            # complete() 内部死循环重试同一个 fallback。
            return self._post_once(stripped, deadline, request_trace)

    def _post_once(self, payload: dict, deadline: float, request_trace: dict) -> dict:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise LLMProviderError(
                "timeout calling LLM API before fallback request",
                is_retryable=True,
            )
        request_trace["request_count"] += 1
        body = self._post_chat_completion(payload, remaining)
        usage = body.get("usage") if isinstance(body, dict) else None
        if isinstance(usage, dict):
            request_trace["usages"].append(usage)
        return body

    @staticmethod
    def _has_usable_output(body: dict) -> bool:
        message = (body.get("choices") or [{}])[0].get("message") or {}
        return bool(message.get("content") or message.get("tool_calls"))

    @classmethod
    def _attach_aggregate_usage(cls, body: dict, request_trace: dict) -> dict:
        """Expose physical-request usage to the outer metering decorator."""

        usages = request_trace.get("usages") or []
        aggregate = dict(body.get("usage") or {})
        aggregate["input_tokens"] = sum(
            cls._usage_int(usage, "input_tokens", "prompt_tokens")
            for usage in usages
        )
        aggregate["output_tokens"] = sum(
            cls._usage_int(usage, "output_tokens", "completion_tokens")
            for usage in usages
        )
        aggregate["request_count"] = max(
            1, int(request_trace.get("request_count") or 1)
        )
        aggregate["successful_request_count"] = len(usages)

        cached_tokens = 0
        cache_write_tokens = 0
        for usage in usages:
            details = (
                usage.get("input_tokens_details")
                or usage.get("prompt_tokens_details")
                or {}
            )
            cached_tokens += cls._usage_int(
                details, "cached_tokens", fallback=usage.get("cached_tokens")
            )
            cache_write_tokens += cls._usage_int(
                details,
                "cache_write_tokens",
                fallback=usage.get("cache_write_tokens"),
            )
        aggregate["input_tokens_details"] = {
            "cached_tokens": cached_tokens,
            "cache_write_tokens": cache_write_tokens,
        }

        merged = dict(body)
        merged["usage"] = aggregate
        return merged

    @staticmethod
    def _usage_int(
        usage: dict,
        primary: str,
        alternate: str | None = None,
        *,
        fallback=None,
    ) -> int:
        value = usage.get(primary)
        if not value and alternate:
            value = usage.get(alternate)
        if not value:
            value = fallback
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _is_reasoning_budget_exhausted(body: dict) -> bool:
        """判断响应是否为“思考耗尽预算、正文为空”。"""

        choices = body.get("choices") or []
        if not choices:
            return False
        choice = choices[0]
        if choice.get("finish_reason") != "length":
            return False
        if (choice.get("message") or {}).get("content"):
            return False
        usage = body.get("usage") or {}
        completion = usage.get("output_tokens") or usage.get("completion_tokens")
        details = (
            usage.get("output_tokens_details")
            or usage.get("completion_tokens_details")
            or {}
        )
        reasoning = details.get("reasoning_tokens")
        if not isinstance(completion, int) or not isinstance(reasoning, int):
            return False
        return completion > 0 and reasoning >= completion

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
