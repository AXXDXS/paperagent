"""Token 递减式重试 + 指数退避（复用 DeepCode 的健壮性策略）。

复用来源：
    Token 递减策略直接复用 DeepCode
    ``workflows/agent_orchestration_engine.py::_adjust_params_for_retry``
    的思路（见该文件 365-396 行）：当遇到"上下文长度超限"错误时，
    正确的应对是**减少** ``max_tokens``（为 input 腾出空间），而不是
    增大它；每次重试同时降低 ``temperature`` 以提升输出的稳定性和
    可预测性。具体递减节奏沿用原实现：
        第 1 次重试 → retry_max_tokens（通常是 base 的 75%）
        第 2 次重试 → retry_max_tokens × 0.9
        第 3 次及以后 → retry_max_tokens × 0.8
    温度衰减：``max(temperature - retry_count * 0.15, 0.05)``。

    对于非"上下文超限"的普通网络错误/限流错误，采用标准指数退避
    （复用 Pi 项目里"API 调用韧性层"的思路，见
    ``doc/pi-项目分析.md`` 中对话历史相关部分），与 token 递减策略
    独立叠加：既降低 token 预算，也在两次尝试之间等待递增的时间。
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import replace
from typing import Callable

from repro_agent.providers.base import (
    LLMMessage,
    LLMProviderError,
    LLMRequestParams,
    LLMResponse,
)

logger = logging.getLogger(__name__)


def adjust_params_for_retry(
    params: LLMRequestParams, retry_count: int, *, retry_max_tokens: int | None = None
) -> LLMRequestParams:
    """按 DeepCode 的 token 递减策略调整下一次重试的请求参数。"""

    base_retry_tokens = retry_max_tokens or max(1, int(params.max_tokens * 0.75))
    if retry_count == 0:
        new_max_tokens = base_retry_tokens
    elif retry_count == 1:
        new_max_tokens = int(base_retry_tokens * 0.9)
    else:
        new_max_tokens = int(base_retry_tokens * 0.8)

    new_temperature = max(params.temperature - (retry_count * 0.15), 0.05)

    logger.info(
        "adjusting LLM params for retry %d: max_tokens %d -> %d, temperature %.2f -> %.2f",
        retry_count + 1,
        params.max_tokens,
        new_max_tokens,
        params.temperature,
        new_temperature,
    )
    return replace(params, max_tokens=new_max_tokens, temperature=new_temperature)


def exponential_backoff_delay(
    attempt: int, *, base_delay: float = 1.0, max_delay: float = 30.0
) -> float:
    """标准指数退避 + 抖动，避免多个并发子智能体同时重试造成惊群效应。"""

    delay = min(max_delay, base_delay * (2**attempt))
    jitter = random.uniform(0, delay * 0.3)
    return delay + jitter


def call_with_retry(
    fn: Callable[[list[LLMMessage], LLMRequestParams], LLMResponse],
    messages: list[LLMMessage],
    params: LLMRequestParams,
    *,
    max_retries: int = 3,
    retry_max_tokens: int | None = None,
) -> LLMResponse:
    """执行一次 LLM 调用，失败时按错误类型选择重试策略。

    - 上下文长度超限错误：应用 token 递减策略（不等待，立即用更小
      的 max_tokens 重试，因为等待不会改变"超限"这个事实）；
    - 其他可重试错误（网络抖动/限流）：指数退避后用原参数重试；
    - 不可重试错误：立即抛出，不做任何重试尝试。
    """

    current_params = params
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return fn(messages, current_params)
        except LLMProviderError as exc:
            last_error = exc
            if not exc.is_retryable or attempt == max_retries:
                raise
            if exc.is_context_length_error:
                current_params = adjust_params_for_retry(
                    current_params, attempt, retry_max_tokens=retry_max_tokens
                )
            else:
                delay = exponential_backoff_delay(attempt)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s; retrying after %.1fs",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)

    # 理论上不会到达这里（循环内要么 return 要么 raise），加一道保险。
    assert last_error is not None
    raise last_error
