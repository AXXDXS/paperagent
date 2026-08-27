"""通用基础工具：时间、ID 生成。

设计取舍说明：
    设计文档没有规定 ID 的具体生成规则，这里参考 DeepCode
    （``core/domain/common.py`` 中的 ``new_id``/``utc_now`` 模式）采用
    "前缀 + uuid4 十六位摘要"的可读 ID，并统一使用带时区的 UTC 时间，
    避免跨进程/跨机器出现时区不一致导致超时判断错误。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone


def utc_now() -> datetime:
    """返回带时区信息的当前 UTC 时间。"""

    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    """生成形如 ``{prefix}_{12位十六进制}`` 的可读 ID。"""

    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def iso(dt: datetime | None) -> str | None:
    """把 datetime 序列化为 ISO-8601 字符串，None 保持 None。"""

    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat()
