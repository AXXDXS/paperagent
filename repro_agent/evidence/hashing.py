"""基于 SHA-256 的文件哈希与目录快照工具。

复用来源：
    直接复用 paper-replication-paper 项目 ``paper_replication.py`` 中
    "以文件 SHA-256 作为证据链锚点"的思路（见
    ``doc/paper-replication-paper_架构分析.md`` 第 5.1/5.2 节）：
    产物、代码、配置、论文依据引用文档的哈希互相锁定在一份
    provenance 记录里，事后任何一环被篡改都能通过重新计算哈希
    立刻发现。本模块提供最基础的哈希原语，供
    ``evidence/provenance.py`` 组装成完整的证据记录。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


_CHUNK_SIZE = 1024 * 1024


def sha256_of_file(path: str | Path) -> str:
    """计算文件的 SHA-256 十六进制摘要（流式读取，避免大文件占用大量内存）。"""

    p = Path(path)
    hasher = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(_CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of_directory(path: str | Path) -> str:
    """对目录做确定性快照哈希：按相对路径排序后拼接每个文件的哈希，
    保证目录内容不变时哈希稳定，任何文件增删改都会导致哈希变化。
    """

    p = Path(path)
    entries = []
    for file_path in sorted(p.rglob("*")):
        rel = str(file_path.relative_to(p))
        if file_path.is_symlink():
            entries.append(f"{rel}:symlink:{os.readlink(file_path)}")
        elif file_path.is_file():
            entries.append(f"{rel}:{sha256_of_file(file_path)}")
    combined = "\n".join(entries)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def files_are_identical(path_a: str | Path, path_b: str | Path) -> bool:
    """比较两个文件内容是否完全一致（用于反作弊检查：产物是否直接
    复制了论文自带的图/代码，见 evidence/anti_cheat.py）。
    """

    a, b = Path(path_a), Path(path_b)
    if not a.exists() or not b.exists():
        return False
    if a.stat().st_size != b.stat().st_size:
        return False
    return sha256_of_file(a) == sha256_of_file(b)
