"""候选记忆的验证流水线（设计文档 §15.2：验证 → 冲突检查 → 敏感信息检查）。

设计要点：
    - 冲突检查：同一 ``topic`` 下如果已有正式记忆条目，且新旧内容
      对同一字段给出不同结论，不能静默覆盖（§9.4 的"存在冲突时不能
      静默覆盖"原则同样适用于记忆系统，不只是实验规格字段）。
    - 敏感信息检查：借鉴 DeerFlow DeerMem 记忆后端的
      "提示注入防御"思路（``doc/DeerFlow_架构分析.md`` 第 5.2 节
      ``_escape_memory_for_prompt``）——但本系统的记忆完全来自子
      智能体对论文/代码/日志的分析结果，不存在"用户可控输入注入
      prompt"的攻击面，因此这里的"敏感信息检查"聚焦于更贴合本场景
      的风险：避免把 API Key / 凭证 / 内网地址等误写入 Markdown
      记忆文件（这些文件可能会被最终报告直接引用并展示给用户）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from repro_agent.memory.candidate import CandidateMemory

_SENSITIVE_PATTERNS = [
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9\-_]{16,}"),
    re.compile(r"(?i)secret[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9\-_]{8,}"),
    re.compile(r"-----BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(r"(?i)password\s*[:=]\s*['\"]?\S{6,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9\-_.]{20,}"),
]


@dataclass
class MemoryValidationResult:
    accepted: bool
    conflicts: list[str] = field(default_factory=list)
    sensitive_findings: list[str] = field(default_factory=list)
    rejection_reason: str = ""


def scan_sensitive_content(text: str) -> list[str]:
    findings = []
    for pattern in _SENSITIVE_PATTERNS:
        if pattern.search(text):
            findings.append(f"检测到疑似敏感信息模式: {pattern.pattern[:40]}...")
    return findings


def detect_conflict(
    candidate: CandidateMemory, existing_entries: list[CandidateMemory]
) -> list[str]:
    """检查同一 topic 下是否存在结论冲突的既有记忆条目。

    这里的"冲突"判定是保守且显式的：只要 details 中出现同名字段但
    取值不同，就上报冲突，交给主智能体人工判断谁对谁错（对应 §9.4
    "存在冲突时不能静默覆盖"），不在这里自动仲裁。
    """

    conflicts = []
    for existing in existing_entries:
        if existing.topic != candidate.topic:
            continue
        for key, value in candidate.details.items():
            if key in existing.details and existing.details[key] != value:
                conflicts.append(
                    f"字段 '{key}' 冲突: 既有记忆={existing.details[key]!r}, "
                    f"新候选={value!r} (existing candidate_id={existing.candidate_id})"
                )
    return conflicts


def validate_candidate(
    candidate: CandidateMemory, existing_entries: list[CandidateMemory]
) -> MemoryValidationResult:
    """§15.2 完整验证流水线：冲突检查 + 敏感信息检查。

    只有两项检查都通过才允许"转正"写入正式 Markdown 记忆；存在冲突
    时仍然允许写入（因为冲突需要暴露给主智能体而不是拒绝记录），
    但会在结果里标注 ``conflicts``，供主智能体决定如何处理；存在
    敏感信息则直接拒绝，因为记忆文件可能被最终报告引用展示给用户。
    """

    text_blob = candidate.to_markdown()
    sensitive = scan_sensitive_content(text_blob)
    if sensitive:
        return MemoryValidationResult(
            accepted=False,
            sensitive_findings=sensitive,
            rejection_reason="候选记忆包含疑似敏感信息，已拒绝写入正式记忆",
        )

    conflicts = detect_conflict(candidate, existing_entries)
    return MemoryValidationResult(accepted=True, conflicts=conflicts)
