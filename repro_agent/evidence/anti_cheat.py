"""反作弊检查：禁用词扫描 + "抄论文素材"检测。

复用来源：
    直接复用 paper-replication-paper 项目的两项反作弊机制（详见
    ``doc/paper-replication-paper_架构分析.md`` 第 5 节）：

    1. **禁用词扫描**（``SUSPICIOUS_BASELINE_MARKERS``）：在方法标签、
       实现摘要、执行命令、代码内容里扫描"pattern generator"/
       "fit to paper"/"hard-coded paper" 等短语，命中即拒绝把该产物
       登记为"忠实复现"（baseline_faithful）。这是针对"用曲线拟合
       论文数字来伪装真实实验结果"这种已知作弊手法的黑名单防御。
    2. **产物与论文素材哈希比对**：如果复现产物的哈希与论文自带的
       图片/代码素材完全一致，说明是直接复制的论文原图/原代码，
       不能算作复现产物。

为什么要把这套机制迁移到 ReproAgent：
    设计文档要求"子智能体不能自行宣布整个实验复现成功"（§3 原则
    21），并且反思智能体需要排查"论文理解是否正确""代码路径是否
    正确"等维度（§11.3）——但设计文档没有显式提到"如何防止子智能体
    伪造实验结果/直接复制论文图表来蒙混过关"这个风险点。这属于
    LLM Agent 执行长任务时的已知失败模式（偷懒/幻觉完成），
    paper-replication 项目已经用真实的 12 次案例研究验证过这套机制
    的有效性（见该项目 analysis 部分），因此直接复用，用于结果验证
    子智能体（§9.9）和反思智能体（§9.10）的核验环节。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from repro_agent.evidence.hashing import files_are_identical

# 直接沿用 paper-replication-paper 的禁用词表（见该项目
# ``skill/claude-code/paper-replication/scripts/paper_replication.py``
# 第 83-98 行 ``SUSPICIOUS_BASELINE_MARKERS``），按论文复现场景补充了
# 两个通用的中文等价表达，方便扫描中文实现摘要。
SUSPICIOUS_BASELINE_MARKERS: tuple[str, ...] = (
    "artifact generator",
    "pattern generator",
    "matches reported figure/table patterns",
    "match reported figure/table patterns",
    "matches reported figure patterns",
    "reported figure/table patterns",
    "reported figure patterns",
    "paper pattern",
    "fit to paper",
    "fitted to paper",
    "curve-fit to paper",
    "curve fit to paper",
    "hard-coded paper",
    "hard coded paper",
    "拟合论文数字",
    "硬编码论文结果",
)


@dataclass
class AntiCheatFinding:
    passed: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "reasons": self.reasons}


def scan_suspicious_markers(*texts: str) -> AntiCheatFinding:
    """扫描一组文本（方法标签/实现摘要/执行命令/代码内容等）是否命中
    禁用词表。命中任意一处即判定为不通过。
    """

    reasons = []
    for text in texts:
        if not text:
            continue
        lowered = text.lower()
        for marker in SUSPICIOUS_BASELINE_MARKERS:
            if marker.lower() in lowered:
                reasons.append(f"命中可疑标记词: '{marker}'")
    return AntiCheatFinding(passed=len(reasons) == 0, reasons=reasons)


def check_not_copied_from_paper_assets(
    artifact_path: str | Path, paper_asset_paths: list[str | Path]
) -> AntiCheatFinding:
    """检查复现产物是否与论文自带素材（图片/代码）字节级完全一致。

    对应 paper-replication 项目 ``matched_row_errors`` 中的
    ``artifact_matches_reference`` 检查（§5.2）：如果完全一致，
    说明可能是直接复制了论文的图/代码，而不是真实复现出来的。
    """

    reasons = []
    for asset_path in paper_asset_paths:
        if files_are_identical(artifact_path, asset_path):
            reasons.append(
                f"产物 {artifact_path} 与论文素材 {asset_path} 字节完全一致，"
                "疑似直接复制论文原始素材而非真实复现"
            )
    return AntiCheatFinding(passed=len(reasons) == 0, reasons=reasons)


def check_code_path_not_inside_paper_or_artifacts(
    code_path: str, forbidden_dir_markers: tuple[str, ...] = ("artifacts/", "paper/")
) -> AntiCheatFinding:
    """检查"实现代码"路径不能位于 artifacts/、paper/ 等目录内
    （§5.2：code_path/config_path 不能位于 artifacts/ 或 paper/
    目录或论文源码树内，防止把论文自带代码伪装成自己的实现）。
    """

    normalized = code_path.replace("\\", "/")
    for marker in forbidden_dir_markers:
        if marker in normalized:
            return AntiCheatFinding(
                passed=False,
                reasons=[
                    f"实现代码路径 '{code_path}' 位于禁止目录 '{marker}' 下，"
                    "可能是论文自带代码或生成产物的伪装"
                ],
            )
    return AntiCheatFinding(passed=True)
