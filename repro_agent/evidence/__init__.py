"""证据链与反作弊模块（复用 paper-replication-paper 的哈希证据链设计）。

对应设计文档 §3 原则 25："每个结果都必须能够追溯到代码、配置、数据、
日志和实验运行"，以及 §10.5 正式实验的可追溯七元组要求。
"""

from repro_agent.evidence.anti_cheat import (
    AntiCheatFinding,
    check_code_path_not_inside_paper_or_artifacts,
    check_not_copied_from_paper_assets,
    scan_suspicious_markers,
)
from repro_agent.evidence.hashing import (
    files_are_identical,
    sha256_of_directory,
    sha256_of_file,
    sha256_of_text,
)
from repro_agent.evidence.provenance import (
    ArtifactProvenance,
    ComparisonEvidence,
    ProvenanceError,
    RunRecord,
    register_artifact_provenance,
    verify_provenance,
)

__all__ = [
    "AntiCheatFinding",
    "ArtifactProvenance",
    "ComparisonEvidence",
    "ProvenanceError",
    "RunRecord",
    "check_code_path_not_inside_paper_or_artifacts",
    "check_not_copied_from_paper_assets",
    "files_are_identical",
    "register_artifact_provenance",
    "scan_suspicious_markers",
    "sha256_of_directory",
    "sha256_of_file",
    "sha256_of_text",
    "verify_provenance",
]
