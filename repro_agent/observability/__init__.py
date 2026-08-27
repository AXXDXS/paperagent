"""最终报告生成与复现结论判定（设计文档 §2、§20）。"""

from repro_agent.observability.conclusion import (
    ConclusionInputs,
    determine_final_status,
    render_conclusion_text,
)
from repro_agent.observability.report import FinalReportGenerator, ReportInputs
from repro_agent.observability.result_query import (
    JobResult,
    JobResultIntegrityError,
    JobResultNotFoundError,
    JobResultService,
    VerifiedArtifact,
)

__all__ = [
    "ConclusionInputs",
    "FinalReportGenerator",
    "JobResult",
    "JobResultIntegrityError",
    "JobResultNotFoundError",
    "JobResultService",
    "ReportInputs",
    "VerifiedArtifact",
    "determine_final_status",
    "render_conclusion_text",
]
