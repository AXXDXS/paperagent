"""Main-agent-only tool for accurate retrieval of one persisted job result."""

from __future__ import annotations

from typing import Any

from repro_agent.observability.result_query import JobResultService


class GetJobResultTool:
    name = "get_job_result"
    description = (
        "Load the current job's persisted result, verify active-attempt artifacts "
        "against their SHA-256 evidence, and return job-scoped report paths."
    )

    def __init__(self, service: JobResultService, *, current_job_id: str):
        self._service = service
        self._current_job_id = current_job_id

    @classmethod
    def to_openai_tool(cls) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Exact job id bound to the current main agent.",
                        }
                    },
                    "required": ["job_id"],
                    "additionalProperties": False,
                },
            },
        }

    def call(self, *, job_id: str) -> dict[str, Any]:
        if job_id != self._current_job_id:
            raise PermissionError(
                "get_job_result is bound to the current main-agent job and cannot query another job"
            )
        return self._service.get(job_id).to_dict()
