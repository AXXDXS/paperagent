from __future__ import annotations

from types import SimpleNamespace

from repro_agent.agents.environment.agent import EnvironmentBuildAgent


def _build_result(*, cache_hit: bool, digest_char: str) -> dict:
    digest = "sha256:" + digest_char * 64
    return {
        "image_ref": digest,
        "image_digest": digest,
        "exit_code": 0,
        "stdout": "cached" if cache_hit else "built",
        "stderr": "",
        "cache_hit": cache_hit,
        "environment_fingerprint": digest_char * 64,
        "cache_ref": f"repro-agent/env-cache:{digest_char * 64}",
    }


def test_cached_image_that_fails_smoke_test_is_force_rebuilt() -> None:
    agent = object.__new__(EnvironmentBuildAgent)
    agent.task = SimpleNamespace(
        task_id="environment-task",
        definition=SimpleNamespace(
            inputs={
                "repository_path": ".",
                "dependencies_hint": "",
                "base_image": "scratch",
            }
        ),
    )
    agent._attempt_id = "attempt-1"
    agent._read_dependency_files = lambda root: ({}, [])
    agent._analyze_dependencies = lambda root, hint, files: "none"
    agent._generate_lockfile = lambda files: ""
    agent._generate_import_smoke_test = lambda lockfile: "print('ok')\n"
    agent._generate_dockerfile = lambda base, analysis, wheels: "FROM scratch\n"
    agent._guarded_write_file = lambda path, content: None
    agent.write_json_output = lambda filename, payload: None
    agent.write_candidate_memory = lambda content: None

    build_results = [
        _build_result(cache_hit=True, digest_char="a"),
        _build_result(cache_hit=False, digest_char="b"),
    ]
    build_calls: list[bool] = []

    def build_image(
        *,
        image_tag: str,
        force_rebuild: bool = False,
        network_enabled: bool = False,
    ) -> dict:
        build_calls.append(force_rebuild)
        return build_results.pop(0)

    smoke_results = iter([False, True])
    agent._build_environment_image = build_image
    agent._run_import_smoke_test = lambda: next(smoke_results)

    result = agent.run()

    assert result.succeeded is True
    assert build_calls == [False, True]
    assert result.outputs["cache_hit"] is False
    assert result.outputs["cache_rebuilt"] is True
    assert result.outputs["environment_fingerprint"] == "b" * 64
    assert result.outputs["import_test_passed"] is True
