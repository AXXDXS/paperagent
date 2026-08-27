# ReproAgent P0 Reliability and Sandbox Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing partial orchestration scaffold into a fail-closed, tiered paper-reproduction workflow that produces verified, evidence-backed reports and never executes real agent commands directly on the host.

**Architecture:** Keep the existing scheduler, repositories, agents, tier gate, and SQLite storage. Add versioned result contracts, validated dependency binding, deterministic phase coordination, an attempt-scoped Docker execution boundary, and repository-backed report assembly. Mock execution remains available only for explicitly mock jobs and traverses the same workflow phases.

**Tech Stack:** Python 3.10+, dataclasses, SQLite, `subprocess` for invoking the fixed Docker CLI only, JSON Schema-compatible tool metadata, pytest, optional `pypdf` for PDF text extraction.

## Global Constraints

- Real experiment commands must use Docker and must never fall back to host execution.
- Formal experiment containers run with network disabled, read-only source/input mounts, read-only root filesystem, bounded CPU/memory/PIDs, and no inherited secrets.
- Every production behavior change starts with a failing regression test.
- Downstream tasks consume only validated dependency results.
- An attempt result is accepted only when its `attempt_id` is still active.
- Job completion comes only from conclusion calculation after verification.
- Mock runs are labelled mock and cannot produce a real reproduction conclusion.
- The target directory is not a Git work tree; replace commit steps with explicit test checkpoints and do not perform destructive cleanup in this phase.

---

## File structure

New focused modules:

- `repro_agent/schemas/results.py`: common result envelope, artifact references, parsing, and validation errors.
- `repro_agent/orchestrator/artifacts.py`: validated dependency result loading and input binding.
- `repro_agent/orchestrator/phases.py`: deterministic phase decisions and task creation.
- `repro_agent/execution/backend.py`: execution request/result protocols and resource policy.
- `repro_agent/execution/docker.py`: Docker availability, argv construction, execution, stop, and kill.
- `repro_agent/execution/mock.py`: deterministic execution for explicit mock jobs.
- `repro_agent/domain/verification.py`: persisted verification verdict and metric comparisons.
- `repro_agent/observability/assembler.py`: repository-backed report inputs and JSON report assembly.
- `repro_agent/paper_input.py`: separate text and PDF extraction.

Existing modules remain responsible for scheduling, persistence, task-specific analysis, and report rendering.

### Task 1: Preserve behavior-affecting state and attempt identity

**Files:**
- Modify: `repro_agent/domain/task.py`
- Modify: `repro_agent/domain/job.py`
- Modify: `repro_agent/domain/reflection.py`
- Modify: `repro_agent/storage/repository.py`
- Test: `tests/test_persistence_roundtrip.py`
- Test: `tests/test_liveness_and_termination.py`

**Interfaces:**
- Produces: `Task.active_attempt_id: str`, complete `to_dict()` round-trips, and separate push/pull heartbeat persistence.
- Consumes: existing `JobRepository`, `TaskRepository`, `ReflectionRepository`, and domain dataclasses.

- [ ] **Step 1: Write failing round-trip tests**

```python
def test_task_roundtrip_preserves_behavior_fields(database):
    task = make_task(liveness_grace_seconds=7, priority=9)
    task.active_attempt_id = "attempt_current"
    task.last_activity_signature = "log:42"
    task.heartbeat = Heartbeat(eta_seconds=12, reported_by="pull")
    repo = TaskRepository(database)
    repo.save(task)
    loaded = repo.get(task.task_id)
    assert loaded.definition.liveness_grace_seconds == 7
    assert loaded.definition.priority == 9
    assert loaded.active_attempt_id == "attempt_current"
    assert loaded.last_activity_signature == "log:42"
    assert loaded.heartbeat.eta_seconds == 12
    assert loaded.heartbeat.reported_by == "pull"
```

- [ ] **Step 2: Run the tests and verify field-loss failures**

Run: `python -m pytest tests/test_persistence_roundtrip.py tests/test_liveness_and_termination.py -q`

Expected: failures show missing `active_attempt_id`, priority/liveness fields, heartbeat ETA/source, job final status, or reflection fields.

- [ ] **Step 3: Add complete serialization and deserialization**

Add `active_attempt_id`, `last_push_heartbeat`, and `last_pull_heartbeat` to `Task`; preserve every field listed in the design specification. Deserialize reflection reports into `ReflectionReport` objects rather than dictionaries.

- [ ] **Step 4: Prevent pull probes from overwriting push heartbeats**

Change `MainAgent.get_subagent_status()` to update `last_pull_heartbeat` while leaving `last_push_heartbeat` intact. Make liveness freshness read only `last_push_heartbeat`.

- [ ] **Step 5: Run the focused and full regression suites**

Run: `python -m pytest tests/test_persistence_roundtrip.py tests/test_liveness_and_termination.py -q`

Run: `python -m pytest -q`

Expected: all tests pass.

### Task 2: Validate result envelopes and bind dependency artifacts

**Files:**
- Create: `repro_agent/schemas/results.py`
- Create: `repro_agent/orchestrator/artifacts.py`
- Modify: `repro_agent/orchestrator/validator.py`
- Modify: `repro_agent/agents/base.py`
- Modify: task-specific agents under `repro_agent/agents/`
- Test: `tests/test_result_contracts.py`
- Test: `tests/test_artifact_resolver.py`

**Interfaces:**
- Produces: `TaskResultEnvelope.from_file(path, expected_task_id, expected_attempt_id)`, `ArtifactResolver.resolve(task) -> dict[str, object]`, and task-specific payload validators.
- Consumes: task output directories, task repository, and sandbox manager.

- [ ] **Step 1: Write failing envelope validation tests**

```python
def test_result_envelope_rejects_stale_attempt(tmp_path):
    path = write_envelope(tmp_path, task_id="t1", attempt_id="old")
    with pytest.raises(ResultValidationError, match="attempt"):
        TaskResultEnvelope.from_file(path, expected_task_id="t1", expected_attempt_id="current")

def test_validator_rejects_empty_payload_for_specification(task, sandbox):
    write_valid_common_envelope(sandbox, task, payload={})
    result = OutputValidator(sandbox.manager).validate(task, agent_succeeded=True)
    assert not result.passed
```

- [ ] **Step 2: Run tests and verify the missing-contract failures**

Run: `python -m pytest tests/test_result_contracts.py -q`

Expected: import or validation failures because the envelope contract does not exist.

- [ ] **Step 3: Implement the common envelope and artifact hashing**

Implement dataclasses for `ArtifactReference` and `TaskResultEnvelope`. Reject unknown versions, identity mismatches, paths outside output, missing files, empty files, and digest mismatches.

- [ ] **Step 4: Implement task-specific payload validators**

Require meaningful fields for paper, code, resource, specification, environment, experiment, verification, reflection, and coding results. Keep optional fields explicit and preserve warnings.

- [ ] **Step 5: Write failing dependency binding tests**

```python
def test_specification_receives_validated_dependency_payloads(resolver, spec_task):
    resolved = resolver.resolve(spec_task)
    assert resolved["paper_findings"]["expected_metrics"] == {"accuracy": 0.91}
    assert resolved["code_findings"]["entry_points"] == ["train.py"]
    assert resolved["resource_findings"]["docker_available"] is True
```

- [ ] **Step 6: Implement `ArtifactResolver` and dispatcher integration**

Resolve dependency results immediately before sandbox staging. Preserve immutable task-definition inputs and pass a separate resolved-input mapping to the agent. Fail the task with `DEPENDENCY_ERROR` when a required validated result is absent.

- [ ] **Step 7: Run focused and full tests**

Run: `python -m pytest tests/test_result_contracts.py tests/test_artifact_resolver.py -q`

Run: `python -m pytest -q`

Expected: all tests pass.

### Task 3: Add deterministic phase coordination and tier gates

**Files:**
- Create: `repro_agent/orchestrator/phases.py`
- Create: `repro_agent/domain/verification.py`
- Modify: `repro_agent/orchestrator/main_agent.py`
- Modify: `repro_agent/orchestrator/planner.py`
- Modify: `repro_agent/evaluation/tier_gate.py`
- Modify: `repro_agent/orchestrator/reflection_controller.py`
- Modify: `repro_agent/storage/database.py`
- Modify: `repro_agent/storage/repository.py`
- Modify: `repro_agent/agents/verification/agent.py`
- Test: `tests/test_phase_coordinator.py`
- Test: `tests/test_reflection_loop.py`

**Interfaces:**
- Produces: `VerificationRecord`, `VerificationRepository`, `PhaseCoordinator.advance(job, tasks, runs, verifications, reflections) -> PhaseDecision`, and idempotent task creation keys.
- Consumes: repositories, `TierGate`, task factory, and validated result payloads.

- [ ] **Step 1: Write failing phase-transition tests**

```python
def test_environment_success_does_not_finish_job(coordinator, job, analysis_tasks):
    decision = coordinator.advance(job, analysis_tasks + [successful_environment_task()], [], [], [])
    assert decision.terminal_status is None
    assert decision.tasks_to_create[0].definition.inputs["tier"] == "static_check"

def test_full_run_requires_all_prior_tiers(coordinator, job):
    decision = coordinator.advance(job, [], runs_for("static_check", "unit_test"), [], [])
    assert all(t.definition.inputs.get("tier") != "full_experiment" for t in decision.tasks_to_create)
```

- [ ] **Step 2: Run tests and verify premature-completion failures**

Run: `python -m pytest tests/test_phase_coordinator.py -q`

Expected: failures because there is no phase coordinator and current job completion depends only on current task counts.

- [ ] **Step 3: Implement pure phase decisions and idempotent creation keys**

Add decisions for analysis, resources, specification, environment, all experiment tiers, verification, reflection/audit/repair/rerun, conclusion, blocked, and failed paths. Store a deterministic `creation_key` in task inputs so repeated main-loop steps cannot duplicate tasks.

- [ ] **Step 4: Integrate the coordinator into `MainAgent.step()`**

Replace `_update_job_status()` task-count completion with coordinator decisions. Persist created tasks and job status in a single logical transition, recording an audit event for each phase change.

- [ ] **Step 5: Persist strict verification records**

Add a `verification_records` table and repository. Store the source full-run ID, expected and observed metric names, comparisons, missing metrics, execution-evidence verdict, provenance verdict, anti-cheat verdict, gap fingerprint, and completion timestamp. Make the verification agent fail its task when required metrics or evidence are missing instead of returning a successful empty comparison.

- [ ] **Step 6: Make reflection idempotent and fully bound**

Compute a stable gap fingerprint from verification comparisons. Include source paper/code/resource/specification/run/verification identifiers in audit and repair tasks. Create minimum reruns only with a concrete tier, command, image digest, and parent run.

- [ ] **Step 7: Run phase, reflection, and full tests**

Run: `python -m pytest tests/test_phase_coordinator.py tests/test_reflection_loop.py -q`

Run: `python -m pytest -q`

Expected: all tests pass.

### Task 4: Enforce the Docker execution boundary

**Files:**
- Create: `repro_agent/execution/backend.py`
- Create: `repro_agent/execution/docker.py`
- Create: `repro_agent/execution/mock.py`
- Modify: `repro_agent/execution/__init__.py`
- Modify: `repro_agent/tools/write_tools.py`
- Modify: `repro_agent/tools/registry.py`
- Modify: `repro_agent/orchestrator/dispatcher.py`
- Modify: `repro_agent/agents/experiment/agent.py`
- Modify: `repro_agent/agents/environment/agent.py`
- Test: `tests/test_docker_execution.py`
- Test: `tests/test_execution_fail_closed.py`

**Interfaces:**
- Produces: `ExecutionRequest`, `ExecutionResourcePolicy`, `ExecutionResult`, `ExecutionBackend`, `DockerExecutionBackend`, and `MockExecutionBackend`.
- Consumes: resolved experiment specification, task sandbox paths, and explicit mock-job configuration.

- [ ] **Step 1: Write failing fail-closed and argv tests**

```python
def test_real_execution_blocks_when_docker_is_missing(monkeypatch, request):
    backend = DockerExecutionBackend(docker_binary="missing-docker")
    with pytest.raises(ExecutionUnavailable):
        backend.execute(request)

def test_docker_argv_is_offline_read_only_and_bounded(backend, request):
    argv = backend.build_run_argv(request)
    assert ["--network", "none"] == adjacent_pair(argv, "--network")
    assert "--read-only" in argv
    assert "--memory" in argv and "--cpus" in argv and "--pids-limit" in argv
    assert all("docker.sock" not in value for value in argv)
```

- [ ] **Step 2: Run tests and verify missing-backend failures**

Run: `python -m pytest tests/test_docker_execution.py tests/test_execution_fail_closed.py -q`

Expected: imports fail because execution backends do not exist.

- [ ] **Step 3: Implement execution dataclasses and Docker argv construction**

Validate mount containment and allowlisted environment variables. Build Docker CLI argv with network none, read-only root/input, writable attempt workspace/output, bounded tmpfs, memory, CPU, PID, no-new-privileges, dropped capabilities, non-root user configuration, and deterministic container name.

- [ ] **Step 4: Implement Docker availability, execution, logs, stop, and kill**

Invoke only fixed Docker CLI operations. Capture container identifier and image digest. On timeout, issue stop, wait for terminal state, issue kill if required, collect logs, and return an explicit termination reason.

- [ ] **Step 5: Remove host arbitrary-command authorization**

Replace the `execute_command` tool handler with a backend-bound command tool. Real jobs receive only the Docker backend. Explicit mock jobs receive only `MockExecutionBackend`. Environment image preparation uses a separate audited operation and does not authorize network for experiment containers.

- [ ] **Step 6: Persist experiment run records**

Convert each execution result into `ExperimentRun`, including attempt identity, argv, image digest, source/data/config digests, resources, timestamps, exit code, logs, metrics, artifact hashes, and mock flag.

- [ ] **Step 7: Run focused, default, and optional Docker tests**

Run: `python -m pytest tests/test_docker_execution.py tests/test_execution_fail_closed.py -q`

Run: `python -m pytest -q`

When Docker is available, run: `python -m pytest -m docker -q`

Expected: all applicable tests pass; missing Docker produces a tested blocked result rather than host execution.

### Task 5: Make task attempts, timeout handling, and the main loop race-safe

**Files:**
- Modify: `repro_agent/orchestrator/dispatcher.py`
- Modify: `repro_agent/orchestrator/main_agent.py`
- Modify: `repro_agent/scheduler/scheduler.py`
- Modify: `repro_agent/scheduler/lease.py`
- Modify: `repro_agent/scheduler/timeout_policy.py`
- Modify: `repro_agent/context/snapshot.py`
- Test: `tests/test_attempt_races.py`
- Test: `tests/test_main_loop_outcome.py`

**Interfaces:**
- Produces: `RunLoopOutcome`, attempt-scoped handles, stale-result rejection, bounded event waiting, and state-change snapshotting.
- Consumes: execution backend cancellation, task repository, and phase coordinator.

- [ ] **Step 1: Write failing stale-attempt and incomplete-outcome tests**

```python
def test_old_attempt_result_cannot_complete_retried_task(agent, task):
    task.active_attempt_id = "new"
    accepted = agent.accept_attempt_result(task, attempt_id="old", result=successful_result())
    assert accepted is False
    assert task.status == TaskStatus.RUNNING

def test_iteration_limit_returns_incomplete(agent):
    outcome = agent.run_until_finished(max_iterations=1)
    assert outcome.completed is False
    assert outcome.reason == "iteration_limit"
```

- [ ] **Step 2: Run tests and verify race/outcome failures**

Run: `python -m pytest tests/test_attempt_races.py tests/test_main_loop_outcome.py -q`

Expected: failures because attempt identity and structured run-loop outcomes do not exist.

- [ ] **Step 3: Make dispatch and collection attempt-scoped**

Generate an attempt ID before dispatch, include it in the sandbox and handle, and compare it before validation or state mutation. Archive late results and record a `stale_attempt_result_rejected` event.

- [ ] **Step 4: Couple timeout retries to backend termination**

Do not make a timed-out task retryable until its execution backend reports terminal process/container state. Release the lease only after termination or record terminal failure if termination cannot be confirmed.

- [ ] **Step 5: Add bounded waiting and state-change snapshots**

Use a completion event with a small configurable poll timeout. Calculate a decision-state fingerprint and save a snapshot only when the fingerprint changes. Enforce snapshot retention.

- [ ] **Step 6: Return structured main-loop outcomes**

Return completed, terminal status, reason, and iteration count. Treat `VERIFIED_REPRODUCTION_GAP` as terminal. Do not mutate success on limits.

- [ ] **Step 7: Run focused and full tests**

Run: `python -m pytest tests/test_attempt_races.py tests/test_main_loop_outcome.py tests/test_async_dispatch_and_validation.py -q`

Run: `python -m pytest -q`

Expected: all tests pass.

### Task 6: Make OpenAI-compatible tools valid and bounded

**Files:**
- Modify: `repro_agent/tools/base.py`
- Modify: `repro_agent/tools/registry.py`
- Modify: `repro_agent/tools/authorization.py`
- Modify: `repro_agent/providers/base.py`
- Modify: `repro_agent/providers/openai_compatible.py`
- Modify: `repro_agent/agents/base.py`
- Test: `tests/test_openai_tool_protocol.py`
- Test: `tests/test_tool_authorization.py`

**Interfaces:**
- Produces: `ToolSpec.to_openai_tool()`, JSON Schema argument validation, finite tool-call loop, cancellation checks, and redacted audit arguments.
- Consumes: existing tool registry, provider response parser, and task risk budgets.

- [ ] **Step 1: Write failing protocol and argument tests**

```python
def test_tool_spec_emits_openai_function_schema(read_file_spec):
    tool = read_file_spec.to_openai_tool()
    assert tool["type"] == "function"
    assert tool["function"]["parameters"]["additionalProperties"] is False

def test_unauthorized_or_invalid_arguments_never_reach_handler(auth):
    with pytest.raises(ToolPermissionError):
        auth.invoke("read_file", path="x", unexpected=True)
```

- [ ] **Step 2: Run tests and verify schema failures**

Run: `python -m pytest tests/test_openai_tool_protocol.py tests/test_tool_authorization.py -q`

Expected: failures because tools expose internal metadata rather than function schemas and arguments are not schema-validated.

- [ ] **Step 3: Add schemas to every registered tool**

Define required properties, types, enum constraints, and `additionalProperties: false`. Keep risk metadata internal and send only OpenAI-supported fields to the provider.

- [ ] **Step 4: Validate and redact tool calls**

Validate arguments before authorization and execution. Enforce `forbidden_actions`, maximum calls, network policy, and risk budgets. Redact values whose keys match secret/token/password/key patterns in audit events.

- [ ] **Step 5: Add a finite tool-call loop with cancellation**

Check cancellation before and after each model/tool call. Append tool results with the correct `tool_call_id`. Stop at the task tool-call budget and return a structured failure instead of looping indefinitely.

- [ ] **Step 6: Run focused and full tests**

Run: `python -m pytest tests/test_openai_tool_protocol.py tests/test_tool_authorization.py tests/test_llm_assisted_decision.py -q`

Run: `python -m pytest -q`

Expected: all tests pass.

### Task 7: Assemble evidence-backed reports and truthful CLI outcomes

**Files:**
- Create: `repro_agent/observability/assembler.py`
- Modify: `repro_agent/observability/report.py`
- Modify: `repro_agent/observability/conclusion.py`
- Modify: `repro_agent/cli/main.py`
- Test: `tests/test_report_assembler.py`
- Test: `tests/test_cli_outcomes.py`
- Test: `tests/test_final_report_parameters_section.py`

**Interfaces:**
- Produces: `ReportAssembler.build(job_id) -> ReportInputs`, `ReportAssembler.to_json(inputs) -> dict`, and distinct CLI exit codes.
- Consumes: all repositories and `RunLoopOutcome`.

- [ ] **Step 1: Write failing report and CLI tests**

```python
def test_report_assembler_includes_runs_metrics_and_evidence(assembler, completed_job):
    inputs = assembler.build(completed_job.job_id)
    assert inputs.experiment_runs
    assert inputs.metric_comparisons
    assert inputs.evidence_records

def test_cli_does_not_report_completion_at_iteration_limit(run_cli):
    result = run_cli("run", "--mock", "--max-iterations", "1")
    assert result.exit_code != 0
    assert "完成" not in result.stdout
```

- [ ] **Step 2: Run tests and verify empty-report/false-success failures**

Run: `python -m pytest tests/test_report_assembler.py tests/test_cli_outcomes.py -q`

Expected: failures because CLI passes only a job object and always prints completion.

- [ ] **Step 3: Implement repository-backed report assembly**

Load specification, experiment runs, persisted verification records and comparisons, evidence, reflections, events, unresolved issues, and budgets. Render safe Markdown and write a complete JSON representation with explicit mock/real status.

- [ ] **Step 4: Centralize final status calculation**

Use `determine_final_status` only after verification. Correct descriptions for environment-ready, pipeline-only, full-run-completed, fully reproduced, partially reproduced, and verified gap outcomes.

- [ ] **Step 5: Make CLI exit status truthful**

Use the structured run-loop outcome. Return distinct non-zero codes for incomplete, blocked, failed, and cancelled jobs. Write diagnostic partial reports without claiming completion.

- [ ] **Step 6: Run focused and full tests**

Run: `python -m pytest tests/test_report_assembler.py tests/test_cli_outcomes.py tests/test_final_report_parameters_section.py -q`

Run: `python -m pytest -q`

Expected: all tests pass.

### Task 8: Handle PDF inputs explicitly and complete end-to-end verification

**Files:**
- Create: `repro_agent/paper_input.py`
- Modify: `repro_agent/agents/paper/agent.py`
- Modify: `pyproject.toml`
- Create: `README.md`
- Modify: `docs/IMPLEMENTATION_NOTES.md`
- Test: `tests/test_paper_input.py`
- Test: `tests/test_end_to_end_p0.py`

**Interfaces:**
- Produces: `extract_paper_text(path) -> PaperText`, explicit PDF input errors, and a complete mock end-to-end workflow.
- Consumes: text files, `pypdf.PdfReader`, phase coordinator, mock execution backend, verifier, and report assembler.

- [ ] **Step 1: Write failing text/PDF input tests**

```python
def test_image_only_pdf_fails_explicitly(image_only_pdf):
    with pytest.raises(PaperInputError, match="extractable text"):
        extract_paper_text(image_only_pdf)

def test_text_input_preserves_source_reference(tmp_path):
    paper = tmp_path / "paper.txt"
    paper.write_text("method", encoding="utf-8")
    extracted = extract_paper_text(paper)
    assert extracted.text == "method"
    assert extracted.references == ["paper.txt:1"]
```

- [ ] **Step 2: Run tests and verify binary-text handling failures**

Run: `python -m pytest tests/test_paper_input.py -q`

Expected: failures because there is no format-specific paper reader.

- [ ] **Step 3: Implement text and PDF readers**

Use `pypdf` page extraction for PDFs, retain page references, report encrypted/malformed/image-only files, and record truncation warnings. Do not add OCR in this phase.

- [ ] **Step 4: Write the failing full mock workflow test**

```python
def test_mock_job_reaches_verified_report(tmp_path, sample_repo, sample_paper):
    result = run_mock_job(tmp_path, sample_repo, sample_paper)
    assert result.outcome.completed is True
    assert result.job.status in {JobStatus.FULLY_REPRODUCED, JobStatus.VERIFIED_REPRODUCTION_GAP}
    assert result.markdown_report.stat().st_size > 0
    report = json.loads(result.json_report.read_text())
    assert report["runs"] and report["verification"]
    assert report["execution_mode"] == "mock"
```

- [ ] **Step 5: Wire the complete mock workflow through production boundaries**

Construct `ExperimentRunRepository`, `VerificationRepository`, `ReflectionRepository`, `ArtifactResolver`, `PhaseCoordinator`, and `ReportAssembler` in `MainAgent`. Select `MockExecutionBackend` only when the job configuration is explicitly mock; otherwise construct `DockerExecutionBackend`. Persist mock run records after every tier, persist the verification record after verification validation, invoke the coordinator once per accepted result, and assemble both reports after conclusion calculation.

- [ ] **Step 6: Update package and operating documentation**

Document installation, Docker requirements, offline experiment policy, optional image-build networking, CLI exit codes, mock limitations, reports, and resume behavior. Remove dependency declarations only when import and test evidence proves they are unused; keep `pypdf` as the explicit PDF dependency.

- [ ] **Step 7: Run all verification commands**

Run: `python -m pytest -q`

Run: `python -m compileall -q repro_agent tests`

Run: `python -m repro_agent.cli.main run --mock --paper-path <fixture-paper> --repository-path <fixture-repo> --target-experiment main --work-dir <temporary-directory> --max-iterations 500`

When Docker is available, run: `python -m pytest -m docker -q`

Expected: unit and mock end-to-end suites pass, compilation succeeds, mock CLI reaches a truthful terminal report, and Docker conformance tests pass where Docker is present.

---

## Plan completion criteria

- Every task above has a recorded red test, green focused test, and green full regression test.
- Real command execution cannot reach host `subprocess.run` except for fixed Docker CLI management operations.
- A successful real conclusion requires a persisted full run and verification evidence.
- Iteration exhaustion, missing Docker, failed validation, and stale attempts cannot be reported as success.
- Markdown and JSON reports are populated from persisted records.
- The final code and documentation agree with the approved design specification.
