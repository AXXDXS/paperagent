# ReproAgent P0 Reliability and Sandbox Design

## 1. Status and scope

This specification defines the first repair phase for `paper_agent`. It covers every P0 issue confirmed during the code audit:

- the workflow ends after environment preparation instead of running and verifying experiments;
- dependency tasks control ordering but do not pass validated data to their consumers;
- the CLI reports unfinished jobs as complete and produces an empty final report;
- the main loop busy-spins and writes snapshots without state changes;
- host subprocess execution is presented as a sandbox even though it does not isolate the host, environment, or network;
- timed-out attempts can continue running and race with retries;
- task output validation checks file presence instead of schema and semantics;
- OpenAI-compatible function tool definitions do not follow the API schema;
- reflection and minimum rerun tasks lack required inputs and idempotency;
- persistence drops state required for correct recovery;
- PDF inputs can be silently treated as text bytes.

The chosen safety policy is **Docker fail-closed**. Real experiment commands must run in a Docker container. If Docker or the required image is unavailable, the job becomes blocked; the system must never silently fall back to arbitrary host command execution.

This phase does not introduce a distributed workflow service, a web UI, remote workers, gVisor, Kubernetes, MLflow, or broad repository cleanup. Generated caches and dead files will be handled in a later cleanup phase after the repaired workflow is verified.

## 2. Chosen approach

The implementation will incrementally close the existing architecture rather than replacing it.

Existing SQLite repositories, the task scheduler, task-specific agents, the tier gate, evidence models, reflection controller, and report generator remain the foundation. The repair adds focused boundaries around them:

1. `ArtifactResolver` binds validated dependency results into downstream task inputs.
2. `PhaseCoordinator` creates the next task phase only after the current phase satisfies explicit gates.
3. `ExecutionBackend` separates command intent from command execution.
4. `DockerExecutionBackend` is the only backend authorized for real commands.
5. Versioned result envelopes and task-specific validators replace file-existence-only success.
6. Attempt-scoped execution and compare-before-accept logic prevent stale results from overwriting current state.
7. A report assembler loads persisted domain records before producing Markdown and JSON reports.

This approach limits churn while fixing the observed failure modes. A complete state-machine rewrite or Temporal migration would improve long-term maintainability but would expand this repair beyond a single testable phase.

## 3. Target workflow

The repaired workflow is:

```text
PAPER_ANALYSIS + CODE_ANALYSIS
                |
                v
         RESOURCE_CHECK
                |
                v
          SPECIFICATION
                |
                v
      ENVIRONMENT_PREPARATION
                |
                v
 STATIC_CHECK -> SMOKE_RUN -> REDUCED_RUN -> FULL_RUN
       each transition is authorized by TierGate
                |
                v
           VERIFICATION
                |
       +--------+---------+
       |                  |
  within tolerance   verified gap/error
       |                  |
       |        REFLECTION -> AUDIT -> REPAIR
       |                         -> MINIMUM RERUN
       +------------------+---------+
                          v
                      CONCLUSION
                          |
                          v
              MARKDOWN + JSON REPORT
```

`PhaseCoordinator` owns phase transitions. It is deterministic and does not call the LLM. A phase transition is a pure decision based on persisted task results, experiment runs, verification records, reflection budgets, and terminal job state.

The scheduler continues to decide which ready tasks can run concurrently. It no longer decides that a job is complete merely because all currently known tasks succeeded.

## 4. Phase gates

### 4.1 Analysis to specification

Paper, code, and resource tasks must each produce a valid versioned result envelope. `ArtifactResolver` loads their validated payloads and supplies the specification task with:

- paper parameters, expected metrics, experimental claims, datasets, models, and unresolved paper ambiguities;
- repository entry points, configuration sources, supported commands, code-level defaults, and commit or tree digest;
- available datasets, models, checkpoints, Docker availability, accelerator information, and blocking resource issues.

Missing required inputs block specification creation. Optional missing information is represented explicitly as an unresolved field; it is never converted into an empty successful result.

### 4.2 Specification to environment

The specification validator requires an experiment identifier, executable command intent, parameter provenance, expected metrics, comparison tolerances, seed policy, dataset references, model references, and resource requirements.

Environment preparation selects or builds a Docker image and records its immutable digest. A comment-only dependency lockfile, a dry-run install, or a single unrelated import test cannot establish environment readiness.

### 4.3 Tiered experiment execution

Experiment tiers are `STATIC_CHECK`, `SMOKE_RUN`, `REDUCED_RUN`, and `FULL_RUN`. Each successful run is persisted as an `ExperimentRunRecord`. `TierGate` authorizes only the immediately following tier and checks the preceding record for execution success, required artifacts, and tier-specific criteria.

Mock mode may use a dedicated deterministic mock execution backend. It must still traverse every phase and produce structurally valid run and verification records. Mock mode is visibly labelled in all reports and cannot be classified as a real reproduction.

### 4.4 Verification, reflection, and conclusion

Verification requires at least one real full-run record for a real reproduction conclusion. It checks metric completeness, tolerance comparisons, provenance completeness, artifact digests, execution evidence, and anti-cheat rules.

Reflection is keyed by `(verification_id, gap_fingerprint)`. The same verified gap cannot create multiple concurrent or repeated reflection rounds. Audit and repair tasks receive the original validated paper, repository, resource, specification, run, and verification inputs through `ArtifactResolver`.

Only a confirmed repair can create a minimum-scope rerun. The rerun contains a concrete command, tier, image digest, resource policy, and parent run identifier.

Conclusion calculation is the single authority for final reproduction status.

## 5. Data contracts

### 5.1 Task result envelope

Every successful agent task writes `output/result.json` using this logical contract:

```json
{
  "schema_version": 1,
  "task_id": "task_...",
  "attempt_id": "attempt_...",
  "task_type": "paper_analysis",
  "outcome": "succeeded",
  "payload": {},
  "artifacts": [
    {
      "path": "output/result.json",
      "size_bytes": 123,
      "sha256": "..."
    }
  ],
  "evidence_refs": [],
  "warnings": []
}
```

The common validator checks envelope identity, active attempt identity, schema version, JSON structure, artifact containment, file size, and SHA-256. A task-specific validator then checks the payload. Unknown schema versions fail closed.

### 5.2 Experiment run record

Every command execution records:

- `run_id`, `task_id`, `attempt_id`, experiment identifier, and tier;
- exact argv without shell interpolation;
- container image name and immutable image digest;
- repository commit when available and repository tree digest otherwise;
- dataset, model, checkpoint, configuration, and source artifact digests;
- seed, resolved parameters, CPU, memory, PID, GPU, timeout, and network policy;
- start and completion timestamps, duration, exit code, and termination reason;
- stdout/stderr artifact references, parsed metrics, output artifacts, and their digests;
- whether the run is mock or real.

Verification must reject a real-run claim when required provenance is absent.

### 5.3 Persistence completeness

Serialization round-trips must preserve all behavior-affecting fields, including:

- job final reproduction status and budget counters;
- task liveness grace, priority, last activity signature, current attempt, and lease fields;
- heartbeat ETA and `reported_by` source;
- reflection likely source, repair task identifiers, rerun status, and gap fingerprint.

Schema migration is explicit: the database stores a schema version and upgrades older rows with deterministic defaults. A newer unsupported version is rejected rather than partially loaded.

## 6. Docker execution boundary

### 6.1 Interface

Agents produce an `ExecutionRequest`; they do not call `subprocess.run` directly. The request contains argv, image reference, mounts, environment allowlist, working directory, timeout, and resource policy. `ExecutionBackend.execute(request)` returns an `ExecutionResult` and a persisted run record.

The production configuration registers only `DockerExecutionBackend`. Host execution of arbitrary agent-generated commands is removed from the authorized tool path. Tests may use `MockExecutionBackend` with an explicit mock job flag.

### 6.2 Container policy

A real experiment container uses these defaults:

- `--network none`;
- `--read-only` root filesystem;
- source/input mounts read-only;
- attempt-specific workspace and output mounts read-write;
- `tmpfs` for `/tmp` with a bounded size;
- explicit memory, CPU, PID, and process timeout limits;
- `no-new-privileges` and capability drop where supported;
- no host Docker socket, SSH agent, cloud credentials, home directory, or unrestricted environment inheritance;
- an allowlist of non-secret environment variables;
- a non-root container user when the image supports it.

Paths are resolved and checked before constructing Docker argv. Commands are always passed as an argument vector and never through `shell=True`.

### 6.3 Image preparation and network

Formal experiment runs are always offline. Environment image preparation is a separate phase:

- the default path selects a locally available image by digest and uses local dependency caches;
- optional build networking requires an explicit configuration flag, is recorded in provenance, receives no experiment secrets, and does not authorize network access for later experiment runs;
- if the image cannot be prepared under the configured policy, the job becomes `BLOCKED_BY_MISSING_RESOURCE`.

### 6.4 Cancellation and timeout

Each attempt receives a unique container name or identifier. On cancellation or timeout, the backend sends `docker stop` with a bounded grace period and then `docker kill` if necessary. It waits for terminal container state, captures final logs, records the termination reason, and only then permits a retry.

Python thread cancellation is not treated as process termination. A stale attempt result is archived for audit but cannot update the active task.

## 7. Liveness, retries, and main-loop behavior

Push heartbeat state and pull probe state are stored separately. A pull probe must never replace the timestamp or payload of the most recent push heartbeat.

Retries create a new `attempt_id`, sandbox directory, container, lease, and result namespace. Result acceptance uses the active attempt identifier and terminal task state as a compare-before-accept guard.

The main loop waits for a task-completion/heartbeat event or a bounded poll interval. It saves a context snapshot only when persisted decision state changes. Snapshot retention has a configurable limit.

`run_until_finished` returns a structured outcome. Reaching the iteration or wall-clock limit produces an incomplete outcome; it does not mutate the job into a successful state.

Terminal job states are:

- `FULLY_REPRODUCED`;
- `VERIFIED_REPRODUCTION_GAP`;
- `FAILED`;
- `BLOCKED_BY_MISSING_RESOURCE`;
- `CANCELLED`.

`USER_REPORT_READY` is a reporting phase, not proof of successful reproduction.

## 8. OpenAI-compatible tools

Tool descriptions sent to Chat Completions use the official function tool structure:

```json
{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "Read a staged input file",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": [],
      "additionalProperties": false
    },
    "strict": true
  }
}
```

Each tool owns a JSON Schema. Tool-call arguments are parsed and validated before authorization and execution. Unknown fields, missing required fields, malformed JSON, unauthorized tools, and risk-budget violations are rejected and audited with redacted arguments.

The LLM interaction loop has a finite tool-call budget and cancellation checks before and after every model and tool call.

## 9. PDF input handling

Text and PDF inputs are handled by separate readers. PDF input is parsed page by page with page references retained in extracted evidence. Empty, encrypted, image-only, or malformed PDFs produce explicit input errors unless a supported OCR path is configured. This phase does not add OCR.

The paper payload records extraction warnings and truncation. Required sections or target claims lost to truncation prevent the paper-analysis task from being treated as fully valid.

## 10. Reporting and CLI behavior

`ReportAssembler` loads the job, resolved specification, experiment runs, metric comparisons, provenance verdict, reflection reports, unresolved issues, and budget usage from repositories.

It produces:

- `final_report.md` for people;
- `final_report.json` for automation.

Both reports identify mock versus real execution and distinguish complete, failed, blocked, cancelled, and incomplete outcomes.

The CLI returns zero only for a terminal, successfully processed job outcome. Failed, blocked, cancelled, and incomplete runs return non-zero codes with distinct diagnostic messages. It never prints “complete” after an iteration-limit exit.

## 11. Validation and acceptance tests

Implementation follows red-green-refactor. Required regression coverage includes:

1. validated upstream payloads appear in specification inputs;
2. environment success alone cannot finish a job;
3. a mock end-to-end job traverses all phases and produces populated Markdown and JSON reports;
4. iteration or wall-clock exhaustion returns an incomplete, non-zero CLI result;
5. malformed, empty, mismatched-attempt, or semantically incomplete outputs fail validation;
6. persistence round-trips every behavior-affecting Job, Task, Heartbeat, ExperimentRun, and Reflection field;
7. a pull probe does not replace the last push heartbeat;
8. a stale attempt cannot overwrite a current attempt;
9. missing Docker blocks real execution without host fallback;
10. Docker argv includes offline networking, read-only inputs/root, bounded resources, and security options;
11. a Docker timeout stops or kills the attempt container before retry becomes eligible;
12. OpenAI tool descriptions match the function-tool schema and invalid arguments are rejected;
13. a PDF with no extractable text fails explicitly;
14. verification rejects missing metrics, provenance, logs, or artifacts;
15. one verification gap creates at most one active reflection for the same fingerprint.

Unit and mock end-to-end tests run in the default suite. Docker conformance tests use a `docker` marker and run when Docker is available. The product behavior itself remains fail-closed when Docker is unavailable.

The repaired default test suite must pass without warnings attributable to the project. The existing 31-test baseline remains green, except where an existing test encoded the defective behavior and is intentionally replaced by the new contract.

## 12. Documentation and compatibility

The implementation updates the CLI help, package README, implementation notes, status descriptions, and configuration examples. It documents Docker as mandatory for real execution and explains the separate opt-in image-build network policy.

Backward compatibility is maintained for reading existing jobs where deterministic migration is possible. Existing incomplete jobs are not automatically labelled successful; they resume from the first unmet phase gate or become blocked with a diagnostic reason.

## 13. Evidence for the design

- PaperBench supports hierarchical, independently gradable reproduction criteria: <https://arxiv.org/abs/2504.01848>.
- ReAct supports interleaving reasoning with externally observed actions: <https://arxiv.org/abs/2210.03629>.
- Reflexion supports reflection grounded in explicit feedback and episodic memory: <https://arxiv.org/abs/2303.11366>.
- The MAST taxonomy identifies specification, inter-agent alignment, verification, and termination as major multi-agent failure categories: <https://arxiv.org/abs/2503.13657>.
- Temporal's architecture motivates deterministic orchestration, durable event history, and idempotent side-effecting activities: <https://github.com/temporalio/temporal/blob/main/docs/architecture/README.md>.
- Docker documents offline networking and resource constraints: <https://docs.docker.com/engine/network/drivers/none/> and <https://docs.docker.com/engine/containers/resource_constraints/>.
- SLSA defines provenance in terms of builder, process, inputs, and artifact identity: <https://slsa.dev/spec/v1.2/provenance>.
- OpenAI documents the function tool envelope and JSON Schema parameters: <https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create>.

